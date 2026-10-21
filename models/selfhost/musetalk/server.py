#!/usr/bin/env python3
"""MuseTalk 1.5 (Tencent Music / TMElyralab, real-time mode) behind the StreamEval self-host protocol
(docs/protocol.md).

Composition system: the shared client's composition layer runs ASR -> LLM -> TTS and streams the TTS speech to
this server as `audio_in` (s16le 16 kHz, 10 ms blocks, real-time pace, silence between turns).

MuseTalk is a lip-sync model: it re-renders only the lower half of the face of a reference video. The benchmark
gives one portrait, so the avatar is prepared exactly as upstream's real-time script prepares an avatar
(`scripts/realtime_inference.py`, `Avatar.prepare_material`, v1.5 settings) from a one-frame reference video made
of the portrait: DWPose face landmarks + S3FD face box -> crop box (extra_margin 10), VAE latents of the masked /
reference crop, BiSeNet face-parsing blend mask ("jaw" mode, cheek width 90). Every output frame is that portrait
with a newly generated mouth region pasted back, so there is NO head motion, blinking or expression change: only
the lips and jaw move.

* `ready` after the avatar is prepared (audio_in keeps being read meanwhile);
* audio timeline indexed by wall clock since `ready` (first `audio_in` anchors the stream pts);
* 5 frames (200 ms) per step; frame f uses upstream's whisper window (features [2f-4, 2f+6) at 50 Hz, i.e. audio
  up to 80 ms after the frame), computed with whisper-tiny over the last <= 10 s of received audio (left-aligned in
  the 30 s encoder window as upstream does for a clip shorter than 30 s); the step waits in real time until that
  audio has arrived -> no pre-reading, no offline replay;
* UNet (single step, fp16) + VAE decode for the 5 frames, paste-back with the prepared mask;
* frames go out at 25 fps on an absolute clock, frame f paired with the 40 ms of speech [f*640, (f+1)*640) as
  `audio_out` (48 kHz); silence renders the closed / resting mouth continuously;
* text events are acked `applied`, reference/control `unsupported`;
* a portrait in which no face is found cannot be lip-synced: the server then streams the portrait unchanged with
  the speech (`animated: false` in ready.config and in the server log).

    python server.py --port 18793 --weights $STREAMEVAL_MODEL_ROOT/musetalk/weights
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
import types

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
from streameval.runtime import protocol as sp  # noqa: E402

MODEL = "musetalk"
SR = 16000
FPS = 25
SPF = SR // FPS            # 640
FEAT = 320                 # whisper feature hop (50 Hz)
STEP = 5                   # frames per render step (200 ms)
PAD_L = PAD_R = 2          # upstream audio_padding_length_left / right
EXTRA_MARGIN = 10          # upstream v1.5 default
DEFAULT_ROOT = os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/musetalk"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def to48k(x16: np.ndarray) -> bytes:
    n = len(x16)
    if n == 0:
        return b""
    y = np.interp(np.arange(3 * n) / 3.0, np.arange(n), x16.astype(np.float64))
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()


def read_hashes(path: Path) -> dict:
    if not path.exists():
        return {}
    return {p: h for h, p in (line.split(None, 1) for line in path.read_text().split("\n") if line.strip())}


def fit_portrait(image: Image.Image, max_side: int) -> Image.Image:
    """Center-crop to square (benchmark portraits already are) and downscale to max_side (never upscale)."""
    w, h = image.size
    s = min(w, h)
    image = image.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    if s > max_side:
        image = image.resize((max_side, max_side), Image.LANCZOS)
    return image


def stub_xtcocotools():
    """mmpose imports xtcocotools in dataset / evaluation modules that inference never runs; it has no wheel for
    this python, so register an inert stub when it is missing."""
    try:
        import xtcocotools  # noqa: F401
        return "installed"
    except Exception:  # noqa: BLE001
        for name in ("xtcocotools", "xtcocotools.coco", "xtcocotools.cocoeval", "xtcocotools.mask"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["xtcocotools.coco"].COCO = object
        sys.modules["xtcocotools.cocoeval"].COCOeval = object
        return "stub"


def stub_mmcv_ops():
    """DWPose's RTMPose config builds its CSPNeXt backbone from the mmdet registry; importing mmdet.models pulls in
    mmcv.ops (compiled CUDA/C++ extensions: NMS, RoIAlign, deformable conv, ...), which mmcv-lite does not ship
    and which upstream's mmcv 2.0.1 wheels do not provide for this torch / GPU. Top-down pose inference on the whole
    portrait never calls any of these ops (backbone = plain convolutions, head = RTMCC/SimCC), so serve every
    mmcv.ops.* module as a placeholder whose attributes raise if they are ever used."""
    import importlib.abc
    import importlib.machinery
    try:
        import mmcv._ext  # noqa: F401
        return "compiled"
    except Exception:  # noqa: BLE001
        pass

    class _Unavailable:
        def __init__(self, *a, **k):
            raise RuntimeError("mmcv.ops is not available (mmcv-lite); this op is not used by RTMPose inference")

        def __call__(self, *a, **k):
            raise RuntimeError("mmcv.ops is not available (mmcv-lite)")

    def _getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (_Unavailable,), {})

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "mmcv.ops" or fullname.startswith("mmcv.ops."):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            mod = types.ModuleType(spec.name)
            mod.__path__ = []
            mod.__getattr__ = _getattr
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _Finder())
    return "placeholder (mmcv-lite, ops unused by RTMPose inference)"


class Paster:
    """Upstream blending.get_image_blending (PIL: crop the expanded face box, paste the generated face, paste back
    through the face-parsing mask) restricted to the mask box, in numpy with PIL's exact mask blend
    out = DIV255(dst * (255 - m) + src * m); identical output, ~10x less CPU per frame at 1024 px."""

    def __init__(self, frame: np.ndarray, bbox, mask: np.ndarray, crop_box):
        x1, y1, x2, y2 = bbox
        xs, ys, xe, ye = crop_box
        h, w = frame.shape[:2]
        self.frame = frame
        self.face = (slice(y1 - ys, y2 - ys), slice(x1 - xs, x2 - xs))
        self.size = (x2 - x1, y2 - y1)
        large = np.zeros((ye - ys, xe - xs, 3), np.uint8)              # PIL crop pads outside the image with 0
        cy0, cy1, cx0, cx1 = max(ys, 0), min(ye, h), max(xs, 0), min(xe, w)
        self.dst = (slice(cy0, cy1), slice(cx0, cx1))
        self.src = (slice(cy0 - ys, cy1 - ys), slice(cx0 - xs, cx1 - xs))
        large[self.src] = frame[self.dst]
        self.large = large
        m = mask.astype(np.uint32)
        if m.ndim == 3:
            m = m[..., 0]
        self.m = m[self.src][..., None]

    def __call__(self, face: np.ndarray) -> np.ndarray:
        import cv2
        large = self.large.copy()
        large[self.face] = cv2.resize(face.astype(np.uint8), self.size)
        out = self.frame.copy()
        v = out[self.dst].astype(np.uint32) * (255 - self.m) + large[self.src].astype(np.uint32) * self.m + 128
        out[self.dst] = ((v + (v >> 8)) >> 8).astype(np.uint8)
        return out


class Model:
    def __init__(self, args):
        root = Path(args.root)
        repo = root / "repo"
        os.chdir(repo)                                  # upstream loads ./models/... and ./musetalk/... relative
        sys.path.insert(0, str(repo))
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")   # mmengine checkpoints (torch >= 2.6)
        self.xtcocotools = stub_xtcocotools()
        self.mmcv_ops = stub_mmcv_ops()
        import torch
        from transformers import WhisperModel
        t0 = time.time()
        from musetalk.utils.preprocessing import get_landmark_and_bbox     # loads DWPose + S3FD on import
        from musetalk.utils.blending import get_image_prepare_material, get_image_blending
        from musetalk.utils.utils import load_all_model
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing
        self.torch = torch
        self.get_landmark_and_bbox, self.prepare_material = get_landmark_and_bbox, get_image_prepare_material
        self.blend = get_image_blending
        self.device = torch.device("cuda:0")
        self.vae, self.unet, self.pe = load_all_model(unet_model_path="./models/musetalkV15/unet.pth", vae_type="sd-vae",
                                                      unet_config="./models/musetalkV15/musetalk.json", device=self.device)
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        self.timesteps = torch.tensor([0], device=self.device)
        self.audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        self.whisper = WhisperModel.from_pretrained("./models/whisper").to(device=self.device, dtype=torch.float16).eval()
        self.whisper.requires_grad_(False)
        self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        self.context = int(args.context_s * SR) // FEAT * FEAT
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(STEP)
        # every GPU call (warm-up, avatar preparation, rendering) runs on ONE long-lived thread: per-thread CUDA
        # library state (cuBLAS / cuDNN handles and autotuning) is then warmed once instead of costing ~1.3 s on the
        # first step of every session
        self.gpu = ThreadPoolExecutor(1)
        self.max_side = args.max_side
        log(f"models loaded in {time.time() - t0:.1f}s")
        commit = root / "repo_commit.txt"
        revision = root / "weights_revision.txt"
        import diffusers
        import transformers
        self.config = {
            "model": "MuseTalk 1.5 (real-time mode)", "code_commit": commit.read_text().strip() if commit.exists() else None,
            "weights_repo": revision.read_text().strip().split("\n") if revision.exists() else None,
            "weights_sha256": read_hashes(root / "weights_sha256.txt"), "version": "v15", "dtype": "fp16",
            "avatar": "upstream realtime Avatar preparation on a one-frame reference video = the portrait "
                      "(bbox_shift 0, extra_margin 10, parsing jaw, cheek width 90); no head motion, only the mouth "
                      "region is generated",
            "audio": {"encoder": "openai/whisper-tiny (all 5 hidden states)", "window_frames": "[2f-4, 2f+6) at 50 Hz",
                      "lookahead_ms": 80, "context_s": args.context_s, "step_frames": STEP},
            "fps": FPS, "max_side": self.max_side, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "diffusers": diffusers.__version__, "transformers": transformers.__version__, "xtcocotools": self.xtcocotools,
            "mmcv_ops": self.mmcv_ops,
        }
        self._warm()

    def _warm(self):
        t0 = time.time()
        img = fit_portrait(Image.open("assets/demo/musk/musk.png").convert("RGB"), self.max_side)
        tmp = Path(tempfile.mkdtemp(prefix="musetalk_")) / "warm.png"
        img.save(tmp)
        avatar = self.gpu.submit(self.prepare, str(tmp), 0).result()
        audio = np.zeros(SR * 3, np.float32)
        for k in range(0, 50, STEP):
            self.gpu.submit(self.render, avatar, audio, k).result()
        if avatar is not None:   # the numpy paste-back must equal upstream's PIL get_image_blending exactly
            import cv2
            face = cv2.resize(avatar["frame"][100:356, 100:356], (256, 256))
            x1, y1, x2, y2 = avatar["bbox"]
            ref = self.blend(avatar["frame"], cv2.resize(face, (x2 - x1, y2 - y1)), [x1, y1, x2, y2], avatar["mask"],
                             avatar["crop_box"])
            diff = int(np.abs(ref.astype(int) - avatar["paste"](face).astype(int)).max())
            self.config["paste_back"] = f"numpy twin of upstream get_image_blending (max abs diff {diff})"
            log("paste-back check: max abs diff vs upstream", diff)
        log(f"warm-up done in {time.time() - t0:.1f}s (face found: {avatar is not None})")

    def prepare(self, path: str, seed: int):
        """Upstream Avatar.prepare_material for a one-frame reference video; None when no face is found."""
        import cv2
        torch = self.torch
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        random.seed(seed)
        try:
            coord_list, frame_list = self.get_landmark_and_bbox([path], 0)
        except ZeroDivisionError:   # upstream averages landmark ranges over detected faces: none were detected
            return None
        bbox, frame = coord_list[0], frame_list[0]
        if bbox == (0.0, 0.0, 0.0, 0.0):
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        with torch.no_grad():
            latents = self.vae.get_latents_for_unet(crop)
        mask, crop_box = self.prepare_material(frame, [x1, y1, x2, y2], fp=self.fp, mode="jaw")
        return {"frame": frame, "bbox": (x1, y1, x2, y2), "latents": latents, "mask": mask, "crop_box": crop_box,
                "paste": Paster(frame, (x1, y1, x2, y2), mask, crop_box)}

    def render(self, avatar, timeline: np.ndarray, k: int, quality: int = 90) -> tuple[list[bytes], dict]:
        """Frames k..k+STEP-1 as JPEG from timeline[:e], e = (k+STEP)*640 + 1280 (80 ms look-ahead), plus the
        stage timings (s): whisper features, UNet + VAE, paste-back + JPEG (parallel over the frames)."""
        import cv2
        from einops import rearrange
        torch = self.torch
        t0 = time.monotonic()
        e = (k + STEP) * SPF + 2 * SPF
        s = max(0, e - self.context)
        segment = timeline[s:e]
        if len(segment) < e - s:
            segment = np.concatenate([segment, np.zeros(e - s - len(segment), np.float32)])
        with torch.no_grad():
            # log-mel on the GPU (transformers' torch STFT path, device="cuda"): the CPU path stalled steps when
            # the co-located client was busy
            feats = self.audio_processor.feature_extractor(segment, return_tensors="pt", sampling_rate=SR,
                                                           device="cuda").input_features
            hidden = self.whisper.encoder(feats.to(self.device, torch.float16), output_hidden_states=True).hidden_states
            hidden = torch.stack(hidden, dim=2)[0]                       # (1500, 5, 384), 50 Hz from s
            n_valid = (e - s) // FEAT
            hidden = hidden[:n_valid]
            base = s // FEAT                                             # global feature index of hidden[0]
            width = 2 * (PAD_L + PAD_R + 1)                              # 10 features per frame
            prompts = []
            for f in range(k, k + STEP):
                g0 = 2 * f - 2 * PAD_L                                   # global start (upstream left zero padding)
                idx = np.arange(g0, g0 + width) - base
                clip = torch.zeros((width, hidden.shape[1], hidden.shape[2]), dtype=hidden.dtype, device=self.device)
                ok = (idx >= 0) & (idx < n_valid)
                clip[torch.from_numpy(np.nonzero(ok)[0]).to(self.device)] = hidden[torch.from_numpy(idx[ok]).to(self.device)]
                prompts.append(clip)
            audio = rearrange(torch.stack(prompts), "b c h w -> b (c h) w")    # (STEP, 50, 384)
            audio = self.pe(audio)
            torch.cuda.synchronize()
            t1 = time.monotonic()
            latent = avatar["latents"].to(device=self.device, dtype=self.unet.model.dtype).repeat(STEP, 1, 1, 1)
            pred = self.unet.model(latent, self.timesteps, encoder_hidden_states=audio).sample
            recon = self.vae.decode_latents(pred.to(dtype=self.vae.vae.dtype))
        t2 = time.monotonic()
        x1, y1, x2, y2 = avatar["bbox"]

        def paste(res):
            return cv2.imencode(".jpg", avatar["paste"](res), [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()

        out = list(self.pool.map(paste, list(recon)))
        t3 = time.monotonic()
        return out, {"audio": t1 - t0, "unet_vae": t2 - t1, "paste_jpeg": t3 - t2}


class Session:
    def __init__(self, conn: sp.ServerConnection, model: Model, args, loop):
        self.conn, self.m, self.args, self.loop = conn, model, args, loop
        self.stop = threading.Event()
        self.lock = threading.Condition()
        self.timeline = np.zeros(0, dtype=np.float32)
        self.received_to = 0
        self.anchor = None
        self.t_ready = None
        self.avatar = None
        self.static_jpeg = None
        self.out: asyncio.Queue = asyncio.Queue()
        self.stats = {"steps": 0, "gen_s": [], "frames": 0}

    def audio_in(self, pts_ns: int, payload: bytes):
        pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        with self.lock:
            if self.t_ready is None:
                return
            if self.anchor is None:
                self.anchor = int(round((time.monotonic() - self.t_ready) * SR)) - int(pts_ns * SR // 1_000_000_000)
            start = self.anchor + int(pts_ns * SR // 1_000_000_000)
            end = start + len(pcm)
            if start < 0:
                pcm, start = pcm[-start:], 0
            if end > len(self.timeline):
                self.timeline = np.concatenate([self.timeline, np.zeros(max(end - len(self.timeline), SR * 10), np.float32)])
            self.timeline[start:end] = pcm
            self.received_to = max(self.received_to, end)
            self.lock.notify_all()

    def wait_audio(self, end: int) -> np.ndarray:
        deadline = self.t_ready + end / SR + self.args.grace
        with self.lock:
            while not self.stop.is_set() and self.received_to < end:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self.lock.wait(min(left, 0.01))
            if len(self.timeline) < end:
                self.timeline = np.concatenate([self.timeline, np.zeros(end - len(self.timeline) + SR * 10, np.float32)])
            return self.timeline[:end].copy()

    def render(self):
        k = 0
        try:
            while not self.stop.is_set():
                tl = self.wait_audio((k + STEP) * SPF + 2 * SPF)
                if self.stop.is_set():
                    break
                t0 = time.monotonic()
                if self.avatar is not None:
                    frames, parts = self.m.gpu.submit(self.m.render, self.avatar, tl, k, self.args.jpeg_quality).result()
                    for name, v in parts.items():
                        self.stats.setdefault(name, []).append(v)
                    if self.stats["steps"] < 3:
                        log("step", k, json.dumps({n: round(1000 * v) for n, v in parts.items()}))
                else:
                    frames = [self.static_jpeg] * STEP
                self.stats["gen_s"].append(time.monotonic() - t0)
                self.stats["steps"] += 1
                for j, jpg in enumerate(frames):
                    f = k + j
                    self.loop.call_soon_threadsafe(self.out.put_nowait, (jpg, to48k(tl[f * SPF:(f + 1) * SPF])))
                k += STEP
        except Exception as exc:  # noqa: BLE001
            log("render error", repr(exc))
            self.loop.call_soon_threadsafe(self.out.put_nowait, ("error", repr(exc)))

    async def sender(self):
        """Absolute 25 fps clock; frame k and its 40 ms of speech go out together. A frame more than 100 ms late
        (render stall) re-anchors the clock (as the other self-hosted avatars do): audio_out stays continuous and
        paired with its frames, the stall adds to the latency (counted as `reanchors`)."""
        t0, k = None, 0
        while True:
            item = await self.out.get()
            if item[0] == "error":
                await self.conn.error("render failed: " + item[1])
                return
            now = time.monotonic()
            if t0 is not None and now > t0 + k / FPS + 0.1:
                self.stats["reanchors"] = self.stats.get("reanchors", 0) + 1
                self.stats["reanchor_ms"] = self.stats.get("reanchor_ms", 0) + round(1000 * (now - t0 - k / FPS))
            if t0 is None or now > t0 + k / FPS + 0.1:
                t0, k = now, 0
            wait = t0 + k / FPS - now
            if wait > 0:
                await asyncio.sleep(wait)
            jpg, pcm = item
            await self.conn.frame(jpg)
            await self.conn.audio(pcm)
            self.stats["frames"] += 1
            k += 1

    async def start(self, init: dict):
        import cv2
        img_b64 = init.get("image_b64")
        if not img_b64:
            await self.conn.error("digital_human composition renderer needs init.image_b64 (portrait)")
            return
        raw = base64.b64decode(img_b64)
        image = fit_portrait(Image.open(io.BytesIO(raw)).convert("RGB"), self.args.max_side)
        seed = int(init.get("seed") if init.get("seed") is not None else 42)
        tmp = Path(tempfile.mkdtemp(prefix="musetalk_")) / "portrait.png"
        image.save(tmp)
        reason = None
        try:
            self.avatar = await asyncio.get_running_loop().run_in_executor(self.m.gpu, self.m.prepare, str(tmp), seed)
            if self.avatar is None:
                reason = "no face found in the portrait (S3FD face detector / DWPose landmarks)"
        except Exception as exc:  # noqa: BLE001
            self.avatar, reason = None, f"avatar preparation failed: {exc!r}"[:300]
        if self.avatar is None:
            log("not animated:", reason)
            arr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            self.static_jpeg = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])[1].tobytes()
        if self.stop.is_set():
            return
        w, h = image.size
        with self.lock:
            self.t_ready = time.monotonic()
        cfg = {**self.m.config, "seed": seed, "portrait_sha256": hashlib.sha256(raw).hexdigest(), "resolution": [w, h],
               "animated": self.avatar is not None, "not_animated_reason": reason,
               "face_box": list(self.avatar["bbox"]) if self.avatar else None}
        await self.conn.ready(MODEL, self.m.config["code_commit"] or "unknown", w, h, FPS,
                              protocol=sp.PROTOCOL_VERSION, config=cfg)
        log("ready", json.dumps({"seed": seed, "case": init.get("case_id"), "animated": self.avatar is not None,
                                 "size": [w, h], "face_box": cfg["face_box"]}))
        self.thread = threading.Thread(target=self.render, daemon=True)
        self.thread.start()
        self.send_task = asyncio.create_task(self.sender())

    async def run(self):
        started = None
        try:
            while True:
                msg = await self.conn.recv()
                if isinstance(msg, bytes):
                    tag, pts, payload = sp.unpack(msg)
                    if tag == sp.AUDIO_IN:
                        self.audio_in(pts, payload)
                    continue
                kind = msg.get("type")
                if kind == "init" and not started:     # preparation takes seconds: keep draining audio_in meanwhile
                    started = asyncio.create_task(self.start(msg))
                elif kind == "text":
                    await self.conn.ack(msg["event_id"], "applied",
                                        "renderer-only: speech for this turn arrives as audio_in from the composition layer")
                elif kind in ("reference", "control"):
                    await self.conn.ack(msg["event_id"], "unsupported", f"{MODEL}: no {kind} control")
                elif kind == "end":
                    break
        except sp.ConnectionClosed:
            pass
        finally:
            self.stop.set()
            with self.lock:
                self.lock.notify_all()
            if started:
                await asyncio.gather(started, return_exceptions=True)
            if getattr(self, "thread", None):
                await asyncio.get_running_loop().run_in_executor(None, self.thread.join, 30)
            if getattr(self, "send_task", None):
                self.send_task.cancel()
            g = self.stats["gen_s"]
            parts = {f"{n}_ms_mean": round(1000 * float(np.mean(self.stats[n])), 1)
                     for n in ("audio", "unet_vae", "paste_jpeg") if self.stats.get(n)}
            log("session end", json.dumps({"steps": self.stats["steps"], "frames": self.stats["frames"],
                                           "gen_ms_mean": round(1000 * float(np.mean(g)), 1) if g else None,
                                           "gen_ms_max": round(1000 * float(np.max(g)), 1) if g else None,
                                           "gen_ms_p95": round(1000 * float(np.percentile(g, 95)), 1) if g else None,
                                           "steps_over_150ms": int(sum(x > 0.15 for x in g)),
                                           "reanchors": self.stats.get("reanchors", 0),
                                           "reanchor_ms": self.stats.get("reanchor_ms", 0),
                                           **parts, "animated": self.avatar is not None}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18793)
    ap.add_argument("--weights", default=DEFAULT_ROOT + "/weights", help="its parent holds repo/ (repo/models -> weights)")
    ap.add_argument("--max-side", type=int, default=1024, help="portrait side cap (downscale only)")
    ap.add_argument("--context-s", type=float, default=10.0, help="whisper audio context (<= 30 s)")
    ap.add_argument("--grace", type=float, default=0.03)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    args = ap.parse_args(argv)
    args.root = Path(args.weights).resolve().parent
    model = Model(args)
    busy = asyncio.Lock()

    async def handler(conn):
        async with busy:
            await Session(conn, model, args, asyncio.get_running_loop()).run()

    log(f"{MODEL} listening on ws://{args.host}:{args.port}/session")
    sp.serve(handler, args.host, args.port, health=lambda: {"ok": True, "model": MODEL, **model.config})


if __name__ == "__main__":
    main()
