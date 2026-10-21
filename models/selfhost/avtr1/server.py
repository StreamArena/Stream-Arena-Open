#!/usr/bin/env python3
"""AVTR-1 renderer (Avaturn Renderer, TensorRT) behind the StreamEval self-host protocol (docs/protocol.md).

Composition system: the shared client's composition layer runs ASR -> LLM -> TTS and streams the TTS speech to
this server as `audio_in` (s16le 16 kHz, 10 ms blocks, real-time pace, silence between turns). AVTR-1's own
conversation engines (OpenAI Realtime / Cartesia adapters in avaturn_live_streamer) are NOT used; only the
renderer pipeline (`avtr1_renderer.pipeline.Pipeline`) is driven, so the dialogue stack matches the other
composition systems. The model's second audio stream ("listen", drives active listening) is fed silence because
the composition layer only hands the renderer the avatar's own speech.

* `ready` after the portrait (init.image_b64) is registered (face detect + appearance/motion extraction);
* audio timeline indexed by wall clock since `ready` (first `audio_in` anchors the stream pts);
* one 5-frame chunk (200 ms) per 200 ms of audio; each chunk needs 200 ms + 5 ms of look-ahead audio (model
  design: window = (5 + 5) * 640 + 80 samples), waited for in real time -> no pre-reading, no offline replay;
* frames go out at 25 fps on an absolute clock, each paired with the 40 ms of speech that drove it as
  `audio_out` (48 kHz), so audio and lips stay in sync; silence renders idle/listening motion continuously;
* text events are acked `applied` (the composition layer already turned them into speech), reference/control
  `unsupported`.

    python server.py --port 18791 --weights $STREAMEVAL_MODEL_ROOT/avtr1/weights    [--prepare: build TensorRT engines, exit]
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
import sys
import tempfile
import threading
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
from streameval.runtime import protocol as sp  # noqa: E402

MODEL = "avtr1"
SR = 16000
FPS = 25
SPF = SR // FPS            # 640
DEFAULT_ROOT = os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/avtr1"


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


def square_crop(image: Image.Image, w: int, h: int) -> Image.Image:
    """Center-crop to the output aspect, then resize (the AVTR loader would otherwise stretch)."""
    iw, ih = image.size
    target = w / h
    if iw / ih > target:
        nw = int(round(ih * target))
        image = image.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
    elif iw / ih < target:
        nh = int(round(iw / target))
        image = image.crop((0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh))
    return image.resize((w, h), Image.LANCZOS)


def arch_storage(root: Path) -> Path:
    """TensorRT engines are compute-capability specific: one storage root per arch (storage_sm89, storage_sm120),
    sharing the downloaded (arch-independent) artifacts of storage/main via symlinks. Engines are built on first
    use with the upstream scripts/build_engines.py (~minutes) and reused afterwards."""
    import subprocess
    import torch
    major, minor = torch.cuda.get_device_capability(0)
    base = root / f"storage_sm{major}{minor}"
    main = base / "main"
    main.mkdir(parents=True, exist_ok=True)
    for name in ("renderer_runtime_artifacts", "build_artifacts", "avatars_artifacts"):
        link = main / name
        if not link.exists():
            link.symlink_to(root / "storage" / "main" / name)
    os.environ["AVTR1_LOCAL_STORAGE"] = str(base)
    need = [main / "speech2motion_runtime_artifacts_cc" / f for f in ("avtr1_encode_fp16.engine", "avtr1_decode_fp16.engine", "hubert_lbs_fp16.engine")]
    need += [main / "renderer_runtime_artifacts_cc" / f for f in ("decoder_b5_fp16.engine", "warp_network_b5_fp16.engine",
                                                                  "stitch_network_b5_fp16.engine", "modnet_b5_fp16.engine")]
    missing = [p.name for p in need if not p.is_file()]
    if missing:
        log(f"building TensorRT engines for sm{major}{minor} (missing {missing}) ...")
        t0 = time.time()
        scripts = root / "repo" / "scripts"
        r = subprocess.run([sys.executable, str(scripts / "build_engines.py")], cwd=str(scripts),
                           env={**os.environ, "PYTHONPATH": str(scripts)}, stdout=sys.stderr, stderr=sys.stderr)
        if r.returncode != 0:
            raise SystemExit("TensorRT engine build failed")
        log(f"engines built in {time.time() - t0:.0f}s")
    return base


def ort_on_cpu():
    """The portrait-registration ONNX models (face detection, landmarks, appearance / motion extraction) run once per
    portrait using CPU ONNX Runtime for portability (host copies in and out); the per-frame renderer stays on TensorRT."""
    import onnxruntime as ort
    import torch
    from avtr1_renderer.runtime import onnxrt

    def from_file(cls, onnx_file, input_cls, output_cls, device_id=0):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8        # leave cores for the client on the same machine
        session = ort.InferenceSession(onnx_file, so, providers=["CPUExecutionProvider"])
        return cls(session, input_cls, output_cls, ep_stream=torch.cuda.current_stream(device=device_id))

    def call(self, inputs, out=None):
        feed = {n: getattr(inputs, n).detach().cpu().numpy() for n in self._input_names}
        res = [torch.from_numpy(r) for r in self.session.run(self._output_names, feed)]
        if out is None:
            return self.output_cls(**{n: r.cuda() for n, r in zip(self._output_names, res)})
        for n, r in zip(self._output_names, res):
            getattr(out, n).copy_(r)
        return out

    onnxrt.OnnxRTEngine.from_file = classmethod(from_file)
    onnxrt.OnnxRTEngine.__call__ = call


class Model:
    def __init__(self, args):
        arch_storage(args.root)
        import torch
        ort_on_cpu()
        from avtr1_renderer.pipeline import Pipeline
        from avtr1_renderer.avatar_loader import AvatarLoader
        from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager, get_trt_engine_path
        from avtr1_renderer.types import Chunk, RenderOptions
        self.torch, self.Chunk = torch, Chunk
        self.width = self.height = args.size
        t0 = time.time()
        self.pipe, _ = Pipeline.from_artifacts(avatar_ids=[], out_size=(self.height, self.width))
        mgr = get_artifact_manager()
        self.loader = AvatarLoader(
            engine_files={k: mgr.get_artifact_path(k) for k in
                          ("insightface_det", "landmark106", "landmark203", "appearance_extractor", "motion_extractor")},
            mask_template_path=mgr.get_artifact_path("pasteback_mask"),
            out_h=self.height, out_w=self.width, max_dim=max(self.height, self.width))
        mg = self.pipe._motion_generator
        self.step_samples = mg.chunk_size * mg.frame_len                                      # 3200 (200 ms)
        self.window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift        # 6480
        self.lookahead = self.window - self.step_samples                                      # 3280 (205 ms)
        self.options = RenderOptions(pixel_format="yuv_i420", bg_id=args.bg, stream_frames=True)
        engines = {n: get_trt_engine_path(n).is_file() for n in
                   ("avtr1_encode", "avtr1_decode", "hubert_lbs", "decoder", "warp_network", "stitch_network", "modnet")}
        log(f"pipeline loaded in {time.time() - t0:.1f}s; trt engines: {engines}")
        commit = Path(args.root) / "repo_commit.txt"
        self.gpu = torch.cuda.get_device_name(0)
        self.config = {
            "model": "AVTR-1 (Avaturn Renderer, TensorRT)", "code_commit": commit.read_text().strip() if commit.exists() else None,
            "weights_repo": "avaturn-live/avtr-1 + digital-avatar/ditto-talkinghead ditto_onnx",
            "weights_sha256": read_hashes(Path(args.root) / "weights_sha256.txt"),
            "resolution": [self.width, self.height], "fps": FPS, "chunk_frames": mg.chunk_size,
            "lookahead_ms": round(1000 * self.lookahead / SR), "trt_engines": engines, "gpu": self.gpu,
            "torch": torch.__version__, "listen_track": "silence (composition layer supplies speech only)",
            "cfg": {"self_audio": self.options.cfg_self_audio, "other_audio": self.options.cfg_other_audio,
                    "kp": self.options.cfg_kp}, "background": "portrait's own (RGB portrait, no matting)",
        }
        self._warm()

    def _warm(self):
        t0 = time.time()
        ref = Path(os.environ["AVTR1_LOCAL_STORAGE"]) / "main/avatars_artifacts/reference_frames/maria.png"
        img = Image.open(ref).convert("RGB")
        tmp = Path(tempfile.mkdtemp(prefix="avtr1_")) / "warm.png"
        square_crop(img, self.width, self.height).save(tmp)
        self.set_portrait(str(tmp), 0)
        for _ in range(10):
            self.step(np.zeros(self.window, np.float32))
        log(f"warm-up done in {time.time() - t0:.1f}s")

    def set_portrait(self, path: str, seed: int):
        self.torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        self.avatar = self.loader.load(path, avatar_id="case")
        self.state = None

    def step(self, window: np.ndarray) -> list[np.ndarray]:
        """window: 6480 samples starting at the chunk start -> 5 RGB uint8 frames."""
        import cv2
        chunk = self.Chunk(audio_speech=window.astype(np.float32),
                           audio_listen=np.zeros(self.window, np.float32))
        self.state, frames = self.pipe.process_chunk(self.avatar, chunk, self.state, self.options)
        return [cv2.cvtColor(f.data, cv2.COLOR_YUV2RGB_I420) for f in frames]


class Session:
    def __init__(self, conn: sp.ServerConnection, model: Model, args, loop):
        self.conn, self.m, self.args, self.loop = conn, model, args, loop
        self.stop = threading.Event()
        self.lock = threading.Condition()
        self.timeline = np.zeros(0, dtype=np.float32)
        self.received_to = 0
        self.anchor = None
        self.t_ready = None
        self.out: asyncio.Queue = asyncio.Queue()
        self.stats = {"chunks": 0, "gen_s": [], "frames": 0}

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
        import cv2
        pos = 0
        try:
            while not self.stop.is_set():
                tl = self.wait_audio(pos + self.m.window)
                if self.stop.is_set():
                    break
                t0 = time.monotonic()
                frames = self.m.step(tl[pos:pos + self.m.window])
                self.stats["gen_s"].append(time.monotonic() - t0)
                self.stats["chunks"] += 1
                for j, fr in enumerate(frames):
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
                    speech = tl[pos + j * SPF: pos + (j + 1) * SPF]
                    self.loop.call_soon_threadsafe(self.out.put_nowait, (jpg.tobytes(), to48k(speech)))
                pos += self.m.step_samples
        except Exception as exc:
            log("render error", repr(exc))
            self.loop.call_soon_threadsafe(self.out.put_nowait, ("error", repr(exc)))

    async def sender(self):
        t0, k = None, 0
        while True:
            item = await self.out.get()
            if item[0] == "error":
                await self.conn.error("render failed: " + item[1])
                return
            now = time.monotonic()
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
        img_b64 = init.get("image_b64")
        if not img_b64:
            await self.conn.error("digital_human composition renderer needs init.image_b64 (portrait)")
            return
        raw = base64.b64decode(img_b64)
        image = square_crop(Image.open(io.BytesIO(raw)).convert("RGB"), self.m.width, self.m.height)
        seed = int(init.get("seed") if init.get("seed") is not None else 42)
        tmp = Path(tempfile.mkdtemp(prefix="avtr1_")) / "portrait.png"
        image.save(tmp)
        try:
            await asyncio.get_running_loop().run_in_executor(None, self.m.set_portrait, str(tmp), seed)
        except Exception as exc:
            await self.conn.error(f"portrait registration failed: {exc!r}")
            return
        if self.stop.is_set():
            return
        with self.lock:
            self.t_ready = time.monotonic()
        cfg = {**self.m.config, "seed": seed, "portrait_sha256": hashlib.sha256(raw).hexdigest()}
        await self.conn.ready(MODEL, self.m.config["code_commit"] or "unknown", self.m.width, self.m.height, FPS,
                              protocol=sp.PROTOCOL_VERSION, config=cfg)
        log("ready", json.dumps({"seed": seed, "case": init.get("case_id")}))
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
                if kind == "init" and not started:     # registration takes ~20 s: keep draining audio_in meanwhile
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
            log("session end", json.dumps({"chunks": self.stats["chunks"], "frames": self.stats["frames"],
                                           "gen_ms_mean": round(1000 * float(np.mean(g)), 1) if g else None,
                                           "gen_ms_max": round(1000 * float(np.max(g)), 1) if g else None}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18791)
    ap.add_argument("--weights", default=DEFAULT_ROOT + "/weights", help="its parent holds storage/, weights_sha256.txt")
    ap.add_argument("--prepare", action="store_true", help="build the TensorRT engines for this GPU and exit")
    ap.add_argument("--size", type=int, default=720, help="square output side (benchmark portraits are square)")
    ap.add_argument("--bg", default="plain_white", help="bg id (only used for RGBA portraits)")
    ap.add_argument("--grace", type=float, default=0.03)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    args = ap.parse_args(argv)
    args.root = Path(args.weights).resolve().parent
    if args.prepare:
        arch_storage(args.root)
        return
    model = Model(args)
    busy = asyncio.Lock()

    async def handler(conn):
        async with busy:
            await Session(conn, model, args, asyncio.get_running_loop()).run()

    log(f"{MODEL} listening on ws://{args.host}:{args.port}/session")
    sp.serve(handler, args.host, args.port, health=lambda: {"ok": True, "model": MODEL, **model.config})


if __name__ == "__main__":
    main()
