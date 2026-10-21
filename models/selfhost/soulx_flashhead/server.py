#!/usr/bin/env python3
"""SoulX-FlashHead (Lite, 1.3B) renderer behind the StreamEval self-host protocol (docs/protocol.md).

Composition system: the shared client's composition layer runs ASR -> LLM -> TTS and streams the TTS speech to
this server as `audio_in` (s16le 16 kHz, 10 ms blocks, real-time pace, silence between turns). The server

* sends `ready` after the portrait (init.image_b64) is encoded and the compiled model is warm;
* keeps an audio timeline indexed by wall-clock since `ready` (first `audio_in` anchors the stream pts);
* renders one 24-frame chunk (0.96 s) every time 0.96 s of new audio is available (or its wall-clock deadline
  passes; missing audio = silence), i.e. strictly causal, no look-ahead, no offline replay;
* streams frames at 25 fps on an absolute clock, each frame paired with the 40 ms of speech it was driven by as
  `audio_out` (48 kHz) -> audio and lips stay in sync; silence renders idle/listening frames continuously;
* acks every text/control/reference event: text is `applied` (the composition layer already turned it into
  the speech that arrives on audio_in), reference/control `unsupported`.

    python server.py --port 18790 --weights $STREAMEVAL_MODEL_ROOT/soulx_flashhead/weights
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

MODEL = "soulx_flashhead_lite"
SR = 16000
FPS = 25
SPF = SR // FPS            # 640 audio samples per frame
DEFAULT_ROOT = os.environ.get("SOULX_FLASHHEAD_ROOT") or (
    os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/soulx_flashhead")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def to48k(x16: np.ndarray) -> bytes:
    """float32 [-1,1] 16 kHz -> s16le 48 kHz (linear interpolation, block-local)."""
    n = len(x16)
    if n == 0:
        return b""
    xp = np.arange(n, dtype=np.float64)
    x = np.arange(3 * n, dtype=np.float64) / 3.0
    y = np.interp(x, xp, x16.astype(np.float64))
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()


def read_hashes(path: Path) -> dict:
    """weights_sha256.txt written by setup.sh (sha256sum format) -> {relative path: sha256}."""
    if not path.exists():
        return {}
    return {p: h for h, p in (line.split(None, 1) for line in path.read_text().split("\n") if line.strip())}


def stub_optional_modules(face_crop: bool):
    """Single-GPU Lite inference never touches xfuser (sequence parallel) or mediapipe (only for --face-crop),
    but upstream imports them at module level. If they are not importable in this env, register inert stubs."""
    import importlib
    import types
    try:
        importlib.import_module("xfuser.core.distributed")
        importlib.import_module("xfuser.core.long_ctx_attention")
    except Exception as exc:
        log("xfuser unavailable, using single-GPU stub:", repr(exc)[:200])
        for name in ("xfuser", "xfuser.core", "xfuser.core.distributed", "xfuser.core.long_ctx_attention"):
            sys.modules[name] = types.ModuleType(name)
        d = sys.modules["xfuser.core.distributed"]
        d.get_sequence_parallel_rank = lambda: 0
        d.get_sequence_parallel_world_size = lambda: 1
        d.get_sp_group = lambda: None
        d.initialize_model_parallel = d.init_distributed_environment = lambda *a, **k: None
        sys.modules["xfuser.core.long_ctx_attention"].xFuserLongContextAttention = object
    if not face_crop:
        try:
            importlib.import_module("mediapipe")
        except Exception:
            sys.modules["mediapipe"] = types.ModuleType("mediapipe")


class Model:
    """Loads the FlashHead Lite pipeline once; per session: set portrait + seed, then step()."""

    def __init__(self, args):
        root = Path(args.weights)
        self.ckpt = root / "SoulX-FlashHead-1_3B"
        self.w2v = root / "wav2vec2-base-960h"
        repo = Path(args.repo)
        os.chdir(repo)                      # flash_head/inference.py reads configs relative to cwd
        sys.path.insert(0, str(repo))
        stub_optional_modules(args.face_crop)
        import torch
        from flash_head import inference as fi
        self.torch, self.fi = torch, fi
        t0 = time.time()
        self.pipe = fi.get_pipeline(world_size=1, ckpt_dir=str(self.ckpt), model_type="lite",
                                    wav2vec_dir=str(self.w2v))
        self.params = fi.get_infer_params()
        self.frame_num = self.params["frame_num"]                  # 33
        self.motion = self.params["motion_frames_num"]             # 9
        self.slice = self.frame_num - self.motion                  # 24 new frames per chunk
        self.step_samples = self.slice * SPF                       # 15360 (0.96 s)
        self.cache = self.params["cached_audio_duration"] * SR     # 8 s wav2vec context
        self.end_idx = self.params["cached_audio_duration"] * FPS
        self.start_idx = self.end_idx - self.frame_num
        self.width, self.height = self.params["width"], self.params["height"]
        self.face_crop = args.face_crop
        log(f"pipeline loaded in {time.time() - t0:.1f}s")
        self.weights_sha = read_hashes(root.parent / "weights_sha256.txt")
        self.gpu = torch.cuda.get_device_name(0)
        commit = Path(args.repo).parent / "repo_commit.txt"
        self.config = {
            "model": "SoulX-FlashHead-1_3B Model_Lite", "code_commit": commit.read_text().strip() if commit.exists() else None,
            "weights_repo": "Soul-AILab/SoulX-FlashHead-1_3B", "weights_sha256": self.weights_sha,
            "audio_encoder": "facebook/wav2vec2-base-960h", "resolution": [self.width, self.height], "fps": FPS,
            "steps": self.params["sample_steps"], "chunk_frames": self.slice, "motion_frames": self.motion,
            "audio_context_s": self.params["cached_audio_duration"], "gpu": self.gpu,
            "torch": torch.__version__, "attention": self._attn(), "face_crop": self.face_crop,
        }
        self._warm()

    def _attn(self):
        from flash_head.src.modules import flash_head_model as m
        return ("sage" if m.SAGE_ATTN_AVAILABLE else "flash3" if m.FLASH_ATTN_3_AVAILABLE
                else "flash2" if m.FLASH_ATTN_2_AVAILABLE else "sdpa")

    def _warm(self):
        """Compile + warm with the bundled example portrait so the first session is real-time."""
        t0 = time.time()
        self.set_portrait(str(Path(self.fi.__file__).parents[1] / "examples" / "girl.png"), seed=0)
        for _ in range(3):
            self.step(np.zeros(self.cache, dtype=np.float32))
        log(f"warm-up done in {time.time() - t0:.1f}s")

    def set_portrait(self, path: str, seed: int):
        self.torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        self.fi.get_base_data(self.pipe, cond_image_path_or_dir=path, base_seed=seed, use_face_crop=self.face_crop)

    def step(self, audio_ctx: np.ndarray) -> np.ndarray:
        """audio_ctx: last `cache` samples ending at the chunk end -> (24, H, W, 3) uint8 new frames."""
        emb = self.fi.get_audio_embedding(self.pipe, audio_ctx, self.start_idx, self.end_idx)
        video = self.fi.run_pipeline(self.pipe, emb)
        return video[self.motion:].to(self.torch.uint8).cpu().numpy()


class Session:
    def __init__(self, conn: sp.ServerConnection, model: Model, args, loop):
        self.conn, self.m, self.args, self.loop = conn, model, args, loop
        self.stop = threading.Event()
        self.lock = threading.Condition()
        self.timeline = np.zeros(0, dtype=np.float32)   # 16 kHz, index 0 = ready instant
        self.received_to = 0                            # highest timeline index written
        self.anchor = None                              # timeline index of audio_in pts 0
        self.t_ready = None
        self.out: asyncio.Queue = asyncio.Queue()
        self.stats = {"chunks": 0, "gen_s": [], "late_chunks": 0, "frames": 0}

    # ---- input ----
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

    def window(self, end: int) -> np.ndarray:
        """Wait until audio up to `end` arrived or its wall-clock deadline (+grace) passed; return timeline[:end]."""
        deadline = self.t_ready + end / SR + self.args.grace
        with self.lock:
            while not self.stop.is_set() and self.received_to < end:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self.lock.wait(min(left, 0.02))
            if len(self.timeline) < end:
                self.timeline = np.concatenate([self.timeline, np.zeros(end - len(self.timeline) + SR * 10, np.float32)])
            return self.timeline[:end].copy()

    # ---- render thread ----
    def render(self):
        import cv2
        pos = 0
        step = self.m.step_samples
        try:
            while not self.stop.is_set():
                end = pos + step
                tl = self.window(end)
                if self.stop.is_set():
                    break
                ctx = tl[max(0, end - self.m.cache):end]
                if len(ctx) < self.m.cache:
                    ctx = np.concatenate([np.zeros(self.m.cache - len(ctx), np.float32), ctx])
                t0 = time.monotonic()
                frames = self.m.step(ctx)
                dt = time.monotonic() - t0
                self.stats["chunks"] += 1
                self.stats["gen_s"].append(dt)
                if time.monotonic() > self.t_ready + end / SR + dt + 0.5:
                    self.stats["late_chunks"] += 1
                for j, fr in enumerate(frames):
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
                    speech = tl[pos + j * SPF: pos + (j + 1) * SPF]
                    self.loop.call_soon_threadsafe(self.out.put_nowait, (jpg.tobytes(), to48k(speech)))
                pos = end
        except Exception as exc:  # fatal model error -> protocol error
            log("render error", repr(exc))
            self.loop.call_soon_threadsafe(self.out.put_nowait, ("error", repr(exc)))

    async def sender(self):
        """Absolute 25 fps clock; frame k and its 40 ms speech go out together. Underrun -> re-anchor."""
        t0, k = None, 0
        while True:
            item = await self.out.get()
            if item[0] == "error":
                await self.conn.error("render failed: " + item[1])
                return
            now = time.monotonic()
            if t0 is None or now > t0 + k / FPS + 0.2:
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
        image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        seed = int(init.get("seed") if init.get("seed") is not None else 42)
        tmp = Path(tempfile.mkdtemp(prefix="flashhead_")) / "portrait.png"
        image.save(tmp)
        await asyncio.get_running_loop().run_in_executor(None, self.m.set_portrait, str(tmp), seed)
        with self.lock:
            self.t_ready = time.monotonic()
        cfg = {**self.m.config, "seed": seed, "portrait_sha256": hashlib.sha256(base64.b64decode(img_b64)).hexdigest()}
        await self.conn.ready(MODEL, self.m.config["code_commit"] or "unknown", self.m.width, self.m.height, FPS,
                              protocol=sp.PROTOCOL_VERSION, config=cfg)
        log("ready", json.dumps({"seed": seed, "case": init.get("case_id")}))
        self.thread = threading.Thread(target=self.render, daemon=True)
        self.thread.start()
        self.send_task = asyncio.create_task(self.sender())

    async def run(self):
        started = False
        try:
            while True:
                msg = await self.conn.recv()
                if isinstance(msg, bytes):
                    tag, pts, payload = sp.unpack(msg)
                    if tag == sp.AUDIO_IN:
                        self.audio_in(pts, payload)
                    continue
                kind = msg.get("type")
                if kind == "init" and not started:
                    started = True
                    await self.start(msg)
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
            if getattr(self, "thread", None):
                await asyncio.get_running_loop().run_in_executor(None, self.thread.join, 30)
            if getattr(self, "send_task", None):
                self.send_task.cancel()
            g = self.stats["gen_s"]
            log("session end", json.dumps({"chunks": self.stats["chunks"], "frames": self.stats["frames"],
                                           "gen_ms_mean": round(1000 * float(np.mean(g)), 1) if g else None,
                                           "gen_ms_max": round(1000 * float(np.max(g)), 1) if g else None,
                                           "late_chunks": self.stats["late_chunks"]}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18790)
    ap.add_argument("--weights", default=DEFAULT_ROOT + "/weights")
    ap.add_argument("--repo", default=DEFAULT_ROOT + "/repo")
    ap.add_argument("--grace", type=float, default=0.06, help="s to wait past an audio window's wall-clock end")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--face-crop", action="store_true", help="crop the portrait around the face (mediapipe)")
    args = ap.parse_args(argv)
    model = Model(args)
    busy = asyncio.Lock()

    async def handler(conn):
        async with busy:           # one session at a time on one GPU
            await Session(conn, model, args, asyncio.get_running_loop()).run()

    log(f"{MODEL} listening on ws://{args.host}:{args.port}/session")
    sp.serve(handler, args.host, args.port, health=lambda: {"ok": True, "model": MODEL, **model.config})


if __name__ == "__main__":
    main()
