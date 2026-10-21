#!/usr/bin/env python3
"""SoulX-FlashTalk 14B renderer behind the StreamEval self-host protocol (docs/protocol.md).

Launch (N GPUs, xfuser Ulysses sequence parallel as upstream; rank 0 serves the WebSocket, all ranks run the model):

    torchrun --nproc_per_node=4 --master_port=29661 server.py --port 18801 --size 512*512

Composition renderer for digital_human_audio / digital_human_text: the client's shared ASR -> LLM -> TTS stack
streams the TTS speech as ``audio_in`` (s16le 16 kHz, 10 ms, real-time pace, silence between turns). The server

* sends ``ready`` once the portrait (init.image_b64) is encoded (CLIP + VAE, text prompt) and the compiled model is warm;
* keeps an audio timeline indexed by wall-clock since ``ready`` (the first ``audio_in`` anchors the stream pts);
* renders one 28-frame chunk (1.12 s at 25 fps; 33-frame window with 5 motion frames from the previous chunk, upstream
  ``generate_video.py --audio_encode_mode stream``) every time 1.12 s of new audio has arrived or its wall-clock
  deadline passed (missing audio = silence): strictly causal, no look-ahead, no offline replay. The wav2vec features
  come from the last 8 s of audio ending at the chunk end (upstream's rolling deque);
* streams frames at 25 fps, each frame paired with the 40 ms of speech it was driven by as ``audio_out`` (48 kHz), so
  audio and lips stay in sync; silence renders idle/listening frames continuously. When generation is slower than
  real time, chunks arrive late (each chunk re-anchors the 25 fps clock); nothing is dropped or re-timed;
* acks ``text`` as applied (the composition layer already turned it into the speech arriving on ``audio_in``);
  ``reference`` / ``control`` are unsupported (FlashTalk's prompt does not control motion: upstream issues #29, #37).
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
import re
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("SOULX_FLASHTALK_ROOT") or (
    os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/soulx_flashtalk"))
sys.path.insert(0, str(HERE.parents[2]))
from streameval.runtime import protocol as sp  # noqa: E402

for _k, _v in {"TORCHINDUCTOR_CACHE_DIR": str(MODEL_DIR / "inductor_cache"),
               "TRITON_CACHE_DIR": str(MODEL_DIR / "triton_cache")}.items():
    os.environ.setdefault(_k, _v)

MODEL = "soulx_flashtalk_14b"
SR = 16000
FPS = 25
SPF = SR // FPS            # 640 audio samples per frame
DEFAULT_PROMPT = "A person is talking. Only the foreground characters are moving, the background remains static."


def log(*a):
    if int(os.getenv("RANK", 0)) == 0:
        print(time.strftime("%H:%M:%S"), "[flashtalk]", *a, file=sys.stderr, flush=True)


def to48k(x16: np.ndarray) -> bytes:
    n = len(x16)
    if n == 0:
        return b""
    y = np.interp(np.arange(3 * n, dtype=np.float64) / 3.0, np.arange(n, dtype=np.float64), x16.astype(np.float64))
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()


def gradient_image(w, h):
    from PIL import Image
    x = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.stack([np.tile(x, (h, 1)), np.tile(x[::-1], (h, 1)), np.full((h, w), 128, np.uint8)], -1)
    return Image.fromarray(arr)


class Engine:
    """Pipeline state on every rank; rank 0 drives all GPU calls in lockstep over a gloo control group."""

    def __init__(self, args):
        self.args = args
        self.rank = int(os.getenv("RANK", 0))
        self.world = int(os.getenv("WORLD_SIZE", 1))
        self.width, self.height = [int(v) for v in args.size.split("*")]
        repo = Path(args.repo)
        os.chdir(repo)                       # flash_talk/inference.py reads its yaml relative to cwd
        sys.path[:0] = [str(HERE / "shims"), str(repo)]
        import torch
        import torch.distributed as dist
        self.torch, self.dist = torch, dist
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        from flash_talk import inference as fi
        self.fi = fi
        fi.target_size = (self.height, self.width)
        p = fi.infer_params
        self.frame_num, self.motion = p["frame_num"], p["motion_frames_num"]      # 33, 5
        self.slice = self.frame_num - self.motion                                 # 28 new frames per chunk
        self.step_samples = self.slice * SPF                                      # 17920 (1.12 s)
        self.cache = p["cached_audio_duration"] * SR                              # 8 s wav2vec context
        self.end_idx = p["cached_audio_duration"] * FPS
        self.start_idx = self.end_idx - self.frame_num
        t0 = time.time()
        self.pipe = fi.get_pipeline(world_size=self.world, ckpt_dir=args.ckpt_dir, wav2vec_dir=args.wav2vec_dir)
        self.ctrl = None
        if self.world > 1:
            from datetime import timedelta
            self.ctrl = dist.new_group(backend="gloo", timeout=timedelta(days=7))
        self.device = self.pipe.device
        self.load_s = time.time() - t0
        log(f"pipeline loaded in {self.load_s:.1f}s size={self.width}x{self.height} world={self.world}")
        self.gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        self.timing = {}

    # ---- lockstep commands ----------------------------------------------------------------------------------
    def start(self, cmd):
        from PIL import Image
        image = cmd["image"]
        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        self.torch.manual_seed(int(cmd["seed"]))
        np.random.seed(int(cmd["seed"]) % (2 ** 32))
        self.fi.get_base_data(self.pipe, input_prompt=cmd["prompt"], cond_image=image, base_seed=int(cmd["seed"]))

    def chunk(self, cmd):
        torch = self.torch
        t0 = time.perf_counter()
        emb = self.fi.get_audio_embedding(self.pipe, np.asarray(cmd["ctx"], dtype=np.float32), self.start_idx,
                                          self.end_idx)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        video = self.fi.run_pipeline(self.pipe, emb)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        self.timing = {"audio_ms": round((t1 - t0) * 1000, 1), "model_ms": round((t2 - t1) * 1000, 1)}
        if self.rank != 0:
            return None
        return video[self.motion:].to(torch.uint8).cpu().numpy()

    def run(self, cmd):
        op = cmd["op"]
        if op == "start":
            return self.start(cmd)
        if op == "chunk":
            return self.chunk(cmd)
        return None

    def bcast(self, cmd):
        if self.world > 1:
            box = [cmd]
            self.dist.broadcast_object_list(box, src=0, group=self.ctrl)
            return box[0]
        return cmd

    def worker_loop(self):
        while True:
            cmd = self.bcast(None)
            if cmd is None or cmd["op"] == "idle":
                continue
            if cmd["op"] == "quit":
                return
            self.run(cmd)

    def warmup(self, chunks):
        """Compile + warm (rank 0 drives) so the first session is not slowed by torch.compile."""
        t0 = time.time()
        self.run(self.bcast({"op": "start", "image": gradient_image(self.width, self.height), "prompt": DEFAULT_PROMPT,
                             "seed": 0}))
        rng = np.random.default_rng(0)
        for n in range(chunks):
            t = time.time()
            ctx = np.zeros(self.cache, np.float32) if n % 2 == 0 else (rng.standard_normal(self.cache) * 0.05).astype(np.float32)
            self.run(self.bcast({"op": "chunk", "ctx": ctx}))
            log(f"warmup chunk {n}: {time.time() - t:.2f}s {self.timing}")
        log(f"warm-up done in {time.time() - t0:.1f}s")


class Session:
    def __init__(self, conn, server, loop):
        self.conn, self.server, self.loop = conn, server, loop
        self.eng, self.args = server.engine, server.args
        self.stop = threading.Event()
        self.lock = threading.Condition()
        self.timeline = np.zeros(0, dtype=np.float32)   # 16 kHz, index 0 = ready instant
        self.received_to = 0
        self.anchor = None
        self.last_pts = -1
        self.t_ready = None
        self.out: asyncio.Queue = asyncio.Queue()
        self.stats = {"chunks": [], "frames": 0, "late_chunks": 0, "reanchors": 0}

    # ---- input ----
    def audio_in(self, pts_ns: int, payload: bytes):
        pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        with self.lock:
            if self.t_ready is None:
                return
            if self.anchor is None or pts_ns < self.last_pts:
                self.anchor = int(round((time.monotonic() - self.t_ready) * SR)) - int(pts_ns * SR // 1_000_000_000)
            self.last_pts = pts_ns
            start = self.anchor + int(pts_ns * SR // 1_000_000_000)
            if start < 0:
                pcm, start = pcm[-start:], 0
            end = start + len(pcm)
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

    # ---- generation thread (rank 0, holds the GPU lock) ----
    def render(self, init):
        import cv2
        eng = self.eng
        try:
            with self.server.gpu_lock:
                eng.run(eng.bcast({"op": "start", **init}))
                with self.lock:
                    self.t_ready = time.monotonic()
                self.loop.call_soon_threadsafe(self.ready_evt.set)
                pos = 0
                step = eng.step_samples
                while not self.stop.is_set():
                    end = pos + step
                    tl = self.window(end)
                    if self.stop.is_set():
                        break
                    ctx = tl[max(0, end - eng.cache):end]
                    if len(ctx) < eng.cache:
                        ctx = np.concatenate([np.zeros(eng.cache - len(ctx), np.float32), ctx])
                    t0 = time.monotonic()
                    frames = eng.run(eng.bcast({"op": "chunk", "ctx": ctx}))
                    t1 = time.monotonic()
                    late = t1 - (self.t_ready + end / SR)
                    self.stats["chunks"].append({"n": len(self.stats["chunks"]), "audio_end_s": round(end / SR, 3),
                                                 "start_s": round(t0 - self.t_ready, 3),
                                                 "done_s": round(t1 - self.t_ready, 3),
                                                 "compute_ms": round((t1 - t0) * 1000, 1), **eng.timing})
                    if late > step / SR + 0.5:
                        self.stats["late_chunks"] += 1
                    for j, fr in enumerate(frames):
                        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
                        speech = tl[pos + j * SPF: pos + (j + 1) * SPF]
                        self.loop.call_soon_threadsafe(self.out.put_nowait, (jpg.tobytes(), to48k(speech)))
                    pos = end
        except Exception as exc:  # fatal model error -> protocol error
            import traceback
            traceback.print_exc()
            self.error = f"{type(exc).__name__}: {exc}"
            self.loop.call_soon_threadsafe(self.ready_evt.set)
            self.loop.call_soon_threadsafe(self.out.put_nowait, ("error", self.error))

    async def sender(self):
        """Absolute 25 fps clock; frame k and its 40 ms of speech go out together. Underrun -> re-anchor."""
        t0, k = None, 0
        while True:
            item = await self.out.get()
            if item[0] == "error":
                await self.conn.error("render failed: " + item[1])
                return
            now = time.monotonic()
            if t0 is None or now > t0 + k / FPS + 0.2:
                if t0 is not None:
                    self.stats["reanchors"] += 1
                t0, k = now, 0
            wait = t0 + k / FPS - now
            if wait > 0:
                await asyncio.sleep(wait)
            jpg, pcm = item
            now_ns = time.monotonic_ns()
            await self.conn.frame(jpg, now_ns)
            await self.conn.audio(pcm, now_ns)
            self.stats["frames"] += 1
            k += 1

    async def start(self, init_msg):
        img_b64 = init_msg.get("image_b64")
        if not img_b64:
            await self.conn.error("digital_human composition renderer needs init.image_b64 (portrait)")
            return False
        raw = base64.b64decode(img_b64)
        seed = int(init_msg.get("seed") if init_msg.get("seed") is not None else self.args.seed)
        self.error = None
        self.ready_evt = asyncio.Event()
        self.stats.update({"case_id": init_msg.get("case_id"), "track": init_msg.get("track"), "seed": seed})
        self.thread = threading.Thread(target=self.render, args=({"image": raw, "prompt": self.args.prompt, "seed": seed},),
                                       daemon=True)
        self.thread.start()
        await self.ready_evt.wait()
        if self.error:
            await self.conn.error(self.error)
            return False
        cfg = {**self.server.config, "seed": seed, "portrait_sha256": hashlib.sha256(raw).hexdigest()}
        await self.conn.ready(MODEL, self.server.config["repo_commit"] or "unknown", self.eng.width, self.eng.height, FPS,
                              protocol=sp.PROTOCOL_VERSION, config=cfg)
        log("ready", json.dumps({"seed": seed, "case": init_msg.get("case_id")}))
        self.send_task = asyncio.create_task(self.sender())
        return True

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
                    if not await self.start(msg):
                        break
                elif kind == "text":
                    await self.conn.ack(msg.get("event_id"), "applied",
                                        "renderer-only: speech for this turn arrives as audio_in from the composition layer")
                elif kind in ("reference", "control"):
                    await self.conn.ack(msg.get("event_id"), "unsupported",
                                        f"{MODEL}: no {kind} control (the text prompt does not steer motion)")
                elif kind == "end":
                    break
        except sp.ConnectionClosed:
            pass
        finally:
            self.stop.set()
            with self.lock:
                self.lock.notify_all()
            if getattr(self, "send_task", None):
                self.send_task.cancel()
            try:
                await self.conn.close()
            except Exception:
                pass
            if getattr(self, "thread", None):
                await asyncio.get_running_loop().run_in_executor(None, self.thread.join, 120)
            self.server.write_stats(self)


class Server:
    def __init__(self, engine, args):
        self.engine, self.args = engine, args
        self.gpu_lock = threading.Lock()
        self.busy = asyncio.Lock()
        eng = engine
        weights = json.loads((MODEL_DIR / "weights_hash.json").read_text()) if (MODEL_DIR / "weights_hash.json").exists() else {}
        commit = (MODEL_DIR / "repo_commit.txt").read_text().strip() if (MODEL_DIR / "repo_commit.txt").exists() else None
        rev = MODEL_DIR / "weights/SoulX-FlashTalk-14B/.complete"
        self.config = {
            "model": "SoulX-FlashTalk-14B", "repo": "https://github.com/Soul-AILab/SoulX-FlashTalk", "repo_commit": commit,
            "hf_weights": "Soul-AILab/SoulX-FlashTalk-14B" + (f"@{rev.read_text().strip()}" if rev.exists() else ""),
            "audio_encoder": "TencentGameMate/chinese-wav2vec2-base", "weights": weights,
            "width": eng.width, "height": eng.height, "fps": FPS, "steps": eng.fi.infer_params["sample_steps"],
            "sample_shift": eng.fi.infer_params["sample_shift"], "cfg": "none (distilled)",
            "chunk_frames": eng.slice, "motion_frames": eng.motion, "audio_context_s": eng.cache / SR,
            "prompt": self.args.prompt, "gpu": eng.gpu, "gpu_count": eng.world,
            "parallel": f"xfuser ulysses x{eng.world}" if eng.world > 1 else "single GPU",
            "torch": eng.torch.__version__, "torch_compile": True, "load_s": round(eng.load_s, 1),
            "xformers": "shim: torch SDPA (exact) for the audio cross-attention",
        }

    def write_stats(self, sess):
        s = {k: v for k, v in sess.stats.items()}
        c = s["chunks"]
        if c:
            frames = len(c) * self.engine.slice
            s["summary"] = {"chunks": len(c), "frames_generated": frames, "frames_sent": s["frames"],
                            "mean_compute_ms": round(float(np.mean([x["compute_ms"] for x in c[1:] or c])), 1),
                            "max_compute_ms": round(float(np.max([x["compute_ms"] for x in c])), 1),
                            "generation_fps": round(frames / max(1e-6, sum(x["compute_ms"] for x in c) / 1000), 2),
                            "realtime_budget_ms": round(self.engine.slice / FPS * 1000, 1),
                            "max_lag_s": round(max(x["done_s"] - x["audio_end_s"] for x in c), 3),
                            "late_chunks": s["late_chunks"], "reanchors": s["reanchors"]}
            log("session summary " + json.dumps(s["summary"]))
        d = Path(self.args.stats_dir)
        d.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(s.get("case_id") or "session"))
        (d / f"{name}.{int(time.time())}.json").write_text(json.dumps({"config": self.config, **s}, indent=1))

    async def handler(self, conn):
        async with self.busy:                    # one session at a time
            await Session(conn, self, asyncio.get_running_loop()).run()

    def health(self):
        return {"ok": True, "model": MODEL, "busy": self.busy.locked()}


def idle_ticker(server):
    """Keep the gloo control channel exercised between sessions."""
    while True:
        time.sleep(1.0)
        if server.gpu_lock.acquire(blocking=False):
            try:
                server.engine.bcast({"op": "idle"})
            finally:
                server.gpu_lock.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18801)
    ap.add_argument("--repo", default=str(MODEL_DIR / "repo"))
    ap.add_argument("--ckpt_dir", default=str(MODEL_DIR / "weights/SoulX-FlashTalk-14B"))
    ap.add_argument("--wav2vec_dir", default=str(MODEL_DIR / "weights/chinese-wav2vec2-base"))
    ap.add_argument("--size", default="512*512", help="W*H (upstream default 416*720; benchmark portraits are square)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grace", type=float, default=0.06, help="s to wait past an audio window's wall-clock end")
    ap.add_argument("--warmup_chunks", type=int, default=4)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--stats_dir", default=str(MODEL_DIR / "logs/sessions"))
    args = ap.parse_args()
    engine = Engine(args)
    if engine.rank != 0:
        engine.worker_loop()
        return
    engine.warmup(args.warmup_chunks)
    server = Server(engine, args)
    if engine.world > 1:
        threading.Thread(target=idle_ticker, args=(server,), daemon=True).start()
    log(f"listening on ws://{args.host}:{args.port}/session")
    sp.serve(server.handler, args.host, args.port, health=server.health)


if __name__ == "__main__":
    main()
