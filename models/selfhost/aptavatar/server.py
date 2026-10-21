#!/usr/bin/env python3
"""AptAvatar 14B (Taobao Live AIGC, Alibaba) behind the StreamEval self-host protocol (docs/protocol.md).

Launch (N GPUs, xfuser Ulysses sequence parallel as upstream; rank 0 serves the WebSocket, all ranks run the model):

    torchrun --nproc_per_node=4 --master_port=29681 server.py --port 18881 --size 512*512

AptAvatar is Wan2.1-I2V-14B + the InfiniteTalk audio adapter, distilled to 2 steps without CFG (endpoint-anchored
DMD). Upstream ``generate_video.py`` is chunk-wise autoregressive: a 73-frame window whose first 9 frames are the
VAE-encoded last 9 frames of the previous chunk (the first chunk: the portrait), 64 new frames per chunk (2.56 s at
25 fps), and a text prompt per chunk (``set_chunk_prompt``). This server runs the same loop live:

* ``ready`` once the portrait is encoded (CLIP + VAE) and the compiled model is warm;
* audio: an audio timeline indexed by wall-clock since ``ready`` (the first ``audio_in`` anchors the stream pts).
  Chunk k uses the audio of output frames 64k .. 64k+72 (upstream's chunk alignment); it starts once audio up to
  frame 64k+73 has arrived or its wall-clock deadline passed (missing audio = silence). The wav2vec features come
  from the last ``--audio_context`` s of audio ending at that point (loudness-normalised as upstream), and the
  per-frame +-2 feature window is clamped at the end, so nothing later than the chunk is read: strictly causal;
* video: chunk 0 sends all 73 frames (frame 0 ~ portrait), later chunks their 64 new frames, at 25 fps, each with the
  40 ms of speech that drove it as ``audio_out`` (48 kHz). Latency from speech to lips is about one chunk (2.56 s)
  plus compute. When generation is slower than real time, chunks arrive late (each re-anchors the clock); nothing is
  dropped or re-timed;
* avatar_performance (native): ``init.prompt`` (rendering_context, benchmark boilerplate stripped) is the base
  prompt; a visual control (``text`` user_prompt / ``control``) is appended to the prompt of the next chunk (control
  map ``aptavatar-control-map/1``, same vocabulary as the other self-hosted avatar models). AptAvatar's prompts are
  motion descriptions (upstream examples: step-by-step hand / arm gestures), so this is upstream's own steering path.
  Camera / scene / appearance / speech-content controls are acked unsupported;
* digital_human composition: the shared ASR -> LLM -> TTS stack pushes the TTS speech as ``audio_in``; dialogue
  ``text`` is acked applied (it already became the speech on ``audio_in``); the prompt is upstream's default.
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
MODEL_DIR = Path(os.path.expanduser(os.environ.get("APTAVATAR_ROOT") or (
    os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models") + "/aptavatar")))
sys.path.insert(0, str(HERE.parents[2]))
from streameval.runtime import protocol as sp  # noqa: E402

for _k, _v in {"TORCHINDUCTOR_CACHE_DIR": str(MODEL_DIR / "inductor_cache"),
               "TRITON_CACHE_DIR": str(MODEL_DIR / "triton_cache")}.items():
    os.environ.setdefault(_k, _v)

MODEL = "aptavatar_14b"
SR = 16000
FPS = 25
SPF = SR // FPS            # 640 audio samples per frame
DEFAULT_PROMPT = "A person is talking. Only the foreground characters are moving, the background remains static."
CONTROL_MAPPING_VERSION = "aptavatar-control-map/1"


def log(*a):
    if int(os.getenv("RANK", 0)) == 0:
        print(time.strftime("%H:%M:%S"), "[aptavatar]", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Control mapping (same vocabulary as livetalk-control-map/1 / liveact-control-map/1): applied when the control asks
# for a visible movement / pose of the body, gaze, or a facial expression; camera, scene, lighting, props, clothing,
# style and speaking content have no counterpart in a portrait animation prompt and are acked unsupported.
ACTION_WORDS = r"""hand hands arm arms palm palms fist fists finger fingers index thumb thumbs wrist wrists elbow elbows
shoulder shoulders torso body chest waist hip hips lean leans leaning step steps stepping knee knees foot feet leg legs
toe toes heel stance posture pose poses stand standing sit sitting squat crouch head nod nods nodding tilt tilts tilted
turn turns turning rotate rotates bow bows bowing wave waves waving sweep sweeps raise raises raised lift lifts lifted
lower lowers lowered drop drops dropped cross crossed clap claps point points pointing reach reaches extend extends
extended gesture gestures grip grab hold holding pinch press presses pressing fold shrug shrugs neck spine back
upright slump""".split()
GAZE_WORDS = "gaze gazes look looks looking glance glances stare stares eye eyes track tracks".split()
EMOTION_WORDS = r"""smile smiles smiling grin grins frown frowns frowning brow brows eyebrow eyebrows expression
expressions face facial squint squints squinting laugh laughs laughing sad sadness happy joy joyful angry anger
surprise surprised calm serious neutral curious alert wink winks pout pouts cheek cheeks lips jaw worried confident
relaxed tense excited scowl grimace sneer""".split()
UNSUPPORTED_PATTERNS = [r"\bzoom\w*", r"\bdolly\b", r"\bcamera (?:moves?|pans?|angle|cuts?|shot|orbits?|tilts?)\b",
                        r"\b(?:pan|cut|switch) (?:the )?camera\b", r"\blighting\b", r"\bbackground\b", r"\bscenery\b",
                        r"\b(?:outfit|clothes|clothing|costume|hairstyle)\b", r"\b(?:art|render(?:ing)?) style\b",
                        r"\b(?:say|says|saying|speak|speaks|sing|sings)\b"]
BOILERPLATE = [re.compile(p, re.I) for p in (
    r"Lips follow only the supplied audio;?\s*stay silent otherwise\.?",
    r"Mouth shape follows only the current actual input audio;[^.]*\.",
)]


def _hits(words, text):
    return sorted({w for w in words if re.search(r"\b" + re.escape(w) + r"\b", text, re.I)})


def strip_boilerplate(text: str) -> str:
    for p in BOILERPLATE:
        text = p.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def map_control(text: str) -> dict:
    clean = strip_boilerplate(text or "")
    action, gaze, emotion = _hits(ACTION_WORDS, clean), _hits(GAZE_WORDS, clean), _hits(EMOTION_WORDS, clean)
    other = sorted({m.group(0).lower() for p in UNSUPPORTED_PATTERNS for m in re.finditer(p, clean, re.I)})
    channels = (["action"] if action or gaze else []) + (["emotion"] if emotion else [])
    if not channels:
        why = f"no action/emotion content (matched {other})" if other else "no action/emotion content"
        return {"status": "unsupported", "channels": [], "prompt": None,
                "detail": f"{CONTROL_MAPPING_VERSION}: {why}; AptAvatar is steered only by its per-chunk text prompt"}
    detail = f"{CONTROL_MAPPING_VERSION}: appended to the text prompt ({'+'.join(channels)}) from the next chunk"
    if other:
        detail += f"; parts without a counterpart ignored ({', '.join(other)})"
    return {"status": "applied", "channels": channels, "prompt": clean, "detail": detail}


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
        os.chdir(repo)                       # AptAvatar/inference.py reads its yaml relative to cwd
        sys.path[:0] = [str(HERE.parent / "soulx_flashtalk" / "shims"), str(repo)]
        import torch
        import torch.distributed as dist
        self.torch, self.dist = torch, dist
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        from AptAvatar import inference as ai
        self.ai = ai
        ai.target_size = (self.height, self.width)
        p = ai.infer_params
        self.frame_num, self.motion = p["frame_num"], p["motion_frames_num"]      # 73, 9
        self.slice = self.frame_num - self.motion                                 # 64 new frames per chunk
        self.ctx = int(args.audio_context * SR)
        t0 = time.time()
        self.pipe = ai.get_pipeline(world_size=self.world, ckpt_dir=args.ckpt_dir, wav2vec_dir=args.wav2vec_dir)
        self.ctrl = None
        if self.world > 1:
            from datetime import timedelta
            self.ctrl = dist.new_group(backend="gloo", timeout=timedelta(days=7))
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
        self.ai.get_base_data(self.pipe, input_prompt=cmd["prompt"], cond_image=image, base_seed=int(cmd["seed"]))
        self.prompt = cmd["prompt"]

    def chunk(self, cmd):
        torch = self.torch
        t0 = time.perf_counter()
        if cmd["prompt"] != self.prompt:
            self.pipe.set_input_prompt(cmd["prompt"])
            self.prompt = cmd["prompt"]
        emb = self.ai.get_audio_embedding(self.pipe, np.asarray(cmd["ctx"], dtype=np.float32))
        emb = emb[:, -self.frame_num:].contiguous()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        video = self.ai.run_pipeline(self.pipe, emb)                               # [73, H, W, 3] float 0..255
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        self.timing = {"audio_ms": round((t1 - t0) * 1000, 1), "model_ms": round((t2 - t1) * 1000, 1)}
        if self.rank != 0:
            return None
        keep = video if cmd["first"] else video[self.motion:]
        return keep.round().clamp(0, 255).to(torch.uint8).cpu().numpy()

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
            ctx = np.zeros(self.ctx, np.float32) if n % 2 == 0 else (rng.standard_normal(self.ctx) * 0.05).astype(np.float32)
            prompt = DEFAULT_PROMPT if n < 2 else DEFAULT_PROMPT + " The person raises the right hand to the chest."
            self.run(self.bcast({"op": "chunk", "ctx": ctx, "prompt": prompt, "first": n == 0}))
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
        self.control_prompt = None
        self.track = ""
        self.out: asyncio.Queue = asyncio.Queue()
        self.stats = {"chunks": [], "events": [], "frames": 0, "late_chunks": 0, "reanchors": 0}

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
                k = 0
                while not self.stop.is_set():
                    f0 = k * eng.slice                                  # first frame of the chunk's 73-frame window
                    end = (f0 + eng.frame_num) * SPF                    # audio needed up to the window's last frame
                    tl = self.window(end)
                    if self.stop.is_set():
                        break
                    ctx = tl[max(0, end - eng.ctx):end]
                    if len(ctx) < eng.ctx:
                        ctx = np.concatenate([np.zeros(eng.ctx - len(ctx), np.float32), ctx])
                    prompt = init["prompt"] if not self.control_prompt else f"{init['prompt']} {self.control_prompt}"
                    t0 = time.monotonic()
                    frames = eng.run(eng.bcast({"op": "chunk", "ctx": ctx, "prompt": prompt, "first": k == 0}))
                    t1 = time.monotonic()
                    late = t1 - (self.t_ready + end / SR)
                    self.stats["chunks"].append({"n": k, "first_frame": 0 if k == 0 else f0 + eng.motion,
                                                 "frames": len(frames), "audio_end_s": round(end / SR, 3),
                                                 "start_s": round(t0 - self.t_ready, 3),
                                                 "done_s": round(t1 - self.t_ready, 3),
                                                 "compute_ms": round((t1 - t0) * 1000, 1), **eng.timing,
                                                 "prompt_changed": prompt != init["prompt"]})
                    if late > eng.slice / FPS + 0.5:
                        self.stats["late_chunks"] += 1
                    first = 0 if k == 0 else f0 + eng.motion
                    for j, fr in enumerate(frames):
                        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
                        i = first + j
                        self.loop.call_soon_threadsafe(self.out.put_nowait,
                                                       (jpg.tobytes(), to48k(tl[i * SPF:(i + 1) * SPF])))
                    k += 1
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

    async def start(self, m):
        img_b64 = m.get("image_b64")
        if not img_b64:
            await self.conn.error("init.image_b64 (portrait) required")
            return False
        self.track = m.get("track") or ""
        raw = base64.b64decode(img_b64)
        seed = int(m.get("seed") if m.get("seed") is not None else self.args.seed)
        base = strip_boilerplate((m.get("prompt") or "").strip()) if self.track == "avatar_performance" else ""
        prompt = base or self.args.prompt
        self.error = None
        self.ready_evt = asyncio.Event()
        self.stats.update({"case_id": m.get("case_id"), "track": self.track, "seed": seed, "prompt": prompt})
        self.thread = threading.Thread(target=self.render, args=({"image": raw, "prompt": prompt, "seed": seed},),
                                       daemon=True)
        self.thread.start()
        await self.ready_evt.wait()
        if self.error:
            await self.conn.error(self.error)
            return False
        cfg = {**self.server.config, "seed": seed, "prompt": prompt, "portrait_sha256": hashlib.sha256(raw).hexdigest()}
        await self.conn.ready(MODEL, self.server.config["repo_commit"] or "unknown", self.eng.width, self.eng.height, FPS,
                              protocol=sp.PROTOCOL_VERSION, config=cfg)
        log("ready", json.dumps({"seed": seed, "case": m.get("case_id"), "track": self.track}))
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
                elif kind in ("text", "control"):
                    eid = msg.get("event_id")
                    text = msg.get("text") or ""
                    if kind == "control":
                        params = msg.get("params") or {}
                        text = " ".join(str(x) for x in (msg.get("action"), params.get("text") or params.get("prompt")) if x)
                    if self.track.startswith("digital_human") and kind == "text":
                        res = {"status": "applied",
                               "detail": "renderer-only: speech for this turn arrives as audio_in from the composition layer"}
                    else:
                        res = map_control(text)
                        if res["status"] == "applied":
                            self.control_prompt = res["prompt"]
                    t = None if self.t_ready is None else round(time.monotonic() - self.t_ready, 3)
                    self.stats["events"].append({"event_id": eid, "at_s": t, **res})
                    await self.conn.ack(eid, res["status"], res["detail"])
                elif kind == "reference":
                    await self.conn.ack(msg.get("event_id"), "unsupported", "AptAvatar has no reference-image editing")
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
        rev = MODEL_DIR / "weights/AptAvatar/.complete"
        p = eng.ai.infer_params
        self.config = {
            "model": "AptAvatar (14B)", "repo": "https://github.com/TaoLiveAIGC/AptAvatar", "repo_commit": commit,
            "hf_weights": "TaoLiveAIGC/AptAvatar" + (f"@{rev.read_text().strip()}" if rev.exists() else ""),
            "audio_encoder": "TencentGameMate/chinese-wav2vec2-base", "weights": weights,
            "width": eng.width, "height": eng.height, "fps": FPS, "steps": p["sample_steps"],
            "sample_shift": p["sample_shift"], "cfg": "none (distilled)",
            "color_correction_strength": p["color_correction_strength"],
            "chunk_frames": eng.slice, "window_frames": eng.frame_num, "motion_frames": eng.motion,
            "audio_context_s": args.audio_context, "control_mapping": CONTROL_MAPPING_VERSION,
            "prompt": "avatar_performance: init.prompt (rendering_context, benchmark boilerplate stripped) + latest "
                      "control, per chunk; digital_human: upstream default prompt",
            "gpu": eng.gpu, "gpu_count": eng.world,
            "parallel": f"xfuser ulysses x{eng.world}" if eng.world > 1 else "single GPU",
            "torch": eng.torch.__version__, "torch_compile": True, "load_s": round(eng.load_s, 1),
            "xformers": "shim: torch SDPA (exact) for the audio cross-attention",
        }

    def write_stats(self, sess):
        s = {k: v for k, v in sess.stats.items()}
        c = s["chunks"]
        if c:
            frames = sum(x["frames"] for x in c)
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
    ap.add_argument("--port", type=int, default=18881)
    ap.add_argument("--repo", default=str(MODEL_DIR / "repo"))
    ap.add_argument("--ckpt_dir", default=str(MODEL_DIR / "weights/AptAvatar"))
    ap.add_argument("--wav2vec_dir", default=str(MODEL_DIR / "weights/chinese-wav2vec2-base"))
    ap.add_argument("--size", default="512*512", help="W*H (upstream default 704*1280; benchmark portraits are square)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audio_context", type=float, default=8.0, help="s of past audio for each chunk's wav2vec pass")
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
