"""StreamEval self-host server for JoyAI-Video-Edit (JD, Apache-2.0).

Implements docs/protocol.md on top of the upstream
streaming runtime (xvideo.serving.joyomni_streaming, repo pinned in setup.sh):

  ws://127.0.0.1:<port>/session   one connection = one session

  init          -> ready {model, version, width, height, fps, system_config}
  source_frame  b"SF"+u64 pts_ns+JPEG, pushed at source fps; the causal DiT edits
                every 4-frame chunk (first chunk = 1 frame) as it arrives
  text          instruction V2V: text is the full edit prompt (verbatim, no
                accumulation); the model conditions on one prompt per KV session,
                so the stream restarts its causal session on the next live frame
  reference     RV2V with one reference image (the model has exactly one
                reference slot). The server keeps the authored slot state
                (set replaces, patch merges, remove drops); a resulting state of
                exactly one slot is applied, anything else (2+ slots, or an empty
                state) is acked `unsupported` and the previous state is kept.
  control       acked `unsupported` (no structured control input)
  end           closes the session

  frame         b"VF"+u64 pts_ns+JPEG, pts_ns = time.monotonic_ns() when sent; the
                frames of a decoded chunk are spaced at the source frame period

Before the first instruction / reference (no edit state), the source frame is
passed through unchanged at the output resolution (identity; declared in
system_config). Real time only: frames are never read ahead; when the model
falls behind, incoming source frames are dropped (upstream camera policy,
max 2 chunks in flight) and counted.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

JOY_ROOT = Path(os.environ.get("JOY_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/joyai_video_edit"))
DEPLOY = JOY_ROOT / "repo" / "deploy"
sys.path.insert(0, str(DEPLOY))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repository root -> streameval

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from streameval.runtime import protocol as sp  # noqa: E402

MODEL = "JoyAI-Video-Edit"
DIT_NAME = "joyai_video_edit_dit_0811.pth"
# upstream RV2V phrasing ("Replace ... with ... from the reference image"); the slot scope names the target, the
# benchmark's generic event-level rules (msg.scope) are not passed: they dilute the short prompt the model expects
REFERENCE_TEMPLATE = ("Replace the {target} in the video with the one from the reference image, matching its colors, "
                      "material and shape. Keep everything else, the motion and the camera unchanged.")


def now_ns() -> int:
    return time.monotonic_ns()


def _read(path: Path) -> str | None:
    try:
        return path.read_text().split()[0]
    except Exception:
        return None


def build_identity(args) -> dict:
    w = JOY_ROOT / "weights"
    repo_commit = None
    try:
        repo_commit = subprocess.run(["git", "-C", str(JOY_ROOT / "repo"), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass
    hf_meta = next((w / "JoyAI-Video-Edit/.cache/huggingface/download/dit").glob("*.metadata"), None) \
        if (w / "JoyAI-Video-Edit/.cache/huggingface/download/dit").is_dir() else None
    hf_revision = hf_meta.read_text().splitlines()[0] if hf_meta else None
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "model": MODEL,
        "version": f"dit_0811@{(repo_commit or 'unknown')[:12]}",
        "provider": "self-hosted (JD / jd-opensource)",
        "license": "Apache-2.0",
        "upstream_repo": "https://github.com/jd-opensource/JoyAI-Video-Edit",
        "upstream_commit": repo_commit,
        "weights": {
            "hf_repo": "jdopensource/JoyAI-Video-Edit", "hf_revision": hf_revision,
            "dit": DIT_NAME, "dit_sha256": _read(w / "dit_0811.sha256"),
            "vae_sha256": _read(w / "vae.sha256"),
            "text_encoder": "XiaomiMiMo/MiMo-VL-7B-RL-2508",
        },
        "resolution": {"landscape": [args.width, args.height], "portrait": [args.height, args.width]},
        "fps": args.fps,
        "num_inference_steps": args.steps,
        "seed_policy": "init.seed (fallback --seed) seeds torch generator of every causal session",
        "gpu": gpu,
        "torch": torch.__version__,
        "env_flags": {k: os.environ.get(k) for k in (
            "JOYOMNI_SAGE_ATTN", "JOYOMNI_FP8_IMG", "JOYOMNI_FP8_TXT", "JOYOMNI_FP8_FAST_ACCUM",
            "JOYOMNI_LOW_VRAM", "JOYOMNI_CUDA_GRAPH")},
        "streaming": {"chunk_frames": "1 then 4 (causal, VAE temporal factor)", "max_inflight_chunks": args.max_inflight,
                      "kv_reset_frames": args.kv_reset_frames, "max_temporal_ids": 8, "freeze_kv_on_static": True,
                      "presence_gates": "off", "prompt_enhancement": "off",
                      "output_jpeg_quality": args.jpeg_quality},
        "instruction_events": "text sent verbatim as the full prompt; causal session restarted on the next live frame",
        "reference_events": {"slots": 1,
                             "prompt": "event text if given, else the template over the slot scope's target phrase",
                             "template": REFERENCE_TEMPLATE,
                             "unsupported": "state with 2+ slots or empty state; previous state kept"},
        "before_first_edit": "identity passthrough of the live source at output resolution",
        "control_events": "unsupported",
    }


class Engine:
    """Owns the upstream runtime; one active streaming session at a time."""

    def __init__(self, args):
        from xvideo.inductor_autotune_fix import install as _fix
        _fix()
        from xvideo.serving.joyomni_streaming import JoyOmniRuntime, JoyOmniV2VStreamingSession, StreamingSettings
        self.StreamingSettings = StreamingSettings
        self.args = args
        ck = JOY_ROOT / "weights"
        t0 = time.time()
        # a cold compile cache keeps the load-time warmup session's VAE-decode worker busy for minutes; upstream
        # closes it with a 5 s timeout and aborts the load, so let warmup sessions finish compiling instead
        close = JoyOmniV2VStreamingSession.close
        JoyOmniV2VStreamingSession.close = lambda sess, timeout=5.0: close(sess, max(timeout, 1800.0))
        try:
            self.runtime = JoyOmniRuntime.load(
                str(ck / "JoyAI-Video-Edit/dit" / DIT_NAME), vae_ckpt=str(ck / "JoyAI-Video-Edit/vae"),
                text_encoder_ckpt=str(ck / "MiMo-VL-7B-RL-2508"), device=args.device,
                vae_encode_device=args.device, vae_decode_device=args.device, vae_pseudo_device=args.device,
                postprocess_device=args.device, seed=args.seed, warmup_height=args.height, warmup_width=args.width)
        finally:
            JoyOmniV2VStreamingSession.close = close
        self.runtime.output_quality = args.jpeg_quality
        self.load_s = time.time() - t0
        self.align = self.runtime.pipeline.vae.stem.stride * 8
        self.lock = threading.Lock()   # one session per server (single GPU)


class Session:
    def __init__(self, engine: Engine, conn: sp.ServerConnection, loop: asyncio.AbstractEventLoop):
        self.e, self.conn, self.loop = engine, conn, loop
        self.args = engine.args
        self.prompt: str | None = None
        self.slots: dict[str, dict] = {}
        self.ref_image: Image.Image | None = None
        self.model_session = None
        self.gen = 0            # edit-state generation, bumped by every applied event
        self.applied_gen = -1
        self.state_lock = threading.Lock()
        self.frames: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.stats = {"frames_in": 0, "frames_out": 0, "frames_passthrough": 0, "frames_dropped_backlog": 0,
                      "frames_dropped_inflight": 0, "session_restarts": 0, "chunks": 0, "chunk_latency_ms": []}
        self.frames_since_reset = 0

    # ---- sending (worker threads -> event loop) ------------------------------
    def emit_frame(self, jpeg: bytes):
        fut = asyncio.run_coroutine_threadsafe(self.conn.frame(jpeg, now_ns()), self.loop)
        fut.add_done_callback(lambda f: f.exception() and self.stop.set())
        self.stats["frames_out"] += 1

    # ---- init ----------------------------------------------------------------
    def configure(self, msg: dict):
        a = self.args
        self.seed = int(msg.get("seed", a.seed))
        sw, sh = int(msg.get("width") or a.width), int(msg.get("height") or a.height)
        long_, short = max(a.width, a.height), min(a.width, a.height)
        long_ -= long_ % self.e.align
        short -= short % self.e.align
        self.out_w, self.out_h = (short, long_) if sh > sw else (long_, short)
        self.track = msg.get("track")
        self.period_ns = int(1e9 / min(float(msg.get("fps") or a.fps), a.fps))
        self.settings = self.e.StreamingSettings(height=self.out_h, width=self.out_w, num_inference_steps=a.steps,
                                                 seed=self.seed, max_temporal_ids=8, freeze_kv_on_static=True,
                                                 static_diff_thresh=0.5, output_codec="mjpeg")
        with self.state_lock:
            if msg.get("image_b64") and self.track and "reference" in str(self.track):
                self.ref_image = decode_image(msg["image_b64"])
                self.slots = {"init": {"scope": "", "image": self.ref_image}}
            if msg.get("prompt"):
                self.prompt = str(msg["prompt"])
                self.gen += 1

    # ---- events --------------------------------------------------------------
    def on_text(self, msg: dict) -> dict:
        text = str(msg.get("text") or "").strip()
        if not text:
            return {"status": "error", "detail": "empty text"}
        with self.state_lock:
            self.prompt = text
            self.gen += 1
        return {"status": "applied"}

    def on_reference(self, msg: dict) -> dict:
        """Single-slot state machine (same rule as edit_v1.reference_dispositions)."""
        op = msg.get("op")
        slots = msg.get("slots") or []
        state = dict(self.slots)
        if op == "set":
            state = {str(s["slot"]): dict(s) for s in slots}
        elif op == "patch":
            for s in slots:
                prev = state.get(str(s["slot"]), {})
                state[str(s["slot"])] = {**prev, **{k: v for k, v in s.items() if v is not None}}
        elif op == "remove":
            for s in slots:
                state.pop(str(s["slot"] if isinstance(s, dict) else s), None)
            for s in msg.get("remove") or []:
                state.pop(str(s), None)
        elif op == "reset":
            state = {}
        else:
            return {"status": "unsupported", "detail": f"reference op {op!r}"}
        if len(state) != 1:
            return {"status": "unsupported",
                    "detail": "multiple_independent_reference_slots_not_supported" if state
                    else "empty_state_restore_has_no_expressible_input"}
        (slot, spec), = state.items()
        image = spec.get("image")
        if spec.get("image_b64"):
            image = decode_image(spec["image_b64"])
        if image is None:
            return {"status": "error", "detail": f"slot {slot} has no image"}
        prompt = msg.get("text") or msg.get("prompt")
        if not prompt:
            target = str(spec.get("scope") or "main subject").split(". Apply")[0].strip().rstrip(".")
            prompt = REFERENCE_TEMPLATE.format(target=target)
        with self.state_lock:
            self.slots = {slot: {"scope": spec.get("scope", ""), "image": image}}
            self.ref_image = image
            self.prompt = prompt
            self.gen += 1
        return {"status": "applied", "detail": f"slot={slot}"}

    # ---- worker threads ------------------------------------------------------
    def _close_model_session(self):
        # detach first: the output pump treats errors of the current session as fatal, of a replaced one as benign
        s, self.model_session = self.model_session, None
        if s is not None:
            try:
                s.request_stop()
                s.close(timeout=30.0)
            except Exception as exc:
                print(f"[joyai] close error {exc!r}", flush=True)

    def worker(self):
        """Feeds live frames into the causal session; restarts it on new edit state."""
        try:
            while not self.stop.is_set():
                try:
                    item = self.frames.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    break
                # real time: never let the backlog grow, keep only the newest frame
                while self.frames.qsize() > 0:
                    nxt = self.frames.get_nowait()
                    if nxt is None:
                        self.stop.set()
                        break
                    self.stats["frames_dropped_backlog"] += 1
                    item = nxt
                pts, jpeg = item
                frame = Image.open(io.BytesIO(jpeg)).convert("RGB")
                with self.state_lock:
                    gen, prompt, ref = self.gen, self.prompt, self.ref_image
                if prompt is None:
                    out = frame.resize((self.out_w, self.out_h), Image.BICUBIC)
                    buf = io.BytesIO()
                    out.save(buf, format="JPEG", quality=self.args.jpeg_quality)
                    self.emit_frame(buf.getvalue())
                    self.stats["frames_passthrough"] += 1
                    continue
                if gen != self.applied_gen or (self.args.kv_reset_frames and
                                            self.frames_since_reset >= self.args.kv_reset_frames):
                    with self.out_lock:
                        self._close_model_session()
                        self.model_session = self.e.runtime.create_v2v_session(
                            prompt=prompt, settings=self.settings, ref_image=ref)
                    self.applied_gen = gen
                    self.frames_since_reset = 0
                    self.stats["session_restarts"] += 1
                s = self.model_session
                if s.initialized and s.inflight_chunks() >= self.args.max_inflight:
                    self.stats["frames_dropped_inflight"] += 1
                    continue
                s.push_frame(frame, frame_meta={"seq": self.stats["frames_in"], "t_capture_ms": pts / 1e6,
                                                "recv_ns": now_ns()}, drain_results=False)
                self.frames_since_reset += 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.fatal = repr(exc)
            self.stop.set()

    def output_pump(self):
        while not self.stop.is_set():
            with self.out_lock:
                s = self.model_session
            if s is None:
                time.sleep(0.005)
                continue
            try:
                r = s.wait_async_result(0.02)
            except Exception as exc:
                if s is self.model_session:
                    self.fatal = repr(exc)
                    self.stop.set()
                continue
            if r is None:
                continue
            jpegs = r.jpegs[: r.valid_count] if r.valid_count is not None else r.jpegs
            done = now_ns()
            metas = r.source_metas or []
            if metas and metas[-1].get("recv_ns"):
                self.stats["chunk_latency_ms"].append((done - metas[-1]["recv_ns"]) / 1e6)
            self.stats["chunks"] += 1
            for j in jpegs:
                self.paced.put(bytes(j))

    def pacer(self):
        """A chunk decodes 8 frames at once; send them spaced at the source frame period (the chunk's first frame
        is not delayed), catching up without spacing when more than a chunk is queued."""
        last = 0
        while not self.stop.is_set():
            try:
                jpeg = self.paced.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.paced.qsize() < 8:
                time.sleep(max(0.0, (last + self.period_ns - now_ns()) / 1e9))
            last = now_ns()
            self.emit_frame(jpeg)

    def start_threads(self):
        self.fatal = None
        self.out_lock = threading.Lock()
        self.paced: queue.Queue = queue.Queue()
        self.threads = [threading.Thread(target=t, daemon=True) for t in (self.worker, self.output_pump, self.pacer)]
        for t in self.threads:
            t.start()

    def shutdown(self):
        self.frames.put(None)
        self.stop.set()
        for t in getattr(self, "threads", []):
            t.join(timeout=10)
        self._close_model_session()


def decode_image(b64: str) -> Image.Image:
    if "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


async def handle(engine: Engine, identity: dict, conn: sp.ServerConnection):
    if not engine.lock.acquire(blocking=False):
        await conn.error("server busy (one session per GPU)")
        return
    sess = Session(engine, conn, asyncio.get_running_loop())
    started = False
    try:
        while True:
            try:
                m = await conn.recv()
            except sp.ConnectionClosed:
                break
            if sess.stop.is_set() and getattr(sess, "fatal", None):
                await conn.error(sess.fatal)
                break
            if isinstance(m, bytes):
                tag, pts, payload = sp.unpack(m)
                if tag == sp.SOURCE_FRAME and started:
                    sess.stats["frames_in"] += 1
                    sess.frames.put((pts, payload))
                continue
            kind, eid = m.get("type"), m.get("event_id")
            if kind == "init":
                sess.configure(m)
                sess.start_threads()
                started = True
                await conn.ready(MODEL, identity["version"], sess.out_w, sess.out_h, engine.args.fps,
                                 protocol=sp.PROTOCOL_VERSION, config=identity)
            elif kind == "text":
                ack = sess.on_text(m)
                await conn.ack(eid, ack["status"], ack.get("detail"))
            elif kind == "reference":
                try:
                    ack = sess.on_reference(m)
                except Exception as exc:
                    ack = {"status": "error", "detail": repr(exc)}
                await conn.ack(eid, ack["status"], ack.get("detail"))
            elif kind == "control":
                await conn.ack(eid, "unsupported", "JoyAI-Video-Edit has no structured control input")
            elif kind == "end":
                break
            else:
                await conn.error(f"unknown message type {kind!r}")
    finally:
        await asyncio.to_thread(sess.shutdown)
        engine.lock.release()
        print(f"[joyai] session closed {json.dumps(session_stats(sess))}", flush=True)


def session_stats(sess: Session) -> dict:
    lat = sess.stats["chunk_latency_ms"]
    out = {k: v for k, v in sess.stats.items() if k != "chunk_latency_ms"}
    if lat:
        out["chunk_latency_ms"] = {"n": len(lat), "p50": float(np.percentile(lat, 50)),
                                   "p90": float(np.percentile(lat, 90)), "max": float(max(lat))}
    return out


def configure_env():
    """Upstream per-GPU launch flags (DEPLOYMENT.md §4) + per-GPU compile caches, unless already set."""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0].strip()
    except Exception:
        gpu = "unknown"
    if "5090" in gpu:
        for k, v in {"JOYOMNI_SAGE_ATTN": "1", "JOYOMNI_FP8_FAST_ACCUM": "1", "JOYOMNI_LOW_VRAM": "1"}.items():
            os.environ.setdefault(k, v)
    try:   # light joyomni_ops build (no FP8 GEMM for this arch) -> bf16 Linears
        import joyomni_ops
        has_fp8 = bool(joyomni_ops.has_fp8())
    except Exception:
        has_fp8 = False
    if not has_fp8:
        os.environ.setdefault("JOYOMNI_FP8_IMG", "0")
        os.environ.setdefault("JOYOMNI_FP8_TXT", "0")
    for k, v in {"JOYOMNI_FP8_IMG": "1", "JOYOMNI_FP8_TXT": "1", "JOYOMNI_CUDA_GRAPH": "1",
                 "JOYOMNI_SAGE_ATTN": "0", "JOYOMNI_FP8_FAST_ACCUM": "0", "JOYOMNI_LOW_VRAM": "0"}.items():
        os.environ.setdefault(k, v)
    cache = JOY_ROOT / ("cache_" + "".join(c if c.isalnum() else "_" for c in gpu.lower()))
    for k, sub in (("TORCHINDUCTOR_CACHE_DIR", "torchinductor"), ("TRITON_CACHE_DIR", "triton"),
                   ("CUDA_CACHE_PATH", "nv_compute")):
        os.environ.setdefault(k, str(cache / sub))
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env_bin = JOY_ROOT / "env" / "bin"
    os.environ.setdefault("CUDA_HOME", str(JOY_ROOT / "env"))
    if str(env_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{env_bin}:{os.environ.get('PATH', '')}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8774)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--width", type=int, default=int(os.environ.get("JOYOMNI_WIDTH", 840)))
    p.add_argument("--height", type=int, default=int(os.environ.get("JOYOMNI_HEIGHT", 480)))
    p.add_argument("--fps", type=int, default=int(os.environ.get("JOYOMNI_FPS", 24)))
    p.add_argument("--steps", type=int, default=2, help="denoising steps (upstream default 2)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-inflight", type=int, default=2)
    p.add_argument("--kv-reset-frames", type=int, default=600, help="upstream default periodic KV reset")
    p.add_argument("--jpeg-quality", type=int, default=90)
    args = p.parse_args()
    configure_env()
    import logging
    logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
    engine = Engine(args)
    identity = build_identity(args)
    print(f"[joyai] runtime loaded in {engine.load_s:.1f}s {json.dumps(identity)}", flush=True)
    health = lambda: {"ok": True, "model": MODEL, "version": identity["version"], "gpu": identity["gpu"],
                      "load_s": round(engine.load_s, 1), "busy": engine.lock.locked()}
    sp.serve(lambda conn: handle(engine, identity, conn), args.host, args.port, health=health)


if __name__ == "__main__":
    main()
