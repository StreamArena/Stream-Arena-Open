"""Matrix-Game 3.5 (distilled first-person causal model) self-hosted server, docs/protocol.md.

One WebSocket connection = one session:
  init (prompt + image_b64)  -> build the one-scene input (anchor image, static camera, prompt), VAE-encode the
                                anchor and set up the rollout (C0 depth registration, KV/memory) up to chunk 0, then
                                `ready` + the anchor frame; the causal rollout runs chunk by chunk in a worker thread
                                (3 latents = 12 frames @ 16 fps per chunk) and every chunk is decoded and pushed
                                as JPEG `frame`s as soon as it is generated. Every chunk also registers its frames
                                into Patch Memory (DA3 depth + VAE encode), as upstream does.
  text kind=user_prompt      -> prompt switch: the next chunk is conditioned on
                                "<init prompt>\n<latest instruction>" (T5 re-encode; ack "applied" when the chunk that
                                first uses it starts).
  text kind=camera_control   -> ack "unsupported": the model is driven by metric 6-DoF camera poses; free-text
                                camera language (pan/orbit/follow ...) has no exact pose, so nothing is mapped.
  reference / control        -> ack "unsupported".
  init without image_b64     -> `error`: the released checkpoint is image-anchored (TI2V); upstream documents no
                                text-only / text-to-image start, so core text-only cases are not supported.

Upstream code: Riemann-Dynamics/Matrix-Game-3.5 (unpacked by setup.sh into $MG_REPO); causal_rollout.py is loaded with two
one-line hooks (per-chunk prompt context, per-chunk decoded frames) and otherwise runs unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import io
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repository root -> streameval
from streameval.runtime import protocol as proto  # noqa: E402

MODEL_DIR = Path(os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/matrix_game_35")
MG_REPO = Path(os.environ.get("MG_REPO", MODEL_DIR / "repo"))  # upstream code, see setup.sh
WEIGHTS = MODEL_DIR / "weights"                                  # overridden by --weights
MODEL = "matrix_game_35"
FPS = 16  # Wan2.2 TI2V / Matrix-Game output frame rate (video_io._write_video default)


class SessionEnd(Exception):
    pass


def log(*a):
    print(time.strftime("%H:%M:%S"), "[mg35]", *a, flush=True)


# ---------------------------------------------------------------------------------------------- model loading

def _load_patched_rollout():
    """Load causal_rollout.py with two hook calls; everything else is the upstream kernel."""
    import examples.wanvideo.pipeline.mosaic.causal_rollout as orig
    src = Path(orig.__file__).read_text()
    a = "        context = section_contexts[section_idx]\n"
    b = "        gt_prefix_frames = None\n        gt_prefix_frame_indices = None\n"
    assert src.count(a) == 1 and src.count(b) == 1, "upstream causal_rollout.py changed; re-check hooks"
    src = src.replace(a, "        context = _SA_HOOK.context(i, section_contexts[section_idx], _enc, neg_prompt,"
                         " cfg_scale, device, dtype)\n")
    src = src.replace(b, "        generated_frames = _SA_HOOK.chunk_done(i, latents, generated_frames, all_clean,"
                         " initial_c0, pipe, device, vae_decode_tiled, cs)\n" + b)
    mod = types.ModuleType(orig.__name__ + "_sa")
    mod.__file__ = orig.__file__
    mod.__package__ = orig.__package__
    mod.__dict__["_SA_HOOK"] = None
    exec(compile(src, orig.__file__, "exec"), mod.__dict__)
    return mod


class Engine:
    def __init__(self, args):
        os.chdir(MG_REPO)
        sys.path.insert(0, str(MG_REPO))
        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        os.environ["DA3_MODEL_PATH"] = str(WEIGHTS / "DA3NESTED-GIANT-LARGE-1.1")
        import accelerate
        import torch
        import infer as base_infer
        from distilled_config import load_inference_config
        from examples.wanvideo.pipeline.mosaic.causal_config import build_runtime_args
        import importlib.util
        spec = importlib.util.spec_from_file_location("mg35_run_distilled",
                                                      MG_REPO / "tools/run_distilled_inference.py")
        rdi = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rdi)
        build_model = rdi.build_model
        self.torch, self.base_infer = torch, base_infer
        self.build_runtime_args = build_runtime_args
        self.config = dataclasses.replace(load_inference_config(MG_REPO / "configs/infer_distilled.yaml"),
                                          num_blocks=int(args.max_blocks))
        self.ckpt = WEIGHTS / "Matrix-Game-3.5-Distilled/first-person.safetensors"
        self.wan_dir = WEIGHTS / "Wan2.2-TI2V-5B"
        self.tok_dir = self.wan_dir / "google/umt5-xxl"
        self.work = Path(tempfile.mkdtemp(prefix="mg35_"))
        self.rollout = _load_patched_rollout()
        from examples.wanvideo.pipeline.mosaic import causal_inference as ci
        self.ci = ci
        self.tiled = bool(args.vae_tiled)
        # model construction needs a parsed arg namespace; any valid one-scene workspace works
        boot = self._workspace("boot", MG_REPO / "samples/distilled/suburban_street_6blocks/input.png", "boot")
        rt = self._runtime_args(boot, self.config)
        self.accelerator = accelerate.Accelerator()
        t0 = time.time()
        self.model = build_model(rt, self.accelerator)
        self.model.eval()
        self.model.vae_decode_tiled = self.tiled
        log(f"model loaded in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}")
        self.estimator = ci._init_registration_estimator(rt, device=self.model.pipe.device)
        self.gpu = torch.cuda.get_device_name(0)
        self.weights_sha = self._weights_hash()

    def _weights_hash(self):
        f = WEIGHTS / "weights.sha256"
        if f.exists():
            return f.read_text().split()[0]
        sums = self.ckpt.parent / "SHA256SUMS"
        return sums.read_text().split()[0] if sums.exists() else "unknown"

    def _workspace(self, tag, image_path, prompt):
        root = self.work / tag
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        import numpy as np
        from PIL import Image
        w, h = Image.open(image_path).size
        # static camera: identity pose; pinhole intrinsics with the sample's field of view (fx ~ 0.495 W)
        extr = np.eye(4, dtype=np.float32)[None]
        intr = np.array([[0.495 * w, 0.495 * w, w / 2.0, h / 2.0]], np.float32)
        cam = root / "camera.npz"
        np.savez(cam, extrinsics_c2w=extr, intrinsics=intr)
        ns = types.SimpleNamespace(person="first", image=str(image_path), camera=str(cam), caption="",
                                   prompt=prompt, prompt_file="", refs="", num_blocks=self.config.num_blocks,
                                   camera_convention="c2w")
        ws = Path(self.base_infer.build_input_workspace(ns, str(root)))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(MG_REPO) + os.pathsep + env.get("PYTHONPATH", "")
        index = Path(self.base_infer.build_index(str(ws), sys.executable, env))
        return {"root": root, "ws": ws, "index": index}

    def _runtime_args(self, space, config):
        rt = self.build_runtime_args(config, checkpoint=self.ckpt, dataset_index=space["index"],
                                     workspace=space["ws"], wan_dir=self.wan_dir, tokenizer_dir=self.tok_dir,
                                     memory_cache_dir=space["root"] / "memory_latents")
        rt.dataset_pass_id, rt.start_epoch, rt.start_global_step, rt.rank = 0, 0, 0, 0
        return rt

    def system_config(self):
        c = self.config
        return {"model_version": "Matrix-Game-3.5-Distilled first-person (3-step causal, CFG 3)",
                "weights_sha256": self.weights_sha, "width": c.width, "height": c.height, "fps": FPS,
                "steps": len(c.schedule), "schedule": list(c.schedule), "cfg_scale": c.student_cfg_scale,
                "chunk_frames": c.chunk_size * 4, "gpu": self.gpu, "camera": "static identity pose",
                "vae_decode_tiled": self.tiled}


# ---------------------------------------------------------------------------------------------- one session

class Session:
    """Drives one rollout in a worker thread; talks to the event loop through thread-safe queues."""

    def __init__(self, engine: Engine, init: dict, conn, loop, out_q: asyncio.Queue):
        self.e, self.conn, self.loop, self.out_q = engine, conn, loop, out_q
        self.base_prompt = (init.get("prompt") or "").strip()
        self.seed = int(init.get("seed", engine.config.seed))
        self.pending = queue.Queue()         # (event_id, text)
        self.stop = threading.Event()
        self.primed = threading.Event()      # rollout set up (C0 depth registration, KV/memory init), at chunk 0
        self.go = threading.Event()          # `ready` sent: generation may start
        self.t_ready = None
        self.chunk_s = engine.config.chunk_size * 4 / FPS
        self.cur_context = None
        self.stats = {"chunks": 0, "chunk_s": []}
        self._t_chunk = None

    def _emit(self, item):
        self.loop.call_soon_threadsafe(self.out_q.put_nowait, item)

    # hooks called from inside the upstream rollout loop ------------------------------------------------------
    def context(self, i, default_ctx, enc, neg_prompt, cfg_scale, device, dtype):
        if i == 0:  # per-session setup is done before `ready`, so the clock starts at generation
            self.primed.set()
            while not (self.go.wait(0.05) or self.stop.is_set()):
                pass
        if self.stop.is_set():
            raise SessionEnd()
        # real time: do not run more than one chunk ahead of the wall clock
        if self.t_ready is not None:
            ahead = self.t_ready + (i - 1) * self.chunk_s - time.monotonic()
            while ahead > 0 and not self.stop.is_set():
                time.sleep(min(ahead, 0.05))
                ahead = self.t_ready + (i - 1) * self.chunk_s - time.monotonic()
        latest, applied = None, []
        while True:
            try:
                eid, text = self.pending.get_nowait()
            except queue.Empty:
                break
            latest = text
            applied.append(eid)
        if latest is not None:
            prompt = f"{self.base_prompt}\n{latest}" if self.base_prompt else latest
            pos = enc(prompt).to(device=device, dtype=dtype)
            self.cur_context = pos if cfg_scale == 1.0 else self.e.torch.cat(
                [enc(neg_prompt), pos], dim=0).to(device=device, dtype=dtype)
            for eid in applied:  # sent at once, not queued behind the paced frames
                asyncio.run_coroutine_threadsafe(self.conn.ack(eid, "applied", f"prompt switched at chunk {i} "
                                                               f"(frame {1 + i * 12})"), self.loop)
        self._t_chunk = time.monotonic()
        return self.cur_context if self.cur_context is not None else default_ctx

    def chunk_done(self, i, latents, generated_frames, all_clean, initial_c0, pipe, device, tiled, cs):
        frames = generated_frames
        if frames is None:
            prev = initial_c0 if i == 0 else all_clean[-2][:, :, -1:]
            prefix = self.e.torch.cat([prev.detach().cpu().to(latents.dtype), latents.detach().cpu()], dim=2)
            frames = self.e.rollout._decode_generated_chunk_from_prefix_latents(
                pipe, prefix, noisy_start=0, noisy_count=cs, device=device, tiled=tiled,
                context="[selfhost] chunk decode")
        done = time.monotonic_ns()
        self.stats["chunks"] += 1
        if i == 0 and self.t_ready:
            self.stats["first_chunk_s"] = round(time.monotonic() - self.t_ready, 3)
        self.stats["t_last"] = time.monotonic()
        if self._t_chunk:
            self.stats["chunk_s"].append(time.monotonic() - self._t_chunk)
        self._emit(("frames", done, frames))
        if self.stop.is_set():
            raise SessionEnd()
        return generated_frames

    # worker ----------------------------------------------------------------------------------------------------
    def prepare(self, image_bytes: bytes):
        e = self.e
        tag = "s%d" % int(time.time() * 1000)
        img = e.work / (tag + ".png")
        from PIL import Image
        Image.open(io.BytesIO(image_bytes)).convert("RGB").save(img)
        space = e._workspace(tag, img, self.base_prompt or "a scene")
        cfg = dataclasses.replace(e.config, seed=self.seed)
        rt = e._runtime_args(space, cfg)
        from examples.wanvideo.pipeline.mosaic.datasets import build_mosaic_validation_dataset
        ds = build_mosaic_validation_dataset(rt, train_dataset=None)
        model = e.model
        data = ds[0]
        if data.get("needs_vae_materialization") or data.get("needs_first_latent_materialization"):
            data = model._materialize_data(data)
        hist = model._ensure_batched_latents(data["clean_latents"]).detach().cpu()
        lh, lw = hist.shape[-2:]
        handler = e.ci._build_handler(data, rt, height=int(lh) * 16, width=int(lw) * 16, latent_h=lh,
                                      latent_w=lw, estimator=e.estimator)
        c0 = e.rollout._decode_latents_to_numpy_frames(model.pipe, hist[:, :, -1:], model.pipe.device,
                                                       tiled=e.tiled)
        self.job = (rt, ds, data, handler, hist, int(data["noisy_latent_indices"].shape[-1]), int(lh), int(lw))
        self.space = space
        return c0

    def run(self):
        e, torch = self.e, self.e.torch
        rt, ds, data, handler, hist, lws, lh, lw = self.job
        e.rollout._SA_HOOK = self
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        try:
            with torch.no_grad():
                e.rollout._run_causal_kv_rollout(e.model, rt, data, ds, handler, hist, lh, lw, lh * 16, lw * 16,
                                                 e.tiled, int(rt.num_validation_blocks), lws,
                                                 bool(e.model.enable_mosaic and not e.model.only_prope), 0,
                                                 accelerator=e.accelerator)
            self._emit(("done", "rollout reached max_blocks"))
        except SessionEnd:
            self._emit(("done", "ended"))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._emit(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self.primed.set()
            e.rollout._SA_HOOK = None
            shutil.rmtree(self.space["root"], ignore_errors=True)
            torch.cuda.empty_cache()


def _jpeg(frame, quality):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------- protocol

class Server:
    def __init__(self, engine: Engine, jpeg_quality: int):
        self.e, self.q = engine, jpeg_quality
        self.busy = asyncio.Lock()

    def health(self):
        return {"ok": True, "model": MODEL, "busy": self.busy.locked(), **self.e.system_config()}

    async def handle(self, conn):
        async with self.busy:  # one GPU session at a time
            await self._session(conn)

    async def _session(self, conn):
        msg = await conn.recv()
        if not isinstance(msg, dict) or msg.get("type") != "init":
            await conn.error("first message must be init")
            return
        if not msg.get("image_b64"):
            await conn.error("unsupported: Matrix-Game 3.5 needs an anchor image (image+prompt start); text-only "
                             "start is not supported by the released checkpoints")
            return
        loop = asyncio.get_running_loop()
        out_q: asyncio.Queue = asyncio.Queue()
        sess = Session(self.e, msg, conn, loop, out_q)
        try:
            c0 = await loop.run_in_executor(None, sess.prepare, base64.b64decode(msg["image_b64"]))
        except Exception as exc:  # noqa: BLE001
            await conn.error(f"prepare failed: {type(exc).__name__}: {exc}")
            return
        worker = threading.Thread(target=sess.run, daemon=True)
        worker.start()
        await loop.run_in_executor(None, sess.primed.wait)
        sender = asyncio.create_task(self._pump(conn, out_q, sess))
        if worker.is_alive():
            cfg = self.e.config
            await conn.ready(MODEL, "Matrix-Game-3.5-Distilled", cfg.width, cfg.height, FPS,
                             system_config=self.e.system_config())
            sess.t_ready = time.monotonic()
            sess.go.set()
            await conn.frame(_jpeg(c0[0], self.q), time.monotonic_ns())  # anchor = first frame of the output video
        try:
            while True:
                m = await conn.recv()
                if isinstance(m, bytes):
                    continue
                t = m.get("type")
                if t == "end":
                    break
                eid = m.get("event_id", "")
                if t == "text":
                    kind = m.get("kind", "user_prompt")
                    if kind in ("user_prompt", "text_control"):
                        sess.pending.put((eid, m.get("text", "")))
                    else:
                        await conn.ack(eid, "unsupported", f"{kind}: " + (
                            "free-text camera instructions have no exact mapping to the model's metric 6-DoF "
                            "camera-pose input" if kind == "camera_control" else "not an input of Matrix-Game 3.5"))
                elif t in ("reference", "control"):
                    await conn.ack(eid, "unsupported", f"{t} events are not supported by Matrix-Game 3.5")
        except proto.ConnectionClosed:
            pass
        finally:
            sess.stop.set()
            await loop.run_in_executor(None, worker.join)
            sender.cancel()
            while not sess.pending.empty():  # every event is acked, even if no chunk was left to apply it
                eid, _ = sess.pending.get_nowait()
                try:
                    await conn.ack(eid, "error", "session ended before the next chunk could use the prompt")
                except (proto.ConnectionClosed, ConnectionError):
                    break
            st = sess.stats
            span = st.get("t_last", 0) - (sess.t_ready or 0)
            log("session stats", json.dumps({"chunks": st["chunks"], "first_chunk_s": st.get("first_chunk_s"),
                                             "mean_denoise_decode_s": _mean(st["chunk_s"]),
                                             "gen_fps": round(12 * st["chunks"] / span, 2) if span > 0 else None}))

    async def _pump(self, conn, out_q, sess):
        interval = 1.0 / FPS
        next_t = time.monotonic()
        while True:
            item = await out_q.get()
            kind = item[0]
            if kind == "frames":
                pts, frames = item[1], item[2]
                for f in frames:
                    now = time.monotonic()
                    if next_t > now:
                        await asyncio.sleep(next_t - now)
                    await conn.frame(_jpeg(f, self.q), pts)
                    next_t = max(next_t, time.monotonic() - interval) + interval
            elif kind == "error":
                await conn.error(item[1])
                await conn.close()
                return
            elif kind == "done":
                # after max_blocks keep the connection open; the client ends the session
                return


def _mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def main():
    global WEIGHTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-blocks", type=int, default=16, help="rollout length cap: 16 blocks = 80 s")
    ap.add_argument("--vae-tiled", action="store_true", help="upstream tiled VAE decode (slower)")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    args = ap.parse_args()
    WEIGHTS = args.weights
    engine = Engine(args)
    server = Server(engine, args.jpeg_quality)
    log(f"serving ws://{args.host}:{args.port}/session")
    proto.serve(server.handle, args.host, args.port, health=server.health)


if __name__ == "__main__":
    main()
