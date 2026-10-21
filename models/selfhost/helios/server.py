"""Helios 14B (PKU-YuanGroup, Helios-Distilled) self-hosted server, docs/protocol.md.

One WebSocket connection = one session, generated live chunk by chunk. The loop is upstream
`HeliosPipeline.__call__` (helios/diffusers_version/pipeline_helios_diffusers.py) unrolled so inputs can change
between chunks, with the settings of upstream scripts/inference/helios-distilled_{t2v,i2v}.sh:
  640x384, 24 fps, 33 frames (9 latents) per chunk, multi-term memory history [16, 2, 1] + first-frame prefix,
  pyramid stage 2 with [2, 2, 2] DMD steps (first chunk amplified), guidance 1.0 (no CFG), bf16 DiT, fp32 Wan VAE.

  init (prompt [+ image_b64]) -> umT5 prompt encoding; with an image: center-crop to 640x384, VAE-encode the image as
                         the first-frame anchor and the static-clip latent as the last history frame (upstream i2v,
                         image noise sigma 0.111-0.135), then `ready`. A worker thread generates chunk after chunk:
                         stage-2 pyramid sampling -> VAE decode of the 9 new latents -> 33 JPEG `frame`s paced at 24 fps.
                         The worker starts a chunk only when the frames still buffered ahead of the wall clock cover
                         its expected compute time (+ margin), so every event is applied at the latest chunk boundary.
  text user_prompt / text_control -> scene channel; text camera_control -> camera channel (Helios takes camera language
                         through its text condition, like upstream prompts "...The camera remains stationary"). The next
                         chunk is conditioned on "<init prompt>\\n<latest scene instruction>\\n<latest camera instruction>"
                         (umT5 re-encode; Reactor `set_prompt` semantics: takes effect on the next chunk); ack "applied"
                         when that chunk starts.
  reference op=set (general_interaction_image) -> the new image replaces the first-frame anchor (Reactor `set_image`:
                         "replaced during generation; a new image applies from the next chunk onward"); its instruction
                         joins the scene channel. Recent history is kept, so the scene stays continuous.
  other reference ops / control / dialogue text -> ack "unsupported".

Processes: the model runs in a spawned child process (one CUDA thread); this process only serves the socket, encodes
JPEGs and paces frames. In one process the GPU thread held the GIL for ~300 ms at a time during DiT forwards and the
frame pump stalled 3-4 times per chunk (bursty delivery although generation kept up).

Upstream code: PKU-YuanGroup/Helios (see setup.sh), imported unchanged from $HELIOS_REPO.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import io
import multiprocessing as mp
import json
import os
import queue
import sys
import threading
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repository root -> streameval
from streameval.runtime import protocol as proto  # noqa: E402

MODEL_DIR = Path(os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/helios")
REPO = Path(os.environ.get("HELIOS_REPO", MODEL_DIR / "repo"))
MODEL = "helios"
VERSION = "Helios-Distilled (BestWishYsh/Helios-Distilled)"
FPS = 24
W, H = 640, 384
LATENTS_PER_CHUNK = 9
FRAMES_PER_CHUNK = (LATENTS_PER_CHUNK - 1) * 4 + 1   # 33
HISTORY_SIZES = [16, 2, 1]                           # long, mid, short (upstream infer_helios.py)
PYRAMID_STEPS = [2, 2, 2]
NEG_PROMPT = None                                    # guidance 1.0: no negative branch
IMAGE_SIGMA = (0.111, 0.135)                         # upstream image_noise_sigma_min/max (and video_* for the fake latent)
# Every CUDA call runs on the child's main thread: cuDNN v8 execution plans (cudnn.benchmark, set by torch.compile) are
# cached per thread; a fresh thread per session re-benchmarked the VAE convolutions (4 s on each session's first decode).
CATCHUP_S = 0.5                                      # output schedule slack caught up after a stall (see Server._pump)


class SessionEnd(Exception):
    pass


class _NoBar:
    def update(self, *a, **k):
        pass


def log(*a):
    print(time.strftime("%H:%M:%S"), "[helios]", *a, flush=True)


# ---------------------------------------------------------------------------------------------- model

class Engine:
    def __init__(self, args):
        self.args = args
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "yes")
        os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "8")
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(MODEL_DIR / "inductor_cache"))   # --compile: reuse kernels
        os.environ.setdefault("TRITON_CACHE_DIR", str(MODEL_DIR / "triton_cache"))
        sys.path.insert(0, str(REPO))
        # helios.modules.helios_kernels/__init__ imports attention_dispatch, which fetches flash-attn kernels from the
        # HF hub at import time (only used by the training / varlen paths); inference needs none of it.
        stub = types.ModuleType("helios.modules.helios_kernels.attention_dispatch")
        stub.attn_varlen_func = stub.create_navit_attention_masks = None
        sys.modules[stub.__name__] = stub
        import torch
        from diffusers.models import AutoencoderKLWan
        from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
        from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
        from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
        self.torch = torch
        dev = torch.device("cuda")
        ckpt = Path(args.weights) / "Helios-Distilled"
        t0 = time.time()
        transformer = HeliosTransformer3DModel.from_pretrained(ckpt, subfolder="transformer", torch_dtype=torch.bfloat16)
        if args.compile == "off":  # upstream: triton norm / rope kernels only without torch.compile
            from helios.modules.helios_kernels import (replace_all_norms_with_flash_norms, replace_rmsnorm_with_fp32,
                                                       replace_rope_with_flash_rope)
            if args.triton_kernels:
                transformer = replace_rmsnorm_with_fp32(transformer)
                transformer = replace_all_norms_with_flash_norms(transformer)
                replace_rope_with_flash_rope()
        if args.attn != "default":
            transformer.set_attention_backend(args.attn)
        vae = AutoencoderKLWan.from_pretrained(ckpt, subfolder="vae", torch_dtype=torch.float32)
        scheduler = HeliosScheduler.from_pretrained(ckpt, subfolder="scheduler")
        self.pipe = HeliosPipeline.from_pretrained(ckpt, transformer=transformer, vae=vae, scheduler=scheduler,
                                                   torch_dtype=torch.bfloat16).to(dev)
        p = self.pipe
        for m in (p.transformer, p.text_encoder, p.vae):  # inference only (grad mode is per thread; this is global)
            m.eval().requires_grad_(False)
        p.set_progress_bar_config(disable=True)
        p._guidance_scale, p._attention_kwargs, p._interrupt, p._current_timestep = 1.0, None, False, None
        assert p.config.is_distilled, "expected the distilled checkpoint (model_index is_distilled)"
        if args.compile != "off":
            torch.backends.cudnn.benchmark = True
            p.transformer.compile(mode=args.compile, dynamic=False)
            if args.compile_vae:
                p.vae.compile(mode=args.compile, dynamic=False)
        self.lat_mean = torch.tensor(p.vae.config.latents_mean).view(1, 16, 1, 1, 1).to(dev, p.vae.dtype)
        self.lat_std = 1.0 / torch.tensor(p.vae.config.latents_std).view(1, 16, 1, 1, 1).to(dev, p.vae.dtype)
        # constant index layout (keep_first_frame): prefix | long 16 | mid 2 | 1x 1 | 9 new latents
        idx = torch.arange(0, sum([1, *HISTORY_SIZES, LATENTS_PER_CHUNK]))
        pre, i_long, i_mid, i_1x, i_new = idx.split([1, *HISTORY_SIZES, LATENTS_PER_CHUNK], dim=0)
        self.indices = {"indices_hidden_states": i_new.unsqueeze(0),
                        "indices_latents_history_short": torch.cat([pre, i_1x]).unsqueeze(0),
                        "indices_latents_history_mid": i_mid.unsqueeze(0),
                        "indices_latents_history_long": i_long.unsqueeze(0)}
        self.gpu = torch.cuda.get_device_name(0)
        self.weights = self._weights_hash()
        self.chunk_s = None   # EMA of chunk compute (generate + decode), drives the just-in-time chunk start
        log(f"model loaded in {time.time() - t0:.0f}s on {self.gpu}; attn={args.attn} compile={args.compile} "
            f"mem={torch.cuda.memory_allocated() / 2**30:.1f}GiB")
        self._warmup()

    def _weights_hash(self):
        f = Path(self.args.weights).parent / "weights_hash.json"
        if f.exists():
            d = json.loads(f.read_text())
            return {"transformer_combined_sha256": d.get("transformer_combined_sha256"), "files": len(d.get("files", {}))}
        return {"transformer_combined_sha256": "unknown"}

    def _warmup(self):
        from PIL import Image
        t0 = time.time()
        for image in (None, Image.new("RGB", (W, H), (120, 140, 160))):
            s = Session(self, {"prompt": "warmup: a calm lake at sunrise", "seed": 0})
            with self.torch.no_grad():
                s.prepare(image)
            s.warmup, s.max_chunks = True, self.args.warmup_chunks
            # a mid-stream reference swap + prompt switch, so their kernels are warm too
            s.pending.put(("reference", "w", {"image": _png(Image.new("RGB", (512, 512), (200, 90, 60))), "text": "x"}))
            s.run()
            log(f"warmup ({'i2v' if image else 't2v'}) {time.time() - t0:.1f}s; chunk stats {json.dumps(s.stats)}")
        tail = s.stats["chunk_s"][1:] or s.stats["chunk_s"]
        self.chunk_s = sorted(tail)[len(tail) // 2]

    def note_chunk(self, seconds: float):
        self.chunk_s = seconds if self.chunk_s is None else 0.8 * self.chunk_s + 0.2 * seconds

    def system_config(self):
        a = self.args
        return {"model_version": VERSION, "weights": self.weights, "width": W, "height": H, "fps": FPS,
                "chunk_latents": LATENTS_PER_CHUNK, "chunk_frames": FRAMES_PER_CHUNK, "history_sizes": HISTORY_SIZES,
                "pyramid_steps": PYRAMID_STEPS, "amplify_first_chunk": True, "guidance_scale": 1.0,
                "dtype": "bf16 DiT / bf16 umT5 / fp32 VAE", "gpu": self.gpu, "gpus": 1, "attention": a.attn,
                "compile": a.compile, "triton_kernels": bool(a.triton_kernels and a.compile == "off"),
                "lookahead": f"next chunk starts when buffered video <= expected chunk compute + {a.margin_s:g} s",
                "image_noise_sigma": list(IMAGE_SIGMA)}


class Session:
    """One live autoregressive rollout in the model process. emit(item) sends ("ack", eid, status, detail),
    ("frames", k, uint8 THWC), ("error", msg) or ("done", reason) to the serving process; `sent` is the shared count of
    frames the serving process has put on the socket."""

    def __init__(self, engine: Engine, init: dict, emit=None, stop=None, pending=None, sent=None):
        self.e, self.emit_fn, self.sent_v = engine, emit, sent
        self.base_prompt = (init.get("prompt") or "").strip()
        self.scene, self.camera = "", ""
        self.seed = int(init.get("seed") or 0)
        self.pending = pending or queue.Queue()  # ("scene"|"camera"|"reference", event_id, payload)
        self.stop = stop or threading.Event()
        self.t_ready = None
        self.produced = 0
        self.warmup = False
        self.max_chunks = int(engine.args.max_chunks)
        self.stats = {"chunks": 0, "gen_s": [], "decode_s": [], "chunk_s": [], "wait_s": [], "prompt_s": []}

    @property
    def sent(self):
        return self.sent_v.value if self.sent_v is not None else self.produced

    def _emit(self, item):
        if self.emit_fn is not None:
            self.emit_fn(item)

    def _prompt_text(self):
        return "\n".join(t for t in (self.base_prompt or "a scene", self.scene, self.camera) if t)

    def _encode(self, text):
        p = self.e.pipe
        emb, _ = p.encode_prompt(prompt=text, negative_prompt=None, do_classifier_free_guidance=False,
                                 num_videos_per_prompt=1, max_sequence_length=512, device=self.e.torch.device("cuda"))
        return emb.to(p.transformer.dtype)

    def _image_latents(self, image, with_fake: bool):
        """Upstream prepare_image_latents + add_noise_to_image_latents (sigma drawn per call from the session RNG)."""
        torch, p = self.e.torch, self.e.pipe
        from PIL import Image, ImageOps
        image = ImageOps.fit(image.convert("RGB"), (W, H), Image.LANCZOS)
        x = p.video_processor.preprocess(image, height=H, width=W)
        dev = torch.device("cuda")
        if with_fake:
            lat, fake = p.prepare_image_latents(x, latents_mean=self.e.lat_mean, latents_std=self.e.lat_std,
                                                num_latent_frames_per_chunk=LATENTS_PER_CHUNK, dtype=torch.float32,
                                                device=dev, generator=self.gen)
        else:  # mid-stream anchor swap: only the image latent (the static-clip latent seeds history at start only)
            x5 = x.unsqueeze(2).to(device=dev, dtype=p.vae.dtype)
            lat = p.vae.encode(x5).latent_dist.sample(generator=self.gen)
            lat = ((lat - self.e.lat_mean) * self.e.lat_std).to(torch.float32)
            fake = None
        lo, hi = IMAGE_SIGMA
        from diffusers.utils.torch_utils import randn_tensor
        s = torch.rand(1, device=dev, generator=self.gen) * (hi - lo) + lo
        lat = s * randn_tensor(lat.shape, generator=self.gen, device=dev) + (1 - s) * lat
        if fake is not None:
            s2 = torch.rand(1, device=dev, generator=self.gen) * (hi - lo) + lo
            fake = s2 * randn_tensor(fake.shape, generator=self.gen, device=dev) + (1 - s2) * fake
        return lat, fake

    def prepare(self, image):
        """image: PIL or None (text start)."""
        torch = self.e.torch
        self.task = "i2v" if image is not None else "t2v"
        self.gen = torch.Generator(device="cuda").manual_seed(self.seed)
        self.prompt_embeds = self._encode(self._prompt_text())
        n_hist = sum(HISTORY_SIZES)
        self.history = torch.zeros(1, 16, n_hist, H // 8, W // 8, device="cuda", dtype=torch.float32)
        self.image_latents = None
        if image is not None:
            self.image_latents, fake = self._image_latents(image, with_fake=True)
            self.history = torch.cat([self.history[:, :, :-1], fake], dim=2)

    def _apply_pending(self, k, events):
        if not events:
            return
        from PIL import Image
        t0 = time.monotonic()
        ref_ids, text_ids, swapped = [], [], False
        for kind, eid, payload in events:
            if kind == "reference":
                self.image_latents, _ = self._image_latents(Image.open(io.BytesIO(payload["image"])), with_fake=False)
                swapped = True
                if payload.get("text"):
                    self.scene = payload["text"]
                ref_ids.append(eid)
            elif kind == "camera":
                self.camera = payload
                text_ids.append(eid)
            else:
                self.scene = payload
                text_ids.append(eid)
        self.prompt_embeds = self._encode(self._prompt_text())
        self.stats["prompt_s"].append(round(time.monotonic() - t0, 3))
        frame = k * FRAMES_PER_CHUNK
        for eid in text_ids:
            self._emit(("ack", eid, "applied", f"text condition switched at chunk {k} (frame {frame})"))
        for eid in ref_ids:
            self._emit(("ack", eid, "applied", f"reference image is the first-frame anchor from chunk {k} (frame {frame}); "
                                               "instruction joins the text condition"))
        if not self.warmup:
            log(f"chunk {k}: applied {len(events)} event(s){' + anchor swap' if swapped else ''}; "
                f"prompt tail: {self._prompt_text()[-160:]!r}")

    def _drain(self):
        events = []
        while True:
            try:
                events.append(self.pending.get_nowait())
            except queue.Empty:
                return events

    def _chunk(self, k):
        """One chunk of upstream __call__ (keep_first_frame, stage 2) -> the 9 new clean latents."""
        torch, p, e = self.e.torch, self.e.pipe, self.e
        dev = torch.device("cuda")
        long_, mid_, x1 = self.history[:, :, -sum(HISTORY_SIZES):].split(HISTORY_SIZES, dim=2)
        prefix = self.image_latents if self.image_latents is not None else torch.zeros_like(x1)
        short = torch.cat([prefix, x1], dim=2)
        latents = p.prepare_latents(1, 16, H, W, FRAMES_PER_CHUNK, dtype=torch.float32, device=dev, generator=self.gen)
        latents = p.stage2_sample(
            latents=latents, pyramid_num_stages=len(PYRAMID_STEPS), pyramid_num_inference_steps_list=PYRAMID_STEPS,
            prompt_embeds=self.prompt_embeds, negative_prompt_embeds=None, guidance_scale=1.0,
            latents_history_short=short, latents_history_mid=mid_, latents_history_long=long_, attention_kwargs=None,
            device=dev, transformer_dtype=p.transformer.dtype, generator=self.gen, use_zero_init=True, zero_steps=1,
            is_amplify_first_chunk=(k == 0), callback_on_step_end=None, progress_bar=_NoBar(), **e.indices)
        if self.image_latents is None:  # text start: the first generated latent becomes the first-frame anchor
            self.image_latents = latents[:, :, 0:1]
        self.history = torch.cat([self.history, latents], dim=2)[:, :, -sum(HISTORY_SIZES):]
        return latents

    def _decode(self, latents):
        torch, p = self.e.torch, self.e.pipe
        z = latents[:, :, -LATENTS_PER_CHUNK:].to(p.vae.dtype) / self.e.lat_std + self.e.lat_mean
        v = p.vae.decode(z, return_dict=False)[0][0]                         # C T H W in [-1, 1]
        v = ((v.float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
        return v.permute(1, 2, 3, 0).contiguous().cpu().numpy()             # T H W C

    def _wait_turn(self):
        """Just in time: start the next chunk once the video still buffered ahead of playback covers its compute."""
        t0 = time.monotonic()
        while not self.stop.is_set() and self.sent_v is not None:
            ahead_s = (self.produced - self.sent) / FPS
            need_s = (self.e.chunk_s or 0) + self.e.args.margin_s
            if ahead_s <= need_s:
                break
            time.sleep(0.01)
        return time.monotonic() - t0

    def run(self):
        torch = self.e.torch
        torch.manual_seed(self.seed)
        t_start = time.monotonic()
        try:
            with torch.no_grad():
                for k in range(self.max_chunks):
                    wait = self._wait_turn() if k else 0.0
                    if self.stop.is_set():
                        raise SessionEnd()
                    self._apply_pending(k, self._drain())
                    t0 = time.monotonic()
                    lat = self._chunk(k)
                    torch.cuda.synchronize()
                    t1 = time.monotonic()
                    frames = self._decode(lat)
                    t2 = time.monotonic()
                    if k:
                        self.e.note_chunk(t2 - t0)
                    st = self.stats
                    st["chunks"] += 1
                    st["gen_s"].append(round(t1 - t0, 3))
                    st["decode_s"].append(round(t2 - t1, 3))
                    st["chunk_s"].append(round(t2 - t0, 3))
                    st["wait_s"].append(round(wait, 3))
                    self.produced += len(frames)
                    self._emit(("frames", k, frames))
                    if not self.warmup and (k < 3 or k % 20 == 0):
                        log(f"chunk {k}: {len(frames)} frames gen {t1 - t0:.2f}s decode {t2 - t1:.2f}s "
                            f"({len(frames) / (t2 - t0):.1f} fps) waited {wait:.2f}s")
            self._emit(("done", "max_chunks reached"))
        except SessionEnd:
            self._emit(("done", "ended"))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._emit(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self.stats["wall_s"] = round(time.monotonic() - t_start, 2)
            self.history = self.image_latents = self.prompt_embeds = None
            for kind, eid, _ in self._drain():  # released too late for any chunk of this session
                self._emit(("ack", eid, "error", "session ended before the next chunk boundary"))


def _png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(frame, quality):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


# ---------------------------------------------------------------------------------------------- model process

def model_process(args, cmd_q, out_q, stop, sent):
    """Child process: load + warm the model, then run one session per ("session", init, image) command. Session
    events arrive on cmd_q as ("event", (kind, event_id, payload)); outputs go to out_q."""
    try:
        engine = Engine(args)
    except BaseException as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        out_q.put(("fatal", f"{type(exc).__name__}: {exc}"))
        return
    out_q.put(("loaded", engine.system_config()))
    while True:
        cmd = cmd_q.get()
        if cmd is None:
            return
        if cmd[0] != "session":   # an "end" / event that arrived after its session had already finished
            continue
        _, init, raw = cmd
        pending: queue.Queue = queue.Queue()
        sess = Session(engine, init, emit=out_q.put, stop=stop, pending=pending, sent=sent)
        t0 = time.monotonic()
        try:
            from PIL import Image
            with engine.torch.no_grad():
                sess.prepare(Image.open(io.BytesIO(raw)) if raw else None)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            out_q.put(("prepare_failed", f"{type(exc).__name__}: {exc}"))
            continue
        log(f"session {init.get('case_id')} prepared in {time.monotonic() - t0:.2f}s task={sess.task} seed={sess.seed}")
        out_q.put(("prepared", {"task": sess.task, "seed": sess.seed}))
        # events are forwarded into the session's queue by a small thread (the run loop drains it per chunk)
        fwd_stop = threading.Event()

        def forward():
            while not fwd_stop.is_set():
                try:
                    item = cmd_q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None or item[0] == "end":
                    stop.set()
                    if item is None:
                        cmd_q.put(None)
                    return
                pending.put(item[1])

        fwd = threading.Thread(target=forward, daemon=True)
        fwd.start()
        sess.run()
        fwd_stop.set()
        fwd.join()
        s = sess.stats
        out_q.put(("stats", {"task": sess.task, "seed": sess.seed, "chunks": s["chunks"],
                             "mean_gen_s": _mean(s["gen_s"][1:]), "mean_decode_s": _mean(s["decode_s"][1:]),
                             "mean_chunk_s": _mean(s["chunk_s"][1:]), "first_chunk_s": (s["chunk_s"] or [None])[0],
                             "mean_wait_s": _mean(s["wait_s"][1:]), "prompt_switch_s": s["prompt_s"],
                             "wall_s": s.get("wall_s"),
                             "gen_fps": round(FRAMES_PER_CHUNK / _mean(s["chunk_s"][1:]), 2) if s["chunk_s"][1:] else None,
                             "per_chunk": s}))


# ---------------------------------------------------------------------------------------------- protocol

class Server:
    def __init__(self, args):
        self.args, self.q = args, args.jpeg_quality
        ctx = mp.get_context("spawn")
        self.cmd_q, self.out_q = ctx.Queue(), ctx.Queue()
        self.stop, self.sent = ctx.Event(), ctx.Value("q", 0, lock=False)
        self.proc = ctx.Process(target=model_process, args=(args, self.cmd_q, self.out_q, self.stop, self.sent),
                                daemon=True, name="helios-model")
        self.proc.start()
        msg = self.out_q.get()
        if msg[0] != "loaded":
            raise SystemExit(f"model process failed: {msg[1:]}")
        self.config = msg[1]
        self.busy = asyncio.Lock()
        self.jpeg_pool = concurrent.futures.ThreadPoolExecutor(3, thread_name_prefix="jpeg")
        (MODEL_DIR / "logs/sessions").mkdir(parents=True, exist_ok=True)

    def health(self):
        return {"ok": self.proc.is_alive(), "model": MODEL, "busy": self.busy.locked(), **self.config}

    async def _out(self, timeout=None):
        loop = asyncio.get_running_loop()
        while True:
            try:
                return await loop.run_in_executor(None, self.out_q.get, True, 0.5 if timeout is None else timeout)
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError("model process died")
                if timeout is not None:
                    raise

    async def handle(self, conn):
        async with self.busy:  # one GPU session at a time
            await self._session(conn)

    async def _session(self, conn):
        msg = await conn.recv()
        if not isinstance(msg, dict) or msg.get("type") != "init":
            await conn.error("first message must be init")
            return
        raw = base64.b64decode(msg["image_b64"]) if msg.get("image_b64") else None
        self.stop.clear()
        self.sent.value = 0
        init = {k: msg.get(k) for k in ("prompt", "seed", "case_id")}
        self.cmd_q.put(("session", init, raw))
        reply = await self._out()
        if reply[0] != "prepared":
            await conn.error(f"prepare failed: {reply[1]}")
            return
        info = reply[1]
        await conn.ready(MODEL, VERSION, W, H, FPS, protocol=proto.PROTOCOL_VERSION,
                         system_config={**self.config, **info})
        stats = {}
        pump = asyncio.create_task(self._pump(conn, stats))
        try:
            while True:
                recv = asyncio.ensure_future(conn.recv())
                done, _ = await asyncio.wait({recv, pump}, return_when=asyncio.FIRST_COMPLETED)
                if pump in done:      # model finished (error / max chunks): stop reading events
                    recv.cancel()
                    break
                m = recv.result()
                if isinstance(m, bytes):
                    continue
                t = m.get("type")
                if t == "end":
                    break
                eid = m.get("event_id", "")
                if t == "text":
                    kind = m.get("kind") or "user_prompt"
                    text = (m.get("text") or "").strip()
                    if kind in ("user_prompt", "text_control") and text:
                        self.cmd_q.put(("event", ("scene", eid, text)))
                    elif kind == "camera_control" and text:
                        self.cmd_q.put(("event", ("camera", eid, text)))
                    else:
                        await conn.ack(eid, "unsupported", f"text kind {kind} not supported" if text else "empty text")
                elif t == "reference":
                    slots = m.get("slots") or []
                    img = slots[0].get("image_b64") if slots else None
                    if m.get("op", "set") == "set" and img:
                        self.cmd_q.put(("event", ("reference", eid, {"image": base64.b64decode(img),
                                                                     "text": m.get("text")})))
                    else:
                        await conn.ack(eid, "unsupported", f"reference op {m.get('op')} not supported "
                                                           "(Helios takes one anchor image)")
                elif t == "control":
                    await conn.ack(eid, "unsupported", "structured control events are not supported")
                elif eid:
                    await conn.ack(eid, "unsupported", f"message type {t} not supported")
        except proto.ConnectionClosed:
            pass
        finally:
            self.stop.set()
            self.cmd_q.put(("end", None))
            try:
                await asyncio.wait_for(pump, timeout=60)   # drains the model's final acks and its stats
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pump.cancel()
            summary = {"case_id": msg.get("case_id"), "frames_sent": self.sent.value,
                       **{k: v for k, v in stats.get("model", {}).items() if k != "per_chunk"},
                       "pump": {k: round(v, 3) for k, v in stats.get("pump", {}).items()}}
            log("session stats", json.dumps(summary))
            try:
                stamp = time.strftime("%Y%m%dT%H%M%S")
                (MODEL_DIR / f"logs/sessions/{msg.get('case_id')}.{stamp}.json").write_text(
                    json.dumps({**summary, "per_chunk": stats.get("model", {}).get("per_chunk")}, indent=1))
            except OSError:
                pass

    async def _pump(self, conn, stats):
        """Model outputs -> socket. Acks / errors go out as they arrive; frames go to the pacer. Returns after the
        model's ("stats", ...) for this session and the pacer has finished."""
        frames_q: asyncio.Queue = asyncio.Queue()
        state = {"open": True}
        pacer = asyncio.create_task(self._pace(conn, frames_q, stats, state))
        try:
            while True:
                item = await self._out()
                kind = item[0]
                try:
                    if kind == "frames":
                        frames_q.put_nowait(item[2])
                    elif kind == "ack" and state["open"]:
                        await conn.ack(item[1], item[2], item[3])
                    elif kind == "error" and state["open"]:
                        await conn.error(item[1])
                        state["open"] = False
                except (ConnectionError, proto.ConnectionClosed):
                    state["open"] = False
                if kind == "stats":
                    stats["model"] = item[1]
                    return
                if kind == "fatal":
                    raise RuntimeError(item[1])
        finally:
            frames_q.put_nowait(None)
            try:
                await asyncio.wait_for(pacer, timeout=5)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pacer.cancel()

    async def _pace(self, conn, frames_q, stats, state):
        """Frames go out on a 24 fps schedule; a late frame (generation behind) goes out at once and up to CATCHUP_S
        of the schedule is caught up by sending the following frames back to back; a longer gap re-anchors it."""
        loop = asyncio.get_running_loop()
        interval, next_t = 1.0 / FPS, None
        st = stats["pump"] = {"late_100ms": 0, "max_late_s": 0.0}
        while True:
            frames = await frames_q.get()
            if frames is None:
                return
            if not state["open"] or self.stop.is_set():
                continue
            jpegs = [loop.run_in_executor(self.jpeg_pool, _jpeg, f, self.q) for f in frames]
            for fut in jpegs:
                data = await fut
                if self.stop.is_set() or not state["open"]:
                    break
                now = time.monotonic()
                next_t = now if next_t is None else next_t
                if next_t > now:
                    await asyncio.sleep(next_t - now)
                late = time.monotonic() - next_t
                st["late_100ms"] += late > 0.1
                st["max_late_s"] = max(st["max_late_s"], late)
                try:
                    await conn.frame(data, time.monotonic_ns())
                except (ConnectionError, proto.ConnectionClosed):
                    state["open"] = False
                    break
                self.sent.value += 1
                next_t = max(next_t, time.monotonic() - CATCHUP_S) + interval


def bench(args, n: int):
    """Offline timing (no socket): n chunks text start + n chunks image start with a prompt switch."""
    from PIL import Image
    engine = Engine(args)
    for image in (None, Image.new("RGB", (W, H), (90, 120, 160))):
        s = Session(engine, {"prompt": "A red kite flies over a green hill under a blue sky.", "seed": 1})
        with engine.torch.no_grad():
            s.prepare(image)
        s.warmup, s.max_chunks = True, n
        s.pending.put(("scene", "b1", "The kite climbs higher."))
        s.run()
        st = s.stats
        print(json.dumps({"bench": "i2v" if image else "t2v", "chunks": st["chunks"], "first_chunk_s": st["chunk_s"][0],
                          "mean_gen_s": _mean(st["gen_s"][1:]), "mean_decode_s": _mean(st["decode_s"][1:]),
                          "mean_chunk_s": _mean(st["chunk_s"][1:]),
                          "gen_fps": round(FRAMES_PER_CHUNK / _mean(st["chunk_s"][1:]), 2),
                          "prompt_s": st["prompt_s"], "max_mem_gib": round(engine.torch.cuda.max_memory_allocated() / 2**30, 1),
                          **{k: engine.system_config()[k] for k in ("attention", "compile", "triton_kernels")}}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--weights", type=Path, default=MODEL_DIR / "weights")
    ap.add_argument("--max-chunks", type=int, default=400, help="rollout cap: 400 chunks = 550 s")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--attn", default="default",
                    help="diffusers attention backend; default uses torch SDPA dispatch")
    ap.add_argument("--compile", default="max-autotune-no-cudagraphs",
                    help="torch.compile mode: max-autotune-no-cudagraphs | default | off")
    ap.add_argument("--compile-vae", action="store_true")
    ap.add_argument("--no-triton-kernels", dest="triton_kernels", action="store_false")
    ap.add_argument("--margin-s", type=float, default=0.5, help="just-in-time start margin over the chunk compute EMA")
    ap.add_argument("--warmup-chunks", type=int, default=3)
    ap.add_argument("--bench", type=int, default=0, help="offline timing of N chunks per task, then exit")
    args = ap.parse_args()
    if args.bench:
        return bench(args, args.bench)
    server = Server(args)
    log(f"serving ws://{args.host}:{args.port}/session")
    proto.serve(server.handle, args.host, args.port, health=server.health)


if __name__ == "__main__":
    main()
