#!/usr/bin/env python3
"""EditaLive (U. Macau / vivo BlueImage Lab / GVC Lab; Wan2.2-Animate-14B + EditaLive LoRAs, 2-step causal rollout)
self-hosted instruction-edit server, docs/protocol.md.

Upstream: github.com/GVCLab/EditaLive (pinned in setup.sh), LoRAs huaichang/EditaLive. The server follows upstream's
live path, the webcam service (`webcam/service.py`, `webcam/workers.py`), with the benchmark's live source in place of
the camera:

  * model: `EditaLiveStreamingPipeline.from_pretrained` as the webcam generation worker builds it (the three LoRAs
    fused in the documented order, FP8 transformer, Flash-VAED decoder) with dense attention (see below), then
    `prepare_stream_model` (torch.compile warm-up of every streaming shape), `prepare_stream(prompt)` (T5) and
    `stream_start(reference, prompt, generator)`; every chunk is `StreamingSession.push((pose, face), noise)` with
    upstream's noise draw, 2 steps at timesteps [1000, 250] (shift 5, UniPC), 3 latent frames per chunk
    (9 source frames for the first chunk of a session, then 12), rolling KV cache with the reference + first chunk as
    sink;
  * per-chunk conditioning: upstream's `webcam/extractor.OnlineExtractor` (YOLOv10m person detection + ViTPose-H
    whole-body, ONNX Runtime) on exactly the source frames of the chunk: rendered pose skeleton at the output size
    and a 512x512 face crop per frame. Extraction of chunk k+1 overlaps generation of chunk k (upstream runs it in a
    second process for the same reason);
  * reference: as upstream's "Start editing", a fresh live frame becomes the character reference when a session
    starts: the first source frame of the session's first chunk (received after the instruction, so no look-ahead);
  * resolution: upstream's webcam supports landscape output only, 672x384 (default) or 832x480; the source is
    centre-cropped to that aspect (the webcam crop box), portrait sources included.

Instructions: the instruction text is the prompt verbatim. Upstream changes a prompt by Reset + Prepare prompt +
Start editing (a new session with a fresh camera reference), so an instruction change starts a new session on the
next live frames. The source is passed through until the first instruction. Streaming, pacing and acks:
models/selfhost/edit_stream.py.

Attention: this adapter uses upstream's dense setting (`vsa_sparsity=0`) and the portable torch SDPA
implementation in models/selfhost/_shims/flash_attn. Dense attention differs from the upstream sparse-attention
configuration and is recorded as part of the model settings. Pose ONNX models use the TensorRT provider in FP32.
Person-less frames are handled as upstream does (the detector falls back to the whole frame); a frame whose face crop
is empty reuses the previous frame's face crop (upstream raises and ends the session).

    python server.py --port 8772
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import edit_stream  # noqa: E402
from t2v_stream import sha256_files  # noqa: E402  (follows symlinked shared weights)

ROOT = Path(os.environ.get("EDITALIVE_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/editalive"))
REPO = ROOT / "repo"
W = ROOT / "weights"
LOG = logging.getLogger("editalive")
FIRST_CHUNK, CHUNK = 9, 12
LORAS = ["editalive_edit.safetensors", "lightx2v.safetensors", "editalive_streaming.safetensors"]
SIZES = {384: (672, 384), 480: (832, 480)}       # webcam RuntimeConfig.size


def tensorrt_libs_on_path():
    """ONNX Runtime's TensorRT provider loads TensorRT through the dynamic linker: put the pip tensorrt_libs directory on
    LD_LIBRARY_PATH and re-exec once (preloading it with ctypes after torch crashes ONNX Runtime)."""
    import importlib.util
    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is None:
        return
    libs = str(Path(spec.origin).parent)
    if libs not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in (libs, os.environ.get("LD_LIBRARY_PATH")) if p)
        os.execv(sys.executable, [sys.executable, *sys.argv])


def use_tensorrt_for_onnx():
    """Prefer TensorRT in FP32, with an engine cache under the model root.

    Keep CUDA as a fallback for upstream's GPU heatmap decoding and I/O binding.
    """
    import onnxruntime as ort
    trt = ("TensorrtExecutionProvider", {"trt_engine_cache_enable": True, "trt_timing_cache_enable": True,
                                         "trt_engine_cache_path": str(ROOT / "trt_cache"), "trt_fp16_enable": False})
    base = ort.InferenceSession

    class Session(base):
        def __init__(self, path, sess_options=None, providers=None, **kw):
            if providers and any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in providers):
                providers = [trt, *providers]
            super().__init__(path, sess_options=sess_options, providers=providers, **kw)
    ort.InferenceSession = Session


class Extractor:
    """upstream webcam/extractor.OnlineExtractor on decoded frames (uint8 HxWx3 RGB at the output size)."""

    def __init__(self, device, size):
        use_tensorrt_for_onnx()
        from webcam.extractor import OnlineExtractor
        self.x = OnlineExtractor(str(W / "Wan-Animate"), device, size)
        self.x.warmup((FIRST_CHUNK, CHUNK))
        self.providers = self.x.pose.model.session.get_providers()
        self.last_face = None
        self.face_fallbacks = 0

    def reset(self):
        self.x.reset()
        self.last_face = None

    def __call__(self, images):
        import cv2
        x = self.x
        poses, faces = [], []
        for image, meta in zip(images, x.pose(images)):
            x0, x1, y0, y1 = x.face_box(meta['keypoints_face'][:, :2], scale=1.3, image_shape=image.shape[:2])
            face = image[y0:y1, x0:x1]
            if face.size == 0:                       # upstream: ValueError("Empty face crop.") ends the session
                self.face_fallbacks += 1
                if self.last_face is None:
                    h, w = image.shape[:2]
                    s = min(h, w)
                    self.last_face = cv2.resize(image[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2], (512, 512))
                faces.append(self.last_face)
            else:
                self.last_face = cv2.resize(face, (512, 512))
                faces.append(self.last_face)
            canvas = np.zeros((x.size[1], x.size[0], 3), dtype=np.uint8)
            poses.append(x.draw(canvas, x.meta_class.from_humanapi_meta(meta).resize(*x.size)))
        return np.stack(poses), np.stack(faces)


class Engine:
    name = "EditaLive"

    def __init__(self, args):
        self.args = args
        self.size = SIZES[args.resolution]
        cache = ROOT / "compile_cache" / f"{args.resolution}-landscape-{args.fp8}-flash-False"
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache))
        os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "triton_cache"))
        os.environ.setdefault("WAN_ORT_INTRA_THREADS", "4")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        sys.path.insert(0, str(HERE.parent / "_shims"))          # flash_attn -> exact SDPA
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)
        import torch
        self.torch = torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        from editalive.compile_config import configure_compile
        configure_compile(args.compile)                          # before the model modules are imported
        from editalive.configs import editalive_14B
        from editalive.pipeline.editalive_streaming import EditaLiveStreamingPipeline
        self.fps = args.model_fps
        self.device = torch.device("cuda:0")
        t0 = time.time()
        self.pipe = EditaLiveStreamingPipeline.from_pretrained(
            str(W / "Wan-Animate"), editalive_14B, device=self.device, vae_device=self.device,
            lora_paths=[str(W / "EditaLive" / f) for f in LORAS], fp8=args.fp8, fast_decode=True,
            fast_decoder_pth=str(W / "Flash-VAED/Flash_VAED_Wan.pth"), vsa_sparsity=args.vsa_sparsity,
            vsa_backend=args.vsa_backend)
        t1 = time.time()
        width, height = self.size
        self.pipe.prepare_stream_model(height, width, fast_decode=True,
                                       progress_callback=lambda m: LOG.info("prepare: %s", m))
        # upstream moves T5 (and CLIP) to the GPU for every prompt / session start and back to the CPU afterwards
        # (memory saving for 24-80 GB cards); this card has room for both, so they stay resident and a prompt
        # change does not wait for ~13 GB of host <-> device copies. Same encoders, same outputs.
        self.pipe.text_encoder.to(self.device)
        self.pipe.clip.to(self.device)
        for enc in (self.pipe.text_encoder, self.pipe.clip):
            enc.cpu = lambda *a, **k: None
        self.extractor = Extractor(self.device, self.size)
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"editalive@{self.commit[:12]} Wan2.2-Animate-14B + EditaLive LoRAs"
        self.weights_sha256 = sha256_files(
            [W / "EditaLive" / f for f in LORAS] + [W / "Flash-VAED/Flash_VAED_Wan.pth"]
            + sorted((W / "Wan-Animate").glob("diffusion_pytorch_model-*.safetensors"))
            + [W / "Wan-Animate" / f for f in ("Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                                               "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                                               "process_checkpoint/det/yolov10m.onnx")], W)
        pose = W / "Wan-Animate/process_checkpoint/pose2d/vitpose_h_wholebody.onnx"   # ONNX graph + external data
        files = sha256_files(sorted(pose.iterdir()), W)
        self.weights_sha256[f"{pose.relative_to(W)}/ ({len(files)} files: sha256 of the sorted 'sha256  path' list)"] = \
            hashlib.sha256("".join(f"{v}  {k}\n" for k, v in sorted(files.items())).encode()).hexdigest()
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.0fs, stream warm-up %.0fs on %s; pose providers %s; mem %.1f GiB",
                 self.version, t1 - t0, time.time() - t1, self.gpu, self.extractor.providers,
                 torch.cuda.memory_allocated() / 2**30)
        # every model call runs on one long-lived thread (warm-up included): compiled graphs are guarded on
        # thread-local state, and a fresh session thread otherwise re-enters compilation (~5 s on its first chunks)
        self.gpu_thread = ThreadPoolExecutor(1, thread_name_prefix="editalive-gpu")
        self.on_gpu(self.warmup)

    def on_gpu(self, fn, *a):
        def call():
            with self.torch.no_grad():
                return fn(*a)
        return self.gpu_thread.submit(call).result()

    def out_size(self, init):
        return self.size                                         # webcam: landscape only

    # ---- one live session of upstream's StreamingSession -------------------------------------------------
    def _start(self, reference, prompt, seed):
        torch, p = self.torch, self.pipe
        width, height = self.size
        prepared = p.prepare_stream(prompt, height, width, fast_decode=True)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        sess = p.stream_start(reference, prompt, generator=gen, fast_decode=True, prepared=prepared,
                              offload_model=False)
        return sess, gen

    def _push(self, sess, gen, pose, face):
        torch = self.torch
        noise = torch.randn(sess.noise_shape, device=self.device, dtype=torch.float32, generator=gen)
        frames, ready = sess.push((pose, face), noise, display=True)
        ready.synchronize()
        return list(frames.numpy())

    def warmup(self):
        """Real sessions on synthetic frames: extraction, session start, first chunk and steady chunks."""
        width, height = self.size
        rng = np.random.default_rng(0)
        t0 = time.time()
        for prompt in ("Change her hair to blonde.", "Add a black fedora on his head."):
            frames = [rng.integers(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(CHUNK)]
            self.extractor.reset()
            t = time.time()
            sess, gen = self._start(frames[0], prompt, 0)
            times = [("start", round(time.time() - t, 2))]
            try:
                for i in range(5):
                    t = time.time()
                    n = FIRST_CHUNK if i == 0 else CHUNK
                    pose, face = self.extractor(frames[:n])
                    te = time.time() - t
                    self._push(sess, gen, pose, face)
                    times.append((round(te, 2), round(time.time() - t - te, 2)))
            finally:
                sess.close()
        LOG.info("warmup %.0fs; (extract s, generate s) per chunk %s", time.time() - t0, times)

    def system_config(self):
        a = self.args
        return {"model": self.name, "model_variant": "EditaLive: Wan2.2-Animate-14B with the editalive_edit, lightx2v and "
                                                     "editalive_streaming LoRAs fused (upstream order), 2-step causal",
                "version": self.version, "repo": "https://github.com/GVCLab/EditaLive", "repo_commit": self.commit,
                "license": "Apache-2.0 (code and LoRAs; README: academic research only); Wan2.2-Animate-14B Apache-2.0; "
                           "Flash-VAED: no licence stated",
                "weights_sha256": self.weights_sha256, "resolution": list(self.size), "orientation": "landscape only",
                "model_fps": a.model_fps, "chunk_frames": {"first": FIRST_CHUNK, "next": CHUNK}, "chunk_latents": 3,
                "local_attn_latents": 6, "kv_sink": "reference + first chunk (upstream update_kv)",
                "denoising_steps": 2, "custom_timesteps": [1000, 250], "shift": 5.0, "solver": "unipc",
                "fp8": a.fp8, "decoder": "Flash-VAED", "vsa_sparsity": a.vsa_sparsity,
                "vsa_backend": a.vsa_backend if a.vsa_sparsity > 0 else "off (dense attention)",
                "compile": a.compile, "attention": "torch SDPA (models/selfhost/_shims/flash_attn)",
                "pose": f"YOLOv10m + ViTPose-H whole-body (ONNX Runtime {self.extractor.providers[0]}, FP32), "
                        "upstream webcam OnlineExtractor, face crop 512x512",
                "reference": "first live source frame of each session (upstream: fresh camera frame at Start editing)",
                "gpu": self.gpu, "gpu_count": 1, "dtype": "bfloat16 (FP8 transformer linears)",
                "prompt_policy": "instruction text verbatim as the prompt; an instruction change starts a new session "
                                 "(upstream Reset + Prepare prompt + Start editing) on the next live frames",
                "input_policy": f"center crop to {self.size[0]}x{self.size[1]}, subsample to {a.model_fps} fps; newest "
                                "frames win when behind"}

    def run(self, s: edit_stream.EditSession):
        with self.torch.no_grad():
            self._run(s)

    def _run(self, s: edit_stream.EditSession):
        """Two threads, as upstream's extraction and generation workers: this one generates chunk k while the
        extraction thread takes and extracts chunk k+1 (one chunk in flight)."""
        work = queue.Queue(maxsize=1)
        stop = threading.Event()

        def extract():
            active = None
            try:
                while not stop.is_set():
                    got = s.take(lambda p: CHUNK if p == active else FIRST_CHUNK)
                    if got is None:
                        break
                    frames, prompt, event = got
                    images = [((f + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8) for f in frames]
                    restart = prompt != active
                    if restart:
                        self.extractor.reset()
                        active = prompt
                    t = time.perf_counter()
                    pose, face = self.extractor(images)
                    item = (images, prompt, event, restart, pose, face, time.perf_counter() - t)
                    while not stop.is_set():
                        try:
                            work.put(item, timeout=0.2)
                            break
                        except queue.Full:
                            pass
            except Exception as exc:  # noqa: BLE001 - re-raised in the session thread
                work.put(exc)
                return
            work.put(None)

        th = threading.Thread(target=extract, daemon=True)
        th.start()
        sess = gen = None
        try:
            while True:
                item = work.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                images, prompt, event, restart, pose, face, ext_s = item
                t0 = time.perf_counter()
                if restart:
                    if sess is not None:
                        self.on_gpu(sess.close)
                        sess = None
                    sess, gen = self.on_gpu(self._start, images[0], prompt, s.seed)
                t1 = time.perf_counter()
                out = self.on_gpu(self._push, sess, gen, pose, face)
                LOG.info("chunk %d: extract %.0f ms, start %.0f ms, generate %.0f ms", s.stats["chunks"],
                         ext_s * 1e3, (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3)
                s.emit(out)
                ms = (time.perf_counter() - t0) * 1e3
                s.stats["chunks"] += 1
                s.stats["chunk_ms"].append(ms)
                s.stats.setdefault("extract_ms", []).append(round(ext_s * 1e3))
                s.stats["face_fallbacks"] = self.extractor.face_fallbacks
                s.applied(event, f"prompt = instruction text verbatim; new session with a fresh live reference frame "
                                 f"({ms:.0f} ms)" if restart else "identical to the active prompt")
        finally:
            stop.set()
            s.close()
            th.join(10)
            if sess is not None:
                self.on_gpu(sess.close)
            ext = s.stats.pop("extract_ms", [])
            if ext:
                s.stats["extract_ms_mean"] = round(float(np.mean(ext)), 1)


def bench(engine, n):
    """Offline speed check on a synthetic stream (no client): per-chunk extract and generate times."""
    width, height = engine.size
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(CHUNK)]
    engine.extractor.reset()
    sess, gen = engine.on_gpu(engine._start, frames[0], "Change his shirt to bright red.", 1)
    ext, genr = [], []
    for i in range(n):
        k = FIRST_CHUNK if i == 0 else CHUNK
        t = time.perf_counter()
        pose, face = engine.extractor(frames[:k])
        ext.append(round(time.perf_counter() - t, 3))
        t = time.perf_counter()
        engine.on_gpu(engine._push, sess, gen, pose, face)
        genr.append(round(time.perf_counter() - t, 3))
    engine.on_gpu(sess.close)
    steady = sorted(genr[2:] or genr)[len(genr[2:] or genr) // 2]
    print({"size": engine.size, "extract_s": ext, "generate_s": genr, "median_generate_s": steady,
           "gen_fps": round(CHUNK / steady, 2), "peak_gib": round(engine.torch.cuda.max_memory_allocated() / 2**30, 1)},
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--weights", help="unused; weights are under $EDITALIVE_ROOT/weights (setup.sh)")
    ap.add_argument("--resolution", type=int, choices=[384, 480], default=384, help="webcam default 384 (672x384)")
    ap.add_argument("--model_fps", type=int, default=16, help="upstream --fps default")
    ap.add_argument("--no-fp8", dest="fp8", action="store_false", help="webcam default FP8")
    ap.add_argument("--no-compile", dest="compile", action="store_false", help="webcam default compiles")
    ap.add_argument("--vsa_sparsity", type=float, default=0.0,
                    help="0 = dense attention (upstream option); upstream's default 0.75 needs FastVideo's Hopper-only "
                         "kernel, its portable sdpa fallback runs at 8 fps on the v1 GPU")
    ap.add_argument("--vsa_backend", default="sdpa", choices=["sdpa", "auto", "kernel"],
                    help="only with --vsa_sparsity > 0")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--bench", type=int, default=0, help="offline: N chunks on a synthetic stream, print times, exit")
    args = ap.parse_args(argv)
    tensorrt_libs_on_path()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    engine = Engine(args)
    if args.bench:
        bench(engine, args.bench)
        return
    edit_stream.serve(engine, args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
