#!/usr/bin/env python3
"""Yume 1.5 (stdstu12/YUME, Yume-5B-720P; Shanghai AI Lab / Fudan) self-hosted server, docs/protocol.md.

One WebSocket session = one live rollout of upstream's long-video loop (fastvideo/sample/sample_5b.py, the same loop
as webapp_single_gpu.py `long_generate`), unrolled so every segment is generated, decoded and sent before the next
one starts. Upstream settings (scripts/inference/sample_5b.sh): 1280x704, 16 fps, segments of 8 latents (32 frames,
`frame_zero` / `latent_frame_zero`), 4 Euler steps on the distilled model (sigma shift 7.0), no CFG, Wan2.2 VAE.
  * history: all generated latents stay in the context; upstream's model compresses older latents at growing ratios
    (Yume's FramePack-style memory, up to 1370 latents = 340 s), so a session never outgrows it.
  * per-segment text: upstream conditions every segment on its own caption (`caption[step_sample]`, re-encoded
    through `Yume.generate` each segment). The segment caption is '<init prompt>\\n<latest instruction>' (see
    selfhost/t2v_stream.py); a new instruction applies from the next segment and is acked `applied` when it starts.
    Yume's camera / keyboard control is text ("The camera pushes forward (W)."), so camera_control text goes into
    the caption verbatim.
  * attention: upstream calls flash-attn 2; selfhost/_shims implements it with torch SDPA.
  * text start: upstream T2V first segment (`flag=False`, no history); image start: upstream I2V first segment (the
    image, center-cropped to 1280x704, as frame 0 of a 32-frame clip with 16 copies in front, VAE-encoded).

    python server.py --port 18840 [--weights $STREAMEVAL_MODEL_ROOT/yume/weights/Yume-5B-720P]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
import t2v_stream  # noqa: E402

ROOT = Path(os.environ.get("YUME_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/yume"))
REPO = ROOT / "repo"
LOG = logging.getLogger("yume")
W, H, FPS = 1280, 704, 16
LFZ, FZ = 8, 32              # latent_frame_zero / frame_zero (upstream)
STEPS, SHIFT = 4, 7.0        # sample_5b.sh --num_euler_timesteps 4; get_sampling_sigmas(sample_step, 7.0)
MAX_LATENTS = 1370 - LFZ     # upstream compression tiers end at 2+4+16+64+256+1024 history latents


class Engine:
    name = "Yume-5B-720P"
    width, height, fps = W, H, FPS
    image_start = True

    def __init__(self, args):
        self.args = args
        sys.path.insert(0, str(HERE.parent / "_shims"))   # flash_attn on torch SDPA
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import numpy as np
        import torch
        import wan23
        from wan23.configs import WAN_CONFIGS
        self.torch, self.np = torch, np
        torch.set_grad_enabled(False)
        t0 = time.time()
        self.wan = wan23.Yume(config=WAN_CONFIGS["ti2v-5B"], checkpoint_dir=str(args.weights), device_id=0)
        self.model = self.wan.model.eval().requires_grad_(False).to(torch.bfloat16).to("cuda")   # sample_5b.py
        self.wan.text_encoder.model.to("cuda")
        self.sigmas = self._sigmas(STEPS, SHIFT)
        self.gpu = torch.cuda.get_device_name(0)
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"stdstu12/YUME@{self.commit[:12]} Yume-5B-720P"
        wdir = Path(args.weights)
        self.weights_sha256 = t2v_stream.sha256_files(
            [wdir / "diffusion_pytorch_model.safetensors", wdir / "Wan2.2_VAE.pth",
             wdir / "models_t5_umt5-xxl-enc-bf16.pth"], wdir.parent)
        LOG.info("loaded %s in %.0fs on %s, mem %.1f GiB", self.version, time.time() - t0, self.gpu,
                 torch.cuda.memory_allocated() / 2**30)
        if args.warmup:
            self.warmup()

    @staticmethod
    def _sigmas(steps, shift):
        import numpy as np
        sigma = np.linspace(1, 0, steps + 1)[:steps]
        return shift * sigma / (1 + (shift - 1) * sigma)

    def system_config(self):
        return {"model": self.name, "model_variant": "Yume 1.5, Yume-5B-720P (Wan2.2-TI2V-5B based, distilled)",
                "version": self.version, "repo": "https://github.com/stdstu12/YUME", "repo_commit": self.commit,
                "license": "code Apache-2.0; weights Apache-2.0 (stdstu123/Yume-5B-720P)",
                "weights_sha256": self.weights_sha256, "width": W, "height": H, "fps": FPS,
                "segment_latents": LFZ, "segment_frames": FZ, "steps": STEPS, "sigma_shift": SHIFT,
                "guidance_scale": 1.0, "dtype": "bfloat16 DiT (autocast), Wan2.2 VAE", "gpu": self.gpu, "gpu_count": 1,
                "history": "all latents, compressed by upstream's multi-ratio patch embeddings",
                "image_start": "upstream I2V first segment (center-crop to 1280x704)",
                "prompt_switch": "per-segment caption (upstream caption[step_sample]) from the next segment",
                "prompt_policy": t2v_stream.PROMPT_POLICY}

    # ---- rollout (upstream sample_5b.py loop) ------------------------------------------------------------
    def _image_clip(self, image):
        from PIL import Image, ImageOps
        torch, np = self.torch, self.np
        img = ImageOps.fit(image.convert("RGB"), (W, H), Image.LANCZOS)
        ten = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        vid = torch.zeros(3, FZ, H, W)                          # webapp create_video_from_image
        vid[:, 0] = (ten - 0.5) * 2
        vid = torch.cat([vid[:, 0:1].repeat(1, 16, 1, 1), vid], dim=1).to("cuda")   # 16 head copies
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            a = self.wan.vae.encode([vid[:, :-FZ]])[0]
            b = self.wan.vae.encode([vid[:, -FZ:]])[0]
        if self.args.debug:
            LOG.info("image clip: %s + %s latents", tuple(a.shape), tuple(b.shape))
        return torch.cat([a, b], dim=1)

    def _decode(self, model_input):
        torch = self.torch
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            video = self.wan.vae.decode([model_input[:, -LFZ:].to(torch.float32)])[0]      # C T H W, [-1, 1]
        video = video[:, -FZ:]
        v = ((video.float().clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
        return list(v.permute(1, 2, 3, 0).contiguous().cpu().numpy())

    def _segment(self, prompt, model_input, seed):
        """One upstream segment: returns the new model_input (all latents so far)."""
        torch, wan, sig = self.torch, self.wan, self.sigmas
        if model_input is None:                                 # text start (sample_5b t2v branch)
            arg_c, _, noise = wan.generate(prompt, frame_num=FZ, max_area=W * H, latent_frame_zero=LFZ,
                                           seed=seed, offload_model=False)
            latent = noise
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(STEPS):
                    t = torch.tensor([sig[i] * 1000]).to("cuda")
                    pred = self.model([latent], t=t, flag=False, **arg_c)[0]
                    nxt = sig[i + 1] if i + 1 < STEPS else 0.0
                    latent = latent + (nxt - sig[i]) * pred
            return latent
        frame_num = (model_input.shape[1] - 1) * 4 + 1 + FZ
        arg_c, _, noise, mask2, img = wan.generate(prompt, img=model_input, frame_num=frame_num, max_area=W * H,
                                                   latent_frame_zero=LFZ, seed=seed, offload_model=False)
        if self.args.debug:
            LOG.info("segment: model_input %s img %s noise %s mask2 %s seq_len %s", tuple(model_input.shape),
                     tuple(img[0].shape), tuple(noise.shape), tuple(mask2[0].shape), arg_c["seq_len"])
        base = torch.cat([model_input, model_input.new_zeros(model_input.shape[0], LFZ, *model_input.shape[2:])], dim=1)
        latent = torch.cat([img[0][:, :-LFZ], torch.randn_like(base)[:, -LFZ:]], dim=1)
        hist_ts = mask2[0][0][:-LFZ, ::2, ::2].flatten()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(STEPS):
                t = torch.tensor([sig[i] * 1000]).to("cuda")
                ts = torch.cat([hist_ts, hist_ts.new_ones(arg_c["seq_len"] - hist_ts.size(0)) * t]).unsqueeze(0)
                pred = self.model([latent], t=ts, **arg_c)[0]
                nxt = sig[i + 1] if i + 1 < STEPS else 0.0
                x0 = latent[:, -LFZ:] + (nxt - sig[i]) * pred[:, -LFZ:]
                latent = torch.cat([base[:, :-LFZ], x0], dim=1)
        return torch.cat([model_input, latent[:, -LFZ:]], dim=1)

    def rollout(self, prompt, seed, image=None, s: t2v_stream.GenSession | None = None, max_segments=None):
        torch = self.torch
        torch.manual_seed(seed)
        model_input, first_i2v = None, False
        if image is not None:
            model_input, first_i2v = self._image_clip(image), True
        times, k = [], 0
        try:
            while max_segments is None or k < max_segments:
                if s is not None:
                    got = s.next_block()
                    if got is None:
                        return times
                    prompt, _, events = got
                    s.applied(events, f"caption of segment {k}")
                t0 = time.time()
                if first_i2v:                                   # sample_5b i2v first segment: replace the tail
                    first_i2v = False
                    hist = model_input[:, :-LFZ]
                    out = self._segment(prompt, hist, seed)
                else:
                    out = self._segment(prompt, model_input, seed + k)
                model_input = out
                frames = self._decode(model_input)
                torch.cuda.synchronize()
                times.append(round(time.time() - t0, 3))
                if s is not None:
                    s.emit(frames)
                k += 1
                if model_input.shape[1] > MAX_LATENTS:
                    raise RuntimeError("history reached upstream's compression limit")
            return times
        finally:
            torch.cuda.empty_cache()

    def warmup(self):
        from PIL import Image
        for image in (None, Image.new("RGB", (W, H), (120, 140, 160))):
            t0 = time.time()
            with self.torch.inference_mode():
                times = self.rollout("First-person perspective. A calm lake at sunrise.", 0, image, max_segments=3)
            LOG.info("warmup (%s) %.0fs; segment times %s; peak mem %.1f GiB", "image" if image else "text",
                     time.time() - t0, times, self.torch.cuda.max_memory_allocated() / 2**30)

    def run(self, s: t2v_stream.GenSession):
        image = None
        if s.init.get("image_b64"):
            from PIL import Image
            image = Image.open(io.BytesIO(base64.b64decode(s.init["image_b64"])))
        with self.torch.inference_mode():
            self.rollout(s.prompt(), s.seed, image, s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18840)
    ap.add_argument("--weights", default=str(ROOT / "weights/Yume-5B-720P"))
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    ap.add_argument("--bench-image", action="store_true", help="--bench with an image start")
    ap.add_argument("--bench", type=int, default=0, help="offline: N segments text start, print times, exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    engine = Engine(args)
    if args.bench:
        with engine.torch.inference_mode():
            from PIL import Image
            image = Image.new("RGB", (W, H), (90, 120, 160)) if args.bench_image else None
            times = engine.rollout("First-person perspective. A red kite flies over a green hill.", 1, image,
                                   max_segments=args.bench)
        print(json.dumps({"segment_s": times}), flush=True)
        return
    t2v_stream.serve(engine, args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
