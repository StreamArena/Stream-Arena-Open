#!/usr/bin/env python3
"""Zing-0.5 (5B causal world model on Wan2.2-TI2V-5B, Seedleap.ai) self-hosted server, docs/protocol.md.

One WebSocket session = one live autoregressive rollout of upstream `InferencePipeline.generate`
(seedleap/zing-world-model, `config/zing.yaml`, `generator/model.pt`), unrolled so that each causal block is
generated, decoded and sent before the next one starts. Upstream's layout: latent 0 is its own block (T2I for a text
start; the clean image latent for an image start), then blocks of 4 latent frames (16 video frames at 24 fps):

  * denoising: 4 DMD steps (1000/750/500/250, warped, shift 5.0), no CFG, then the clean-context pass ("final",
    timestep 0) into the KV cache, exactly upstream's per-chunk loop;
  * decoding: upstream decodes the whole rollout once with the native Wan2.2 VAE (diffusers AutoencoderKLWan), which is
    frame-by-frame causal internally; here the same decoder runs per block with its causal feature cache kept across
    blocks (identical arithmetic to one full decode).

Online interaction uses the model's own mechanism (upstream `text_prompt_interval`): a changed prompt takes effect at
the next block with `prompt_switch=True` on its first step, which replaces the text K/V cache, keeps the visual K/V
cache, and pins the block's K/V when it would leave the window. Block prompt policy: selfhost/t2v_stream.py
('<init prompt>\\n<latest instruction>'). No keyboard input is invented: camera_control texts are prompt text and
the action input is absent, as in upstream's text-only rollouts.

Speed: upstream's `compile_fusion` option (torch.compile of the norm / AdaLN segments) and cudnn.benchmark for the VAE
decoder; blocks are delivered as soon as generation finishes.

Long sessions: the online cache configuration recommended upstream (README: 33/5 window "for online real-time
inference"), i.e. 33 visible latent frames including a 5-latent sink; RoPE follows Seedleap's realtime server
(Zing-SGLang): `block_relative` temporal positions with gaps between visible frames clamped to 12, so the positional
table (1024) is never exceeded (absolute positions would run out after ~170 s at 24 fps).

Image start (general_interaction_image): upstream TI2V (`reference_frame_count = 1`): the opening image (center-crop to
832:480, then resize; upstream resizes without cropping) is VAE-encoded as latent 0 and written into the cache as
clean context; its decoded frame is the first output frame.

    python server.py --port 18876
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import t2v_stream  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ZING_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/zing"))
LOG = logging.getLogger("zing")
W, H, FPS = 832, 480, 24          # the paper's real-time setting (832x480, 24 fps)
LOCAL_ATTN, SINK = 33, 5          # upstream README: online real-time window
ROPE_MAX_GAP = 12                 # Zing-SGLang online configuration (rope_position_mode block_relative)


def center_crop(img, w, h):
    """Crop to the w:h aspect ratio around the centre, then resize to w x h (RGB)."""
    from PIL import Image
    img = img.convert("RGB")
    iw, ih = img.size
    if iw * h > ih * w:
        nw = round(ih * w / h)
        img = img.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
    else:
        nh = round(iw * h / w)
        img = img.crop((0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh))
    return img.resize((w, h), Image.BICUBIC)


def block_relative_positions(self, current):
    """CausalKVCache.attention_positions with Zing-SGLang's block_relative RoPE: visible frames keep their order and
    spacing, but each gap between consecutive visible latent frames is clamped to ROPE_MAX_GAP."""
    import torch
    if self.positions is None:
        keys = current
    else:
        end = self.active_start if self.active_start is not None else int(self.positions.shape[0])
        keys = torch.cat((self.positions[:end], current), dim=0)
    keys = keys.clone()
    frames, index = torch.unique_consecutive(keys[:, 0], return_inverse=True)
    gap = torch.cat((frames.new_zeros(1), frames[1:] - frames[:-1])).clamp(max=ROPE_MAX_GAP)
    keys[:, 0] = torch.cumsum(gap, dim=0)[index]
    return keys[-current.shape[0]:], keys


class Engine:
    name = "Zing-0.5"
    width, height, fps = W, H, FPS
    image_start = True

    def __init__(self, args):
        sys.path.insert(0, str(HERE.parent / "_shims"))   # flash_attn on torch SDPA
        sys.path.insert(0, str(ROOT / "repo" / "src"))
        import torch
        from zing_v0_5.config import load_config, with_cache_window
        from zing_v0_5.pipeline import InferencePipeline
        from zing_v0_5.scheduler import DmdScheduler
        from zing_v0_5.model import kv_cache

        kv_cache.CausalKVCache.attention_positions = block_relative_positions
        self.torch, self.DmdScheduler = torch, DmdScheduler
        from dataclasses import replace
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True             # VAE decoder conv3d: 0.253 -> 0.223 s per block
        cfg = with_cache_window(load_config(ROOT / "repo/config/zing.yaml"), LOCAL_ATTN, SINK)
        # upstream's own option: torch.compile of the norm / AdaLN segments (DiT block 0.466 -> 0.44 s)
        cfg = replace(cfg, generator=replace(cfg.generator, compile_fusion=True))
        wdir = ROOT / "weights/Zing-0.5"
        t0 = time.time()
        pipe = InferencePipeline(cfg, wdir / "pretrained", wdir / "generator/model.pt")
        pipe.text_encoder.to("cuda")                      # upstream swaps encoder / VAE to the CPU; keep resident
        pipe.vae.to("cuda")
        self.pipe, self.cfg, self.dev = pipe, cfg, pipe.device
        self.vae = pipe.vae.model
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"zing-world-model@{self.commit[:12]} Zing-0.5 generator/model.pt"
        self.weights_sha256 = t2v_stream.sha256_files(
            [wdir / "generator/model.pt", *sorted((wdir / "pretrained/text_encoder").glob("*.safetensors")),
             wdir / "pretrained/vae/diffusion_pytorch_model.safetensors"], ROOT / "weights")
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s", self.version, time.time() - t0, self.gpu)
        self.warmup()

    def system_config(self):
        inf = self.cfg.inference
        return {"model": self.name,
                "model_variant": "Zing-0.5 5B causal world model (4-step DMD) on Wan2.2-TI2V-5B, generator/model.pt",
                "version": self.version, "repo": "https://github.com/seedleap/zing-world-model",
                "repo_commit": self.commit, "license": "Apache-2.0", "weights_sha256": self.weights_sha256,
                "width": W, "height": H, "fps": FPS, "denoising_steps": len(inf.denoising_steps),
                "denoising_step_list": list(inf.denoising_steps), "timestep_shift": inf.timestep_shift,
                "block_latents": inf.frames_per_block, "block_frames": 4 * inf.frames_per_block,
                "first_block": "latent 0 alone (T2I for text start, clean image latent for image start)",
                "local_attn_latents": LOCAL_ATTN, "sink_latents": SINK,
                "rope": f"block_relative (gaps between visible latent frames clamped to {ROPE_MAX_GAP}; Zing-SGLang online)",
                "prompt_switch": "upstream mechanism: text K/V cache replaced at the next block (prompt_switch on its "
                                 "first step), visual K/V cache kept, the block pinned when it leaves the window",
                "actions": "none (no keyboard input; camera_control as prompt text)",
                "vae": "Wan2.2 VAE (diffusers AutoencoderKLWan), causal cached decode per block",
                "text_encoder": "UMT5-XXL (pretrained/text_encoder)", "dtype": "bfloat16",
                "attention": "torch SDPA (models/selfhost/_shims/flash_attn)", "cfg": "none",
                "compile_fusion": True, "cudnn_benchmark": True,
                "gpu": self.gpu, "gpu_count": 1, "prompt_policy": t2v_stream.PROMPT_POLICY}

    # ---- rollout ---------------------------------------------------------------------------------------
    def _context(self, prompt):
        flat, lengths = self.pipe.text_encoder.encode([prompt])
        return flat.to(device=self.dev, dtype=self.torch.bfloat16), lengths.to(self.dev)

    def _flow(self, x, t, ctx, cache, mode, switch):
        return self.pipe._model_flow(x, t, ctx, cache, mode, None, switch)

    def _block(self, n, ctx, cache, switch):
        """Generate n latent frames (upstream per-chunk loop); returns the clean latents [1, n, 48, h, w]."""
        torch = self.torch
        lat = torch.randn((1, n, 48, H // 16, W // 16), device=self.dev, dtype=torch.bfloat16)
        sched = self.DmdScheduler(self.cfg.inference).to(self.dev)
        for i, t in enumerate(sched.timesteps):
            step_t = t * torch.ones((1, n), device=self.dev, dtype=torch.float32)
            pred = self._flow(lat, step_t, ctx, cache, "active", switch and i == 0)
            lat, _ = sched.step(pred, lat)
        self._flow(lat, torch.zeros((1, n), device=self.dev, dtype=torch.float32), ctx, cache, "final", False)
        return lat

    def _decode(self, lat):
        """Causal Wan2.2 VAE decode of these latents with the decoder cache kept (diffusers AutoencoderKLWan._decode
        unrolled); returns uint8 frames [T, H, W, 3]."""
        torch, vae = self.torch, self.vae
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
        mean, std = self.pipe.vae._stats(lat)
        z = lat.permute(0, 2, 1, 3, 4) * std + mean
        x = vae.post_quant_conv(z)
        outs = []
        for i in range(x.shape[2]):
            vae._conv_idx = [0]
            outs.append(vae.decoder(x[:, :, i:i + 1], feat_cache=vae._feat_map, feat_idx=vae._conv_idx,
                                    first_chunk=self._first_decode))
            self._first_decode = False
        out = torch.cat(outs, 2)
        if vae.config.patch_size is not None:
            out = unpatchify(out, patch_size=vae.config.patch_size)
        video = (out.clamp(-1, 1)[0].float() * 0.5 + 0.5).clamp(0, 1)          # C T H W
        return (video.permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu().numpy()

    def _image_latent(self, image):
        torch = self.torch
        import numpy as np
        px = torch.from_numpy(np.asarray(center_crop(image, W, H)).copy()).permute(2, 0, 1)[None, :, None]
        return self.pipe.vae.encode(px.to(self.dev)).to(device=self.dev, dtype=torch.bfloat16)   # [1, 1, 48, h, w]

    def _start(self, seed):
        torch = self.torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.vae.clear_cache()
        self._first_decode = True
        return self.pipe.generator.make_kv_cache()

    def warmup(self):
        from PIL import Image
        torch = self.torch
        for image in (None, Image.new("RGB", (1024, 768), (90, 140, 200))):
            t0 = time.time()
            with torch.no_grad():
                lat0 = self._image_latent(image) if image is not None else None
                cache = self._start(0)
                ctx = self._context("warmup: a calm lake at sunrise")
                times = []
                if lat0 is not None:
                    self._flow(lat0, torch.zeros((1, 1), device=self.dev), ctx, cache, "final", False)
                    self._decode(lat0)
                else:
                    self._decode(self._block(1, ctx, cache, False))
                for i in range(12):                                         # past the window (9 blocks)
                    t = time.time()
                    frames = self._decode(self._block(4, ctx, cache, i == 6))
                    torch.cuda.synchronize()
                    times.append(round(time.time() - t, 3))
            LOG.info("warmup (%s) %.1fs; block times %s; last block %d frames -> %.1f fps",
                     "image" if image else "text", time.time() - t0, times, len(frames), len(frames) / times[-1])

    def run(self, s: t2v_stream.GenSession):
        torch = self.torch
        image = None
        if s.init.get("image_b64"):
            from PIL import Image
            image = Image.open(io.BytesIO(base64.b64decode(s.init["image_b64"])))
        with torch.no_grad():                          # grad mode is per thread; this runs in the session thread
            lat0 = self._image_latent(image) if image is not None else None
            cache = self._start(s.seed)
            ctx, first = None, True
            while True:
                got = s.next_block()
                if got is None:
                    return
                prompt, changed, events = got
                switch = ctx is not None and changed
                if ctx is None or changed:
                    ctx = self._context(prompt)
                s.applied(events, "prompt switch (upstream): text K/V cache replaced, visual K/V cache kept"
                          if switch else "")
                if first:                              # latent 0: the image latent, or T2I; then the first 4-latent block
                    if lat0 is not None:
                        self._flow(lat0, torch.zeros((1, 1), device=self.dev), ctx, cache, "final", False)
                        frames = list(self._decode(lat0))
                    else:
                        frames = list(self._decode(self._block(1, ctx, cache, False)))
                    frames += list(self._decode(self._block(4, ctx, cache, False)))
                    first = False
                else:
                    frames = list(self._decode(self._block(4, ctx, cache, switch)))
                s.emit(frames)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18876)
    ap.add_argument("--weights", help="unused; weights are read from $ZING_ROOT/weights (setup.sh)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t2v_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
