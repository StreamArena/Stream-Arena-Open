#!/usr/bin/env python3
"""LongLive 2.0 (NVlabs, LongLive-2.0-5B BF16: causal Wan2.2-TI2V-5B, 4-step DMD) self-hosted server.

Protocol: docs/protocol.md. One WebSocket session = one live autoregressive rollout of upstream
`CausalDiffusionInferencePipeline._inference_inner` (LongLive2.0/pipeline/causal_diffusion_inference.py), unrolled
so each 8-latent block (32 frames at 24 fps, 1280x704; the first block 29 frames) is generated, decoded and sent
before the next one starts, with the upstream `configs/inference.yaml` settings: 4 UniPC flow steps (shift 5.0),
no CFG, local attention over 32 latent frames, attention sink 8 frames, multi-shot sink on, clean-context KV pass
after each block. Model: the released merged generator `model_bf16.pt` (AR generator + DMD LoRA, so no adapter).

Live additions (nothing in the model is changed):
  * per-block prompts are upstream's multi-prompt input (`conditional_dict_list`, cross-attention cache reset every
    block). A new instruction becomes the prompt of the next block; no scene-cut prefix, so the shot continues
    (the benchmark asks for one continuous scene). KV-recache on a switch is a LongLive 1.0 feature that the 2.0
    release does not have, so it is not used.
  * streaming VAE decode: the release's Wan2.2 VAE has no `cached_decode` (its `streaming_vae` path only works for
    LightVAE), so each block is decoded with upstream `WanVAE_.decode`'s own per-latent loop while the decoder
    feature cache is kept across blocks (`first_chunk` only for the first latent) - the same frames as decoding the
    whole video at once.
  * the temporal RoPE table (upstream 1024 latent positions = 170 s at 24 fps) is recomputed with the same formula
    for 4096 positions; positions below 1024 are bit-identical, so sessions up to ~170 s are exactly upstream.

Streaming, pacing, prompt policy and acks: selfhost/t2v_stream.py.

    python server.py --port 18810 [--weights $STREAMEVAL_MODEL_ROOT/longlive/weights]
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import t2v_stream  # noqa: E402

ROOT = Path(os.environ.get("LONGLIVE_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/longlive"))
REPO = ROOT / "repo/LongLive2.0"
LOG = logging.getLogger("longlive")
W, H, FPS = 1280, 704, 24
BLOCK = 8                      # latent frames per block (configs/inference.yaml num_frame_per_block)
ROPE_POSITIONS = 4096


class Engine:
    name = "LongLive-2.0-5B"
    width, height, fps = W, H, FPS

    def __init__(self, args):
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)                                 # upstream resolves wan_models/ relative to cwd
        import torch
        import torchvision.io as tvio
        if not hasattr(tvio, "write_video"):           # only imported by upstream's training pipeline module
            tvio.write_video = None
        from omegaconf import OmegaConf
        from pipeline import CausalDiffusionInferencePipeline
        from utils.config import normalize_config
        from utils.nvfp4_checkpoint import unwrap_generator_state_dict
        from wan_5b.modules.model import rope_params
        from wan_5b.modules.vae2_2 import unpatchify

        self.torch, self.args, self.unpatchify = torch, args, unpatchify
        torch.set_grad_enabled(False)
        cfg = normalize_config(OmegaConf.load("configs/inference.yaml"))
        t0 = time.time()
        pipe = CausalDiffusionInferencePipeline(cfg, device=torch.device("cuda"))
        self.ckpt = Path(args.weights) / "LongLive-2.0-5B/model_bf16.pt"
        state = unwrap_generator_state_dict(torch.load(self.ckpt, map_location="cpu", weights_only=False))
        pipe.generator.load_state_dict(state, strict=True)                  # merged generator (upstream quick start)
        del state
        pipe = pipe.to(device="cuda", dtype=torch.bfloat16)
        pipe.generator.model.eval().requires_grad_(False)
        # session-invariant model settings, as upstream `inference()` sets them before every rollout
        dit = pipe._dit_model
        dit.local_attn_size = pipe.local_attn_size
        pipe._set_all_modules_max_attention_size(pipe.local_attn_size)
        pipe._set_all_modules_sink_size(pipe.sink_size)
        pipe._set_all_modules_global_sink_size(pipe.global_sink_size)
        dit.use_relative_rope = pipe.use_relative_rope
        dit.rope_temporal_offset = 0.0
        d = dit.dim // dit.num_heads
        dit.freqs = torch.cat([rope_params(ROPE_POSITIONS, d - 4 * (d // 6)),     # rows < 1024 unchanged
                               rope_params(ROPE_POSITIONS, 2 * (d // 6)), rope_params(ROPE_POSITIONS, 2 * (d // 6))],
                              dim=1)
        self.pipe, self.cfg, self.seq = pipe, cfg, pipe.frame_seq_length
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"LongLive2.0@{self.commit[:12]} LongLive-2.0-5B model_bf16.pt"
        wan = Path(args.weights) / "wan_models/Wan2.2-TI2V-5B"
        self.weights_sha256 = t2v_stream.sha256_files(
            [self.ckpt, wan / "models_t5_umt5-xxl-enc-bf16.pth", wan / "Wan2.2_VAE.pth"], Path(args.weights))
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s; local_attn=%s sink=%s global_sink=%s mem=%.1f GiB", self.version,
                 time.time() - t0, self.gpu, pipe.local_attn_size, pipe.sink_size, pipe.global_sink_size,
                 torch.cuda.memory_allocated() / 2**30)
        self.warmup()

    def system_config(self):
        p = self.pipe
        return {"model": self.name, "model_variant": "LongLive-2.0-5B BF16 (merged AR generator + DMD LoRA) on "
                                                     "Wan2.2-TI2V-5B, text-to-video",
                "version": self.version, "repo": "https://github.com/NVlabs/LongLive (LongLive2.0/)",
                "repo_commit": self.commit, "license": "code Apache-2.0; weights NVIDIA Open Model License",
                "weights_sha256": self.weights_sha256, "width": W, "height": H, "fps": FPS,
                "denoising_steps": int(p.sampling_steps), "solver": "UniPC flow, shift 5.0", "guidance_scale": 1.0,
                "block_latents": BLOCK, "block_frames": BLOCK * 4, "local_attn_latents": int(p.local_attn_size),
                "sink_latents": int(p.sink_size), "multi_shot_sink": bool(p.multi_shot_sink),
                "rope_positions": ROPE_POSITIONS, "vae": "Wan2.2 VAE, causal decode per block (decoder cache kept)",
                "dtype": "bfloat16", "attention": "torch SDPA (no flash-attn installed)", "gpu": self.gpu,
                "gpu_count": 1,
                "prompt_switch": "upstream per-block prompts: the next block uses the new prompt (cross-attention "
                                 "recomputed); no scene-cut prefix, no KV-recache (not in the 2.0 release)",
                "prompt_policy": t2v_stream.PROMPT_POLICY}

    # ---- rollout ---------------------------------------------------------------------------------------
    def _reset(self):
        torch, p = self.torch, self.pipe
        p.clear_cache()
        p._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=torch.device("cuda"))
        p._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=torch.device("cuda"))
        p._dit_model.rope_temporal_offset = 0.0
        p.vae.model.clear_cache()

    def _encode(self, prompt):
        return self.pipe.text_encoder(text_prompts=[prompt])               # {"prompt_embeds": [1, 512, 4096]}

    def _block(self, cond, index, gen):
        """Upstream _inference_inner body for chunk `index` (T2V, no initial latent): latents [1, 8, 48, 44, 80]."""
        torch, p = self.torch, self.pipe
        start = index * BLOCK * self.seq
        for b in range(p.num_transformer_blocks):     # upstream: reset every chunk (per-chunk prompts)
            p.crossattn_cache_pos[b]["is_init"] = False
            p.crossattn_cache_neg[b]["is_init"] = False
        latents = torch.randn([1, BLOCK, 48, H // 16, W // 16], device="cuda", dtype=torch.bfloat16, generator=gen)
        sched = p._initialize_sample_scheduler(latents)
        for t in sched.timesteps:
            timestep = t * torch.ones([1, BLOCK], device="cuda", dtype=torch.float32)
            flow, _ = p.generator(noisy_image_or_video=latents, conditional_dict=cond, timestep=timestep,
                                  kv_cache=p.kv_cache_pos, crossattn_cache=p.crossattn_cache_pos,
                                  current_start=start, cache_start=start)
            latents = sched.step(flow, t, latents, return_dict=False)[0]
        p.generator(noisy_image_or_video=latents, conditional_dict=cond, timestep=timestep * 0,
                    kv_cache=p.kv_cache_pos, crossattn_cache=p.crossattn_cache_pos,
                    current_start=start, cache_start=start)
        return latents

    def _decode(self, latents, first):
        """WanVAE_.decode loop on this block's latents, decoder feature cache kept: uint8 [T, H, W, 3]."""
        torch, vae = self.torch, self.pipe.vae
        m = vae.model
        mean = vae.mean.to(device="cuda", dtype=latents.dtype).view(1, m.z_dim, 1, 1, 1)
        inv_std = (1.0 / vae.std.to(device="cuda", dtype=latents.dtype)).view(1, m.z_dim, 1, 1, 1)
        z = latents.permute(0, 2, 1, 3, 4) / inv_std + mean
        x = m.conv2(z)
        outs = []
        for i in range(x.shape[2]):
            m._conv_idx = [0]
            outs.append(m.decoder(x[:, :, i:i + 1], feat_cache=m._feat_map, feat_idx=m._conv_idx,
                                  first_chunk=first and i == 0))
        video = self.unpatchify(torch.cat(outs, 2), patch_size=2)[0].float().clamp_(-1, 1)   # C T H W
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return (video.permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu().numpy()

    def warmup(self):
        torch = self.torch
        t0 = time.time()
        with torch.no_grad():
            self._reset()
            gen = torch.Generator("cuda").manual_seed(0)
            cond = self._encode("warmup: a calm lake at sunrise, gentle ripples on the water")
            times = []
            for i in range(32 // BLOCK + 2):                                # past the first cache roll
                t = time.time()
                frames = self._decode(self._block(cond, i, gen), i == 0)
                torch.cuda.synchronize()
                times.append(round(time.time() - t, 3))
        LOG.info("warmup %.1fs; block times %s; last block %d frames -> %.1f fps; peak mem %.1f GiB",
                 time.time() - t0, times, len(frames), len(frames) / times[-1], torch.cuda.max_memory_allocated() / 2**30)

    def run(self, s: t2v_stream.GenSession):
        torch = self.torch
        with torch.no_grad():                          # grad mode is per thread; this runs in the session thread
            self._reset()
            gen = torch.Generator("cuda").manual_seed(s.seed)
            cond, index = None, 0
            while (index + 1) * BLOCK <= ROPE_POSITIONS:
                got = s.next_block()
                if got is None:
                    return
                prompt, changed, events = got
                if cond is None or changed:
                    cond = self._encode(prompt)
                s.applied(events, "upstream per-block prompt")
                s.emit(list(self._decode(self._block(cond, index, gen), index == 0)))
                index += 1
            raise RuntimeError(f"rollout reached the positional table ({ROPE_POSITIONS} latent frames)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18810)
    ap.add_argument("--weights", default=str(ROOT / "weights"))
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t2v_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
