#!/usr/bin/env python3
"""Rolling Forcing (Wan2.1-T2V-1.3B, rolling-window joint denoising, 5-step DMD, attention sink) self-hosted server,
docs/protocol.md.

One WebSocket session = one live rollout of upstream `CausalInferencePipeline.inference_rolling_forcing`
(TencentARC/RollingForcing, `configs/rolling_forcing_dmd.yaml`, `rolling_forcing_dmd.pt` generator_ema), unrolled
window by window. Upstream's rolling window holds 5 blocks of 3 latent frames at staggered noise levels (the 5 warped
steps 1000/800/600/400/200, shift 5.0, no CFG): each window is ONE DiT call over all its blocks, after which every
block is re-noised to its next level, the oldest block is final, and that block is written into the clean KV cache
(`updating_cache=True`). Here each window is one live step: its final block (12 frames at 16 fps) is decoded with the
causal Wan2.1 VAE (decoder cache kept across blocks) and sent before the next window starts. The first output block
needs the 5 start-up windows of upstream's schedule (block 0 is final after window 4).

Long sessions use upstream's mechanism unchanged: the clean KV cache holds 24 latent frames and rolls with the first
block kept as an anchor (attention sink, re-roped just before the working cache); queries see at most 21 latent
frames. The temporal RoPE table (upstream 1024 positions = 256 s) is recomputed with the same formula for 4096
(positions below 1024 identical), as for models/selfhost/self_forcing.

Prompt switch: the released model has no interactive mechanism. At the next window the prompt is re-encoded and the
cross-attention cache reset, so the new prompt conditions the remaining denoising steps of all 5 blocks in the window:
the block output by that window gets only its last step under the new prompt, the next ones progressively more, and the
block that enters the window then (output 4 blocks = 2.4 s of video later) is denoised entirely under it. The clean KV
cache keeps the past as generated.

Text-to-video only: an opening image is refused, so the image + text track is not registered. Streaming, pacing,
prompt policy and acks: selfhost/t2v_stream.py.

    python server.py --port 18873
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

ROOT = Path(os.environ.get("ROLLING_FORCING_ROOT",
                           os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/rolling_forcing"))
LOG = logging.getLogger("rolling_forcing")
W, H, FPS = 832, 480, 16
BLOCK = 3            # latent frames per block (upstream num_frame_per_block)
ROPE_POSITIONS = 4096
CKPT = "checkpoints/rolling_forcing_dmd.pt"


class Engine:
    name = "Rolling Forcing"
    width, height, fps = W, H, FPS

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path.insert(0, str(repo))
        os.chdir(repo)                                 # upstream resolves wan_models/ relative to cwd
        import torch
        from collections import OrderedDict
        from omegaconf import OmegaConf
        from pipeline import CausalInferencePipeline
        from wan.modules.model import rope_params

        self.torch, self.args = torch, args
        torch.set_grad_enabled(False)
        cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                              OmegaConf.load("configs/rolling_forcing_dmd.yaml"))
        t0 = time.time()
        pipe = CausalInferencePipeline(cfg, device=torch.device("cuda"))
        ckpt = ROOT / "weights" / CKPT
        state = torch.load(ckpt, map_location="cpu")
        # upstream inference.py --use_ema: generator_ema with the FSDP wrapper prefix removed
        ema = OrderedDict((k.replace("_fsdp_wrapped_module.", ""), v) for k, v in state["generator_ema"].items())
        pipe.generator.load_state_dict(ema)
        del state, ema
        pipe = pipe.to(dtype=torch.bfloat16)
        pipe.text_encoder.to("cuda")
        pipe.generator.to("cuda")
        pipe.vae.to("cuda")
        model = pipe.generator.model
        d = model.dim // model.num_heads
        model.freqs = torch.cat([rope_params(ROPE_POSITIONS, d - 4 * (d // 6)),     # rows < 1024 unchanged
                                 rope_params(ROPE_POSITIONS, 2 * (d // 6)), rope_params(ROPE_POSITIONS, 2 * (d // 6))],
                                dim=1)
        self.pipe, self.cfg = pipe, cfg
        self.steps = pipe.denoising_step_list                             # warped, 5 steps, noisy -> clean
        self.nwin = len(self.steps)                                       # blocks per rolling window
        attn = model.blocks[0].self_attn
        self.cache_latents = 24                                           # upstream _initialize_kv_cache: 1560 * 24
        self.max_attn_latents = attn.max_attention_size // attn.frame_length
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"rolling-forcing@{self.commit[:12]} rolling_forcing_dmd.pt (generator_ema)"
        wan = ROOT / "weights/wan_models/Wan2.1-T2V-1.3B"
        self.weights_sha256 = t2v_stream.sha256_files(
            [ckpt, wan / "models_t5_umt5-xxl-enc-bf16.pth", wan / "Wan2.1_VAE.pth"], ROOT / "weights")
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s", self.version, time.time() - t0, self.gpu)
        self.warmup()

    def system_config(self):
        return {"model": self.name,
                "model_variant": "Rolling Forcing on Wan2.1-T2V-1.3B, rolling_forcing_dmd.pt (generator_ema)",
                "version": self.version, "repo": "https://github.com/TencentARC/RollingForcing",
                "repo_commit": self.commit, "license": "MIT", "weights_sha256": self.weights_sha256,
                "width": W, "height": H, "fps": FPS, "denoising_steps": self.nwin,
                "denoising_step_list": [round(float(t), 1) for t in self.steps], "timestep_shift": 5.0,
                "block_latents": BLOCK, "block_frames": BLOCK * 4, "rolling_window_blocks": self.nwin,
                "kv_cache_latents": self.cache_latents, "max_attention_latents": self.max_attn_latents,
                "sink": "first block kept as anchor, re-roped before the working cache (upstream)",
                "context_noise": int(self.cfg.context_noise), "rope_positions": ROPE_POSITIONS,
                "startup": f"first output block after {self.nwin} windows (upstream schedule)",
                "vae": "Wan2.1 VAE, causal cached decode per final block", "dtype": "bfloat16",
                "attention": "torch SDPA", "gpu": self.gpu, "gpu_count": 1,
                "prompt_switch": "text re-encoded, cross-attention cache reset at the next window: the new prompt "
                                 "conditions the remaining steps of all blocks in the rolling window (the block output "
                                 "by that window gets its last step under it; the block entering the window then is "
                                 "fully conditioned and is output 4 blocks later); clean KV cache kept (the released "
                                 "model has no prompt-switch mechanism)",
                "prompt_policy": t2v_stream.PROMPT_POLICY}

    # ---- rollout ---------------------------------------------------------------------------------------
    def _reset(self):
        torch, p = self.torch, self.pipe
        p._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device="cuda")
        p._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device="cuda")
        p.vae.model.clear_cache()
        self.noisy = {}                                # block index -> latent re-noised to its next level

    def _window(self, cond, w, gen):
        """Upstream window `w` (blocks max(0, w-4) .. w; block j at step index w-j). Returns the latent of the block
        that is final after this window (w-4), or None during start-up."""
        torch, p, n = self.torch, self.pipe, self.nwin
        seq = p.frame_seq_length
        first = max(0, w - n + 1)
        blocks = list(range(first, w + 1))
        fresh = torch.randn([1, BLOCK, 16, H // 8, W // 8], device="cuda", dtype=torch.bfloat16, generator=gen)
        x = torch.cat([self.noisy.pop(j) for j in blocks[:-1]] + [fresh], dim=1)
        t = torch.cat([torch.full([1, BLOCK], float(self.steps[w - j]), device="cuda") for j in blocks], dim=1)
        _, pred = p.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=t,
                              kv_cache=p.kv_cache_clean, crossattn_cache=p.crossattn_cache,
                              current_start=first * BLOCK * seq)
        for i, j in enumerate(blocks):                 # re-noise every block that has steps left
            k = w - j
            if k == n - 1:
                continue
            pj = pred[:, i * BLOCK:(i + 1) * BLOCK]
            noise = torch.randn(pj.flatten(0, 1).shape, device="cuda", dtype=pj.dtype, generator=gen)
            nxt = self.steps[k + 1].to("cuda") * torch.ones([BLOCK], device="cuda", dtype=torch.long)
            self.noisy[j] = p.scheduler.add_noise(pj.flatten(0, 1), noise, nxt).unflatten(0, pj.shape[:2])
        # upstream: re-run the window's first block at context_noise to update the clean cache
        p.generator(noisy_image_or_video=pred[:, :BLOCK], conditional_dict=cond,
                    timestep=torch.ones_like(t[:, :BLOCK]) * int(self.cfg.context_noise),
                    kv_cache=p.kv_cache_clean, crossattn_cache=p.crossattn_cache,
                    current_start=first * BLOCK * seq, updating_cache=True)
        return pred[:, :BLOCK] if w - first == n - 1 else None

    def _decode(self, latent):
        torch = self.torch
        video = self.pipe.vae.decode_to_pixel(latent, use_cache=True)[0]   # T C H W in [-1, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return (video.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()

    def _encode(self, prompt):
        for c in self.pipe.crossattn_cache:
            c["is_init"] = False
        return self.pipe.text_encoder(text_prompts=[prompt])

    def _step(self, cond, w, gen):
        """Windows up to `w`, starting with the start-up windows when w is the first final one."""
        out = None
        for v in range(max(0, w - self.nwin + 1) if w == self.nwin - 1 else w, w + 1):
            out = self._window(cond, v, gen)
        return list(self._decode(out))

    def warmup(self):
        torch = self.torch
        t0 = time.time()
        with torch.no_grad():
            self._reset()
            gen = torch.Generator("cuda").manual_seed(0)
            cond = self._encode("warmup: a calm lake at sunrise")
            times = []
            w = self.nwin - 1
            for _ in range(self.cache_latents // BLOCK + 3):               # past the first cache roll
                t = time.time()
                frames = self._step(cond, w, gen)
                times.append(round(time.time() - t, 3))
                w += 1
        LOG.info("warmup %.1fs; step times %s; last step %d frames -> %.1f fps", time.time() - t0, times,
                 len(frames), len(frames) / times[-1])

    def run(self, s: t2v_stream.GenSession):
        torch = self.torch
        with torch.no_grad():                          # grad mode is per thread; this runs in the session thread
            self._reset()
            gen = torch.Generator("cuda").manual_seed(s.seed)
            cond, w = None, self.nwin - 1
            while (w + 1) * BLOCK <= ROPE_POSITIONS:
                got = s.next_block()
                if got is None:
                    return
                prompt, changed, events = got
                if cond is None or changed:
                    cond = self._encode(prompt)
                s.applied(events, "text re-encoded, cross-attention cache reset; conditions the remaining steps of "
                                  f"the {self.nwin} blocks in the rolling window (fully from output block "
                                  f"{s.blocks + self.nwin - 1})")
                s.emit(self._step(cond, w, gen))
                w += 1
            raise RuntimeError(f"rollout reached the positional table ({ROPE_POSITIONS} latent frames)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18873)
    ap.add_argument("--weights", help="unused; weights are read from $ROLLING_FORCING_ROOT/weights (default $STREAMEVAL_MODEL_ROOT/rolling_forcing/weights)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t2v_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
