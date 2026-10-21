#!/usr/bin/env python3
"""Self-Forcing (Wan2.1-T2V-1.3B, chunk-wise autoregressive, 4-step DMD) self-hosted server, docs/protocol.md.

One WebSocket session = one live autoregressive rollout of upstream `CausalInferencePipeline.inference`, unrolled
so that each 3-latent block (12 frames at 16 fps; the first block 9) is generated, decoded and sent before the next
one starts: 4 denoising steps (1000/750/500/250, warped, shift 5.0) -> clean-context pass into the KV cache ->
causal Wan2.1 VAE decode of just these latents (`cached_decode`, decoder cache kept across blocks).

Differences from the upstream 5-second script, all needed to run an open-ended interactive session:
  * rolling KV cache: `local_attn_size = 21` latent frames (the trained context length, upstream default cache
    size) enables upstream's own cache rolling in `CausalWanSelfAttention`, so the rollout continues past 21 latent
    frames by evicting the oldest ones. The checkpoint was trained on 5-second clips; long rollouts are an
    extrapolation (see README).
  * the temporal RoPE table (upstream 1024 positions = 256 s) is recomputed with the same formula for 4096.
  * prompt switch: the released model has no interactive mechanism. The block prompt is re-encoded and the
    cross-attention cache reset at the next block; the self-attention KV cache keeps the past as generated.

Streaming, pacing, prompt policy and acks: selfhost/t2v_stream.py.

    python server.py --port 18813        # weights are read from $SELF_FORCING_ROOT/weights
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

ROOT = Path(os.environ.get("SELF_FORCING_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/self_forcing"))
LOG = logging.getLogger("self_forcing")
W, H, FPS = 832, 480, 16
BLOCK = 3            # latent frames per block (upstream num_frame_per_block)
WINDOW = 21          # latent frames of self-attention context (training length)
ROPE_POSITIONS = 4096


class Engine:
    name = "Self-Forcing"
    width, height, fps = W, H, FPS

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path.insert(0, str(repo))
        os.chdir(repo)                                 # upstream resolves wan_models/ relative to cwd
        import torch
        from omegaconf import OmegaConf
        from pipeline import CausalInferencePipeline
        from wan.modules.model import rope_params

        self.torch, self.args = torch, args
        torch.set_grad_enabled(False)
        cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                              OmegaConf.load("configs/self_forcing_dmd.yaml"))
        cfg.model_kwargs.local_attn_size = WINDOW
        t0 = time.time()
        pipe = CausalInferencePipeline(cfg, device=torch.device("cuda"))
        ckpt = ROOT / "weights/checkpoints/self_forcing_dmd.pt"
        state = torch.load(ckpt, map_location="cpu")
        pipe.generator.load_state_dict(state["generator_ema"])      # upstream README: --use_ema
        del state
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
        self.steps = [float(t) for t in pipe.denoising_step_list]
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"self-forcing@{self.commit[:12]} self_forcing_dmd.pt (generator_ema)"
        wan = ROOT / "weights/wan_models/Wan2.1-T2V-1.3B"
        self.weights_sha256 = t2v_stream.sha256_files(
            [ckpt, wan / "models_t5_umt5-xxl-enc-bf16.pth", wan / "Wan2.1_VAE.pth"], ROOT / "weights")
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s", self.version, time.time() - t0, self.gpu)
        self.warmup()

    def system_config(self):
        return {"model": self.name, "model_variant": "Wan2.1-T2V-1.3B causal, self_forcing_dmd.pt (generator_ema)",
                "version": self.version, "repo": "https://github.com/guandeh17/Self-Forcing",
                "repo_commit": self.commit, "license": "Apache-2.0", "weights_sha256": self.weights_sha256,
                "width": W, "height": H, "fps": FPS, "denoising_steps": len(self.steps),
                "denoising_step_list": [round(t, 1) for t in self.steps], "timestep_shift": 5.0,
                "block_latents": BLOCK, "block_frames": BLOCK * 4, "local_attn_latents": WINDOW, "sink_latents": 0,
                "context_noise": int(self.cfg.context_noise), "rope_positions": ROPE_POSITIONS,
                "vae": "Wan2.1 VAE, causal cached decode per block", "dtype": "bfloat16", "attention": "torch SDPA",
                "gpu": self.gpu, "gpu_count": 1,
                "long_rollout": "rolling KV cache over the last 21 latent frames (upstream cache rolling); the "
                                "checkpoint was trained on 21-latent (5 s) clips",
                "prompt_switch": "text re-encoded, cross-attention cache reset at the next block; self-attention KV "
                                 "cache kept (the released model has no prompt-switch mechanism)",
                "prompt_policy": t2v_stream.PROMPT_POLICY}

    # ---- rollout ---------------------------------------------------------------------------------------
    def _reset(self):
        torch, p = self.torch, self.pipe
        p._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device="cuda")
        p._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device="cuda")
        p.vae.model.clear_cache()

    def _block(self, cond, start, gen):
        """One block starting at latent frame `start`: returns uint8 frames [T, H, W, 3]."""
        torch, p = self.torch, self.pipe
        seq = p.frame_seq_length
        x = torch.randn([1, BLOCK, 16, H // 8, W // 8], device="cuda", dtype=torch.bfloat16, generator=gen)
        for i, t in enumerate(p.denoising_step_list):
            timestep = torch.ones([1, BLOCK], device="cuda", dtype=torch.int64) * t
            _, pred = p.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=timestep,
                                  kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache, current_start=start * seq)
            if i < len(p.denoising_step_list) - 1:
                nxt = p.denoising_step_list[i + 1] * torch.ones([BLOCK], device="cuda", dtype=torch.long)
                noise = torch.randn(pred.flatten(0, 1).shape, device="cuda", dtype=pred.dtype, generator=gen)
                x = p.scheduler.add_noise(pred.flatten(0, 1), noise, nxt).unflatten(0, pred.shape[:2])
        p.generator(noisy_image_or_video=pred, conditional_dict=cond,
                    timestep=torch.ones_like(timestep) * int(self.cfg.context_noise),
                    kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache, current_start=start * seq)
        video = p.vae.decode_to_pixel(pred, use_cache=True)[0]              # T C H W in [-1, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return (video.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()

    def _encode(self, prompt):
        for c in self.pipe.crossattn_cache:
            c["is_init"] = False
        return self.pipe.text_encoder(text_prompts=[prompt])

    def warmup(self):
        torch = self.torch
        t0 = time.time()
        with torch.no_grad():
            self._reset()
            gen = torch.Generator("cuda").manual_seed(0)
            cond = self._encode("warmup: a calm lake at sunrise")
            times = []
            for i in range(WINDOW // BLOCK + 2):                             # past the first cache roll
                t = time.time()
                frames = self._block(cond, i * BLOCK, gen)
                times.append(round(time.time() - t, 3))
        LOG.info("warmup %.1fs; block times %s; last block %d frames -> %.1f fps", time.time() - t0, times,
                 len(frames), len(frames) / times[-1])

    def run(self, s: t2v_stream.GenSession):
        torch = self.torch
        with torch.no_grad():                          # grad mode is per thread; this runs in the session thread
            self._reset()
            gen = torch.Generator("cuda").manual_seed(s.seed)
            cond, start = None, 0
            while start + BLOCK <= ROPE_POSITIONS:
                got = s.next_block()
                if got is None:
                    return
                prompt, changed, events = got
                if cond is None or changed:
                    cond = self._encode(prompt)
                s.applied(events, "text re-encoded, cross-attention cache reset")
                s.emit(list(self._block(cond, start, gen)))
                start += BLOCK
            raise RuntimeError(f"rollout reached the positional table ({ROPE_POSITIONS} latent frames)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18813)
    ap.add_argument("--weights", help="unused; weights are read from $SELF_FORCING_ROOT/weights (default $STREAMEVAL_MODEL_ROOT/self_forcing/weights)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t2v_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
