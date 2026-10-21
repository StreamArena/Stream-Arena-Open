#!/usr/bin/env python3
"""Reward Forcing (Wan2.1-T2V-1.3B, chunk-wise autoregressive, 4-step Re-DMD, EMA sink) self-hosted server,
docs/protocol.md.

One WebSocket session = one live autoregressive rollout of upstream `CausalInferencePipeline.inference`
(JaydenLyh/Reward-Forcing, `configs/reward_forcing.yaml`, `rewardforcing.pt`), unrolled so that each 3-latent block
(12 frames at 16 fps; the first block 9) is generated, decoded and sent before the next one starts: 4 denoising steps
(1000/750/500/250, warped, shift 5.0, no CFG) -> clean-context pass into the KV cache -> causal Wan2.1 VAE decode of
just these latents (`decode_to_pixel(use_cache=True)`, decoder cache kept across blocks).

Long sessions use the model's own mechanism unchanged: upstream config `local_attn_size = 9` latent frames with
`sink_size = 3` — the KV cache keeps a 3-latent attention sink that is updated as an exponential moving average of the
evicted frames (EMA sink, alpha 0.999), and RoPE is applied relative to the attention window, so the rollout has no
positional limit (upstream demonstrates 30 s; sessions here run up to 300 s).

Prompt switch: the released model has no interactive mechanism. The block prompt is re-encoded and the cross-attention
cache reset at the next block; the self-attention KV cache (sink included) keeps the past as generated.

Text-to-video only (chunk-wise T2V checkpoint): an opening image is refused, so the image + text track is not
registered. Streaming, pacing, prompt policy and acks: selfhost/t2v_stream.py.

    python server.py --port 18862
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

ROOT = Path(os.environ.get("REWARD_FORCING_ROOT",
                           os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/reward_forcing"))
LOG = logging.getLogger("reward_forcing")
W, H, FPS = 832, 480, 16
BLOCK = 3            # latent frames per block (upstream num_frame_per_block)
CKPT = "Reward-Forcing-T2V-1.3B/rewardforcing.pt"


class Engine:
    name = "Reward Forcing"
    width, height, fps = W, H, FPS

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path.insert(0, str(repo))
        os.chdir(repo)
        import torch
        from omegaconf import OmegaConf
        from pipeline import CausalInferencePipeline

        self.torch, self.args = torch, args
        torch.set_grad_enabled(False)
        cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                              OmegaConf.load("configs/reward_forcing.yaml"))
        cfg.checkpoint_path = str(ROOT / "weights")                 # holds Wan2.1-T2V-1.3B/ (upstream checkpoints/)
        t0 = time.time()
        pipe = CausalInferencePipeline(cfg, device=torch.device("cuda"))
        ckpt = ROOT / "weights" / CKPT
        state = torch.load(ckpt, map_location="cpu")
        # upstream inference.py loads the file as the generator state dict; accept the Self-Forcing layout too
        key = next((k for k in ("generator_ema", "generator") if isinstance(state, dict) and k in state), None)
        self.state_key = key or "(whole file)"
        pipe.generator.load_state_dict(state[key] if key else state)
        del state
        pipe = pipe.to(dtype=torch.bfloat16)
        pipe.text_encoder.to("cuda")
        pipe.generator.to("cuda")
        pipe.vae.to("cuda")
        self.pipe, self.cfg = pipe, cfg
        self.window = int(cfg.model_kwargs.local_attn_size)
        self.sink = int(cfg.model_kwargs.sink_size)
        self.steps = [float(t) for t in pipe.denoising_step_list]
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"reward-forcing@{self.commit[:12]} rewardforcing.pt"
        wan = ROOT / "weights/Wan2.1-T2V-1.3B"
        self.weights_sha256 = t2v_stream.sha256_files(
            [ckpt, wan / "models_t5_umt5-xxl-enc-bf16.pth", wan / "Wan2.1_VAE.pth"], ROOT / "weights")
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s (state %s) in %.1fs on %s", self.version, self.state_key, time.time() - t0, self.gpu)
        self.warmup()

    def system_config(self):
        return {"model": self.name,
                "model_variant": "Reward Forcing T2V 1.3B (rewarded DMD, EMA sink) on Wan2.1-T2V-1.3B, rewardforcing.pt",
                "version": self.version, "repo": "https://github.com/JaydenLyh/Reward-Forcing",
                "repo_commit": self.commit, "license": "Apache-2.0", "weights_sha256": self.weights_sha256,
                "checkpoint_state": self.state_key,
                "width": W, "height": H, "fps": FPS, "denoising_steps": len(self.steps),
                "denoising_step_list": [round(t, 1) for t in self.steps], "timestep_shift": 5.0,
                "block_latents": BLOCK, "block_frames": BLOCK * 4, "local_attn_latents": self.window,
                "sink_latents": self.sink, "sink": "EMA of evicted frames (upstream, alpha 0.999)",
                "rope": "relative to the attention window (upstream)",
                "context_noise": int(self.cfg.context_noise),
                "vae": "Wan2.1 VAE, causal cached decode per block", "dtype": "bfloat16", "attention": "torch SDPA",
                "gpu": self.gpu, "gpu_count": 1,
                "prompt_switch": "text re-encoded, cross-attention cache reset at the next block; self-attention KV "
                                 "cache and sink kept (the released model has no prompt-switch mechanism)",
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
            for i in range(self.window // BLOCK + 3):                        # past the first sink update
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
            while True:
                got = s.next_block()
                if got is None:
                    return
                prompt, changed, events = got
                if cond is None or changed:
                    cond = self._encode(prompt)
                s.applied(events, "text re-encoded, cross-attention cache reset")
                s.emit(list(self._block(cond, start, gen)))
                start += BLOCK


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18862)
    ap.add_argument("--weights", help="unused; weights are read from $REWARD_FORCING_ROOT/weights (default $STREAMEVAL_MODEL_ROOT/reward_forcing/weights)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t2v_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
