#!/usr/bin/env python3
"""Context Forcing (Wan2.1-T2V-1.3B, chunk-wise autoregressive, 4-step contextual DMD, long-context KV memory)
self-hosted server, docs/protocol.md.

One WebSocket session = one live autoregressive rollout of upstream `CausalLongInferencePipeline.inference`
(TIGER-AI-Lab/Context-Forcing, `configs/context_dmd_inference.yaml`, `model.pt` with `--use_ema` as in upstream
`inference_continue.sh`), unrolled so that each 3-latent block (12 frames at 16 fps; the first block 9) is generated,
decoded and sent before the next one starts: 4 denoising steps (1000/750/500/250, warped, shift 5.0, no CFG) ->
clean-context pass into the KV cache -> causal Wan2.1 VAE decode of just these latents (`decode_to_pixel(use_cache=True)`,
decoder cache kept across blocks; upstream decodes the whole video at the end).

Long sessions use the model's own context-management system unchanged (upstream config): a 21-latent KV cache =
3-latent attention sink + slow memory (context latents selected by key similarity, `keyframe_sample: sim`, one new
context latent per selection) + 6-latent fast local window, with RoPE positions anchored inside the 21-latent range,
so the rollout has no positional limit (upstream demonstrates 252 latents ~ 1 min; sessions here run up to 300 s).

Prompt switch: the released model has no interactive mechanism. The block prompt is re-encoded and the cross-attention
cache reset at the next block; the self-attention KV cache (sink, slow and fast memory) keeps the past as generated.

Text-to-video only (the released checkpoint and upstream inference are T2V): an opening image is refused, so the
image + text track is not registered. Streaming, pacing, prompt policy and acks: selfhost/t2v_stream.py.

    python server.py --port 18874
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import logging
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import t2v_stream  # noqa: E402

ROOT = Path(os.environ.get("CONTEXT_FORCING_ROOT",
                           os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/context_forcing"))
LOG = logging.getLogger("context_forcing")
W, H, FPS = 832, 480, 16
BLOCK = 3            # latent frames per block (upstream num_frame_per_block)
CKPT = "checkpoints/model.pt"


class Engine:
    name = "Context Forcing"
    width, height, fps = W, H, FPS

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path.insert(0, str(repo))
        os.chdir(repo)                                 # upstream resolves wan_models/ relative to cwd
        import torch
        from omegaconf import OmegaConf
        from pipeline import CausalLongInferencePipeline

        self.torch, self.args = torch, args
        torch.set_grad_enabled(False)
        cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                              OmegaConf.load("configs/context_dmd_inference.yaml"))
        t0 = time.time()
        # upstream inference.py: the config has `long`, so the long pipeline is used
        pipe = CausalLongInferencePipeline(cfg, device=torch.device("cuda"))
        ckpt = ROOT / "weights" / CKPT
        raw = torch.load(ckpt, map_location="cpu")["generator_ema"]      # upstream --use_ema
        for prefix in ("model._fsdp_wrapped_module.", "model."):         # upstream key cleaning
            if all(k.startswith(prefix) for k in raw):
                raw = OrderedDict((k[len(prefix):], v) for k, v in raw.items())
                break
        res = pipe.generator.model.load_state_dict(raw, strict=False)    # upstream strict=False
        self.load_report = {"missing": len(res.missing_keys), "unexpected": len(res.unexpected_keys)}
        del raw
        pipe = pipe.to(dtype=torch.bfloat16)
        pipe.text_encoder.to("cuda")
        pipe.generator.to("cuda")
        pipe.vae.to("cuda")
        self.pipe, self.cfg = pipe, cfg
        mk = cfg.model_kwargs
        self.memory = {"cache_latents": int(mk.cache_size), "sink_latents": int(mk.sink_size),
                       "local_latents": int(mk.local_attn_size), "new_context_latents": int(mk.num_new_context_tokens),
                       "rope_length": int(mk.rope_length), "keyframe_sample": str(mk.keyframe_sample)}
        self.steps = [float(t) for t in pipe.denoising_step_list]
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"context-forcing@{self.commit[:12]} model.pt (generator_ema)"
        wan = ROOT / "weights/wan_models/Wan2.1-T2V-1.3B"
        self.weights_sha256 = t2v_stream.sha256_files(
            [ckpt, wan / "models_t5_umt5-xxl-enc-bf16.pth", wan / "Wan2.1_VAE.pth"], ROOT / "weights")
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s (keys %s) in %.1fs on %s", self.version, self.load_report, time.time() - t0, self.gpu)
        self.warmup()

    def system_config(self):
        return {"model": self.name,
                "model_variant": "Context Forcing T2V 1.3B (contextual DMD from a long-context teacher, sink + slow / "
                                 "fast KV memory) on Wan2.1-T2V-1.3B, model.pt (generator_ema)",
                "version": self.version, "repo": "https://github.com/TIGER-AI-Lab/Context-Forcing",
                "repo_commit": self.commit, "license": "MIT", "weights_sha256": self.weights_sha256,
                "checkpoint_load": {"state": "generator_ema", "strict": False, **self.load_report},
                "width": W, "height": H, "fps": FPS, "denoising_steps": len(self.steps),
                "denoising_step_list": [round(t, 1) for t in self.steps], "timestep_shift": 5.0,
                "block_latents": BLOCK, "block_frames": BLOCK * 4, "context_memory": self.memory,
                "context_noise": int(self.cfg.context_noise),
                "vae": "Wan2.1 VAE, causal cached decode per block", "dtype": "bfloat16", "attention": "torch SDPA",
                "gpu": self.gpu, "gpu_count": 1,
                "long_rollout": "upstream context management unchanged: 21-latent KV cache = 3-latent sink + "
                                "similarity-selected slow memory + 6-latent fast window, RoPE anchored within 21 latents",
                "prompt_switch": "text re-encoded, cross-attention cache reset at the next block; self-attention KV "
                                 "cache (sink, slow and fast memory) kept (the released model has no prompt-switch "
                                 "mechanism)",
                "prompt_policy": t2v_stream.PROMPT_POLICY}

    # ---- rollout ---------------------------------------------------------------------------------------
    def _reset(self):
        torch, p = self.torch, self.pipe
        p._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device="cuda")       # fresh memory state
        p._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device="cuda")
        p.vae.model.clear_cache()

    def _block(self, cond, start, gen):
        """One block starting at latent frame `start`: returns uint8 frames [T, H, W, 3]."""
        torch, p = self.torch, self.pipe
        seq = p.frame_seq_length
        x = torch.randn([1, BLOCK, 16, H // 8, W // 8], device="cuda", dtype=torch.bfloat16, generator=gen)
        for i, t in enumerate(p.denoising_step_list):
            timestep = torch.ones([1, BLOCK], device="cuda", dtype=torch.int64) * t
            _, pred, p.kv_cache1 = p.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=timestep,
                                               kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                                               current_start=start * seq)
            if i < len(p.denoising_step_list) - 1:
                nxt = p.denoising_step_list[i + 1] * torch.ones([BLOCK], device="cuda", dtype=torch.long)
                noise = torch.randn(pred.flatten(0, 1).shape, device="cuda", dtype=pred.dtype, generator=gen)
                x = p.scheduler.add_noise(pred.flatten(0, 1), noise, nxt).unflatten(0, pred.shape[:2])
        _, _, p.kv_cache1 = p.generator(noisy_image_or_video=pred, conditional_dict=cond,
                                        timestep=torch.ones_like(timestep) * int(self.cfg.context_noise),
                                        kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                                        current_start=start * seq)
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
            for i in range(self.memory["cache_latents"] // BLOCK + 3):       # past the first memory selections
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
    ap.add_argument("--port", type=int, default=18874)
    ap.add_argument("--weights", help="unused; weights are read from $CONTEXT_FORCING_ROOT/weights (default $STREAMEVAL_MODEL_ROOT/context_forcing/weights)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t2v_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
