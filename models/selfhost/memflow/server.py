#!/usr/bin/env python3
"""MemFlow (HKU / Kling; Wan2.1-T2V-1.3B causal, LongLive-style, 4-step, memory bank) self-hosted server,
docs/protocol.md.

One WebSocket session = one live rollout of upstream `InteractiveCausalInferencePipeline.inference`
(KlingAIResearch/MemFlow, configs/interactive_inference.yaml), unrolled so that each 3-latent block (12 frames at
16 fps; the first block 9) is generated, decoded and sent before the next one starts:

  * per block: 4 denoising steps (1000/750/500/250, warped, shift 5.0) with the memory bank queried on the first
    step, then the clean-context pass that writes the KV cache and, every 3 x record_interval latent frames,
    updates the memory bank (upstream `update_bank` rule); causal Wan2.1 VAE decode of just these latents with
    the decoder cache kept across blocks (upstream decodes the whole video at the end; the VAE is causal, so the
    frames are the same);
  * upstream settings: local attention 12 latent frames, sink 3 (global sink), bank size 3, record interval 3,
    sparse memory activation off (as released), base.pt `generator` + lora.pt (rank 256), context noise 0;
  * prompt switch = upstream's interactive switch: when the block prompt changes, the cross-attention cache is
    reset and the KV cache is re-computed from the last <= 12 generated latent frames under the new prompt
    (`_recache_after_switch`, global sink kept), then generation continues;
  * the temporal RoPE table (upstream 1024 positions = 256 s at 16 fps) is recomputed with the same formula for
    4096 positions (rows below 1024 identical), so 300-second sessions do not run off the table.

Streaming, pacing, prompt policy and acks: selfhost/t2v_stream.py.

    python server.py --port 18814
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

ROOT = Path(os.environ.get("MEMFLOW_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/memflow"))
LOG = logging.getLogger("memflow")
W, H, FPS = 832, 480, 16
ROPE_POSITIONS = 4096


class Engine:
    name = "MemFlow"
    width, height, fps = W, H, FPS

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path.insert(0, str(repo))
        os.chdir(repo)                                 # upstream resolves wan_models/ relative to cwd
        import torch
        import peft
        from omegaconf import OmegaConf
        from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
        from utils.lora_utils import configure_lora_for_model
        from wan.modules.model import rope_params

        self.torch, self.args = torch, args
        torch.set_grad_enabled(False)
        cfg = OmegaConf.load("configs/interactive_inference.yaml")
        t0 = time.time()
        pipe = InteractiveCausalInferencePipeline(cfg, device=torch.device("cuda"))
        base = pipe.generator.model
        d = base.dim // base.num_heads
        base.freqs = torch.cat([rope_params(ROPE_POSITIONS, d - 4 * (d // 6)),       # rows < 1024 unchanged
                                rope_params(ROPE_POSITIONS, 2 * (d // 6)), rope_params(ROPE_POSITIONS, 2 * (d // 6))],
                               dim=1)
        ckpt, lora = ROOT / "weights/MemFlow/base.pt", ROOT / "weights/MemFlow/lora.pt"
        state = torch.load(ckpt, map_location="cpu")
        pipe.generator.load_state_dict(state["generator"])                            # use_ema: false
        del state
        pipe.generator.model = configure_lora_for_model(pipe.generator.model, model_name="generator",
                                                        lora_config=cfg.adapter, is_main_process=True)
        lst = torch.load(lora, map_location="cpu")
        peft.set_peft_model_state_dict(pipe.generator.model, lst["generator_lora"] if "generator_lora" in lst else lst)
        del lst
        pipe.is_lora_enabled = True
        pipe = pipe.to(dtype=torch.bfloat16)
        pipe.text_encoder.to("cuda")
        pipe.generator.to("cuda")
        pipe.vae.to("cuda")
        self.pipe, self.cfg = pipe, cfg
        self.block = int(cfg.num_frame_per_block)
        self.window = int(cfg.model_kwargs.local_attn_size)
        self.steps = [float(t) for t in pipe.denoising_step_list]
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"memflow@{self.commit[:12]} base.pt + lora.pt"
        wan = ROOT / "weights/wan_models/Wan2.1-T2V-1.3B"
        self.weights_sha256 = t2v_stream.sha256_files(
            [ckpt, lora, wan / "models_t5_umt5-xxl-enc-bf16.pth", wan / "Wan2.1_VAE.pth"], ROOT / "weights")
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s", self.version, time.time() - t0, self.gpu)
        self.warmup()

    def system_config(self):
        mk = self.cfg.model_kwargs
        return {"model": self.name, "model_variant": "MemFlow base.pt (generator) + lora.pt (rank 256) on Wan2.1-T2V-1.3B",
                "version": self.version, "repo": "https://github.com/KlingAIResearch/MemFlow",
                "repo_commit": self.commit,
                "license": "Apache-2.0 (repository LICENSE); source files derived from LongLive carry a CC BY-NC-SA "
                           "4.0 header; weights KlingTeam/MemFlow (model card states no licence)",
                "weights_sha256": self.weights_sha256,
                "width": W, "height": H, "fps": FPS, "denoising_steps": len(self.steps),
                "denoising_step_list": [round(t, 1) for t in self.steps], "timestep_shift": float(mk.timestep_shift),
                "block_latents": self.block, "block_frames": self.block * 4, "local_attn_latents": self.window,
                "sink_latents": int(mk.sink_size), "global_sink": bool(self.cfg.global_sink),
                "bank_size": int(mk.bank_size), "record_interval": int(mk.record_interval),
                "sparse_memory_activation": bool(mk.SMA), "context_noise": int(self.cfg.context_noise),
                "rope_positions": ROPE_POSITIONS, "vae": "Wan2.1 VAE, causal cached decode per block",
                "dtype": "bfloat16", "attention": "upstream (flex attention / torch SDPA; no flash-attn)",
                "gpu": self.gpu, "gpu_count": 1,
                "prompt_switch": "upstream interactive switch: cross-attention cache reset + KV re-cache of the last "
                                 f"<= {self.window} latent frames under the new prompt (global sink kept)",
                "prompt_policy": t2v_stream.PROMPT_POLICY}

    # ---- rollout ---------------------------------------------------------------------------------------
    def _reset(self):
        torch, p = self.torch, self.pipe
        p._initialize_kv_cache(1, dtype=torch.bfloat16, device="cuda",
                               kv_cache_size_override=self.window * p.frame_seq_length)
        p._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device="cuda")
        p._initialize_kv_bank(batch_size=1, dtype=torch.bfloat16, device="cuda",
                              kv_bank1_size=p.bank_size * p.frame_seq_length)
        p.generator.model.local_attn_size = p.local_attn_size
        p._set_all_modules_max_attention_size(p.local_attn_size)
        p.vae.model.clear_cache()
        self.recent = []                               # last <= window clean latents [(start, tensor 1xBxCxhxw)]

    def _recache(self, cond, start):
        """Upstream `_recache_after_switch` on the rolling buffer of generated latents."""
        torch, p = self.torch, self.pipe
        for blk in p.crossattn_cache:
            blk["k"].zero_(); blk["v"].zero_(); blk["is_init"] = False
        if start == 0 or not self.recent:
            return
        frames = torch.cat([t for _, t in self.recent], dim=1)[:, -min(self.window, start):]
        n = frames.shape[1]
        rs = start - n
        p.generator.model.block_mask = p.generator.model._prepare_blockwise_causal_attn_mask(
            device=frames.device, num_frames=n, frame_seqlen=p.frame_seq_length,
            num_frame_per_block=p.num_frame_per_block, local_attn_size=p.local_attn_size)
        ctx = torch.ones([1, n], device="cuda", dtype=torch.int64) * int(self.cfg.context_noise)
        p.generator(noisy_image_or_video=frames, conditional_dict=cond, timestep=ctx, kv_cache=p.kv_cache1,
                    kv_bank=p.kv_bank1, crossattn_cache=p.crossattn_cache, current_start=rs * p.frame_seq_length,
                    update_bank=False, update_cache=True, q_bank=True, is_recache=True)
        for blk in p.crossattn_cache:
            blk["k"].zero_(); blk["v"].zero_(); blk["is_init"] = False

    def _block(self, cond, start, gen):
        torch, p, B = self.torch, self.pipe, self.block
        seq = p.frame_seq_length
        x = torch.randn([1, B, 16, H // 8, W // 8], device="cuda", dtype=torch.bfloat16, generator=gen)
        update_bank = start % (3 * p.record_interval) == 0
        for i, t in enumerate(p.denoising_step_list):
            timestep = torch.ones([1, B], device="cuda", dtype=torch.int64) * t
            _, pred = p.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=timestep,
                                  kv_cache=p.kv_cache1, kv_bank=p.kv_bank1, crossattn_cache=p.crossattn_cache,
                                  current_start=start * seq, update_bank=False, q_bank=(i == 0))
            if i < len(p.denoising_step_list) - 1:
                nxt = p.denoising_step_list[i + 1] * torch.ones([B], device="cuda", dtype=torch.long)
                noise = torch.randn(pred.flatten(0, 1).shape, device="cuda", dtype=pred.dtype, generator=gen)
                x = p.scheduler.add_noise(pred.flatten(0, 1), noise, nxt).unflatten(0, pred.shape[:2])
        p.generator(noisy_image_or_video=pred, conditional_dict=cond,
                    timestep=torch.ones_like(timestep) * int(self.cfg.context_noise),
                    kv_cache=p.kv_cache1, kv_bank=p.kv_bank1, crossattn_cache=p.crossattn_cache,
                    current_start=start * seq, update_bank=update_bank, q_bank=False, update_cache=True)
        self.recent.append((start, pred))
        while sum(t.shape[1] for _, t in self.recent) - self.recent[0][1].shape[1] >= self.window:
            self.recent.pop(0)
        video = p.vae.decode_to_pixel(pred, use_cache=True)[0]              # T C H W in [-1, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return (video.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()

    def _encode(self, prompt):
        return self.pipe.text_encoder(text_prompts=[prompt])

    def warmup(self):
        torch = self.torch
        t0 = time.time()
        with torch.no_grad():
            self._reset()
            gen = torch.Generator("cuda").manual_seed(0)
            cond = self._encode("warmup: a calm lake at sunrise")
            times = []
            for i in range(8):
                if i == 5:                                                  # exercise the switch path
                    cond = self._encode("warmup: a boat crosses the lake")
                    self._recache(cond, i * self.block)
                t = time.time()
                frames = self._block(cond, i * self.block, gen)
                times.append(round(time.time() - t, 3))
        LOG.info("warmup %.1fs; block times %s; last block %d frames -> %.1f fps", time.time() - t0, times,
                 len(frames), len(frames) / times[-1])

    def run(self, s: t2v_stream.GenSession):
        torch = self.torch
        with torch.no_grad():                          # grad mode is per thread; this runs in the session thread
            self._reset()
            gen = torch.Generator("cuda").manual_seed(s.seed)
            cond, start = None, 0
            while start + self.block <= ROPE_POSITIONS:
                got = s.next_block()
                if got is None:
                    return
                prompt, changed, events = got
                if cond is None:
                    cond = self._encode(prompt)
                elif changed:
                    cond = self._encode(prompt)
                    self._recache(cond, start)
                s.applied(events, "prompt switch: cross-attention reset + KV re-cache of the recent frames")
                s.emit(list(self._block(cond, start, gen)))
                start += self.block
            raise RuntimeError(f"rollout reached the positional table ({ROPE_POSITIONS} latent frames)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18814)
    ap.add_argument("--weights", help="unused; weights are read from $MEMFLOW_ROOT/weights (default $STREAMEVAL_MODEL_ROOT/memflow/weights)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t2v_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
