#!/usr/bin/env python3
"""LiveEdit (ECCV 2026; Wan2.1-T2V-1.3B causal V2V editor, 4-step DMD) self-hosted instruction-edit server,
docs/protocol.md.

Upstream: github.com/cp-cp/LiveEdit (pinned in setup.sh), checkpoint `ar-forcing_002000.pt` (cp-cp/LiveEdit).
Upstream inference (`inference-mm.py --task v2v`, the long-video variant `infer-local-ar-forcing-long.sh`) is an
offline script, but its loop is block-causal: the source video is VAE-encoded, and each block of 3 latent frames
is denoised with the source latents of the same block concatenated to the noisy latents (32-channel patch
embedding), then written into the KV cache with a clean-context pass. This server runs that loop live:

  * every block takes the next source frames from the live stream (9 frames for the first block of a session,
    then 12 = 3 latents), encodes them with the causal Wan2.1 VAE encoder whose temporal cache is kept across
    blocks (exactly the 1, 4, 4, ... chunking of upstream `WanVAE_.encode` over the whole video), runs the
    4 denoising steps + the context pass, and decodes the block with the cached causal decoder;
  * long sessions use upstream's long-video settings: local attention 9 latent frames with 3 persistent sink
    frames and window-relative RoPE (`--local_attn_size 9 --sink_size 3 --window_rope`);
  * pixel range as upstream: source frames in [0, 1] into the VAE (ToTensor, no normalisation), output clamped
    to [0, 1] (`wo_scale=True`); upstream resizes to 480x832, here the frame is centre-cropped to the output
    aspect first (edit_stream) and portrait sources use 480x832 portrait;
  * denoising steps: upstream inference uses the configured list [1000, 750, 500, 250] without the timestep
    warp (the warp is commented out in pipeline/causal_inference.py); kept as is (`--warp` switches it on).

Instructions: the instruction text is the prompt verbatim. The checkpoint edits one instruction per video, so
an instruction change starts a new causal session (text re-encoded; KV, cross-attention and VAE caches reset)
on the next live frames, the same policy as selfhost/joyai_video_edit and selfhost/streamdiffusionv2. The
source is passed through until the first instruction. Streaming, pacing and acks: models/selfhost/edit_stream.py.

    python server.py --port 8768
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edit_stream  # noqa: E402
from t2v_stream import sha256_files  # noqa: E402  (follows symlinked shared weights)

ROOT = Path(os.environ.get("LIVEEDIT_ROOT", os.path.expanduser(os.environ.get("STREAM_ARENA_MODEL_ROOT", "~/.cache/stream_arena/models")) + "/liveedit"))
LOG = logging.getLogger("liveedit")
BLOCK = 3                    # latent frames per block (upstream num_frame_per_block)
FIRST_FRAMES = 1 + 4 * (BLOCK - 1)
NEXT_FRAMES = 4 * BLOCK
CKPT = "checkpoints/liveedit/ar-forcing_002000.pt"


class Engine:
    name = "LiveEdit"

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path.insert(0, str(repo))
        os.chdir(repo)                                 # upstream resolves wan_models/ relative to cwd
        import torch
        from omegaconf import OmegaConf
        from pipeline.causal_inference import CausalInferencePipeline

        self.torch, self.args = torch, args
        torch.set_grad_enabled(False)
        self.fps = args.model_fps
        cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                              OmegaConf.load("configs/wan_mm-ar-forcing-local.yaml"))
        t0 = time.time()
        pipe = CausalInferencePipeline(cfg, device=torch.device("cuda"), local_attn_size=args.local_attn_size,
                                       sink_size=args.sink_size, window_rope=True, expand_patch_embedding=True)
        state = torch.load(repo / CKPT, map_location="cpu")
        sd = state.get("generator", state.get("generator_ema", state))          # upstream: generator unless --use_ema
        sd = {("model." + k.split("._fsdp_wrapped_module.")[-1]) if "_fsdp_wrapped_module" in k else k: v
              for k, v in sd.items()}
        pipe.generator.load_state_dict(sd)
        del state, sd
        pipe = pipe.to(dtype=torch.bfloat16)
        pipe.text_encoder.to("cuda")
        pipe.generator.to("cuda")
        pipe.vae.to("cuda")
        self.pipe, self.cfg = pipe, cfg
        steps = torch.tensor(list(cfg.denoising_step_list), dtype=torch.long)
        if args.warp:
            ts = torch.cat((pipe.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            steps = ts[1000 - steps].long()
        pipe.denoising_step_list = steps
        self.steps = [int(t) for t in steps]
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"liveedit@{self.commit[:12]} ar-forcing_002000.pt"
        wan = ROOT / "weights/wan_models/Wan2.1-T2V-1.3B"
        self.weights_sha256 = sha256_files(
            [ROOT / "weights" / CKPT, wan / "models_t5_umt5-xxl-enc-bf16.pth", wan / "Wan2.1_VAE.pth"],
            ROOT / "weights")
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s; steps %s", self.version, time.time() - t0, self.gpu, self.steps)
        self.warmup()

    def out_size(self, init):
        return (480, 832) if int(init.get("height") or 0) > int(init.get("width") or 1) else (832, 480)

    # ---- causal pieces ---------------------------------------------------------------------------------
    def _reset(self):
        torch, p = self.torch, self.pipe
        p._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device="cuda")
        p._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device="cuda")
        p.vae.model.clear_cache()                      # decoder and encoder temporal caches

    def _encode(self, frames, first):
        """Source frames (float32 HxWx3 in [-1, 1]) -> latents [1, BLOCK, 16, h, w], continuing the causal encoder
        cache (upstream WanVAE_.encode chunks the video 1, 4, 4, ...; same chunks, same cache)."""
        torch, vae = self.torch, self.pipe.vae
        x = torch.from_numpy(np.stack(frames)).to("cuda", torch.bfloat16)          # T H W 3
        x = ((x + 1.0) * 0.5).permute(3, 0, 1, 2).unsqueeze(0)                    # 1 3 T H W in [0, 1]
        m = vae.model
        chunks = [x[:, :, :1]] + [x[:, :, i:i + 4] for i in range(1, x.shape[2], 4)] if first else \
            [x[:, :, i:i + 4] for i in range(0, x.shape[2], 4)]
        outs = []
        for c in chunks:
            m._enc_conv_idx = [0]
            outs.append(m.encoder(c, feat_cache=m._enc_feat_map, feat_idx=m._enc_conv_idx))
        mu = m.conv1(torch.cat(outs, 2)).chunk(2, dim=1)[0]
        mean, std = vae.mean.to("cuda", torch.bfloat16), vae.std.to("cuda", torch.bfloat16)
        mu = (mu - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1)
        return mu.permute(0, 2, 1, 3, 4).to(torch.bfloat16)                       # 1 T C h w

    def _block(self, cond, y, start, gen):
        torch, p = self.torch, self.pipe
        seq = p.frame_seq_length
        _, T, C, h, w = y.shape
        x = torch.randn([1, T, 16, h, w], device="cuda", dtype=torch.bfloat16, generator=gen)
        steps = p.denoising_step_list
        for i, t in enumerate(steps):
            timestep = torch.ones([1, T], device="cuda", dtype=torch.int64) * t
            _, pred = p.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=timestep,
                                  kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache,
                                  current_start=start * seq, y=y)
            if i < len(steps) - 1:
                nxt = steps[i + 1] * torch.ones([T], device="cuda", dtype=torch.long)
                noise = torch.randn(pred.flatten(0, 1).shape, device="cuda", dtype=pred.dtype, generator=gen)
                x = p.scheduler.add_noise(pred.flatten(0, 1), noise, nxt).unflatten(0, pred.shape[:2])
        p.generator(noisy_image_or_video=pred, conditional_dict=cond,
                    timestep=torch.ones_like(timestep) * int(self.cfg.context_noise),
                    kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache, current_start=start * seq, y=y)
        video = p.vae.decode_to_pixel(pred, use_cache=True)[0].clamp(0, 1)        # T C H W (wo_scale)
        return (video.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()

    def _text(self, prompt):
        for c in self.pipe.crossattn_cache:
            c["is_init"] = False
        return self.pipe.text_encoder(text_prompts=[prompt])

    def warmup(self):
        torch = self.torch
        t0 = time.time()
        times = []
        with torch.no_grad():
            for w, h in ((832, 480), (480, 832)):
                self._reset()
                gen = torch.Generator("cuda").manual_seed(0)
                cond = self._text("Make it a watercolor painting.")
                start = 0
                for i in range(6):
                    n = FIRST_FRAMES if i == 0 else NEXT_FRAMES
                    rnd = [np.random.uniform(-1, 1, (h, w, 3)).astype(np.float32) for _ in range(n)]
                    t = time.time()
                    self._block(cond, self._encode(rnd, i == 0), start, gen)
                    times.append(round(time.time() - t, 3))
                    start += BLOCK
        LOG.info("warmup %.1fs; block times %s (12 frames each after the first)", time.time() - t0, times)

    def system_config(self):
        a = self.args
        return {"model": self.name, "model_variant": "LiveEdit ar-forcing_002000.pt (stage-3 DMD, Wan2.1-T2V-1.3B, "
                                                      "32-channel source-concat patch embedding)",
                "version": self.version, "repo": "https://github.com/cp-cp/LiveEdit", "repo_commit": self.commit,
                "license": "Apache-2.0", "weights_sha256": self.weights_sha256,
                "resolution": {"landscape": [832, 480], "portrait": [480, 832]}, "model_fps": a.model_fps,
                "denoising_steps": len(self.steps), "denoising_step_list": self.steps,
                "timestep_warp": bool(a.warp), "block_latents": BLOCK,
                "block_frames": {"first": FIRST_FRAMES, "next": NEXT_FRAMES},
                "local_attn_latents": a.local_attn_size, "sink_latents": a.sink_size, "window_rope": True,
                "context_noise": int(self.cfg.context_noise), "mask_cache": False,
                "vae": "Wan2.1 VAE: causal encoder and decoder caches kept across blocks",
                "pixel_range": "[0, 1] in and out (upstream ToTensor / wo_scale)",
                "attention": "upstream (flex attention / torch SDPA; no flash-attn)",
                "gpu": self.gpu, "gpu_count": 1, "dtype": "bfloat16",
                "prompt_policy": "instruction text verbatim as the prompt; an instruction change starts a new causal "
                                 "session (KV, cross-attention and VAE caches reset) on the next live frames",
                "input_policy": f"center crop to output aspect, resize, subsample to {a.model_fps} fps; newest frames "
                                "win when behind"}

    def run(self, s: edit_stream.EditSession):
        with self.torch.no_grad():                     # grad mode is per thread; this runs in the session thread
            self._run(s)

    def _run(self, s: edit_stream.EditSession):
        torch = self.torch
        gen = torch.Generator("cuda").manual_seed(s.seed)
        active, start, cond = None, 0, None
        while True:
            got = s.take(lambda p: NEXT_FRAMES if p == active else FIRST_FRAMES)
            if got is None:
                break
            frames, prompt, event = got
            t0 = time.perf_counter()
            restart = prompt != active
            if restart:
                self._reset()
                cond, active, start = self._text(prompt), prompt, 0
            out = self._block(cond, self._encode(frames, restart), start, gen)
            start += BLOCK
            s.emit(list(out))
            ms = (time.perf_counter() - t0) * 1e3
            s.stats["chunks"] += 1
            s.stats["chunk_ms"].append(ms)
            s.applied(event, f"prompt = instruction text verbatim; causal session restarted ({ms:.0f} ms)"
                      if restart else "identical to the active prompt")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--weights", help="unused; weights are linked into the upstream repo (setup.sh)")
    ap.add_argument("--model_fps", type=int, default=16, help="upstream writes its outputs at 16 fps")
    ap.add_argument("--local_attn_size", type=int, default=9, help="upstream long-video script default")
    ap.add_argument("--sink_size", type=int, default=3, help="upstream long-video script default")
    ap.add_argument("--warp", action="store_true", help="warp the step list as in training (upstream inference does not)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    edit_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
