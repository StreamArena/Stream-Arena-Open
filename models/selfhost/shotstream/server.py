#!/usr/bin/env python3
"""ShotStream (Wan2.1-T2V-1.3B, causal multi-shot next-shot generation, 4-step DMD) self-hosted server,
docs/protocol.md.

One WebSocket session = one live multi-shot rollout of upstream `CausalInferenceArPipeline.inference`
(KlingAIResearch/ShotStream, `ckpts/shotstream.yaml` + `default_config.yaml`, `shotstream_merged.pt`), unrolled so that
each 3-latent block (12 frames at 16 fps; the first block of a shot 9) is generated, decoded and sent before the next
one starts. Per shot, as upstream:
  * condition: 6 context frames sampled (linspace) from the decoded frames of the earlier shots (`dynamic_sample_frames`,
    `max_context_frames: 6`, split evenly over the shots, the remainder to the latest), each VAE-encoded as a 1-frame
    clip and written into the context KV cache with timestep 0; zeros for the first shot. Shot-aware RoPE
    (`change_rope`) and per-shot captions (`multi_caption`): every latent attends to the caption of its own shot;
  * the shot: 21 latents (81 frames, 5.06 s) in 3-latent blocks, 4 steps (1000/740/500/260 warped, shift 8.0, no CFG),
    clean-context pass into the shot's KV cache after each block, global attention within the shot (upstream);
  * decode: causal Wan2.1 VAE, decoder cache kept across the blocks of a shot and reset at a new shot. This is
    numerically the same frame-by-frame causal decode upstream runs over the whole shot after it is complete.

Interaction = upstream's next-shot mechanism (the model's "interactive storytelling"):
  * caption of shot k (upstream format `global_caption + f"shot{k}:" + shot_caption`): global caption = the session's
    init prompt; shot caption = the latest instruction (user_prompt or camera_control text, verbatim), the init prompt
    for the shots before the first instruction. This replaces t2v_stream's '<init>\\n<instruction>' block prompt
    because the checkpoint was trained on exactly this global + per-shot caption format;
  * an instruction ends the current shot at the next block boundary and starts a new shot with the new caption
    (acked `applied` when that shot's first block starts);
  * a shot that reaches its trained length (21 latents) without an instruction is followed by a new shot with the
    same caption, so sessions longer than 5 s continue with upstream's own continuation mechanism;
  * history window: the context is drawn from the last 5 shots (re-indexed 0..4, the new shot 5): upstream's demos
    have at most 6 shots, so the shot index of RoPE and of the caption list stays inside that range in 300 s sessions.

Text-to-video only (no image input in the release): an opening image is refused, so the image + text track is not
registered. Streaming, pacing and acks: selfhost/t2v_stream.py.

    python server.py --port 18877
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
import types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import t2v_stream  # noqa: E402

ROOT = Path(os.environ.get("SHOTSTREAM_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/shotstream"))
LOG = logging.getLogger("shotstream")
W, H, FPS = 832, 480, 16
BLOCK = 3            # latent frames per block (upstream num_frame_per_block)
SHOT = 21            # latent frames per shot (upstream noise length, 81 frames)
HISTORY = 5          # earlier shots the context is drawn from
CKPT = "ShotStream/shotstream_merged.pt"


class Engine:
    name = "ShotStream"
    width, height, fps = W, H, FPS

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path.insert(0, str(HERE.parent / "_shims"))       # flash_attn API on torch SDPA
        sys.path.insert(0, str(repo))
        os.chdir(repo)
        sys.modules.setdefault("flask", types.SimpleNamespace(config=None))   # model/base.py imports it, unused
        import transformers.utils.import_utils  # noqa: F401  probes decord's spec before the stub below
        sys.modules.setdefault("decord", types.SimpleNamespace())             # utils/dataset.py video reading, unused
        import torch
        from omegaconf import OmegaConf
        from pipeline.causal_inference_ar import CausalInferenceArPipeline

        self.torch, self.args = torch, args
        torch.set_grad_enabled(False)
        cfg = OmegaConf.merge(OmegaConf.load("ckpts/default_config.yaml"), OmegaConf.load("ckpts/shotstream.yaml"))
        cfg.multi_caption = True
        cfg.use_wo_rope_cache = False                          # upstream Inference_Causal.py defaults
        t0 = time.time()
        pipe = CausalInferenceArPipeline(cfg, device=0)
        ckpt = ROOT / "weights" / CKPT
        state = torch.load(ckpt, map_location="cpu")
        pipe.generator.load_state_dict(state["generator"])
        del state
        pipe = pipe.to(dtype=torch.bfloat16)
        self.pipe, self.cfg = pipe, cfg
        self.steps = [float(t) for t in pipe.denoising_step_list]
        self.max_context = int(pipe.max_context_frames)
        assert pipe.dynamic_sample_frames and pipe.change_rope and pipe.num_frame_per_block == BLOCK
        assert pipe.local_attn_size == SHOT          # upstream inference: attention window = one shot (21 latents)
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"shotstream@{self.commit[:12]} shotstream_merged.pt"
        wan = ROOT / "weights/wan_models/Wan2.1-T2V-1.3B"
        self.weights_sha256 = t2v_stream.sha256_files(
            [ckpt, wan / "models_t5_umt5-xxl-enc-bf16.pth", wan / "Wan2.1_VAE.pth"], ROOT / "weights")
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s; steps %s", self.version, time.time() - t0, self.gpu, self.steps)
        self.warmup()

    def system_config(self):
        return {"model": self.name,
                "model_variant": "ShotStream (causal multi-shot, two-stage intra-/inter-shot self-forcing DMD) on "
                                 "Wan2.1-T2V-1.3B, shotstream_merged.pt",
                "version": self.version, "repo": "https://github.com/KlingAIResearch/ShotStream",
                "repo_commit": self.commit, "license": "Apache-2.0", "weights_sha256": self.weights_sha256,
                "width": W, "height": H, "fps": FPS, "denoising_steps": len(self.steps),
                "denoising_step_list": [round(t, 1) for t in self.steps], "timestep_shift": float(self.cfg.timestep_shift),
                "block_latents": BLOCK, "block_frames": BLOCK * 4, "shot_latents": SHOT, "shot_frames": 81,
                "context_frames": self.max_context, "history_shots": HISTORY,
                "context": "dynamic sample of 6 decoded frames over the earlier shots (upstream), VAE-encoded per frame",
                "attention": "global within the shot + context KV (upstream); torch SDPA through models/selfhost/_shims/flash_attn",
                "context_noise": int(self.cfg.context_noise),
                "vae": "Wan2.1 VAE, causal cached decode per block, reset per shot", "dtype": "bfloat16",
                "gpu": self.gpu, "gpu_count": 1,
                "prompt_switch": "an instruction starts a new shot at the next block (upstream next-shot generation); "
                                 "a shot that reaches 21 latents continues as a new shot with the same caption",
                "prompt_policy": "shot caption = '<init prompt>' + 'shot<k>:' + '<latest instruction>' (upstream "
                                 "global + per-shot caption format; the init prompt is the shot caption before the "
                                 "first instruction); user_prompt and camera_control texts are instructions, verbatim, "
                                 "not accumulated; an instruction takes effect at the first block that starts after "
                                 "it arrives"}

    # ---- rollout ---------------------------------------------------------------------------------------
    def _new_shot(self, shots, shot_text, global_caption, gen):
        """Set up a shot after `shots` (list of (caption, frames [f, c, h, w] in [-1, 1])): context KV cache,
        caption cache. Returns (conditional_dict, rope flags of the 21 noise latents, caption)."""
        torch, p = self.torch, self.pipe
        hist = shots[-HISTORY:]
        n = len(hist)
        caption = f"{global_caption}shot{n}:{shot_text}"
        if n == 0:
            counts = [self.max_context]
            flags = [0] * self.max_context
            cond_latents = None
        else:
            base, rem = divmod(self.max_context, n)
            counts = [base] * n
            for i in range(1, rem + 1):
                counts[-i] += 1
            frames, flags = [], []
            for k, (_, f) in enumerate(hist):
                if counts[k] == 0:
                    continue
                idx = torch.linspace(0, f.shape[0] - 1, steps=counts[k]).round().long().tolist()
                frames.append(f[idx])
                flags += [k] * len(idx)
            cond = torch.cat(frames).unsqueeze(2)                              # f c 1 h w
            cond_latents = p.vae.encode_to_latent(cond).to(torch.bfloat16)     # [f, 1, c, h, w]
            cond_latents = cond_latents.permute(1, 0, 2, 3, 4)                 # 1 f c h w
        if cond_latents is None:                                               # first shot: zero condition (upstream)
            cond_latents = torch.zeros([1, self.max_context, 16, H // 8, W // 8], device="cuda", dtype=torch.bfloat16)
        rope = flags + [flags[-1] + 1] * SHOT
        rope = torch.tensor(rope, dtype=torch.int32, device="cuda")
        prompts = [c for c, _ in hist] + [caption]
        cond_dict = p.text_encoder(text_prompts=prompts)
        nctx = cond_latents.shape[1]
        p._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=cond_latents.device,
                               kv_cache_size_override=SHOT * p.frame_seq_length)
        p._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=cond_latents.device)
        p._initialize_context_kv_cache(batch_size=1, dtype=torch.bfloat16, device=cond_latents.device,
                                       kv_cache_size_override=nctx * p.frame_seq_length)
        p.generator.model.local_attn_size = p.local_attn_size
        p._set_all_modules_max_attention_size(p.local_attn_size)
        p.generator(noisy_image_or_video=cond_latents, conditional_dict=cond_dict,
                    timestep=torch.zeros([1, nctx], device="cuda", dtype=torch.int64), kv_cache=None,
                    crossattn_cache=p.crossattn_cache, kv_cache_context=p.kv_cache1_context,
                    shot_flags_for_rope=rope[:nctx], current_start=0)
        p.vae.model.clear_cache()                                              # encode() also clears it
        return cond_dict, rope[nctx:], caption

    def _block(self, cond_dict, rope, start, gen):
        """Block at shot latent `start`: returns decoded frames [T, C, H, W] in [-1, 1] (bf16, on GPU)."""
        torch, p = self.torch, self.pipe
        seq = p.frame_seq_length
        x = torch.randn([1, BLOCK, 16, H // 8, W // 8], device="cuda", dtype=torch.bfloat16, generator=gen)
        r = rope[start:start + BLOCK]
        for i, t in enumerate(p.denoising_step_list):
            timestep = torch.ones([1, BLOCK], device="cuda", dtype=torch.int64) * t
            _, pred = p.generator(noisy_image_or_video=x, conditional_dict=cond_dict, timestep=timestep,
                                  kv_cache=p.kv_cache1, kv_cache_context=p.kv_cache1_context, shot_flags_for_rope=r,
                                  use_wo_rope_cache=False, crossattn_cache=p.crossattn_cache, current_start=start * seq)
            if i < len(p.denoising_step_list) - 1:
                nxt = p.denoising_step_list[i + 1] * torch.ones([BLOCK], device="cuda", dtype=torch.long)
                noise = torch.randn(pred.flatten(0, 1).shape, device="cuda", dtype=pred.dtype, generator=gen)
                x = p.scheduler.add_noise(pred.flatten(0, 1), noise, nxt).unflatten(0, pred.shape[:2])
        p.generator(noisy_image_or_video=pred, conditional_dict=cond_dict,
                    timestep=torch.ones_like(timestep) * int(self.cfg.context_noise), kv_cache=p.kv_cache1,
                    crossattn_cache=p.crossattn_cache, current_start=start * seq, kv_cache_context=p.kv_cache1_context,
                    shot_flags_for_rope=r, use_wo_rope_cache=False)
        return p.vae.decode_to_pixel(pred, use_cache=True)[0].to(torch.bfloat16)   # T C H W in [-1, 1]

    @staticmethod
    def _to_uint8(torch, video):
        video = (video.float() * 0.5 + 0.5).clamp(0, 1)
        return list((video.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy())

    def warmup(self):
        torch = self.torch
        t0 = time.time()
        gen = torch.Generator("cuda").manual_seed(0)
        shots, times = [], []
        with torch.no_grad():
            for k in range(2):                                   # a second shot exercises the context path
                t = time.time()
                cond, rope, cap = self._new_shot(shots, "warmup: a calm lake at sunrise", "A calm lake. ", gen)
                frames = []
                for start in range(0, 2 * BLOCK if k else SHOT, BLOCK):
                    frames.append(self._block(cond, rope, start, gen))
                    times.append(round(time.time() - t, 3))
                    t = time.time()
                shots.append((cap, torch.cat(frames)))
        LOG.info("warmup %.1fs; block times (first incl. shot setup) %s", time.time() - t0, times)

    def run(self, s: t2v_stream.GenSession):
        torch = self.torch
        with torch.no_grad():                          # grad mode is per thread; this runs in the session thread
            gen = torch.Generator("cuda").manual_seed(s.seed)
            glob = s.base_prompt
            shot_text = s.base_prompt
            shots, frames, cond, start, caption = [], [], None, SHOT, None
            while True:
                got = s.next_block()
                if got is None:
                    return
                _, changed, events = got
                new = cond is None or start >= SHOT or bool(events)
                if events:
                    shot_text = s.instruction
                if new:
                    if frames:
                        shots.append((caption, torch.cat(frames)))
                        shots = shots[-HISTORY:]
                    cond, rope, caption = self._new_shot(shots, shot_text, glob, gen)
                    frames, start = [], 0
                    s.shot_index = getattr(s, "shot_index", -1) + 1
                s.applied(events, f"new shot {s.shot_index} (upstream next-shot generation) with the new shot caption")
                out = self._block(cond, rope, start, gen)
                frames.append(out)
                start += BLOCK
                s.emit(self._to_uint8(torch, out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18877)
    ap.add_argument("--weights", help="unused; weights are read from $SHOTSTREAM_ROOT/weights (default $STREAMEVAL_MODEL_ROOT/shotstream/weights)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t2v_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
