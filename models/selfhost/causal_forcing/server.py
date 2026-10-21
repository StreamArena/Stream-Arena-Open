#!/usr/bin/env python3
"""Causal Forcing++ (frame-wise, 2-step, Wan2.1-T2V-1.3B) self-hosted server, docs/protocol.md.

One WebSocket session = one live autoregressive rollout of upstream `CausalInferencePipeline.inference`
(thu-ml/Causal-Forcing, `configs/causal_forcing_dmd_framewise_2step.yaml`, `causal-forcing++/framewise-2step.pt`,
`generator_ema` as upstream's `--use_ema`), unrolled so that each block — ONE latent frame (frame-wise model; 4 video
frames, the very first latent 1 frame) — is generated, decoded and sent before the next one starts:

  * denoising: the first block uses upstream's 4-step first-chunk schedule (1000/750/500/250, warped), every later
    block the 2-step schedule (1000/500, warped), shift 5.0, no CFG; then the clean-context pass into the KV cache
    (context_noise 0), as upstream;
  * causal Wan2.1 VAE decode of just this latent (`decode_to_pixel(use_cache=True)`, decoder cache kept across blocks).

Image start (general_interaction_image): upstream's frame-wise I2V (`inference.py --i2v`): the opening image is
VAE-encoded as the first latent and written into the KV cache as clean context (timestep 0); generation continues
from latent 1. Upstream resizes the image to 832x480 without keeping the aspect ratio; here it is center-cropped to
832:480 first (same as the other image-start entries), then resized. Its decoded frame is the first output frame.

Differences from the upstream 81-frame script, needed for open-ended interactive sessions (same as selfhost/self_forcing,
whose code base Causal Forcing extends):
  * rolling KV cache: `local_attn_size = 21` latent frames (the trained length) enables upstream's own cache rolling
    in `CausalWanSelfAttention`; the oldest latents are evicted after 21. Long rollouts are an extrapolation of a
    5-second model (upstream README says so explicitly) and are scored as generated.
  * the temporal RoPE table (upstream 1024 positions) is recomputed with the same formula for 4096.
  * prompt switch: no interactive mechanism upstream. The block prompt is re-encoded and the cross-attention cache
    reset at the next block; the self-attention KV cache keeps the past as generated.

Streaming, pacing, prompt policy and acks: selfhost/t2v_stream.py.

    python server.py --port 18861
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

ROOT = Path(os.environ.get("CAUSAL_FORCING_ROOT",
                           os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/causal_forcing"))
LOG = logging.getLogger("causal_forcing")
W, H, FPS = 832, 480, 16
BLOCK = 1            # latent frames per block (frame-wise model)
WINDOW = 21          # latent frames of self-attention context (training length)
ROPE_POSITIONS = 4096
CKPT = "checkpoints/causal-forcing++/framewise-2step.pt"
CONFIG = "configs/causal_forcing_dmd_framewise_2step.yaml"


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


class Engine:
    name = "Causal Forcing++"
    width, height, fps = W, H, FPS
    image_start = True

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
        cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CONFIG))
        cfg.model_kwargs.local_attn_size = WINDOW
        t0 = time.time()
        pipe = CausalInferencePipeline(cfg, device=torch.device("cuda"))
        ckpt = ROOT / "weights" / CKPT
        state = torch.load(ckpt, map_location="cpu")
        gen_sd = state["generator_ema"]                               # upstream README: --use_ema
        try:
            pipe.generator.load_state_dict(gen_sd)
        except RuntimeError:                                          # upstream inference.py fallback (FSDP prefix)
            gen_sd = {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in gen_sd.items()}
            missing, unexpected = pipe.generator.load_state_dict(gen_sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        del state, gen_sd
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
        self.steps_first = [float(t) for t in pipe.denoising_step_list_first_chunk]
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"causal-forcing@{self.commit[:12]} causal-forcing++/framewise-2step.pt (generator_ema)"
        wan = ROOT / "weights/wan_models/Wan2.1-T2V-1.3B"
        self.weights_sha256 = t2v_stream.sha256_files(
            [ckpt, wan / "models_t5_umt5-xxl-enc-bf16.pth", wan / "Wan2.1_VAE.pth"], ROOT / "weights")
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s", self.version, time.time() - t0, self.gpu)
        self.warmup()

    def system_config(self):
        return {"model": self.name,
                "model_variant": "Causal Forcing++ frame-wise 2-step on Wan2.1-T2V-1.3B (causal consistency distillation "
                                 "init + asymmetric DMD), causal-forcing++/framewise-2step.pt (generator_ema)",
                "version": self.version, "repo": "https://github.com/thu-ml/Causal-Forcing",
                "repo_commit": self.commit, "license": "Apache-2.0", "weights_sha256": self.weights_sha256,
                "width": W, "height": H, "fps": FPS, "denoising_steps": len(self.steps),
                "denoising_step_list": [round(t, 1) for t in self.steps],
                "denoising_step_list_first_block": [round(t, 1) for t in self.steps_first], "timestep_shift": 5.0,
                "block_latents": BLOCK, "block_frames": BLOCK * 4, "local_attn_latents": WINDOW, "sink_latents": 0,
                "context_noise": int(self.cfg.context_noise), "rope_positions": ROPE_POSITIONS,
                "vae": "Wan2.1 VAE, causal cached decode per block", "dtype": "bfloat16", "attention": "torch SDPA",
                "gpu": self.gpu, "gpu_count": 1,
                "image_start": "upstream frame-wise I2V: image (center-crop to 832:480, resize) VAE-encoded as latent 0, "
                               "clean KV context; its decoded frame is output frame 0",
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

    def _decode(self, latents):
        torch = self.torch
        video = self.pipe.vae.decode_to_pixel(latents, use_cache=True)[0]   # T C H W in [-1, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return (video.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()

    def _image_latent(self, cond, image):
        """Upstream I2V: encode the image as latent 0 and write it into the KV cache as clean context."""
        torch, p = self.torch, self.pipe
        x = torch.from_numpy(__import__("numpy").asarray(center_crop(image, W, H))).to("cuda")
        x = (x.permute(2, 0, 1).float() / 127.5 - 1.0).to(torch.bfloat16)          # Normalize([0.5], [0.5])
        lat = p.vae.encode_to_latent(x[None, :, None]).to(device="cuda", dtype=torch.bfloat16)   # [1, 1, 16, h, w]
        p.generator(noisy_image_or_video=lat, conditional_dict=cond,
                    timestep=torch.zeros([1, 1], device="cuda", dtype=torch.int64),
                    kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache, current_start=0)
        return lat

    def _block(self, cond, start, gen, first):
        """One latent at latent frame `start`: returns uint8 frames [T, H, W, 3]."""
        torch, p = self.torch, self.pipe
        seq = p.frame_seq_length
        steps = p.denoising_step_list_first_chunk if first else p.denoising_step_list
        x = torch.randn([1, BLOCK, 16, H // 8, W // 8], device="cuda", dtype=torch.bfloat16, generator=gen)
        for i, t in enumerate(steps):
            timestep = torch.ones([1, BLOCK], device="cuda", dtype=torch.int64) * t
            _, pred = p.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=timestep,
                                  kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache, current_start=start * seq)
            if i < len(steps) - 1:
                nxt = steps[i + 1] * torch.ones([BLOCK], device="cuda", dtype=torch.long)
                noise = torch.randn(pred.flatten(0, 1).shape, device="cuda", dtype=pred.dtype, generator=gen)
                x = p.scheduler.add_noise(pred.flatten(0, 1), noise, nxt).unflatten(0, pred.shape[:2])
        p.generator(noisy_image_or_video=pred, conditional_dict=cond,
                    timestep=torch.ones_like(timestep) * int(self.cfg.context_noise),
                    kv_cache=p.kv_cache1, crossattn_cache=p.crossattn_cache, current_start=start * seq)
        return self._decode(pred)

    def _encode(self, prompt):
        for c in self.pipe.crossattn_cache:
            c["is_init"] = False
        return self.pipe.text_encoder(text_prompts=[prompt])

    def warmup(self):
        torch = self.torch
        from PIL import Image
        for image in (None, Image.new("RGB", (1024, 768), (90, 140, 200))):
            t0 = time.time()
            with torch.no_grad():
                self._reset()
                gen = torch.Generator("cuda").manual_seed(0)
                cond = self._encode("warmup: a calm lake at sunrise")
                start = 0
                if image is not None:
                    self._decode(self._image_latent(cond, image))
                    start = 1
                times = []
                for i in range(WINDOW + 4):                                # past the first cache roll
                    t = time.time()
                    frames = self._block(cond, start, gen, i == 0)
                    start += BLOCK
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
            self._reset()
            gen = torch.Generator("cuda").manual_seed(s.seed)
            cond, start, first = None, 0, True
            while start + BLOCK <= ROPE_POSITIONS:
                got = s.next_block()
                if got is None:
                    return
                prompt, changed, events = got
                if cond is None or changed:
                    cond = self._encode(prompt)
                s.applied(events, "text re-encoded, cross-attention cache reset")
                if image is not None:                  # first "block": the image latent, then generation from latent 1
                    frames = list(self._decode(self._image_latent(cond, image)))
                    image, start = None, 1
                    frames += list(self._block(cond, start, gen, True))
                else:
                    frames = list(self._block(cond, start, gen, first))
                first = False
                s.emit(frames)
                start += BLOCK
            raise RuntimeError(f"rollout reached the positional table ({ROPE_POSITIONS} latent frames)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18861)
    ap.add_argument("--weights", help="unused; weights are read from $CAUSAL_FORCING_ROOT/weights (default $STREAMEVAL_MODEL_ROOT/causal_forcing/weights)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t2v_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
