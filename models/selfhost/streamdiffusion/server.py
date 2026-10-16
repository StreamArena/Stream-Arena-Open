#!/usr/bin/env python3
"""StreamDiffusion (cumulo-autumn, SD-Turbo img2img) self-hosted instruction-edit server (docs/protocol.md).

Configuration of the upstream real-time img2img demo (demo/realtime-img2img/img2img.py): StreamDiffusionWrapper
with stabilityai/sd-turbo, TAESD (madebyollin/taesd), t_index_list [35, 45], frame_buffer_size 1, denoising
batch, cfg_type "none", 50-step schedule, the demo's negative prompt, img2img per live frame. Acceleration
"none" (diffusers SDPA attention) instead of the demo's TensorRT / xformers (optional accelerators; same math).

StreamDiffusion is prompt-conditioned img2img: a text instruction becomes the prompt verbatim (no rewriting, no
accumulation) from the next frame on (StreamDiffusion.update_prompt; latents / noise buffers kept). Source frames
are passed through until the first instruction. Streaming, pacing and acks: models/selfhost/edit_stream.py.

    python server.py --port 8776
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edit_stream  # noqa: E402

ROOT = Path(os.environ.get("SD1_ROOT", os.path.expanduser(os.environ.get("STREAM_ARENA_MODEL_ROOT", "~/.cache/stream_arena/models")) + "/streamdiffusion"))
W = ROOT / "weights"
LOG = logging.getLogger("streamdiffusion")
NEGATIVE = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"  # demo default


class Engine:
    name = "StreamDiffusion"

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path[:0] = [str(repo), str(repo / "src")]
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from utils.wrapper import StreamDiffusionWrapper

        self.torch, self.args = torch, args
        torch.set_grad_enabled(False)
        self.fps = args.model_fps
        t0 = time.time()
        self.streams = {}
        for w, h in ((args.width, args.height), (args.height, args.width)):   # landscape, portrait
            self.streams[(w, h)] = StreamDiffusionWrapper(
                model_id_or_path=str(W / "sd-turbo"), use_tiny_vae=True, vae_id=str(W / "taesd"),
                device="cuda", dtype=torch.float16, t_index_list=[35, 45], frame_buffer_size=1, width=w, height=h,
                use_lcm_lora=False, output_type="pt", warmup=10, acceleration="none", mode="img2img",
                use_denoising_batch=True, cfg_type="none", use_safety_checker=False, seed=args.seed)
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"streamdiffusion@{self.commit[:12]}+sd-turbo"
        self.weights_sha256 = edit_stream.sha256_files(
            [W / "sd-turbo/unet/diffusion_pytorch_model.safetensors",
             W / "sd-turbo/text_encoder/model.safetensors",
             W / "sd-turbo/vae/diffusion_pytorch_model.safetensors",
             W / "taesd/diffusion_pytorch_model.safetensors"], W)
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s", self.version, time.time() - t0, self.gpu)
        self.warmup()

    def out_size(self, init):
        a = self.args
        return (a.height, a.width) if int(init.get("height") or 0) > int(init.get("width") or 1) else (a.width, a.height)

    def prepare(self, wrapper, prompt, seed):
        wrapper.stream.prepare(prompt, NEGATIVE, num_inference_steps=50, guidance_scale=1.2, delta=1.0,
                               generator=self.torch.Generator(), seed=seed)

    def step(self, wrapper, frame):
        image = Image.fromarray(((frame + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8))
        out = wrapper.stream(image)                                       # 1 x 3 x H x W in [-1, 1]
        return ((out[0].float().clamp(-1, 1) + 1) * 127.5).round().to(self.torch.uint8).permute(1, 2, 0).cpu().numpy()

    def warmup(self):
        t0 = time.time()
        for (w, h), wrapper in self.streams.items():
            self.prepare(wrapper, "a watercolor painting", 0)
            for _ in range(10):
                self.step(wrapper, np.random.uniform(-1, 1, (h, w, 3)).astype(np.float32))
        self.torch.cuda.synchronize()
        LOG.info("warmup done in %.1fs", time.time() - t0)

    def system_config(self):
        a = self.args
        return {"model": self.name, "model_variant": "stabilityai/sd-turbo img2img + TAESD (upstream realtime-img2img demo)",
                "version": self.version, "repo": "https://github.com/cumulo-autumn/StreamDiffusion",
                "repo_commit": self.commit, "license": "Apache-2.0 (code); sd-turbo: Stability AI Community "
                "License; TAESD: MIT", "weights_sha256": self.weights_sha256,
                "resolution": {"landscape": [a.width, a.height], "portrait": [a.height, a.width]},
                "model_fps": a.model_fps, "t_index_list": [35, 45], "num_inference_steps": 50,
                "frame_buffer_size": 1, "use_denoising_batch": True, "cfg_type": "none",
                "negative_prompt": NEGATIVE, "acceleration": "none (diffusers SDPA)", "vae": "TAESD",
                "gpu": self.gpu, "gpu_count": 1, "dtype": "float16",
                "prompt_policy": "instruction text verbatim as the new prompt (StreamDiffusion.update_prompt) from "
                                 "the next frame on; no rewriting or accumulation; CLIP truncates at 77 tokens",
                "input_policy": f"center crop to output aspect, resize, subsample to {a.model_fps} fps; newest frame "
                                "wins when behind"}

    def run(self, s: edit_stream.EditSession):
        with self.torch.no_grad():
            self._run(s)

    def _run(self, s: edit_stream.EditSession):
        wrapper, active = self.streams[(s.w, s.h)], None
        while True:
            got = s.take(lambda p: 1)
            if got is None:
                return
            frames, prompt, event = got
            t0 = time.perf_counter()
            if active is None:
                self.prepare(wrapper, prompt, s.seed)
            elif prompt != active:
                wrapper.stream.update_prompt(prompt)
            changed, active = prompt != active, prompt
            s.emit([self.step(wrapper, frames[0])])
            ms = (time.perf_counter() - t0) * 1e3
            s.stats["chunks"] += 1
            s.stats["chunk_ms"].append(ms)
            s.applied(event, "prompt = instruction text verbatim from the next frame" if changed
                      else "identical to the active prompt")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8776)
    ap.add_argument("--width", type=int, default=640, help="aspect-preserving 512x512-class budget (UNet: /64)")
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--model_fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2, help="upstream default (sessions use init.seed)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    edit_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
