#!/usr/bin/env python3
"""StreamV2V (Jeff-LiangF/streamv2v, SD 1.5 + LCM-LoRA with feature banks) self-hosted instruction-edit server
(docs/protocol.md).

Configuration of the upstream live-camera demo (demo_w_camera/vid2vid.py): StreamV2VWrapper over SD 1.5
(Jiali/stable-diffusion-1.5) with LCM-LoRA (latent-consistency/lcm-lora-sdv1-5) and TAESD, t_index_list
[30, 35, 40, 45], frame_buffer_size 1, denoising batch, guidance 1.0, cached spatio-temporal attention (feature
bank: cache_interval 4, cache_maxframes 1, ToMe cache) with feature injection (strength 0.8, similarity
threshold 0.98). Acceleration "none" = the SDPA variant of the cached attention processor
(CachedSTAttnProcessor2_0) instead of the demo's xformers / TensorRT (optional accelerators; same algorithm).
The demo's optional style LoRAs (Civitai files activated by trigger words) are not loaded.

StreamV2V is prompt-conditioned V2V: a text instruction becomes the prompt verbatim (no rewriting, no
accumulation) from the next frame on (update_prompt; the feature bank is kept). Source frames are passed through
until the first instruction. Streaming, pacing and acks: models/selfhost/edit_stream.py.
Files are read from $SV2V_ROOT (default $STREAM_ARENA_MODEL_ROOT/streamv2v, see setup.sh).

    python server.py --port 8777
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import site
import sys
import time

# a user site-packages directory (~/.local, e.g. with an older peft) can shadow the venv: the venv's site-packages
# go first
_venv = [p for p in site.getsitepackages() if p.startswith(sys.prefix)] + [
    os.path.join(os.environ.get("SV2V_ROOT", os.path.expanduser(os.environ.get("STREAM_ARENA_MODEL_ROOT", "~/.cache/stream_arena/models")) + "/streamv2v"),
                 f"env/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")]
sys.path = list(dict.fromkeys(_venv)) + [p for p in sys.path if p not in _venv]
# a .pth in the user site may already have imported packages from it at interpreter start: drop them so they are
# re-imported from the venv first
_PURGED = sorted(n for n, m in list(sys.modules.items())
                 if (getattr(m, "__file__", None) or "").startswith(site.getusersitepackages()))
for _n in _PURGED:
    del sys.modules[_n]

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edit_stream  # noqa: E402

ROOT = Path(os.environ.get("SV2V_ROOT", os.path.expanduser(os.environ.get("STREAM_ARENA_MODEL_ROOT", "~/.cache/stream_arena/models")) + "/streamv2v"))
W = ROOT / "weights"
LOG = logging.getLogger("streamv2v")


class Engine:
    name = "StreamV2V"

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path[:0] = [str(repo), str(repo / "src")]
        # local paths only; HF offline mode must stay off: diffusers refuses to guess the LCM-LoRA file name of a
        # local directory in offline mode
        os.environ["HF_HUB_OFFLINE"] = "0"
        import peft
        import torch
        from utils.wrapper import StreamV2VWrapper
        LOG.info("peft %s from %s; purged user-site modules %s", peft.__version__, peft.__file__, _PURGED[:20])
        from streamv2v.models.attention_processor import CachedSTAttnProcessor2_0

        self.torch, self.args = torch, args
        torch.set_grad_enabled(False)
        self.fps = args.model_fps
        t0 = time.time()
        self.streams = {}
        for w, h in ((args.width, args.height), (args.height, args.width)):   # landscape, portrait
            wrapper = StreamV2VWrapper(
                model_id_or_path=str(W / "stable-diffusion-1.5"), lcm_lora_id=str(W / "lcm-lora-sdv1-5"),
                vae_id=str(W / "taesd"), t_index_list=[30, 35, 40, 45], frame_buffer_size=1, width=w, height=h,
                warmup=10, acceleration="none", do_add_noise=True, output_type="pt", use_denoising_batch=True,
                use_cached_attn=True, use_feature_injection=True, feature_injection_strength=0.8,
                feature_similarity_threshold=0.98, cache_interval=4, cache_maxframes=1, use_tome_cache=True,
                seed=args.seed)
            procs = wrapper.stream.pipe.unet.attn_processors.values()
            # the wrapper swallows acceleration errors and silently falls back to plain StreamDiffusion
            assert all(isinstance(p, CachedSTAttnProcessor2_0) for p in procs), "cached attention not installed"
            self.streams[(w, h)] = wrapper
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"streamv2v@{self.commit[:12]}+sd15-lcm"
        self.weights_sha256 = edit_stream.sha256_files(
            [W / "stable-diffusion-1.5/unet/diffusion_pytorch_model.safetensors",
             W / "stable-diffusion-1.5/text_encoder/model.safetensors",
             W / "lcm-lora-sdv1-5/pytorch_lora_weights.safetensors",
             W / "taesd/diffusion_pytorch_model.safetensors"], W)
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s", self.version, time.time() - t0, self.gpu)
        self.warmup()

    def out_size(self, init):
        a = self.args
        return (a.height, a.width) if int(init.get("height") or 0) > int(init.get("width") or 1) else (a.width, a.height)

    def prepare(self, wrapper, prompt, seed):
        """New session: fresh noise / latent buffers (StreamV2V.prepare) and an empty feature bank."""
        t = self.torch
        for p in wrapper.stream.pipe.unet.attn_processors.values():
            p.frame_id = t.tensor(0)
            p.cached_key = p.cached_value = p.cached_output = None
        wrapper.stream.frame_idx = 0
        wrapper.stream.prepare(prompt, "", num_inference_steps=50, guidance_scale=1.0,
                               generator=t.manual_seed(seed), seed=seed)

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
        return {"model": self.name, "model_variant": "SD 1.5 (Jiali/stable-diffusion-1.5) + LCM-LoRA + TAESD, "
                "feature-bank cached attention (upstream demo_w_camera configuration)",
                "version": self.version, "repo": "https://github.com/Jeff-LiangF/streamv2v", "repo_commit": self.commit,
                "license": "UT Austin Research License (code); SD 1.5: CreativeML OpenRAIL-M; LCM-LoRA: OpenRAIL++; "
                           "TAESD: MIT", "weights_sha256": self.weights_sha256,
                "resolution": {"landscape": [a.width, a.height], "portrait": [a.height, a.width]},
                "model_fps": a.model_fps, "t_index_list": [30, 35, 40, 45], "num_inference_steps": 50,
                "guidance_scale": 1.0, "frame_buffer_size": 1, "use_denoising_batch": True,
                "cached_attention": {"cache_interval": 4, "cache_maxframes": 1, "use_tome_cache": True,
                                     "feature_injection_strength": 0.8, "feature_similarity_threshold": 0.98},
                "acceleration": "none (CachedSTAttnProcessor2_0, SDPA)", "style_loras": "not loaded",
                "gpu": self.gpu, "gpu_count": 1, "dtype": "float16",
                "prompt_policy": "instruction text verbatim as the new prompt (update_prompt) from the next frame "
                                 "on; feature bank kept; no rewriting or accumulation; CLIP truncates at 77 tokens",
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
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--width", type=int, default=640, help="aspect-preserving 512x512-class budget (UNet: /64)")
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--model_fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1, help="demo default (sessions use init.seed)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    edit_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
