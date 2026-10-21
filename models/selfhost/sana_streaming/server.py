#!/usr/bin/env python3
"""SANA-Streaming (NVlabs, streaming DiT) self-hosted instruction-edit server (docs/protocol.md).

Upstream ships only an offline script (inference_video_scripts/v2v/inference_sana_streaming.py, mode
long_streaming) that encodes a whole source video and then runs the chunk-wise SANAStreamingSampler. This server
runs the same components incrementally on the live stream:

  * source frames -> causal LTX-2 VAE encoder with persistent feature cache (streaming_causal_encode; first
    chunk 1+24 frames, then 24),
  * one sampler chunk (3 latent frames = 24 pixel frames) per step: the per-chunk body of
    SANAStreamingSampler.sample unrolled, same fixed-RoPE KV/state cache, sink token, num_cached_blocks=2,
    4 steps, cfg 1.0,
  * the released causal LTX-2 decoder (ltx2_causal_vae_0516) with persistent cache (CausalVaeStreamingDecoder).

SANA-Streaming's prompt IS an editing instruction ("Transform the entire scene into ..."). A text event is used
verbatim as the editing prompt from the next chunk on (the sampler takes a per-chunk condition); caches are
kept, so the edit changes without restarting the stream. Earlier instructions are not accumulated. Source
frames are passed through until the first instruction. Streaming, pacing and acks: models/selfhost/edit_stream.py.

    python server.py --port 8766
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ["DISABLE_XFORMERS"] = "1"
os.environ.setdefault("USE_CHUNKWISE_GDN", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edit_stream  # noqa: E402

ROOT = Path(os.environ.get("SANA_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/sana_streaming"))
W = ROOT / "weights"
LOG = logging.getLogger("sana_streaming")


class Engine:
    name = "SANA-Streaming"

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path.insert(0, str(repo))
        os.chdir(repo)
        import pyrallis
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from diffusion.model.ltx2 import CausalVaeStreamingDecoder
        from diffusion.model.ltx2.causal_vae import AutoencoderKLCausalLTX2Video
        from diffusion.scheduler.sana_streaming_sampler import SANAStreamingSampler

        spec = importlib.util.spec_from_file_location(
            "sana_v2v_infer", repo / "inference_video_scripts/v2v/inference_sana_streaming.py")
        self.infer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.infer)
        self.torch, self.args, self.Sampler = torch, args, SANAStreamingSampler
        torch.set_grad_enabled(False)
        self.fps = args.model_fps
        self.device, self.dtype = torch.device("cuda"), torch.bfloat16
        c = self.config = pyrallis.parse(config_class=self.infer.InferenceConfig,
                                         config_path=str(repo / "configs/sana_streaming/sana_streaming_2b_720p.yaml"),
                                         args=[])
        self.stride = c.vae.vae_stride                                      # [8, 32, 32]
        self.base_chunk = 24 // self.stride[0]                              # 3 latent frames per chunk (upstream)
        self.chunk_frames = self.base_chunk * self.stride[0]                # 24 pixel frames
        self.flow_shift = c.scheduler.inference_flow_shift or c.scheduler.flow_shift
        t0 = time.time()
        enc = AutoencoderKLCausalLTX2Video.from_pretrained(str(W / "LTX-2"), subfolder="vae", torch_dtype=self.dtype)
        enc.decoder = None                                                  # encoder: Lightricks/LTX-2 vae (config)
        self.enc = enc.to(self.device).eval()
        dec = AutoencoderKLCausalLTX2Video.from_pretrained(str(W / "SANA-Streaming/ltx2_causal_vae_0516"),
                                                           torch_dtype=self.dtype)
        dec.encoder = None                                                  # decoder: released causal artifact
        self.stream_dec = CausalVaeStreamingDecoder(dec.to(self.device).eval())
        self.tokenizer = AutoTokenizer.from_pretrained(str(W / "gemma-2-2b-it"))
        self.tokenizer.padding_side = "right"
        self.text_encoder = (AutoModelForCausalLM.from_pretrained(str(W / "gemma-2-2b-it"), torch_dtype=self.dtype)
                             .get_decoder().to(self.device).eval())
        self.model = self.infer.load_model(c, c.model.image_size // c.vae.vae_downsample_rate, self.device,
                                           self.dtype, str(W / "SANA-Streaming/dit/sana_streaming_ar.pth"))
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"sana@{self.commit[:12]}+sana_streaming_ar"
        self.weights_sha256 = edit_stream.sha256_files([
            W / "SANA-Streaming/dit/sana_streaming_ar.pth",
            W / "SANA-Streaming/ltx2_causal_vae_0516/diffusion_pytorch_model.safetensors",
            W / "LTX-2/vae/diffusion_pytorch_model.safetensors",
            W / "gemma-2-2b-it/model-00001-of-00002.safetensors",
            W / "gemma-2-2b-it/model-00002-of-00002.safetensors"], W)
        self.neg = self.encode_prompt("", chi=False)
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s", self.version, time.time() - t0, self.gpu)
        self.warmup()

    def out_size(self, init):
        a = self.args
        return (a.height, a.width) if int(init.get("height") or 0) > int(init.get("width") or 1) else (a.width, a.height)

    def encode_prompt(self, prompt, chi=True):
        with self.torch.autocast("cuda", dtype=self.dtype):
            return self.infer.encode_prompt(self.tokenizer, self.text_encoder, prompt, self.config, self.device,
                                            use_chi_prompt=chi)

    def warmup(self):
        w, h = self.args.width, self.args.height
        st = Stream(self, w, h, seed=0)
        st.set_prompt("Transform the scene into a watercolor painting.")
        rnd = [np.random.uniform(-1, 1, (h, w, 3)).astype(np.float32) for _ in range(1 + 2 * self.chunk_frames)]
        t0 = time.time()
        st.step(rnd[:1 + self.chunk_frames])
        st.step(rnd[1 + self.chunk_frames:])
        self.torch.cuda.synchronize()
        LOG.info("warmup done in %.1fs", time.time() - t0)

    def system_config(self):
        a = self.args
        return {"model": self.name, "model_variant": "SANA-Streaming streaming DiT (dit/sana_streaming_ar.pth), BF16",
                "version": self.version, "repo": "https://github.com/NVlabs/Sana (upstream docs/sana_streaming.md)",
                "repo_commit": self.commit, "license": "Apache-2.0 (code); weights: see models/selfhost/sana_streaming/README.md",
                "weights_sha256": self.weights_sha256,
                "resolution": {"landscape": [a.width, a.height], "portrait": [a.height, a.width]},
                "model_fps": a.model_fps, "denoising_steps": a.step, "cfg_scale": 1.0,
                "num_cached_blocks": a.num_cached_blocks, "sink_token": True,
                "chunk_latent_frames": self.base_chunk, "chunk_pixel_frames": self.chunk_frames,
                "vae": "encode: Lightricks/LTX-2 vae (causal streaming); decode: SANA-Streaming ltx2_causal_vae_0516",
                "text_encoder": "Efficient-Large-Model/gemma-2-2b-it (upstream chi_prompt)",
                "quantization": "none", "gpu": self.gpu, "gpu_count": 1, "dtype": "bfloat16",
                "server": "StreamEval streaming wrapper around upstream long_streaming components",
                "prompt_policy": "instruction text verbatim as the editing prompt from the next chunk on; no rewriting "
                                 "or accumulation; caches kept across instruction switches",
                "input_policy": f"center crop to output aspect, resize, subsample to {a.model_fps} fps; newest frames "
                                "win when behind"}

    def run(self, s: edit_stream.EditSession):
        with self.torch.no_grad():                     # grad mode is per thread; this runs in the session thread
            self._run(s)

    def _run(self, s: edit_stream.EditSession):
        st = Stream(self, s.w, s.h, s.seed)
        while True:
            got = s.take(lambda p: self.chunk_frames + (1 if st.prompt is None else 0))
            if got is None:
                return
            frames, prompt, event = got
            t0 = time.perf_counter()
            if prompt != st.prompt:
                st.set_prompt(prompt)
            s.emit(list(st.step(frames)))
            ms = (time.perf_counter() - t0) * 1e3
            s.stats["chunks"] += 1
            s.stats["chunk_ms"].append(ms)
            s.applied(event, f"editing prompt = instruction text verbatim from chunk {s.stats['chunks'] - 1} "
                             "(caches kept)")


class Stream:
    """Incremental SANAStreamingSampler.sample + streaming causal VAE encode / decode for one session."""

    def __init__(self, e: Engine, width: int, height: int, seed: int):
        self.e = e
        self.lh, self.lw = height // e.stride[1], width // e.stride[2]
        self.gen = e.torch.Generator(device=e.device).manual_seed(seed)
        self.sampler = e.Sampler(e.model, condition=e.neg[0], uncondition=e.neg[0], cfg_scale=1.0,
                                 flow_shift=e.flow_shift, model_kwargs={}, base_chunk_frames=e.base_chunk,
                                 num_cached_blocks=e.args.num_cached_blocks, cache_strategy="fixed_rope",
                                 efficient_cache=False, sink_token=True)
        self.sampler._spatial_hw = self.lh * self.lw
        self.chunk_indices, self.kv_cache = [0], []
        self.cond = self.mask = self.prompt = None
        e.enc.clear_encoder_cache()
        e.stream_dec.reset()

    def set_prompt(self, prompt: str):
        self.cond, self.mask = self.e.encode_prompt(prompt)
        self.prompt = prompt

    def encode(self, frames):
        torch, e = self.e.torch, self.e
        x = torch.from_numpy(np.stack(frames)).to(e.device, e.dtype).permute(3, 0, 1, 2).unsqueeze(0)  # 1 3 T H W
        r = e.stride[0]
        out, i, step = [], 0, x.shape[2] % r or r                   # first chunk: 1 frame, then groups of 8
        while i < x.shape[2]:
            out.append(e.enc.streaming_causal_encode(x[:, :, i:i + step], causal=True).latent_dist.mode())
            i, step = i + step, r
        z = torch.cat(out, 2)
        mean = e.enc.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        std = e.enc.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        return ((z - mean) * e.enc.config.scaling_factor / std).to(e.dtype)

    def step(self, frames) -> np.ndarray:
        """frames: 1+24 on the first call, then 24; float32 HxWx3 in [-1, 1]. Returns uint8 T x H x W x 3."""
        from diffusers import FlowMatchEulerDiscreteScheduler
        torch, e, s = self.e.torch, self.e, self.sampler
        with torch.autocast("cuda", dtype=e.dtype):
            src = self.encode(frames)                                   # 1 C t h w
            c = len(self.kv_cache)
            self.chunk_indices.append(self.chunk_indices[-1] + src.shape[2])
            s._chunk_indices = self.chunk_indices
            self.kv_cache.append([[None] * 6 for _ in range(s.num_model_blocks)])
            chunk_kv, _, sink_num, num_cached = s.accumulate_kv_cache(self.kv_cache, c)
            start_f, end_f = self.chunk_indices[c], self.chunk_indices[c + 1]
            frame_index, rope_start = None, self.chunk_indices[max(c - s.num_cached_blocks, 0)]
            if sink_num > 0:
                non_sink = num_cached - sink_num + (end_f - start_f)
                frame_index = torch.cat([torch.arange(sink_num, device=e.device),
                                         torch.arange(end_f - non_sink, end_f, device=e.device)])
                rope_start = 0
            latents = torch.randn(1, e.config.vae.vae_latent_dim, src.shape[2], self.lh, self.lw, device=e.device,
                                  generator=self.gen)
            scheduler = FlowMatchEulerDiscreteScheduler(shift=e.flow_shift)
            kw = dict(start_f=rope_start, end_f=end_f, frame_index=frame_index, kv_cache=chunk_kv, mask=self.mask,
                      data_info={"image_vae_embeds": src})
            for t in s._timesteps_for_steps(scheduler, e.args.step, e.device):
                pred, _ = e.model(latents, t.expand(1), self.cond, save_kv_cache=False, **kw)
                pred = pred if isinstance(pred, torch.Tensor) else pred[0]
                latents = scheduler.step(pred, t, latents, return_dict=False)[0].to(latents.dtype)
            _, self.kv_cache[c] = e.model(latents, torch.zeros(1, device=e.device), self.cond, save_kv_cache=True, **kw)
            pix = e.stream_dec.decode_chunk(latents.to(e.dtype))       # 1 3 T H W in [-1, 1]
        pix = ((pix[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return pix.permute(1, 2, 3, 0).contiguous().cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--model_fps", type=int, default=24)
    ap.add_argument("--step", type=int, default=4, choices=[2, 4])
    ap.add_argument("--num_cached_blocks", type=int, default=2)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    edit_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
