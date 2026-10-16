#!/usr/bin/env python3
"""StreamDiffusionV2 (1.3B, single GPU) self-hosted instruction-edit server (docs/protocol.md).

Model semantics follow the upstream online demo (demo/vid2vid.py): StreamDiffusionV2 is prompt-conditioned
video-to-video. A text instruction becomes the new prompt verbatim (no rewriting, no accumulation of earlier
instructions) and, as in the demo, a prompt change starts a new stream session (text encoder re-run, KV / VAE
caches reset) on the next frames; the source stream is never paused. Source frames are passed through until the
first instruction. Streaming, pacing and acks: models/selfhost/edit_stream.py.

    python server.py --port 8775 [--weights $STREAM_ARENA_MODEL_ROOT/streamdiffusionv2/weights]
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

ROOT = Path(os.environ.get("SDV2_ROOT", os.path.expanduser(os.environ.get("STREAM_ARENA_MODEL_ROOT", "~/.cache/stream_arena/models")) + "/streamdiffusionv2"))
LOG = logging.getLogger("sdv2")


class Engine:
    name = "StreamDiffusionV2"

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path.insert(0, str(repo))
        os.chdir(repo)                                 # upstream resolves wan_models/ and ckpts/ relative to cwd
        import torch
        from streamv2v.inference import SingleGPUInferencePipeline
        from streamv2v.inference_common import merge_cli_config

        self.torch, self.args = torch, args
        torch.set_grad_enabled(False)
        self.fps = args.model_fps
        cfg = repo / "configs/wan_causal_dmd_v2v.yaml"
        ckpt = repo / "ckpts/wan_causal_dmd_v2v"
        self.config = merge_cli_config(str(cfg), {
            "config_path": str(cfg), "checkpoint_folder": str(ckpt), "step": args.step,
            "noise_scale": args.noise_scale, "height": 480, "width": 832, "model_type": "T2V-1.3B", "t2v": False,
            "use_taehv": args.use_taehv, "use_tensorrt": False, "fast": False, "profile": args.profile,
            "seed": args.seed,
            "taehv_checkpoint_path": str(repo / "ckpts/taew2_1.pth")})
        t0 = time.time()
        self.pm = SingleGPUInferencePipeline(self.config, torch.device("cuda"))
        self.pm.load_model(str(ckpt))
        self.chunk = self.pm.base_chunk_size * self.pm.pipeline.num_frame_per_block   # 4 frames
        self.dims = None
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"streamdiffusionv2@{self.commit[:12]}"
        wan = ROOT / "weights/wan_models/Wan2.1-T2V-1.3B"
        self.weights_sha256 = edit_stream.sha256_files(
            [ROOT / "weights/ckpts/wan_causal_dmd_v2v/model.pt", wan / "diffusion_pytorch_model.safetensors",
             wan / "models_t5_umt5-xxl-enc-bf16.pth", wan / "Wan2.1_VAE.pth"], ROOT / "weights")
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s", self.version, time.time() - t0, self.gpu)
        self.warmup()

    def out_size(self, init):
        return (480, 832) if int(init.get("height") or 0) > int(init.get("width") or 1) else (832, 480)

    def set_dims(self, width, height):
        """Landscape 832x480 / portrait 480x832, switched between sessions."""
        if self.dims == (width, height):
            return
        p = self.pm.pipeline
        p.height, p.width = height // 16 * 2, width // 16 * 2
        p.frame_seq_length = (height // 16) * (width // 16)
        p.kv_cache_length = p.frame_seq_length * p.num_kv_cache
        p.args.height, p.args.width = height, width
        self.dims = (width, height)

    def tensor(self, frames):
        x = self.torch.from_numpy(np.stack(frames)).unsqueeze(0).permute(0, 4, 1, 2, 3)   # 1 3 T H W
        return x.to(device="cuda", dtype=self.torch.bfloat16)

    def warmup(self):
        from models.util import set_seed
        set_seed(0)
        self.set_dims(832, 480)
        t0 = time.time()
        rnd = [np.random.uniform(-1, 1, (480, 832, 3)).astype(np.float32) for _ in range(1 + self.chunk)]
        sess, _ = self.pm.start_stream_session("warmup", self.tensor(rnd), self.args.noise_scale)
        for _ in range(4):
            self.pm.run_stream_batch(sess, self.tensor(rnd[:self.chunk]))
        self.torch.cuda.synchronize()
        LOG.info("warmup done in %.1fs", time.time() - t0)

    def system_config(self):
        a = self.args
        return {"model": self.name, "model_variant": "Wan2.1-T2V-1.3B causal DMD v2v (wan_causal_dmd_v2v)",
                "version": self.version, "repo": "https://github.com/chenfengxu714/StreamDiffusionV2",
                "repo_commit": self.commit, "license": "Apache-2.0", "weights_sha256": self.weights_sha256,
                "resolution": {"landscape": [832, 480], "portrait": [480, 832]}, "model_fps": a.model_fps,
                "denoising_steps": a.step, "denoising_step_list": [int(x) for x in self.config.denoising_step_list],
                "noise_scale": a.noise_scale, "chunk_frames": self.chunk, "num_kv_cache": int(self.config.num_kv_cache),
                "vae": "Wan2.1 VAE encode + TAEHV decode" if a.use_taehv else "Wan2.1 VAE (streaming)",
                "gpu": self.gpu, "gpu_count": 1, "dtype": "bfloat16",
                "mode": "single-GPU stream-batch (streamv2v.inference.SingleGPUInferencePipeline)",
                "prompt_policy": "instruction text verbatim as the new prompt; no rewriting or accumulation; a prompt "
                                 "change restarts the stream session (upstream demo semantics)",
                "input_policy": f"center crop to output aspect, resize, subsample to {a.model_fps} fps; newest frames "
                                "win when behind"}

    def run(self, s: edit_stream.EditSession):
        with self.torch.no_grad():                     # grad mode is per thread; this runs in the session thread
            self._run(s)

    def _run(self, s: edit_stream.EditSession):
        from models.util import set_seed
        pm = self.pm
        self.set_dims(s.w, s.h)
        set_seed(s.seed)
        session, active = None, None
        while True:
            got = s.take(lambda p: self.chunk if session is not None and p == active else 1 + self.chunk)
            if got is None:
                break
            frames, prompt, event = got
            t0 = time.perf_counter()
            restart = session is None or prompt != active
            if restart:
                session, video = pm.start_stream_session(prompt, self.tensor(frames), self.args.noise_scale)
                active, outs = prompt, [video]
            else:
                outs = pm.run_stream_batch(session, self.tensor(frames))
            for video in outs:
                s.emit(list((np.clip(video, 0, 1) * 255).round().astype(np.uint8)))
            ms = (time.perf_counter() - t0) * 1e3
            s.stats["chunks"] += 1
            s.stats["chunk_ms"].append(ms)
            s.applied(event, f"prompt = instruction text verbatim; stream session restarted ({ms:.0f} ms)"
                      if restart else "identical to the active prompt")
        if self.args.profile:
            LOG.info("stage fps: encode %.1f decode %.1f", np.mean(pm.encode_fps_list or [0]),
                     np.mean(pm.decode_fps_list or [0]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8775)
    ap.add_argument("--weights", help="unused; weights are linked into the upstream repo (setup.sh)")
    ap.add_argument("--step", type=int, default=2, help="denoising steps (upstream default for 1.3B)")
    ap.add_argument("--noise_scale", type=float, default=0.8, help="upstream demo default")
    ap.add_argument("--model_fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_taehv", action="store_true", help="upstream option: TAEHV decoder instead of the Wan VAE")
    ap.add_argument("--profile", action="store_true", help="log upstream per-stage fps (synchronizes the GPU)")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    edit_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
