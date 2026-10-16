#!/usr/bin/env python3
"""Krea Realtime 14B, streaming video-to-video (krea-ai/realtime-video webcam mode): self-hosted instruction-edit
server, docs/protocol.md.

Same weights and upstream code as selfhost/krea_realtime (general track); this server drives upstream
`release_server.GenerationSession` in its **webcam mode** (`webcam_mode=True`), the live V2V path of Krea's own
demo server, imported unchanged from $KREA_ROOT/repo:

  * every block takes the newest live source frames (9 for the first block of a session, then 12 = 3 latents,
    upstream `process_webcam_frames`), encodes them with the causal Wan2.1 VAE encoder whose cache is kept across
    blocks (`encode_video_latent(..., stream=idx > 0)`), noises the latents to the first step of the strength
    schedule (`latents * (1 - s) + noise * s`) and denoises them with the KV-cache-recomputed causal 14B DiT;
  * denoising strength 0.8 (the demo's "Denoising Strength" default for webcam mode), schedule
    `get_denoising_schedule(timesteps, 0.8, steps=4)`; 4 steps as the model card / README speed setting and the
    scored general entry (the demo slider and `GenerateParams` default to 5);
  * instructions: the text verbatim is the prompt. The first instruction starts the session; later ones use the
    webcam demo's live prompt change (`interpolate_prompt_embeds`, "Blend Steps" default 4: weights 0, 1/3, 2/3,
    1 over the next 4 blocks), acked `applied` at the first block with a non-zero weight of the new prompt;
  * 832x480 only (upstream re-encodes the first context frame at a hard-coded 480x832 when the KV window moves):
    portrait sources are center-cropped to landscape; `torch.compile` and SDPA attention as models/selfhost/krea_realtime.

The source is passed through until the first instruction. Streaming, pacing, frame dropping and acks:
models/selfhost/edit_stream.py. Load-time changes shared with selfhost/krea_realtime (DiT from config, the Krea checkpoint
overwrites every weight; latent buffers for 600 blocks) do not alter the computation.

    python server.py --port 8771 [--weights $STREAM_ARENA_MODEL_ROOT/krea_realtime/weights]
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edit_stream  # noqa: E402
from t2v_stream import sha256_files  # noqa: E402

ROOT = Path(os.environ.get("KREA_ROOT", os.path.expanduser(os.environ.get("STREAM_ARENA_MODEL_ROOT", "~/.cache/stream_arena/models")) + "/krea_realtime"))
REPO = ROOT / "repo"
LOG = logging.getLogger("krea_v2v")
W, H, FPS = 832, 480, 16
STEPS = 4
STRENGTH = 0.8           # release_demo.html denoisingStrength default
KV_FRAMES = 3            # upstream GenerateParams.kv_cache_num_frames default
BLEND = 4                # upstream "Blend Steps" default
NUM_BLOCKS = 600
FIRST_FRAMES, NEXT_FRAMES = 9, 12    # upstream process_webcam_frames
CKPT = "checkpoints/krea-realtime-video-14b.safetensors"


class Engine:
    name = "Krea-Realtime-14B-V2V"
    fps = FPS

    def __init__(self, args):
        self.args = args
        os.environ["MODEL_FOLDER"] = str(Path(args.weights) / "wan_models")
        os.environ.setdefault("DO_COMPILE", "true" if args.compile else "false")
        os.environ.setdefault("DISABLE_SAGEATTENTION", "1")
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / "inductor_cache"))
        os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "triton_cache"))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)
        import torch
        import release_server as rs
        from wan.modules.causal_model import CausalWanModel
        import utils.wan_wrapper as ww

        def from_config(path, **kw):
            return CausalWanModel.from_config(CausalWanModel.load_config(path), **kw)
        ww.CausalWanModel = types.SimpleNamespace(from_pretrained=from_config)
        self.torch, self.rs = torch, rs
        logging.getLogger().setLevel(logging.INFO)
        cfg = rs.load_merge_config("configs/self_forcing_server_14b.yaml")
        cfg.checkpoint_path = str(Path(args.weights) / CKPT)
        self.cfg = cfg
        t0 = time.time()
        self.models = rs.load_all(cfg)
        self.gpu = torch.cuda.get_device_name(0)
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"krea-ai/realtime-video@{self.commit[:12]} krea-realtime-video-14b.safetensors (webcam V2V)"
        wan = Path(args.weights) / "wan_models/Wan2.1-T2V-1.3B"
        self.weights_sha256 = sha256_files(
            [Path(cfg.checkpoint_path), wan / "Wan2.1_VAE.pth", wan / "models_t5_umt5-xxl-enc-bf16.pth"],
            Path(args.weights))
        LOG.info("loaded %s in %.0fs on %s (compile=%s)", self.version, time.time() - t0, self.gpu,
                 os.environ["DO_COMPILE"])
        self.warmup()

    def out_size(self, init):
        return W, H

    def _session(self, prompt, seed):
        params = self.rs.GenerateParams(prompt=prompt, width=W, height=H, seed=seed, num_blocks=NUM_BLOCKS,
                                        num_denoising_steps=STEPS, kv_cache_num_frames=KV_FRAMES,
                                        strength=STRENGTH, webcam_mode=True, webcam_fps=FPS)
        sess = self.rs.GenerationSession(params, self.cfg, frame_callback=lambda *a: None, models=self.models)
        self.steps = [round(float(t), 1) for t in sess.denoising_step_list]
        return sess

    def _push(self, sess, frames):
        """float32 HxWx3 in [-1, 1] -> the session's webcam frame queue (fp16 3xHxW on the GPU, as push_frame)."""
        torch = self.torch
        for f in frames:
            sess.frame_queue.put(torch.from_numpy(f).permute(2, 0, 1).to("cuda", torch.float16))

    def _frames(self, pixels):
        torch = self.torch
        x = pixels[0].float().add_(1.0).mul_(0.5).clamp_(0.0, 1.0).mul_(255).to(torch.uint8)
        return list(x.permute(0, 2, 3, 1).contiguous().cpu().numpy())

    def warmup(self):
        torch = self.torch
        t0 = time.time()
        with torch.inference_mode():
            sess = self._session("warmup: a watercolor painting", 0)
            times = []
            for i in range(7):
                if i == 3:
                    sess.params.prompt = "warmup: a pencil sketch"
                    sess.interpolate_prompt_embeds(self.models, sess.params.prompt, BLEND)
                n = FIRST_FRAMES if i == 0 else NEXT_FRAMES
                self._push(sess, [np.random.uniform(-1, 1, (H, W, 3)).astype(np.float32) for _ in range(n)])
                t = time.time()
                sess.generate_block(self.models)
                torch.cuda.synchronize()
                times.append(round(time.time() - t, 2))
            sess.dispose()
        LOG.info("warmup %.0fs; block times %s; steps %s", time.time() - t0, times, self.steps)

    def system_config(self):
        return {"model": self.name, "model_variant": "Krea Realtime 14B (Self-Forcing distilled Wan2.1-T2V-14B), "
                                                     "upstream webcam streaming V2V",
                "version": self.version, "repo": "https://github.com/krea-ai/realtime-video", "repo_commit": self.commit,
                "license": "code CC BY-NC-SA 4.0 (repo LICENSE.md); weights Apache-2.0 (HF model card)",
                "weights_sha256": self.weights_sha256, "resolution": [W, H], "model_fps": FPS,
                "denoising_steps": STEPS, "denoising_strength": STRENGTH, "denoising_step_list": self.steps,
                "timestep_shift": 5.0, "guidance_scale": 1.0, "block_latents": 3,
                "block_frames": {"first": FIRST_FRAMES, "next": NEXT_FRAMES}, "kv_cache_num_frames": KV_FRAMES,
                "kv_recompute": True, "prompt_blend_blocks": BLEND, "dtype": "bfloat16 DiT, fp16 VAE",
                "attention": "torch SDPA (portable attention backend)",
                "compile": os.environ.get("DO_COMPILE"), "gpu": self.gpu, "gpu_count": 1,
                "prompt_policy": "instruction text verbatim as the prompt; the first instruction starts the webcam "
                                 "session, later ones blend the prompt over 4 blocks (upstream live prompt change)",
                "input_policy": f"center crop to 832x480 (portrait sources too), subsample to {FPS} fps; newest "
                                "frames win when behind"}

    def run(self, s: edit_stream.EditSession):
        torch = self.torch
        sess, active, deferred = None, None, None
        try:
            with torch.inference_mode():
                while True:
                    got = s.take(lambda p: NEXT_FRAMES if sess is not None else FIRST_FRAMES)
                    if got is None:
                        return
                    frames, prompt, event = got
                    t0 = time.perf_counter()
                    if deferred is not None:             # this block has the first non-zero weight
                        s.applied(deferred, f"upstream prompt blend over {BLEND} blocks (weights 0, 1/3, 2/3, 1)")
                        deferred = None
                    note = None
                    if sess is None:
                        sess, active = self._session(prompt, s.seed), prompt
                        note = "webcam session started with the instruction as prompt"
                    elif prompt != active:
                        sess.params.prompt, active = prompt, prompt
                        sess.interpolate_prompt_embeds(self.models, prompt, BLEND)
                        deferred, event = event, None
                    elif event is not None:
                        note = "identical to the active prompt"
                    if sess.block_idx >= sess.num_blocks:
                        raise RuntimeError(f"rollout reached {NUM_BLOCKS} blocks")
                    self._push(sess, frames)
                    out = self._frames(sess.generate_block(self.models))
                    s.emit(out)
                    ms = (time.perf_counter() - t0) * 1e3
                    s.stats["chunks"] += 1
                    s.stats["chunk_ms"].append(ms)
                    if event is not None:
                        s.applied(event, f"{note} ({ms:.0f} ms)")
        finally:
            if deferred is not None:
                s._ack(deferred, "error", "session ended before the blended prompt reached a block")
            if sess is not None:
                sess.dispose()
            sess = None
            torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--weights", default=str(ROOT / "weights"))
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    edit_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
