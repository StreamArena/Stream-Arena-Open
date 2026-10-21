#!/usr/bin/env python3
"""Krea Realtime 14B (krea-ai/realtime-video; Self-Forcing-distilled Wan2.1-T2V-14B, causal 3-latent blocks)
self-hosted server, docs/protocol.md.

One WebSocket session = one live rollout of upstream `release_server.GenerationSession` (the code behind Krea's own
WebSocket demo server), imported unchanged from $KREA_ROOT/repo and driven block by block:
  * every block: KV-cache recomputation over the last 3 clean latent frames (upstream `do_kv_recomp`; the first
    latent is re-encoded from the oldest decoded frame once the window has moved, Krea's error-accumulation fix),
    4 denoising steps (upstream `num_denoising_steps`, README / model card "4 inference steps"; schedule from
    `get_denoising_schedule`, timestep shift 5.0), causal Wan2.1 VAE decode with the decoder cache kept
    (`demo_utils/vae_block3.py`). 832x480, 16 fps (Wan2.1), 12 frames per block (the first block 9).
  * prompt changes use upstream's live mechanism (`interpolate_prompt_embeds`, the demo's "Blend Steps" default 4):
    the text embedding is blended from the current one to the new one over 4 blocks (weights 0, 1/3, 2/3, 1).
    The event is acked `applied` when the first block with a non-zero weight of the new prompt starts.
  * image start (general_interaction_image): upstream `start_frame` (`setup_start_frame`: the image, center-cropped
    to 832x480, is VAE-encoded as the clean context the rollout resumes from).
  * `torch.compile` of the DiT and the VAE decoder as upstream `DO_COMPILE=true`; attention is torch SDPA (upstream
    falls back to it without SageAttention / FlashAttention: neither has a build for compute capability 10.3,
    unsupported GPU architectures).

Load-time changes of this wrapper, none of which alters the computation:
  * the DiT is built from the Wan2.1-T2V-14B config (`from_config`) instead of `from_pretrained`: upstream loads the
    57 GB base checkpoint and immediately overwrites every weight with the Krea checkpoint (strict load either way).
  * the latent / noise buffers are sized for 600 blocks (upstream sizes them by the requested `num_blocks`);
    positions stay small because the KV cache is recomputed with relative positions every block.

Streaming, pacing, prompt policy and acks: selfhost/t2v_stream.py.

    python server.py --port 18820 [--weights $STREAMEVAL_MODEL_ROOT/krea_realtime/weights]
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
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import t2v_stream  # noqa: E402

ROOT = Path(os.environ.get("KREA_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/krea_realtime"))
REPO = ROOT / "repo"
LOG = logging.getLogger("krea_realtime")
W, H, FPS = 832, 480, 16
STEPS = 4
KV_FRAMES = 3            # upstream GenerateParams.kv_cache_num_frames default
BLEND = 4                # upstream prompt blend steps (server and demo default)
NUM_BLOCKS = 600         # 300 s at 16 fps needs ~400 blocks
CKPT = "checkpoints/krea-realtime-video-14b.safetensors"


class Engine:
    name = "Krea-Realtime-14B"
    width, height, fps = W, H, FPS
    image_start = True

    def __init__(self, args):
        self.args = args
        os.environ["MODEL_FOLDER"] = str(Path(args.weights) / "wan_models")
        os.environ.setdefault("DO_COMPILE", "true" if args.compile else "false")
        os.environ.setdefault("DISABLE_SAGEATTENTION", "1")
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / "inductor_cache"))
        os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "triton_cache"))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)                                 # upstream resolves configs/ relative to cwd
        import torch
        import release_server as rs                    # sets grad off, CUDA streams, dynamo recompile limit
        from wan.modules.causal_model import CausalWanModel
        import utils.wan_wrapper as ww

        def from_config(path, **kw):                   # every weight is overwritten by the Krea checkpoint
            return CausalWanModel.from_config(CausalWanModel.load_config(path), **kw)
        ww.CausalWanModel = types.SimpleNamespace(from_pretrained=from_config)    # its only use in wan_wrapper
        self.torch, self.rs = torch, rs
        logging.getLogger().setLevel(logging.INFO)     # release_server sets DEBUG globally
        cfg = rs.load_merge_config("configs/self_forcing_server_14b.yaml")
        cfg.checkpoint_path = str(Path(args.weights) / CKPT)
        self.cfg = cfg
        t0 = time.time()
        self.models = rs.load_all(cfg)
        self.gpu = torch.cuda.get_device_name(0)
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"krea-ai/realtime-video@{self.commit[:12]} krea-realtime-video-14b.safetensors"
        wan = Path(args.weights) / "wan_models/Wan2.1-T2V-1.3B"
        self.weights_sha256 = t2v_stream.sha256_files(
            [Path(cfg.checkpoint_path), wan / "Wan2.1_VAE.pth", wan / "models_t5_umt5-xxl-enc-bf16.pth"],
            Path(args.weights))
        LOG.info("loaded %s in %.0fs on %s (compile=%s), mem %.1f GiB", self.version, time.time() - t0, self.gpu,
                 os.environ["DO_COMPILE"], torch.cuda.memory_allocated() / 2**30)
        self.warmup()

    def system_config(self):
        return {"model": self.name, "model_variant": "Krea Realtime 14B (Self-Forcing distilled Wan2.1-T2V-14B, "
                                                     "causal, 3 latent frames per block)",
                "version": self.version, "repo": "https://github.com/krea-ai/realtime-video", "repo_commit": self.commit,
                "license": "code CC BY-NC-SA 4.0 (repo LICENSE.md); weights Apache-2.0 (HF model card)",
                "weights_sha256": self.weights_sha256, "width": W, "height": H, "fps": FPS,
                "denoising_steps": STEPS, "timestep_shift": 5.0, "guidance_scale": 1.0, "block_latents": 3,
                "block_frames": 12, "kv_cache_num_frames": KV_FRAMES, "kv_recompute": True,
                "prompt_blend_blocks": BLEND, "dtype": "bfloat16 DiT, fp16 VAE",
                "attention": "torch SDPA (portable attention backend)",
                "compile": os.environ.get("DO_COMPILE"), "gpu": self.gpu, "gpu_count": 1,
                "image_start": "upstream start_frame (center-crop to 832x480, VAE-encoded clean context)",
                "prompt_switch": "upstream interpolate_prompt_embeds over 4 blocks; acked at the first block with "
                                 "a non-zero weight of the new prompt",
                "prompt_policy": t2v_stream.PROMPT_POLICY}

    # ---- rollout ---------------------------------------------------------------------------------------
    def _session(self, prompt, seed, image=None):
        rs = self.rs
        params = rs.GenerateParams(prompt=prompt or " ", width=W, height=H, seed=seed, num_blocks=NUM_BLOCKS,
                                   num_denoising_steps=STEPS, kv_cache_num_frames=KV_FRAMES)
        if image is not None:
            from PIL import Image, ImageOps
            params.start_frame = ImageOps.fit(image.convert("RGB"), (W, H), Image.LANCZOS)
        return rs.GenerationSession(params, self.cfg, frame_callback=lambda *a: None, models=self.models)

    def _frames(self, pixels):
        torch = self.torch
        x = pixels[0].float().add_(1.0).mul_(0.5).clamp_(0.0, 1.0).mul_(255).to(torch.uint8)   # T 3 H W
        return list(x.permute(0, 2, 3, 1).contiguous().cpu().numpy())

    def warmup(self):
        """Compile every shape a session meets: text and image start, the growing context of the first blocks,
        steady state and a prompt blend."""
        from PIL import Image
        torch = self.torch
        for image in (None, Image.new("RGB", (W, H), (120, 140, 160))):
            t0 = time.time()
            sess = self._session("warmup: a calm lake at sunrise, gentle ripples", 0, image)
            times = []
            for i in range(7):
                if i == 3:
                    sess.params.prompt = "warmup: a red kite over a green hill"
                    sess.interpolate_prompt_embeds(self.models, sess.params.prompt, BLEND)
                t = time.time()
                n = sess.generate_block(self.models).shape[1]
                torch.cuda.synchronize()
                times.append(round(time.time() - t, 2))
            sess.dispose()
            LOG.info("warmup (%s) %.0fs; block times %s; last block %d frames -> %.1f fps; peak mem %.1f GiB",
                     "image" if image else "text", time.time() - t0, times, n, n / times[-1],
                     torch.cuda.max_memory_allocated() / 2**30)

    def run(self, s: t2v_stream.GenSession):
        torch = self.torch
        image = None
        if s.init.get("image_b64"):
            from PIL import Image
            image = Image.open(io.BytesIO(base64.b64decode(s.init["image_b64"])))
        deferred = []
        sess = None
        try:
            with torch.inference_mode():               # grad mode is per thread; this runs in the session thread
                while True:
                    got = s.next_block()
                    if got is None:
                        return
                    prompt, changed, events = got
                    if deferred:                       # this block is the first with weight > 0 of the new prompt
                        s.applied(deferred, f"upstream prompt blend over {BLEND} blocks (weights 0, 1/3, 2/3, 1)")
                        deferred = []
                    if sess is None:                   # first block: the prompt (with any early instruction) is the
                        sess = self._session(prompt, s.seed, image)     # session prompt
                        s.applied(events, "session prompt")
                    elif changed:
                        sess.params.prompt = prompt
                        sess.interpolate_prompt_embeds(self.models, prompt, BLEND)
                        deferred = events
                    else:
                        s.applied(events, "prompt unchanged")
                    if sess.block_idx >= sess.num_blocks:
                        raise RuntimeError(f"rollout reached {NUM_BLOCKS} blocks")
                    s.emit(self._frames(sess.generate_block(self.models)))
        finally:
            for eid, kind in deferred:
                s._ack(eid, "error", "session ended before the blended prompt reached a block")
            if sess is not None:
                sess.dispose()
            del sess
            torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18820)
    ap.add_argument("--weights", default=str(ROOT / "weights"))
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--bench", type=int, default=0, help="offline: N blocks text start, print block times, exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    engine = Engine(args)
    if args.bench:
        torch = engine.torch
        with torch.inference_mode():
            sess = engine._session("A red kite flies over a green hill under a blue sky.", 1)
            times = []
            for _ in range(args.bench):
                t = time.time()
                sess.generate_block(engine.models)
                torch.cuda.synchronize()
                times.append(round(time.time() - t, 3))
        steady = sorted(times[3:] or times)
        print({"block_s": times, "median_block_s": steady[len(steady) // 2],
               "gen_fps": round(12 / steady[len(steady) // 2], 2)}, flush=True)
        return
    t2v_stream.serve(engine, args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
