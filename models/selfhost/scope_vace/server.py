#!/usr/bin/env python3
"""Daydream Scope, StreamDiffusionV2 pipeline with VACE reference conditioning: self-hosted reference-edit server
(docs/protocol.md).

Upstream: github.com/daydreamlive/scope (pinned in setup.sh). The server loads the `streamdiffusionv2` pipeline
the way Scope's PipelineManager does (Wan2.1-T2V-1.3B + StreamDiffusionV2 causal DMD v2v generator, UMT5 FP8
text encoder, Wan VAE, VACE 1.3B module enabled) and drives it the way Scope's PipelineProcessor does in Video
input mode with "Use Input Video" off (the frontend default): every call gets the next 4 live source frames as
`video` (latent initialisation, noise_scale 0.7 with the noise controller, denoising steps [750, 250]), the text
prompt, and on reference changes `vace_ref_images` (VACE R2V hints, context scale 1.0).

Reference events: one reference slot (models/selfhost/edit_stream.py single-slot rule, as for JoyAI-Video-Edit). An
applied reference becomes the VACE reference image and the prompt REFERENCE_TEMPLATE(<slot scope target>).
--ref_policy decides how the image is passed: `once` (Scope's app: `vace_ref_images` is a one-shot parameter
sent with the first chunk after a change; its effect carries through the KV cache) or `every_chunk` (re-sent
with every chunk; the VACE encoding block's documented contract is "encoded fresh each chunk, application layer
manages reuse"). A prompt change resets the KV cache (Scope's manage_cache behaviour with video input).

    python server.py --port 8767
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edit_stream  # noqa: E402

ROOT = Path(os.environ.get("SCOPE_ROOT", os.path.expanduser(os.environ.get("STREAM_ARENA_MODEL_ROOT", "~/.cache/stream_arena/models")) + "/scope_vace"))
MODELS = ROOT / "weights"
LOG = logging.getLogger("scope_vace")
# same RV2V phrasing as selfhost/joyai_video_edit (the benchmark's reference events carry no text)
REFERENCE_TEMPLATE = ("Replace the {target} in the video with the one from the reference image, matching its colors, "
                      "material and shape. Keep everything else, the motion and the camera unchanged.")


def probe_attention():
    """Scope imports flash_attn unconditionally and prefers it (then SageAttention) over SDPA. Prebuilt FA2 /
    Sage wheels may lack kernels for this GPU: run each once and fall
    back to torch SDPA (Scope's own fallback branch) when a kernel is missing."""
    import torch
    from scope.core.pipelines.wan2_1.modules import attention as A
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    used = "sdpa"
    if A.SAGEATTN_AVAILABLE:
        used = "sageattention"
    elif A.FLASH_ATTN_2_AVAILABLE or A.FLASH_ATTN_3_AVAILABLE:
        try:
            A.flash_attention(q, q, q)
            torch.cuda.synchronize()
            used = "flash_attn_3" if A.FLASH_ATTN_3_AVAILABLE else "flash_attn_2"
        except Exception as exc:  # noqa: BLE001 - no kernel image for this arch
            LOG.warning("flash-attn unusable on this GPU (%s); using torch SDPA", str(exc).splitlines()[0][:200])
            A.FLASH_ATTN_2_AVAILABLE = A.FLASH_ATTN_3_AVAILABLE = False
    return used


class Engine:
    name = "Daydream Scope (StreamDiffusionV2 + VACE)"

    def __init__(self, args):
        os.environ["DAYDREAM_SCOPE_MODELS_DIR"] = str(MODELS)
        sys.path.insert(0, str(ROOT / "repo/src"))
        import torch
        from omegaconf import OmegaConf
        self.torch, self.args = torch, args
        torch.set_grad_enabled(False)
        self.attention = probe_attention()
        from scope.core.pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

        self.fps = args.model_fps
        self.chunk = 4                                       # Scope input_size for streamdiffusionv2 (1 latent)
        config = OmegaConf.create({
            "model_dir": str(MODELS),
            "generator_path": str(MODELS / "StreamDiffusionV2/wan_causal_dmd_v2v/model.pt"),
            "text_encoder_path": str(MODELS / "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"),
            "tokenizer_path": str(MODELS / "Wan2.1-T2V-1.3B/google/umt5-xxl"),
            "vace_path": str(MODELS / "WanVideo_comfy/Wan2_1-VACE_module_1_3B_bf16.safetensors"),
            "vace_context_scale": args.vace_context_scale,
            "height": args.height, "width": args.width, "base_seed": args.seed, "vae_type": "wan",
            "_lora_merge_mode": "permanent_merge"})
        t0 = time.time()
        self.pipe = StreamDiffusionV2Pipeline(config, quantization=None, device=torch.device("cuda"),
                                              dtype=torch.bfloat16)
        assert self.pipe.vace_enabled, "VACE module not loaded"
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"scope@{self.commit[:12]}+streamdiffusionv2+vace"
        self.weights_sha256 = edit_stream.sha256_files([
            MODELS / "StreamDiffusionV2/wan_causal_dmd_v2v/model.pt",
            MODELS / "WanVideo_comfy/Wan2_1-VACE_module_1_3B_bf16.safetensors",
            MODELS / "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors",
            MODELS / "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"], MODELS)
        self.gpu = torch.cuda.get_device_name(0)
        self.refdir = Path(tempfile.mkdtemp(prefix="scope_ref_"))
        LOG.info("loaded %s in %.1fs on %s (attention: %s)", self.version, time.time() - t0, self.gpu, self.attention)
        self.warmup()

    def reference_prompt(self, target: str) -> str:
        return REFERENCE_TEMPLATE.format(target=target)

    def out_size(self, init):
        a = self.args
        return (a.height, a.width) if int(init.get("height") or 0) > int(init.get("width") or 1) else (a.width, a.height)

    def frames(self, frames):
        """float32 HxWx3 in [-1, 1] (edit_stream) -> Scope video packets: uint8 1xHxWx3 tensors."""
        t = self.torch
        return [t.from_numpy(((f + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)).unsqueeze(0) for f in frames]

    def call(self, frames, prompt, width, height, seed, init_cache, ref_path):
        kw = dict(video=self.frames(frames), prompts=[{"text": prompt, "weight": 1.0}], init_cache=init_cache,
                  height=height, width=width, base_seed=seed, vace_context_scale=self.args.vace_context_scale)
        if ref_path is not None:
            kw["vace_ref_images"] = [str(ref_path)]
        video = self.pipe(**kw)["video"]                     # T x H x W x 3 float in [0, 1]
        return (video * 255).round().clamp(0, 255).to(self.torch.uint8).cpu().numpy()

    def warmup(self):
        """Compiles flex attention / kernels for both orientations (first sessions are not slowed)."""
        from PIL import Image
        ref = self.refdir / "warmup.png"
        Image.fromarray(np.random.randint(0, 255, (256, 256, 3), np.uint8)).save(ref)
        t0 = time.time()
        for w, h in ((self.args.width, self.args.height), (self.args.height, self.args.width)):
            rnd = [np.random.uniform(-1, 1, (h, w, 3)).astype(np.float32) for _ in range(self.chunk)]
            for i in range(4):
                self.call(rnd, "a watercolor painting", w, h, 0, i == 0, ref if i < 2 else None)
        self.torch.cuda.synchronize()
        LOG.info("warmup done in %.1fs", time.time() - t0)

    def system_config(self):
        a = self.args
        return {"model": self.name, "pipeline": "scope streamdiffusionv2 (StreamDiffusionV2Pipeline) with VACE",
                "version": self.version, "repo": "https://github.com/daydreamlive/scope", "repo_commit": self.commit,
                "license": "CC BY-NC-SA 4.0 (Scope); weights: see models/selfhost/scope_vace/README.md",
                "weights_sha256": self.weights_sha256,
                "resolution": {"landscape": [a.width, a.height], "portrait": [a.height, a.width]},
                "model_fps": a.model_fps, "chunk_frames": self.chunk, "denoising_steps": [750, 250],
                "noise_scale": 0.7, "noise_controller": True, "vace_context_scale": a.vace_context_scale,
                "vace_use_input_video": False, "ref_policy": a.ref_policy, "manage_cache": True,
                "text_encoder": "UMT5-XXL FP8 e4m3fn (daydreamlive/WanVideo_comfy)", "vae": "Wan2.1 VAE",
                "attention": self.attention, "quantization": "none", "gpu": self.gpu, "gpu_count": 1,
                "dtype": "bfloat16",
                "reference_events": {"slots": 1, "prompt_template": REFERENCE_TEMPLATE,
                                     "unsupported": "state with 2+ slots or empty state (remove/reset to nothing); "
                                                    "previous state kept"},
                "instruction_events": "text used verbatim as the prompt (not registered on this track)",
                "input_policy": f"center crop to output aspect, resize, subsample to {a.model_fps} fps; newest "
                                "frames win when behind"}

    def run(self, s: edit_stream.EditSession):
        with self.torch.no_grad():                           # grad mode is per thread
            self._run(s)

    def _run(self, s: edit_stream.EditSession):
        first, sent_gen = True, -1
        while True:
            got = s.take(lambda p: self.chunk)
            if got is None:
                break
            frames, prompt, event = got
            image, gen = s.taken_ref
            ref_path = None
            if image is not None and (gen != sent_gen or self.args.ref_policy == "every_chunk"):
                ref_path = self.refdir / f"ref_{id(s)}_{gen}.png"
                if not ref_path.exists():
                    image.save(ref_path)
            t0 = time.perf_counter()
            out = self.call(frames, prompt, s.w, s.h, s.seed, first, ref_path)
            s.emit(list(out))
            ms = (time.perf_counter() - t0) * 1e3
            s.stats["chunks"] += 1
            s.stats["chunk_ms"].append(ms)
            if event is not None:
                s.applied(event, f"VACE reference generation {gen}, prompt = reference template ({ms:.0f} ms)"
                          if image is not None and gen != sent_gen else f"prompt updated ({ms:.0f} ms)")
            first, sent_gen = False, gen if image is not None else sent_gen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--width", type=int, default=832, help="StreamDiffusionV2 training resolution 832x480")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--model_fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42, help="Scope default base_seed (sessions use init.seed)")
    ap.add_argument("--vace_context_scale", type=float, default=1.0, help="Scope default")
    ap.add_argument("--ref_policy", choices=["once", "every_chunk"], default="once")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    edit_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
