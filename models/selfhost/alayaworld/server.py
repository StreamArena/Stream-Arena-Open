#!/usr/bin/env python3
"""AlayaWorld (AlayaLab; interactive long-horizon world model on LTX-2.3) self-hosted server, docs/protocol.md.

One WebSocket session = one live rollout through upstream's own interactive adapter `reactor/alayaworld.py`
(`AlayaWorld`, the class behind AlayaLab's browser demo), imported unchanged from $ALAYA_ROOT/repo and driven turn by
turn without the Reactor transport:
  * load: `AlayaWorld.load(reactor/alayaworld.yaml)` = upstream serving settings (da3 inference path, 960x544, 24 fps,
    32-frame chunks of 4 latents, 4 sampling steps, torch.compile default, FlashAttention 4, flex attention, TAEHV
    decoder for spatial-memory pixels, spatial memory 320 / 160 frames, one warm-up turn).
  * session start = the adapter's upload path: `_reset_rollout(prompt, seed, UploadedFile)` (image center-cropped to
    960x544 over the upload template's camera calibration, `FlashAlayaPipeline.initialize_cache`).
  * every chunk = `_generate_chunk(prompt, 0, 0, 0, 0, 0, 0)`: generate / finalize / decode one turn; the prompt is
    re-encoded when it changes (the demo's mid-rollout prompt switch, "takes effect on the next turn").
  * camera: the six camera axes are held at zero (a static camera); camera_control text goes into the prompt like any
    instruction (block prompt '<init prompt>\\n<latest instruction>', selfhost/t2v_stream.py).
Image start only (the released interactive path is image-to-video), so only the i2v track is registered.

    python server.py --port 18850 [--host 127.0.0.1] [--weights $ALAYA_ROOT/checkpoints] [--jpeg_quality 90]
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
import t2v_stream  # noqa: E402

ROOT = Path(os.environ.get("ALAYA_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/alayaworld"))
REPO = ROOT / "repo"
LOG = logging.getLogger("alayaworld")
W, H, FPS = 960, 544, 24


def _uploaded(UploadedFile, raw: bytes, mime: str):
    for kw in ({"name": "init_image", "data": raw, "content_type": mime}, {"name": "init_image", "data": raw}):
        try:
            return UploadedFile(**kw)
        except TypeError:
            continue
    obj = object.__new__(UploadedFile)
    obj.__dict__.update(name="init_image", data=raw, content_type=mime)
    return obj


class Engine:
    name = "AlayaWorld"
    width, height, fps = W, H, FPS
    image_start = True

    def __init__(self, args):
        self.args = args
        os.environ.setdefault("REACTOR_WEIGHTS_PATH", str(ROOT / "checkpoints"))
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")           # assets are prepared by setup.sh
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / "inductor_cache"))
        os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "triton_cache"))
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)
        import torch
        from reactor_runtime import UploadedFile
        import reactor.alayaworld as adapter
        self.torch, self.UploadedFile = torch, UploadedFile
        torch.set_grad_enabled(False)
        t0 = time.time()
        self.aw = adapter.AlayaWorld()
        self.aw.load(REPO / "reactor/alayaworld.yaml")
        cfg = self.aw._config
        self.gpu = torch.cuda.get_device_name(0)
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"AlayaLab/AlayaWorld@{self.commit[:12]} merged_infer.safetensors@{cfg.model.revision[:12]}"
        self.cfg_summary = {"compile": cfg.compile_mode, "attention_backend": cfg.attention_backend,
                            "flex_attention": cfg.flex_attention, "bank_taehv": cfg.bank_taehv, "ttc": cfg.ttc,
                            "max_spatial_frames": cfg.max_spatial_frames,
                            "recent_spatial_frames": cfg.recent_spatial_frames,
                            "decode_overlap_latents": cfg.decode_overlap_latents,
                            "chunk_latents": self.aw._chunk_latents, "history_latents": self.aw._history_latents}
        self.weights_sha256 = t2v_stream.sha256_files([cfg.model.path], ROOT / "checkpoints")
        self.mime = "image/png"
        LOG.info("loaded %s in %.0fs on %s (incl. upstream warm-up turn), mem %.1f GiB; %s", self.version,
                 time.time() - t0, self.gpu, torch.cuda.memory_allocated() / 2**30, json.dumps(self.cfg_summary))
        self.warmup()

    def system_config(self):
        return {"model": self.name, "model_variant": "AlayaWorld da3 interactive path (reactor/alayaworld.py)",
                "version": self.version, "repo": "https://github.com/AlayaLab/AlayaWorld", "repo_commit": self.commit,
                "license": "LTX-2 Community License Agreement (repo LICENSE, HF model card); Gemma 3 text encoder: Gemma "
                           "Terms of Use",
                "weights_sha256": self.weights_sha256, "width": W, "height": H, "fps": FPS, "chunk_frames": 32,
                "sampling_steps": 4, **self.cfg_summary, "camera": "six axes held at zero (static camera)",
                "gemma_source": "google/gemma-3-12b-it-qat-q4_0-unquantized", "gpu": self.gpu,
                "gpu_count": 1, "prompt_switch": "adapter _generate_chunk: caption re-encoded from the next turn",
                "prompt_policy": t2v_stream.PROMPT_POLICY}

    def _turn(self, prompt):
        frames = self.aw._generate_chunk(prompt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # uint8 T H W C
        return list(frames)

    def warmup(self):
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (1280, 720), (120, 140, 160)).save(buf, "PNG")
        t0 = time.time()
        self.aw._reset_rollout("A calm lake at sunrise.", 0, _uploaded(self.UploadedFile, buf.getvalue(), "image/png"))
        times = []
        for p in ("A calm lake at sunrise.", "A calm lake at sunrise.\nA boat appears.", "A calm lake at sunrise."):
            t = time.time()
            n = len(self._turn(p))
            times.append(round(time.time() - t, 2))
        self.aw._cache = None
        LOG.info("warmup %.0fs; turn times %s; %d frames per turn -> %.1f fps; peak mem %.1f GiB", time.time() - t0,
                 times, n, n / times[-1], self.torch.cuda.max_memory_allocated() / 2**30)

    def run(self, s: t2v_stream.GenSession):
        raw = base64.b64decode(s.init["image_b64"])
        mime = s.init.get("image_mime") or "image/png"
        try:
            with self.torch.inference_mode():
                self.aw._reset_rollout(s.prompt() or "Continue the visual scene shown in the reference image.",
                                       s.seed, _uploaded(self.UploadedFile, raw, mime))
                while True:
                    got = s.next_block()
                    if got is None:
                        return
                    prompt, _, events = got
                    s.applied(events, "prompt of the next turn (caption re-encoded)")
                    s.emit(self._turn(prompt))
        finally:
            self.aw._cache = None
            self.aw._camera = None
            self.torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18850)
    ap.add_argument("--weights", default=str(ROOT / "checkpoints"))
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    os.environ["REACTOR_WEIGHTS_PATH"] = args.weights
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t2v_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
