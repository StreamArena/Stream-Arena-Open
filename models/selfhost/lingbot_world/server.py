#!/usr/bin/env python3
"""LingBot-World 2.0 1.3B causal-fast (Robbyant, I2V world model on Wan 1.3B, 4-step, camera poses) self-hosted
server, docs/protocol.md.

One WebSocket session = one live autoregressive rollout of upstream `WanI2VCausal._generate_causal_fast`
(Robbyant/lingbot-world-v2, `generate.py` defaults for `--task i2v-1.3B --size 480*832`: chunk 4 latents,
timesteps index [0, 250, 500, 750] of the shift-5.0 schedule, no CFG; `run_fast.sh`: local attention 18 latents with
a 6-latent sink), unrolled so that each chunk (16 frames at 16 fps; the first 13) is generated, decoded and sent before
the next one starts: 4 denoising steps -> clean-context pass (t = 0) into the KV cache -> causal Wan2.1 VAE decode of
just these latents (decoder feature cache kept across chunks; the same per-latent loop as upstream's full decode).

Image start (required: the model is image-to-video only, so only the image + text track is registered): the opening
image is center-cropped to 832:480 and resized (bicubic, as upstream) to 832x480. As upstream, the i2v condition is
mask (latent 0 = image) + VAE encoding of [image, zero frames]; the encoding of the zero frames converges to a steady state
(max |difference| between consecutive latents 1.3 at latent 2, 0.003 at latent 15, decaying geometrically), so it is
computed once for 32 latents and its last latent is repeated for later chunks (the warmup logs the differences).

Long sessions: RoPE positions are absolute frame indices and upstream builds the table for 1024 latent frames
(4093 frames, 256 s); 300 s sessions need 1200, so the same RoPE formula is evaluated for 4096 positions (rows below
1024 are identical). Everything else (rolling KV cache with sink) is upstream's own long-rollout mechanism.

Prompt: the chunk prompt is "<init prompt>\\n<latest instruction>" (selfhost/t2v_stream.py policy). A change is
re-encoded by the umT5 encoder and the cross-attention cache recomputed at the next chunk (upstream's per-generate
cross-attention gate); the self-attention KV cache keeps the past as generated.

Camera: native input is a camera pose per latent (Plucker embedding of the framewise relative pose; upstream
normalises translations so that a constant-speed move is a unit step per latent). Default: static camera (identity
poses), as in models/selfhost/hy_worldplay. `camera_control` text is mapped by the same keyword map as selfhost/hy_worldplay
(translate forward/back/left/right, yaw, pitch, orbit = strafe + counter-yaw, zoom-in ~ dolly-in, hold) onto a unit
translation step / 3 degrees per latent, held until the next camera event; unmappable text is acked `unsupported`.
Intrinsics: upstream examples 00/02/05 (fx 502.9, fy 503.1, cx 415.8, cy 239.8 at 832x480).

    python server.py --port 18875
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import t2v_stream  # noqa: E402

ROOT = Path(os.environ.get("LINGBOT_WORLD_ROOT",
                           os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/lingbot_world"))
LOG = logging.getLogger("lingbot_world")
W, H, FPS = 832, 480, 16
CHUNK = 4                       # latents per chunk (generate.py --chunk_size default)
LOCAL_ATTN, SINK = 18, 6        # run_fast.sh
STEPS_INDEX = [0, 250, 500, 750]
SHIFT = 5.0
Y_LATENTS = 32                  # latents of the [image, zeros] encoding; later latents repeat the last one
ROPE_POSITIONS = 4096
KS = (502.9116, 503.10812, 415.77786, 239.77779)
DIT = "lingbot-world-v2-1.3b-causal-fast"
ASSETS = "Wan2.1-T2V-1.3B"
STEP_T, STEP_DEG = 1.0, 3.0     # translation (normalised units) / rotation (degrees) per latent

# ---- camera language (keyword map of selfhost/hy_worldplay, motions as per-latent pose steps) ----------------------
_DIR = r"(?:\s+\w+){0,3}?\s*(?:to(?:ward)?s?\s+the\s+)?"


def camera_motion(text: str):
    """Free-text camera instruction -> (motion dict {forward, right, yaw, pitch} per latent, label) or None."""
    t = text.lower()
    deg = STEP_DEG
    if re.search(r"\b(orbit\w*|arc|arcs|arcing|circl\w+|revolv\w+)\b", t):
        ccw = re.search(r"counter-?clockwise|anti-?clockwise|rightward|\bright\b", t) and "clockwise" not in \
            re.sub(r"(counter|anti)-?clockwise", "", t)
        if re.search(r"counter-?clockwise|anti-?clockwise", t) or (ccw and "arc" in t):
            return {"right": STEP_T, "yaw": -deg}, "orbit counter-clockwise (strafe right + yaw left)"
        return {"right": -STEP_T, "yaw": deg}, "orbit clockwise (strafe left + yaw right)"
    m = re.search(r"\b(pan|turn|rotate|yaw|swing|sweep|look|swivel)\w*" + _DIR + r"(left|right)\b", t)
    if m:
        return ({"yaw": -deg}, "yaw left") if m.group(2) == "left" else ({"yaw": deg}, "yaw right")
    m = re.search(r"\b(tilt|look|crane|pitch|angle)\w*" + _DIR + r"(up|upward|down|downward)\b", t)
    if m:
        up = m.group(2).startswith("up")
        return ({"pitch": deg}, "pitch up") if up else ({"pitch": -deg}, "pitch down")
    m = re.search(r"\b(truck|strafe|slide|move|dolly|step|drift|shift)\w*" + _DIR + r"(left|right)\b", t)
    if m:
        return ({"right": -STEP_T}, "strafe left") if m.group(2) == "left" else ({"right": STEP_T}, "strafe right")
    if re.search(r"\b(pull\s+(the\s+camera\s+)?back|dolly\s+(out|back)|move\s+back|retreat|back\s+away|zoom\s+out"
                 r"|backward)", t):
        return {"forward": -STEP_T}, "move backward"
    if re.search(r"\b(zoom\s+in|close\s+in|focus\s+tightly|tighten)", t):
        return {"forward": STEP_T}, "move forward (zoom approximated by dolly-in)"
    if re.search(r"\b(push|dolly|advance|approach|forward|fly|glide|move\s+(in|closer|toward)|through|into|enter"
                 r"|follow|track|tracking|chase|walk|continue\s+(along|down|past))", t):
        return {"forward": STEP_T}, "move forward"
    if re.search(r"\bpan\b", t):
        return {"yaw": deg}, "yaw right (pan without direction)"
    if re.search(r"^\s*(hold|settle|stop|freeze|pause|keep\s+(the\s+)?(camera|frame|shot)\s+(still|steady|static|locked))"
                 r"|\b(static|locked-off|stationary)\s+(camera|shot)", t):
        return {}, "hold (no motion)"
    return None


def step_pose(motion: dict):
    """Framewise relative camera-to-world step (OpenCV: x right, y down, z forward) for one latent."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    yaw, pitch = motion.get("yaw", 0.0), motion.get("pitch", 0.0)
    if yaw or pitch:   # about the camera y axis (down): positive turns the view right; about x: positive tilts it up
        T[:3, :3] = Rotation.from_euler("yx", [yaw, pitch], degrees=True).as_matrix()
    T[:3, 3] = [motion.get("right", 0.0), 0.0, motion.get("forward", 0.0)]
    return T


def center_crop(img, w, h):
    img = img.convert("RGB")
    sw, sh = img.size
    if sw * h > sh * w:
        nw = round(sh * w / h)
        img = img.crop(((sw - nw) // 2, 0, (sw - nw) // 2 + nw, sh))
    else:
        nh = round(sw * h / w)
        img = img.crop((0, (sh - nh) // 2, sw, (sh - nh) // 2 + nh))
    return img


class Session(t2v_stream.GenSession):
    """GenSession plus camera events: camera_control is mapped onto the pose input instead of prompt text."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cam_events = []
        self.motion, self.motion_label = {}, "hold"

    def on_text(self, text, event_id, kind):
        if kind != "camera_control":
            return super().on_text(text, event_id, kind)
        mapped = camera_motion(text)
        if mapped is None:
            asyncio.ensure_future(self.conn.ack(event_id, "unsupported", "camera_control: no mapping onto the "
                                                "model's camera-pose input (translate / yaw / pitch)"))
            return
        with self.cv:
            self.cam_events.append((event_id, mapped))
            self.cv.notify_all()

    def take_camera(self):
        with self.cv:
            ev, self.cam_events = self.cam_events, []
        for i, (eid, (motion, label)) in enumerate(ev):
            note = f"camera pose input from block {self.blocks} (output frame {self.frames_made}): {label}"
            if i < len(ev) - 1:
                note += "; replaced by a later camera instruction released before this block started"
            self._ack(eid, "applied", note)
        if ev:
            self.motion, self.motion_label = ev[-1][1]
        return bool(ev)

    def drop_camera(self):
        with self.cv:
            ev, self.cam_events = self.cam_events, []
        for eid, _ in ev:
            self._ack(eid, "error", "session ended before the camera instruction took effect")


class Engine:
    name = "LingBot-World 2.0 1.3B"
    width, height, fps = W, H, FPS
    image_start = True

    def __init__(self, args):
        repo = ROOT / "repo"
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(HERE.parent / "_shims"))      # flash_attn API on torch SDPA (see README)
        import numpy as np
        import torch
        from einops import rearrange
        from wan.configs import WAN_CONFIGS
        from wan.image2video import WanI2VCausal
        from wan.modules.model import rope_params
        from wan.utils.cam_utils import get_plucker_embeddings

        self.np, self.torch, self.rearrange, self.plucker = np, torch, rearrange, get_plucker_embeddings
        torch.set_grad_enabled(False)
        t0 = time.time()
        cfg = WAN_CONFIGS["i2v-1.3B"]
        self.cfg = cfg
        logging.getLogger().setLevel(logging.INFO)
        pipe = WanI2VCausal(config=cfg, checkpoint_dir=str(ROOT / "weights" / DIT), device_id=0, rank=0,
                            t5_cpu=False, local_attn_size=LOCAL_ATTN, sink_size=SINK, infer_mode="causal_fast",
                            assets_dir=str(ROOT / "weights" / ASSETS))
        pipe.text_encoder.model.to("cuda")
        m = pipe.model
        d = m.dim // m.num_heads
        m.freqs = torch.cat([rope_params(ROPE_POSITIONS, d - 4 * (d // 6)), rope_params(ROPE_POSITIONS, 2 * (d // 6)),
                             rope_params(ROPE_POSITIONS, 2 * (d // 6))], dim=1).to("cuda")
        pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=SHIFT)
        self.timesteps = pipe.scheduler.timesteps[STEPS_INDEX]
        self.pipe = pipe
        self.lat_h, self.lat_w = H // 8, W // 8
        self.frame_seqlen = self.lat_h * self.lat_w // 4
        self.kv_size = self.frame_seqlen * LOCAL_ATTN
        Ks = torch.tensor(KS, dtype=torch.float32)
        self.Ks = Ks[None].repeat(CHUNK, 1).cuda()
        self._t5 = {}
        self.commit = (ROOT / "repo_commit.txt").read_text().strip()
        self.version = f"lingbot-world-v2@{self.commit[:12]} {DIT}"
        dit = ROOT / "weights" / DIT
        self.weights_sha256 = t2v_stream.sha256_files(
            sorted(dit.glob("model-*.safetensors")) + [ROOT / "weights" / ASSETS / "models_t5_umt5-xxl-enc-bf16.pth",
                                                      ROOT / "weights" / ASSETS / "Wan2.1_VAE.pth"], ROOT / "weights")
        self.gpu = torch.cuda.get_device_name(0)
        LOG.info("loaded %s in %.1fs on %s; timesteps %s", self.version, time.time() - t0, self.gpu,
                 [round(float(t), 1) for t in self.timesteps])
        self.warmup()

    def system_config(self):
        return {"model": self.name,
                "model_variant": "LingBot-World 2.0 (LingBot-World-Infinity) 1.3B causal-fast, I2V with camera poses",
                "version": self.version, "repo": "https://github.com/Robbyant/lingbot-world-v2",
                "repo_commit": self.commit, "license": "CC BY-NC-SA 4.0", "weights_sha256": self.weights_sha256,
                "width": W, "height": H, "fps": FPS, "denoising_steps": len(STEPS_INDEX),
                "denoising_timesteps": [round(float(t), 1) for t in self.timesteps], "timestep_shift": SHIFT,
                "chunk_latents": CHUNK, "chunk_frames": CHUNK * 4, "local_attn_latents": LOCAL_ATTN,
                "sink_latents": SINK, "rope_positions": ROPE_POSITIONS, "cfg": "none (distilled)",
                "image_condition": f"mask + VAE([image, zeros]) for {Y_LATENTS} latents, last latent repeated",
                "camera": "static identity pose by default; camera_control keyword-mapped (hy_worldplay map) onto "
                          f"per-latent steps: translation {STEP_T} (normalised), rotation {STEP_DEG} deg",
                "intrinsics_fx_fy_cx_cy": KS,
                "vae": "Wan2.1 VAE, causal cached decode per chunk", "dtype": "bfloat16 DiT, float32 VAE",
                "attention": "torch SDPA (flash_attn shim)", "gpu": self.gpu, "gpu_count": 1,
                "prompt_switch": "umT5 re-encode and cross-attention cache recomputed at the next chunk; "
                                 "self-attention KV cache kept",
                "prompt_policy": t2v_stream.PROMPT_POLICY}

    # ---- pieces ------------------------------------------------------------------------------------------
    def _encode_text(self, prompt):
        if prompt not in self._t5:
            self._t5[prompt] = self.pipe.text_encoder([prompt], self.torch.device("cuda"))
        return self._t5[prompt]

    def _image_condition(self, image):
        """[20, Y_LATENTS, h, w]: mask (4 ch) + VAE latents of [image, zeros] (upstream construction)."""
        torch, F = self.torch, 1 + 4 * (Y_LATENTS - 1)
        import torchvision.transforms.functional as TF
        img = TF.to_tensor(center_crop(image, W, H)).sub_(0.5).div_(0.5)
        img = torch.nn.functional.interpolate(img[None], size=(H, W), mode="bicubic").transpose(0, 1)
        y = self.pipe.vae.encode([torch.concat([img, torch.zeros(3, F - 1, H, W)], dim=1).cuda()])[0]
        msk = torch.zeros(4, Y_LATENTS, self.lat_h, self.lat_w, device="cuda")
        msk[:, 0] = 1
        return torch.concat([msk, y])

    def _plucker(self, poses):
        torch, np = self.torch, self.np
        c2ws = torch.from_numpy(np.stack(poses)).float().cuda()
        emb = self.plucker(c2ws, self.Ks[:len(poses)], H, W)
        emb = self.rearrange(emb, "f (h c1) (w c2) c -> (f h w) (c c1 c2)", c1=8, c2=8)[None]
        emb = self.rearrange(emb, "b (f h w) c -> b c f h w", f=len(poses), h=self.lat_h, w=self.lat_w)
        return emb.to(self.cfg.param_dtype)

    def _reset(self):
        p, torch = self.pipe, self.torch
        mc = p.model.config
        hd = mc.dim // mc.num_heads
        self.kv = p._initialize_self_kv_cache(num_layers=mc.num_layers, shape=[1, self.kv_size, mc.num_heads, hd],
                                              dtype=torch.bfloat16, device="cuda")
        self.xkv = p._initialize_crossattn_cache(num_layers=mc.num_layers, shape=[1, 512, mc.num_heads, hd],
                                                 dtype=torch.bfloat16, device="cuda")
        self.cross_new = True

    def _decode(self, z, first):
        """Causal decode of latents z [16, T, h, w] with the decoder cache kept across chunks (upstream decode loop)."""
        torch, vae = self.torch, self.pipe.vae
        m = vae.model
        with torch.amp.autocast("cuda", dtype=vae.dtype):
            if first:
                m.clear_cache()
            z = z[None] / vae.scale[1].view(1, 16, 1, 1, 1) + vae.scale[0].view(1, 16, 1, 1, 1)
            x = m.conv2(z)
            out = []
            for i in range(x.shape[2]):
                m._conv_idx = [0]
                out.append(m.decoder(x[:, :, i:i + 1], feat_cache=m._feat_map, feat_idx=m._conv_idx))
            v = torch.cat(out, 2).float().clamp_(-1, 1)[0]                     # C T H W
        v = (v * 0.5 + 0.5).permute(1, 2, 3, 0)
        return (v * 255).round().to(torch.uint8).cpu().numpy()

    def _chunk(self, context, y_all, poses, start, gen):
        torch, p = self.torch, self.pipe
        ycond = torch.stack([y_all[:, min(start + i, Y_LATENTS - 1)] for i in range(CHUNK)], dim=1)
        x = torch.randn(16, CHUNK, self.lat_h, self.lat_w, dtype=torch.float32, device="cuda", generator=gen)
        kw = {"context": [context[0]], "seq_len": CHUNK * self.frame_seqlen, "y": [ycond],
              "dit_cond_dict": {"c2ws_plucker_emb": self._plucker(poses).chunk(1, dim=0)},
              "kv_cache": self.kv, "crossattn_cache": self.xkv, "current_start": start * self.frame_seqlen,
              "max_attention_size": self.kv_size, "frame_seqlen": self.frame_seqlen}
        with torch.amp.autocast("cuda", dtype=self.cfg.param_dtype):
            for i, t in enumerate(self.timesteps):
                pred = p.model(x=[x], t=torch.stack([t]).cuda(), cross_attn_first_call=self.cross_new, **kw)[0]
                self.cross_new = False
                x0 = p._convert_flow_pred_to_x0(flow_pred=pred, xt=x, timestep=t, scheduler=p.scheduler)
                if i < len(self.timesteps) - 1:
                    noise = torch.randn(x0.shape, generator=gen, device="cuda", dtype=x0.dtype)
                    x = p.scheduler.add_noise(x0, noise, self.timesteps[i + 1])
            p.model(x=[x0], t=torch.stack([self.timesteps[-1] * 0.0]).cuda(), cross_attn_first_call=False, **kw)
        return x0

    # ---- rollout -----------------------------------------------------------------------------------------
    def _rollout(self, image, prompt_fn, seed, on_chunk):
        """Generic loop used by warmup and sessions: prompt_fn(i) -> (prompt, changed, motion) or None to stop."""
        torch, np = self.torch, self.np
        self._reset()
        gen = torch.Generator("cuda").manual_seed(seed)
        y_all = self._image_condition(image)
        context, i = None, 0
        while True:
            got = prompt_fn(i)
            if got is None:
                return y_all
            prompt, changed, motion = got
            if context is None or changed:
                context = self._encode_text(prompt)
                self.cross_new = True
            step = step_pose(motion)
            poses = [np.eye(4) if i == 0 and j == 0 else step for j in range(CHUNK)]   # latent 0: identity
            z = self._chunk(context, y_all, poses, i * CHUNK, gen)
            frames = self._decode(z, first=(i == 0))
            on_chunk(i, frames)
            i += 1

    def warmup(self):
        from PIL import Image
        torch = self.torch
        t0 = time.time()
        times, t = [], [time.time()]

        def prompt_fn(i):
            if i >= LOCAL_ATTN // CHUNK + 3:
                return None
            t[0] = time.time()
            return "warmup: a calm lake at sunrise", False, ({"forward": STEP_T} if i >= 2 else {})

        def on_chunk(i, frames):
            torch.cuda.synchronize()
            times.append((len(frames), round(time.time() - t[0], 3)))

        with torch.no_grad():
            y = self._rollout(Image.new("RGB", (1024, 768), (90, 140, 200)), prompt_fn, 0, on_chunk)
        diffs = [round(float((y[4:, k] - y[4:, k - 1]).abs().max()), 5) for k in range(2, Y_LATENTS)]
        LOG.info("warmup %.1fs; chunks (frames, s) %s; last chunk %.1f fps; y steady-state max|diff| per latent %s",
                 time.time() - t0, times, times[-1][0] / times[-1][1], diffs)

    def run(self, s: Session):
        from PIL import Image
        torch = self.torch
        image = Image.open(io.BytesIO(base64.b64decode(s.init["image_b64"])))

        def prompt_fn(i):
            got = s.next_block()
            if got is None:
                return None
            prompt, changed, events = got
            s.take_camera()
            s.applied(events, "umT5 re-encoded, cross-attention cache recomputed")
            return prompt, changed, s.motion

        def on_chunk(i, frames):
            s.emit(list(frames))

        try:
            with torch.no_grad():
                self._rollout(image, prompt_fn, s.seed, on_chunk)
        finally:
            s.drop_camera()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18875)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    t2v_stream.GenSession = Session          # serve() builds sessions through the module attribute
    t2v_stream.serve(Engine(args), args.host, args.port, args.jpeg_quality)


if __name__ == "__main__":
    main()
