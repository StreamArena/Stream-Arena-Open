#!/usr/bin/env python3
"""AvatarForcing 1.3B (Kling Team, Kuaishou) self-hosted model server (docs/protocol.md).

    python server.py --port 18891

AvatarForcing (arXiv 2603.14331) is a one-step streaming talking-avatar model: a causal OmniAvatar-style DiT
(Wan2.1-T2V-1.3B + LoRA + audio pack, in_dim 33) that denoises a *local-future sliding window* of 4 blocks (4 latent
frames = 16 video frames each) at staggered noise levels; every window step runs the DiT once and finalises the
oldest block, which is then written (clean) into a KV cache of 7 latent frames whose first slot is the portrait
(style anchor, re-RoPE'd as the frame just before the working cache). Upstream ``inference.py`` runs this offline
over a whole audio file (``inference_avatar_forcing``); this server unrolls the same window loop live:

* portrait -> 640x624 (centre crop; 39x40 latent patches = the 1560 tokens per latent frame the model hard-codes,
  upstream's default 832x480 has the same count), VAE latent = frame 0, prefilled into the KV cache at t = 0 exactly
  as upstream (``_prefill_cache_for_rolling`` with ``independent_first_frame``);
* window step w covers blocks max(0, w-3) .. w; block w enters as fresh noise, older blocks carry their re-noised
  previous predictions (upstream's ``noisy_cache``), timesteps are upstream's warped denoise_steps [1000, 800, 400,
  200] (oldest block lowest). From w >= 3 the oldest block (w-3) is final: decoded with the causal Wan VAE (cache
  kept across blocks) and sent as 16 frames at 25 fps;
* audio (``audio_in``, 16 kHz): latent frame i is conditioned on the wav2vec features of video frames 4i-4 .. 4i-1
  (upstream's zero-prefixed per-frame embedding). Window step w therefore needs audio up to the end of block w, i.e.
  **48 frames = 1.92 s past the last frame of the block it finalises** (the model's local-future look-ahead; disclosed,
  it shows up as latency). The step starts once that audio has arrived or its wall-clock deadline passed (missing
  audio = silence). Features come from the last ``--audio_window`` s of audio ending there, so nothing later is read;
* each sent frame carries the 40 ms of speech that drove it as ``audio_out`` (48 kHz); when generation is slower than
  real time, frames arrive late (the clock re-anchors); nothing is dropped;
* text: every window is conditioned on the text prompt (cross-attention). avatar_performance: ``init.prompt``
  (rendering_context, benchmark boilerplate stripped) + the latest visual control (control map
  ``avatarforcing-control-map/1``, same vocabulary as the other self-hosted avatar models), applied from the next
  window step (cross-attention cache reset); blocks already final are not changed. Camera / scene / appearance /
  speech-content controls are acked unsupported. digital_human composition: TTS speech arrives as ``audio_in``,
  dialogue ``text`` is acked applied (handled by the composition layer);
* temporal RoPE table recomputed with the same formula for ``--rope_positions`` latent frames (upstream 1024 = 164 s;
  rows < 1024 unchanged), so long sessions keep valid positions.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("AVATARFORCING_ROOT") or
                 os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/avatarforcing")
sys.path.insert(0, str(HERE.parents[2]))
from streameval.runtime import protocol as sp  # noqa: E402

for _k, _v in {"TORCHINDUCTOR_CACHE_DIR": str(MODEL_DIR / "inductor_cache"),
               "TRITON_CACHE_DIR": str(MODEL_DIR / "triton_cache")}.items():
    os.environ.setdefault(_k, _v)

MODEL = "avatarforcing_1p3b"
SR, FPS = 16000, 25
SPF = SR // FPS                 # 640 samples per frame
BLOCK = 4                       # latent frames per block
BF = 4 * BLOCK                  # 16 video frames per block
WINDOW = 4                      # blocks per sliding window (= len(denoise_steps))
FRAME_SEQ = 1560
WIDTH, HEIGHT = 640, 624        # 40 x 39 patches = 1560 tokens per latent frame
DEFAULT_PROMPT = ("A person faces the camera and speaks in a composed, natural rhythm; lip shapes are clearly "
                  "articulated, with natural facial expressions and slight brow movement. Natural blinks and small "
                  "head shifts keep the portrait alive while maintaining frontal alignment. The camera is locked; "
                  "lighting is soft and balanced, with consistent skin tone and stable colour throughout.")
CONTROL_MAPPING_VERSION = "avatarforcing-control-map/1"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[avatarforcing]", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Control mapping (same vocabulary as livetalk-control-map/1).
ACTION_WORDS = r"""hand hands arm arms palm palms fist fists finger fingers index thumb thumbs wrist wrists elbow elbows
shoulder shoulders torso body chest waist hip hips lean leans leaning step steps stepping knee knees foot feet leg legs
toe toes heel stance posture pose poses stand standing sit sitting squat crouch head nod nods nodding tilt tilts tilted
turn turns turning rotate rotates bow bows bowing wave waves waving sweep sweeps raise raises raised lift lifts lifted
lower lowers lowered drop drops dropped cross crossed clap claps point points pointing reach reaches extend extends
extended gesture gestures grip grab hold holding pinch press presses pressing fold shrug shrugs neck spine back
upright slump""".split()
GAZE_WORDS = "gaze gazes look looks looking glance glances stare stares eye eyes track tracks".split()
EMOTION_WORDS = r"""smile smiles smiling grin grins frown frowns frowning brow brows eyebrow eyebrows expression
expressions face facial squint squints squinting laugh laughs laughing sad sadness happy joy joyful angry anger
surprise surprised calm serious neutral curious alert wink winks pout pouts cheek cheeks lips jaw worried confident
relaxed tense excited scowl grimace sneer""".split()
UNSUPPORTED_PATTERNS = [r"\bzoom\w*", r"\bdolly\b", r"\bcamera (?:moves?|pans?|angle|cuts?|shot|orbits?|tilts?)\b",
                        r"\b(?:pan|cut|switch) (?:the )?camera\b", r"\blighting\b", r"\bbackground\b", r"\bscenery\b",
                        r"\b(?:outfit|clothes|clothing|costume|hairstyle)\b", r"\b(?:art|render(?:ing)?) style\b",
                        r"\b(?:say|says|saying|speak|speaks|sing|sings)\b"]
BOILERPLATE = [re.compile(p, re.I) for p in (
    r"Lips follow only the supplied audio;?\s*stay silent otherwise\.?",
    r"Mouth shape follows only the current actual input audio;[^.]*\.",
)]


def _hits(words, text):
    return sorted({w for w in words if re.search(r"\b" + re.escape(w) + r"\b", text, re.I)})


def strip_boilerplate(text: str) -> str:
    for p in BOILERPLATE:
        text = p.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def map_control(text: str) -> dict:
    clean = strip_boilerplate(text or "")
    action, gaze, emotion = _hits(ACTION_WORDS, clean), _hits(GAZE_WORDS, clean), _hits(EMOTION_WORDS, clean)
    other = sorted({m.group(0).lower() for p in UNSUPPORTED_PATTERNS for m in re.finditer(p, clean, re.I)})
    channels = (["action"] if action or gaze else []) + (["emotion"] if emotion else [])
    if not channels:
        why = f"no action/emotion content (matched {other})" if other else "no action/emotion content"
        return {"status": "unsupported", "channels": [], "prompt": None,
                "detail": f"{CONTROL_MAPPING_VERSION}: {why}; AvatarForcing is steered only by its text prompt"}
    detail = f"{CONTROL_MAPPING_VERSION}: appended to the text prompt ({'+'.join(channels)}) from the next window step"
    if other:
        detail += f"; parts without a counterpart ignored ({', '.join(other)})"
    return {"status": "applied", "channels": channels, "prompt": clean, "detail": detail}


def to48k(x16: np.ndarray) -> bytes:
    n = len(x16)
    if n == 0:
        return b""
    y = np.interp(np.arange(3 * n, dtype=np.float64) / 3.0, np.arange(n, dtype=np.float64), x16.astype(np.float64))
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()


# ---------------------------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, args):
        self.args = args
        repo = Path(args.repo)
        os.chdir(repo)                                   # upstream loads T5 / VAE from ./wan_models/...
        sys.path.insert(0, str(repo))
        import torch
        from omegaconf import OmegaConf
        self.torch = torch
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        from transformers import Wav2Vec2FeatureExtractor
        from pipeline import AvatarForcingInferencePipeline
        from utils.inject import _apply_lora
        from wan.models.wav2vec import Wav2VecModel
        from wan.modules.model import rope_params
        t0 = time.time()
        dev, dt = torch.device("cuda"), torch.bfloat16
        self.dev, self.dt = dev, dt
        cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load("configs/avatarforcing.yaml"))
        self.cfg = cfg
        pipe = AvatarForcingInferencePipeline(cfg, device=dev)
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        pipe.generator.model = _apply_lora(pipe.generator.model, cfg["models"]["lora"])
        missing = pipe.generator.load_state_dict(sd["generator"])
        del sd
        self.pipe = pipe.to(device=dev, dtype=dt)
        self.gen_model = pipe.generator.model
        d = self.gen_model.dim // self.gen_model.num_heads
        n = args.rope_positions
        self.gen_model.freqs = torch.cat([rope_params(n, d - 4 * (d // 6)), rope_params(n, 2 * (d // 6)),
                                          rope_params(n, 2 * (d // 6))], dim=1)
        self.steps = pipe.denoise_steps.to(device=dev, dtype=torch.float32)        # [1000, 800, 400, 200] warped
        assert len(self.steps) == WINDOW and pipe.num_frame_per_block == BLOCK and pipe.independent_first_frame
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec)
        self.audio_encoder = Wav2VecModel.from_pretrained(args.wav2vec, local_files_only=True).to(device=dev, dtype=dt)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.prompt_cache = {}
        self.load_s = time.time() - t0
        self.gpu = torch.cuda.get_device_name(0)
        self.timing = {}
        log(f"loaded in {self.load_s:.1f}s on {self.gpu}; load_state_dict {missing}; steps "
            f"{[round(float(t), 1) for t in self.steps]}; rope rows {n}")

    # ---- per session -----------------------------------------------------------------------------------------
    def encode_prompt(self, text):
        if text not in self.prompt_cache:
            if len(self.prompt_cache) > 64:
                self.prompt_cache.clear()
            self.prompt_cache[text] = self.pipe.text_encoder(text_prompts=[text])["prompt_embeds"].to(self.dt)
        for c in self.pipe.crossattn_cache or []:
            c["is_init"] = False
        return self.prompt_cache[text]

    def start(self, image_bytes, prompt, seed):
        with self.torch.no_grad():
            return self._start(image_bytes, prompt, seed)

    def _start(self, image_bytes, prompt, seed):
        torch, pipe = self.torch, self.pipe
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ar = WIDTH / HEIGHT                                   # centre crop to the output aspect, then resize
        if img.width / img.height > ar:
            w = int(round(img.height * ar))
            img = img.crop(((img.width - w) // 2, 0, (img.width - w) // 2 + w, img.height))
        else:
            h = int(round(img.width / ar))
            img = img.crop((0, (img.height - h) // 2, img.width, (img.height - h) // 2 + h))
        img = img.resize((WIDTH, HEIGHT), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255).to(self.dev)
        x = (x * 2.0 - 1.0)[None, :, None].to(self.dt)                        # [1, 3, 1, H, W]
        lat = pipe.vae.encode_to_latent(x).to(self.dt)                          # [1, 1, 16, h, w]
        self.ref = lat
        img_lat = lat.permute(0, 2, 1, 3, 4)                                    # [1, 16, 1, h, w]
        self.y0 = torch.cat([img_lat, torch.zeros_like(img_lat[:, :1])], 1)     # frame 0: mask 0
        self.y1 = torch.cat([img_lat.repeat(1, 1, WINDOW * BLOCK, 1, 1),
                             torch.ones_like(img_lat[:, :1]).repeat(1, 1, WINDOW * BLOCK, 1, 1)], 1)
        self.h, self.w = lat.shape[-2:]
        assert self.h * self.w // 4 == FRAME_SEQ, (self.h, self.w)
        pipe._reset_or_init_caches(batch_size=1, dtype=self.dt, device=self.dev)
        pipe.vae.model.clear_cache()
        self.prompt = prompt
        self.cond = self.encode_prompt(prompt)
        torch.manual_seed(seed)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.noisy = {}                                                         # block -> re-noised latents
        # prefill frame 0 (upstream _prefill_cache_for_rolling, independent_first_frame)
        a0 = torch.zeros([1, 1, 10752], device=self.dev, dtype=self.dt)
        zero_t = torch.zeros([1, 1], device=self.dev, dtype=torch.int64)
        pipe.generator(noisy_image_or_video=lat, conditional_dict={"prompt_embeds": self.cond, "audio_emb": a0, "y": self.y0},
                       timestep=zero_t, kv_cache=pipe.kv_cache_clean, crossattn_cache=pipe.crossattn_cache,
                       current_start=0)
        frame0 = pipe.vae.decode_to_pixel(lat, use_cache=True)[0]               # [1, 3, H, W] in [-1, 1]
        return self._to_u8(frame0)

    def _to_u8(self, video):
        video = (video.float() * 0.5 + 0.5).clamp(0, 1)
        return (video.permute(0, 2, 3, 1) * 255).round().to(self.torch.uint8).cpu().numpy()

    def audio_features(self, window: np.ndarray, frames: int):
        """window: 16 kHz audio ending at the end of the last needed video frame -> [1, frames, 10752]."""
        torch = self.torch
        n = int(round(len(window) / SPF))
        iv = np.squeeze(self.fe(window, sampling_rate=SR).input_values)
        iv = torch.from_numpy(iv).float().to(device=self.dev, dtype=self.dt)[None]
        hs = self.audio_encoder(iv, seq_len=n, output_hidden_states=True)
        emb = hs.last_hidden_state
        for mid in hs.hidden_states:
            emb = torch.cat((emb, mid), -1)
        return emb[:, -frames:].to(self.dt)

    def step(self, w, window, prompt):
        with self.torch.no_grad():
            return self._step(w, window, prompt)

    def _step(self, w, window, prompt):
        """Window step w (blocks sb..w). Returns the decoded frames of block w-3 once it is final, else None."""
        torch, pipe = self.torch, self.pipe
        t0 = time.perf_counter()
        if prompt != self.prompt:
            self.cond = self.encode_prompt(prompt)
            self.prompt = prompt
        sb = max(0, w - WINDOW + 1)
        nb = w - sb + 1
        audio = self.audio_features(window, nb * BF)                            # video frames 16sb .. 16w+15
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        noise = torch.randn([1, BLOCK, 16, self.h, self.w], device=self.dev, dtype=self.dt, generator=self.gen)
        x = torch.cat([self.noisy[b] for b in range(sb, w)] + [noise], 1)
        levels = [nb - 1 - p for p in range(nb)]                                # oldest block = most denoised
        ts = torch.cat([self.steps[lv].repeat(BLOCK) for lv in levels])[None]
        frame_s = 1 + sb * BLOCK
        cond = {"prompt_embeds": self.cond, "audio_emb": audio, "y": self.y1[:, :, :nb * BLOCK]}
        _, den = pipe.generator(noisy_image_or_video=x, conditional_dict=cond, timestep=ts,
                                kv_cache=pipe.kv_cache_clean, crossattn_cache=pipe.crossattn_cache,
                                current_start=frame_s * FRAME_SEQ)
        for p, b in enumerate(range(sb, w + 1)):
            lv = levels[p]
            blk = den[:, p * BLOCK:(p + 1) * BLOCK]
            if lv == WINDOW - 1:
                self.noisy.pop(b, None)
                continue
            nxt = self.steps[lv + 1] * torch.ones([BLOCK], device=self.dev, dtype=torch.long)
            nz = torch.randn(blk.flatten(0, 1).shape, device=self.dev, dtype=blk.dtype, generator=self.gen)
            self.noisy[b] = pipe.scheduler.add_noise(blk.flatten(0, 1), nz, nxt).unflatten(0, blk.shape[:2])
        ccond = {"prompt_embeds": self.cond, "audio_emb": audio[:, :BF], "y": self.y1[:, :, :BLOCK]}
        ct = torch.ones([1, BLOCK], device=self.dev, dtype=ts.dtype) * float(self.cfg.get("context_noise", 0))
        pipe.generator(noisy_image_or_video=den[:, :BLOCK], conditional_dict=ccond, timestep=ct,
                       kv_cache=pipe.kv_cache_clean, crossattn_cache=pipe.crossattn_cache,
                       current_start=frame_s * FRAME_SEQ, updating_cache=True)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        frames = None
        if nb == WINDOW:
            frames = self._to_u8(pipe.vae.decode_to_pixel(den[:, :BLOCK], use_cache=True)[0])
        t3 = time.perf_counter()
        self.timing = {"audio_ms": round((t1 - t0) * 1000, 1), "dit_ms": round((t2 - t1) * 1000, 1),
                       "vae_ms": round((t3 - t2) * 1000, 1)}
        return frames

    def warmup(self, steps):
        from PIL import Image
        t0 = time.time()
        xs = np.linspace(0, 255, WIDTH, dtype=np.uint8)
        arr = np.stack([np.tile(xs, (HEIGHT, 1)), np.tile(xs[::-1], (HEIGHT, 1)), np.full((HEIGHT, WIDTH), 128, np.uint8)], -1)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "PNG")
        self.start(buf.getvalue(), DEFAULT_PROMPT, 0)
        rng = np.random.default_rng(0)
        for w in range(steps):
            t = time.time()
            win = (rng.standard_normal(int(SR * self.args.audio_window)) * 0.05).astype(np.float32)
            fr = self.step(w, win, DEFAULT_PROMPT if w < 5 else DEFAULT_PROMPT + " The person waves the right hand.")
            log(f"warmup step {w}: {time.time() - t:.2f}s frames={0 if fr is None else len(fr)} {self.timing}")
        log(f"warm-up done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------------------------------------------
class Session:
    def __init__(self, conn, server, loop):
        self.conn, self.server, self.loop = conn, server, loop
        self.eng, self.args = server.engine, server.args
        self.stop = threading.Event()
        self.lock = threading.Condition()
        self.timeline = np.zeros(0, dtype=np.float32)
        self.received_to = 0
        self.anchor = None
        self.last_pts = -1
        self.t_ready = None
        self.control_prompt = None
        self.track = ""
        self.out: asyncio.Queue = asyncio.Queue()
        self.stats = {"steps": [], "events": [], "frames": 0, "reanchors": 0}
        self.error = None

    def audio_in(self, pts_ns, payload):
        pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        with self.lock:
            if self.t_ready is None:
                return
            if self.anchor is None or pts_ns < self.last_pts:
                self.anchor = int(round((time.monotonic() - self.t_ready) * SR)) - int(pts_ns * SR // 1_000_000_000)
            self.last_pts = pts_ns
            start = self.anchor + int(pts_ns * SR // 1_000_000_000)
            if start < 0:
                pcm, start = pcm[-start:], 0
            end = start + len(pcm)
            if end > len(self.timeline):
                self.timeline = np.concatenate([self.timeline, np.zeros(max(end - len(self.timeline), SR * 10), np.float32)])
            self.timeline[start:end] = pcm
            self.received_to = max(self.received_to, end)
            self.lock.notify_all()

    def window(self, end):
        deadline = self.t_ready + end / SR + self.args.grace
        with self.lock:
            while not self.stop.is_set() and self.received_to < end:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self.lock.wait(min(left, 0.02))
            if len(self.timeline) < end:
                self.timeline = np.concatenate([self.timeline, np.zeros(end - len(self.timeline) + SR * 10, np.float32)])
            return self.timeline[:end].copy()

    def _emit(self, frames, first, tl):
        import cv2
        for j, fr in enumerate(frames):
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
            k = first + j                    # decoded frame k (k >= 1) was driven by the audio of video frame k-1
            pcm = tl[(k - 1) * SPF:k * SPF] if k >= 1 else np.zeros(SPF, np.float32)
            self.loop.call_soon_threadsafe(self.out.put_nowait, (jpg.tobytes(), to48k(pcm)))

    def render(self, init):
        eng = self.eng
        try:
            with self.server.gpu_lock:
                frame0 = eng.start(init["image"], init["prompt"], init["seed"])
                with self.lock:
                    self.t_ready = time.monotonic()
                self.loop.call_soon_threadsafe(self.ready_evt.set)
                self._emit(frame0, 0, np.zeros(0, np.float32))
                w = 0
                wlen = int(SR * self.args.audio_window)
                while not self.stop.is_set():
                    end = BF * (w + 1) * SPF                                  # audio up to the end of block w
                    tl = self.window(end)
                    if self.stop.is_set():
                        break
                    win = tl[max(0, end - wlen):end]
                    if len(win) < wlen:
                        win = np.concatenate([np.zeros(wlen - len(win), np.float32), win])
                    prompt = init["prompt"] if not self.control_prompt else f"{init['prompt']} {self.control_prompt}"
                    t0 = time.monotonic()
                    frames = eng.step(w, win, prompt)
                    t1 = time.monotonic()
                    b = w - (WINDOW - 1)
                    self.stats["steps"].append({"w": w, "final_block": b if frames is not None else None,
                                                "audio_end_s": round(end / SR, 3), "start_s": round(t0 - self.t_ready, 3),
                                                "done_s": round(t1 - self.t_ready, 3),
                                                "compute_ms": round((t1 - t0) * 1000, 1), **eng.timing,
                                                "prompt_changed": prompt != init["prompt"]})
                    if frames is not None:
                        self._emit(frames, 1 + BF * b, tl)
                    w += 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = f"{type(exc).__name__}: {exc}"
            self.loop.call_soon_threadsafe(self.ready_evt.set)
            self.loop.call_soon_threadsafe(self.out.put_nowait, ("error", self.error))

    async def sender(self):
        t0, k = None, 0
        while True:
            item = await self.out.get()
            if item[0] == "error":
                await self.conn.error("render failed: " + item[1])
                return
            now = time.monotonic()
            if t0 is None or now > t0 + k / FPS + 0.2:
                if t0 is not None:
                    self.stats["reanchors"] += 1
                t0, k = now, 0
            wait = t0 + k / FPS - now
            if wait > 0:
                await asyncio.sleep(wait)
            jpg, pcm = item
            now_ns = time.monotonic_ns()
            await self.conn.frame(jpg, now_ns)
            await self.conn.audio(pcm, now_ns)
            self.stats["frames"] += 1
            k += 1

    async def start(self, m):
        if not m.get("image_b64"):
            await self.conn.error("init.image_b64 (portrait) required")
            return False
        self.track = m.get("track") or ""
        raw = base64.b64decode(m["image_b64"])
        seed = int(m.get("seed") if m.get("seed") is not None else self.args.seed)
        base = strip_boilerplate((m.get("prompt") or "").strip()) if self.track == "avatar_performance" else ""
        prompt = base or DEFAULT_PROMPT
        self.stats.update({"case_id": m.get("case_id"), "track": self.track, "seed": seed, "prompt": prompt})
        self.ready_evt = asyncio.Event()
        self.thread = threading.Thread(target=self.render, args=({"image": raw, "prompt": prompt, "seed": seed},),
                                       daemon=True)
        self.thread.start()
        await self.ready_evt.wait()
        if self.error:
            await self.conn.error(self.error)
            return False
        cfg = {**self.server.config, "seed": seed, "prompt": prompt, "portrait_sha256": hashlib.sha256(raw).hexdigest()}
        await self.conn.ready(MODEL, self.server.config["repo_commit"] or "unknown", WIDTH, HEIGHT, FPS,
                              protocol=sp.PROTOCOL_VERSION, config=cfg)
        log("ready", json.dumps({"seed": seed, "case": m.get("case_id"), "track": self.track}))
        self.send_task = asyncio.create_task(self.sender())
        return True

    async def run(self):
        started = False
        try:
            while True:
                msg = await self.conn.recv()
                if isinstance(msg, bytes):
                    tag, pts, payload = sp.unpack(msg)
                    if tag == sp.AUDIO_IN:
                        self.audio_in(pts, payload)
                    continue
                kind = msg.get("type")
                if kind == "init" and not started:
                    started = True
                    if not await self.start(msg):
                        break
                elif kind in ("text", "control"):
                    eid = msg.get("event_id")
                    text = msg.get("text") or ""
                    if kind == "control":
                        params = msg.get("params") or {}
                        text = " ".join(str(x) for x in (msg.get("action"), params.get("text") or params.get("prompt")) if x)
                    if self.track.startswith("digital_human") and kind == "text":
                        res = {"status": "applied",
                               "detail": "renderer-only: speech for this turn arrives as audio_in from the composition layer"}
                    else:
                        res = map_control(text)
                        if res["status"] == "applied":
                            self.control_prompt = res["prompt"]
                    t = None if self.t_ready is None else round(time.monotonic() - self.t_ready, 3)
                    self.stats["events"].append({"event_id": eid, "at_s": t, **res})
                    await self.conn.ack(eid, res["status"], res["detail"])
                elif kind == "reference":
                    await self.conn.ack(msg.get("event_id"), "unsupported", "AvatarForcing has no reference-image editing")
                elif kind == "end":
                    break
        except sp.ConnectionClosed:
            pass
        finally:
            self.stop.set()
            with self.lock:
                self.lock.notify_all()
            if getattr(self, "send_task", None):
                self.send_task.cancel()
            try:
                await self.conn.close()
            except Exception:
                pass
            if getattr(self, "thread", None):
                await asyncio.get_running_loop().run_in_executor(None, self.thread.join, 120)
            self.server.write_stats(self)


class Server:
    def __init__(self, engine, args):
        self.engine, self.args = engine, args
        self.gpu_lock = threading.Lock()
        self.busy = asyncio.Lock()
        weights = json.loads((MODEL_DIR / "weights_hash.json").read_text()) if (MODEL_DIR / "weights_hash.json").exists() else {}
        commit = (MODEL_DIR / "repo_commit.txt").read_text().strip() if (MODEL_DIR / "repo_commit.txt").exists() else None
        rev = MODEL_DIR / "weights/AvatarForcing/.complete"
        eng = engine
        self.config = {
            "model": "AvatarForcing (Wan2.1-T2V-1.3B + LoRA, one-step)", "repo": "https://github.com/KlingAIResearch/AvatarForcing",
            "repo_commit": commit,
            "hf_weights": "lycui/AvatarForcing" + (f"@{rev.read_text().strip()}" if rev.exists() else "") +
                          " (model.pt) + Wan-AI/Wan2.1-T2V-1.3B (T5, VAE) + facebook/wav2vec2-base-960h",
            "weights": weights, "width": WIDTH, "height": HEIGHT, "fps": FPS,
            "steps_per_block": 1, "window_blocks": WINDOW, "denoise_steps": [round(float(t), 2) for t in eng.steps],
            "block_latents": BLOCK, "block_frames": BF, "kv_cache_latents": 7, "lookahead_frames": BF * (WINDOW - 1),
            "lookahead_s": BF * (WINDOW - 1) / FPS, "rope_positions": self.args.rope_positions,
            "audio_window_s": self.args.audio_window, "vae": "Wan2.1 VAE, causal cached decode per block",
            "dtype": "bfloat16", "gpu": eng.gpu, "gpu_count": 1, "control_mapping": CONTROL_MAPPING_VERSION,
            "prompt": "avatar_performance: init.prompt (rendering_context, benchmark boilerplate stripped) + latest "
                      "control; digital_human: generic talking-portrait caption in upstream's example style",
            "torch": eng.torch.__version__, "load_s": round(eng.load_s, 1)}

    def write_stats(self, sess):
        s = dict(sess.stats)
        st = s["steps"]
        fin = [x for x in st if x["final_block"] is not None]
        if st:
            frames = BF * len(fin)
            s["summary"] = {"steps": len(st), "frames_generated": frames + 1, "frames_sent": s["frames"],
                            "mean_compute_ms": round(float(np.mean([x["compute_ms"] for x in st[4:] or st])), 1),
                            "generation_fps": round(frames / max(1e-6, sum(x["compute_ms"] for x in st) / 1000), 2),
                            "realtime_budget_ms": round(BF / FPS * 1000, 1),
                            "max_lag_s": round(max(x["done_s"] - x["audio_end_s"] for x in st), 3),
                            "reanchors": s["reanchors"]}
            log("session summary " + json.dumps(s["summary"]))
        d = Path(self.args.stats_dir)
        d.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(s.get("case_id") or "session"))
        (d / f"{name}.{int(time.time())}.json").write_text(json.dumps({"config": self.config, **s}, indent=1))

    async def handler(self, conn):
        async with self.busy:
            await Session(conn, self, asyncio.get_running_loop()).run()

    def health(self):
        return {"ok": True, "model": MODEL, "busy": self.busy.locked()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18891)
    ap.add_argument("--repo", default=str(MODEL_DIR / "repo"))
    ap.add_argument("--checkpoint", default=str(MODEL_DIR / "weights/AvatarForcing/model.pt"))
    ap.add_argument("--wav2vec", default=str(MODEL_DIR / "weights/wav2vec2-base-960h"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audio_window", type=float, default=5.0, help="s of past audio for each step's wav2vec pass")
    ap.add_argument("--rope_positions", type=int, default=8192)
    ap.add_argument("--grace", type=float, default=0.06)
    ap.add_argument("--warmup_steps", type=int, default=8)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--stats_dir", default=str(MODEL_DIR / "logs/sessions"))
    args = ap.parse_args()
    engine = Engine(args)
    engine.warmup(args.warmup_steps)
    server = Server(engine, args)
    log(f"listening on ws://{args.host}:{args.port}/session")
    sp.serve(server.handler, args.host, args.port, health=server.health)


if __name__ == "__main__":
    main()
