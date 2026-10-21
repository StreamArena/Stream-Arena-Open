#!/usr/bin/env python3
"""LiveTalk 1.3B (GAIR-NLP) self-hosted model server (docs/protocol.md).

    python server.py --port 18811

LiveTalk is a Self-Forcing-distilled causal OmniAvatar (Wan2.1-T2V-1.3B + audio adapter): block-wise autoregressive,
3 latent frames per block (12 frames at 16 fps; the first block 9 frames, its frame 0 = the portrait), 4 denoising
steps, KV cache of 15 latent frames with a 9-frame identity sink (upstream ``scripts/inference_example.py``:
``local_attn_size=15``, ``sink_size=9``), clean-context re-forward after each block. The upstream script is offline
(whole wav2vec pass, decode at the end); this server unrolls the same per-block loop live:

* audio: speech arrives as ``audio_in`` (16 kHz, 10 ms); block n covers frames 12n-3 .. 12n+8 (n=0: 0 .. 8) and
  starts once audio up to the end of its last frame has arrived or its wall-clock deadline passed (missing audio =
  silence). Its wav2vec features come from the last ``--audio_window`` s of audio ending at that frame (16 fps
  interpolation as upstream), so nothing is read ahead;
* video: the block's latents go through the causal Wan VAE decoder with its cache kept across blocks
  (``decode_to_pixel(use_cache=True)``), frames are paced at 16 fps together with the 62.5 ms of speech that drove
  each frame (``audio_out``, 48 kHz); when generation is slower than real time chunks arrive late (each re-anchors the
  clock); nothing is dropped;
* text: every block is conditioned on the text prompt. avatar_performance visual controls replace the control part
  of the prompt of the next block (``"<base prompt> <control>"``, control map ``livetalk-control-map/1``): the prompt
  is re-encoded and the cross-attention cache reset; the self-attention KV cache keeps the generated past. The released
  model has no dedicated control mechanism; the prompt is upstream's only steering input (trained on motion-focused
  prompts). Camera / scene / appearance / speech-content controls are acked unsupported;
* digital_human composition: the shared ASR -> LLM -> TTS stack pushes the TTS speech as ``audio_in``; dialogue
  ``text`` is handled client-side (acked applied: it already became the speech on ``audio_in``).
* temporal RoPE table recomputed with the same formula for 4096 latent frames (upstream 1024 = 256 s; rows < 1024
  unchanged), so sessions longer than 256 s keep valid positions.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import types

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("LIVETALK_ROOT") or
                 os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/livetalk")
sys.path.insert(0, str(HERE.parents[2]))
from streameval.runtime import protocol as sp  # noqa: E402

for _k, _v in {"TORCHINDUCTOR_CACHE_DIR": str(MODEL_DIR / "inductor_cache"),
               "TRITON_CACHE_DIR": str(MODEL_DIR / "triton_cache")}.items():
    os.environ.setdefault(_k, _v)

MODEL = "livetalk_1p3b"
SR, FPS = 16000, 16
SPF = SR // FPS                 # 1000 samples per frame
BLOCK = 3                       # latent frames per block
WINDOW, SINK = 15, 9            # upstream local_attn_size / sink_size (latent frames)
FRAME_SEQ = 1024                # tokens per latent frame at 512x512
ROPE_POSITIONS = 4096
DEFAULT_PROMPT = ("A realistic video of a person speaking directly to the camera. The individual maintains steady eye "
                  "contact with clear, expressive facial features. Their facial expressions are naturally animated "
                  "and emotionally engaging, with precise lip movements perfectly synchronized to their speech.")
CONTROL_MAPPING_VERSION = "livetalk-control-map/1"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[livetalk]", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Control mapping (same vocabulary as soulx_liveact's liveact-control-map/1): a control is applied when it asks for
# a visible movement / pose of the performer's body, gaze, or a facial expression; camera, scene, lighting, props,
# clothing, style and speaking content have no counterpart in a portrait animation prompt and are acked unsupported.
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


def map_control(text: str) -> dict:
    clean = text or ""
    for p in BOILERPLATE:
        clean = p.sub("", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    action, gaze, emotion = _hits(ACTION_WORDS, clean), _hits(GAZE_WORDS, clean), _hits(EMOTION_WORDS, clean)
    other = sorted({m.group(0).lower() for p in UNSUPPORTED_PATTERNS for m in re.finditer(p, clean, re.I)})
    channels = (["action"] if action or gaze else []) + (["emotion"] if emotion else [])
    if not channels:
        why = f"no action/emotion content (matched {other})" if other else "no action/emotion content"
        return {"status": "unsupported", "channels": [], "prompt": None,
                "detail": f"{CONTROL_MAPPING_VERSION}: {why}; LiveTalk is steered only by its text prompt"}
    detail = f"{CONTROL_MAPPING_VERSION}: appended to the text prompt ({'+'.join(channels)}) from the next block"
    if other:
        detail += f"; parts without a counterpart ignored ({', '.join(other)})"
    return {"status": "applied", "channels": channels, "prompt": clean, "detail": detail}


def strip_boilerplate(text: str) -> str:
    for p in BOILERPLATE:
        text = p.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def to48k(x16: np.ndarray) -> bytes:
    n = len(x16)
    if n == 0:
        return b""
    y = np.interp(np.arange(3 * n, dtype=np.float64) / 3.0, np.arange(n, dtype=np.float64), x16.astype(np.float64))
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()


def _load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, args):
        self.args = args
        repo = Path(args.repo)
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        # the `model` package __init__ imports the training classes (DMD, CausVid, ...); inference needs only the
        # causal DiT module, so register the package path without running __init__
        pkg = types.ModuleType("model")
        pkg.__path__ = [str(repo / "model")]
        sys.modules["model"] = pkg
        import torch
        self.torch = torch
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        from transformers import Wav2Vec2FeatureExtractor
        from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
        from model.wan_video_dit_omniavatar_helper import precompute_freqs_cis_3d
        # the two OmniAvatar files LiveTalk ships as patches (scripts/add_patch.sh copies them into OmniAvatar/)
        fm = _load_file_module("omniavatar_flow_match", repo / "utils/Omniavatarpatch/flow_match.py")
        w2v = _load_file_module("omniavatar_wav2vec", repo / "utils/Omniavatarpatch/wav2vec.py")
        t0 = time.time()
        dev, dt = torch.device("cuda"), torch.bfloat16
        self.dev, self.dt = dev, dt
        W = Path(args.weights)
        self.scheduler = fm.FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True, num_inference_steps=4)
        timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        self.steps = timesteps[:-1].to(dtype=dt, device=dev)                       # warp_denoising_step: true
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(str(W / "wav2vec2-base-960h"))
        self.audio_encoder = w2v.Wav2VecModel.from_pretrained(str(W / "wav2vec2-base-960h"),
                                                              local_files_only=True).to(device=dev, dtype=dt)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.generator = WanDiffusionWrapper(model_path=str(W / "LiveTalk-1.3B-V0.1/model.safetensors"), is_causal=True,
                                             use_omniavatar_model=False, local_attn_size=WINDOW, sink_size=SINK).model
        self.generator.to(device=dev, dtype=dt).eval().requires_grad_(False)
        self.generator.num_frame_per_block = BLOCK
        self.generator.local_attn_size = WINDOW
        self.generator.independent_first_frame = False
        head_dim = getattr(self.generator, "dim", 1536) // getattr(self.generator, "num_heads", 12)
        self.generator.freqs = precompute_freqs_cis_3d(head_dim, end=ROPE_POSITIONS)
        self.num_blocks = len(self.generator.blocks)
        wan = W / "Wan2.1-T2V-1.3B"
        self.text_encoder = WanTextEncoder(text_encoder_path=str(wan / "models_t5_umt5-xxl-enc-bf16.pth"),
                                           tokenizer_path=str(wan / "google/umt5-xxl") + "/")
        self.text_encoder.to(device=dev, dtype=dt).requires_grad_(False)
        self.vae = WanVAEWrapper(vae_path=str(wan / "Wan2.1_VAE.pth"))
        self.vae.to(device=dev, dtype=dt).requires_grad_(False)
        self.width = self.height = 512
        self.kv_cache = self.cross_cache = None
        self.prompt_cache = {}
        self.load_s = time.time() - t0
        self.gpu = torch.cuda.get_device_name(0)
        self.timing = {}
        log(f"loaded in {self.load_s:.1f}s on {self.gpu}; steps {[round(float(t), 1) for t in self.steps]}; "
            f"{self.num_blocks} DiT blocks, head_dim {head_dim}")

    # ---- caches ----------------------------------------------------------------------------------------------
    def _reset_caches(self):
        torch, dev, dt = self.torch, self.dev, self.dt
        size = WINDOW * FRAME_SEQ
        if self.kv_cache is None:
            self.kv_cache = [{"k": torch.zeros([1, size, 12 * 128], dtype=dt, device=dev),
                              "v": torch.zeros([1, size, 12 * 128], dtype=dt, device=dev),
                              "global_end_index": torch.tensor([0], dtype=torch.long, device=dev),
                              "local_end_index": torch.tensor([0], dtype=torch.long, device=dev)}
                             for _ in range(self.num_blocks)]
            self.cross_cache = [{"k": torch.zeros([1, 512, 12 * 128], dtype=dt, device=dev),
                                 "v": torch.zeros([1, 512, 12 * 128], dtype=dt, device=dev), "is_init": False}
                                for _ in range(self.num_blocks)]
        for c in self.kv_cache:
            c["k"].zero_()
            c["v"].zero_()
            c["global_end_index"].fill_(0)
            c["local_end_index"].fill_(0)
        for c in self.cross_cache:
            c["is_init"] = False
        self.vae.model.clear_cache()

    def encode_prompt(self, text):
        if text not in self.prompt_cache:
            if len(self.prompt_cache) > 128:
                self.prompt_cache.clear()
            self.prompt_cache[text] = self.text_encoder([text])["prompt_embeds"]
        for c in self.cross_cache:
            c["is_init"] = False
        return self.prompt_cache[text]

    # ---- session ---------------------------------------------------------------------------------------------
    def start(self, image_bytes, prompt, seed):
        with self.torch.no_grad():               # grad mode is per thread; sessions run in their own thread
            return self._start(image_bytes, prompt, seed)

    def _start(self, image_bytes, prompt, seed):
        torch = self.torch
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        s = min(img.size)                                     # benchmark portraits are square; center-crop otherwise
        img = img.crop(((img.width - s) // 2, (img.height - s) // 2, (img.width - s) // 2 + s, (img.height - s) // 2 + s))
        img = img.resize((self.width, self.height), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255).to(self.dev)
        x = (x * 2.0 - 1.0)[None, :, None].to(self.dt)                          # [1, 3, 1, H, W]
        lat = self.vae.encode_to_latent(x)                                       # [1, 1, 16, h, w]
        self.ref = lat.to(self.dt)                                               # first-frame latent [1, 1, 16, h, w]
        lat = lat.permute(0, 2, 1, 3, 4).to(self.dt)                             # [1, 16, 1, h, w]
        self.y_first = torch.cat([lat.repeat(1, 1, BLOCK, 1, 1), torch.ones_like(lat[:, :1]).repeat(1, 1, BLOCK, 1, 1)], 1)
        self.y_first[:, 16:, 0] = 0                                              # mask 0 on global frame 0 only
        self.y_rest = torch.cat([lat.repeat(1, 1, BLOCK, 1, 1), torch.ones_like(lat[:, :1]).repeat(1, 1, BLOCK, 1, 1)], 1)
        self._reset_caches()
        self.prompt = None
        self.cond = self.encode_prompt(prompt)
        self.prompt = prompt
        torch.manual_seed(seed)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)

    def audio_embedding(self, window: np.ndarray, n: int):
        """window: 16 kHz audio ending at the end of block n's last frame -> projected audio cond [P, 3, 1, 1, D]."""
        torch = self.torch
        frames = int(round(len(window) / SPF))
        iv = np.squeeze(self.fe(window, sampling_rate=SR).input_values)
        iv = torch.from_numpy(iv).float().to(device=self.dev, dtype=self.dt)[None]
        hs = self.audio_encoder(iv, seq_len=frames, output_hidden_states=True)
        emb = hs.last_hidden_state
        for mid in hs.hidden_states:
            emb = torch.cat((emb, mid), -1)
        emb = emb.permute(0, 2, 1)[:, :, :, None, None]                          # [1, 10752, frames, 1, 1]
        if n == 0:                                                               # frames 0..8, frame 0 repeated x3
            last9 = emb[:, :, -9:]
            emb = torch.cat([last9[:, :, :1].repeat(1, 1, 3, 1, 1), last9], 2)
        else:                                                                    # frames 12n-3 .. 12n+8
            emb = emb[:, :, -12:]
        a = self.generator.audio_proj(emb.to(self.dt))
        return torch.concat([proj(a) for proj in self.generator.audio_cond_projs], 0)

    def _x0(self, flow_pred, xt, timestep):
        torch = self.torch
        dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(lambda x: x.double().to(flow_pred.device),
                                               [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps])
        tid = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        return (xt - sigmas[tid].reshape(-1, 1, 1, 1) * flow_pred).to(dtype)

    def _forward(self, x, cond, y, audio, timestep, start):
        flow = self.generator(x.permute(0, 2, 1, 3, 4), timestep=timestep, context=cond, y=y, audio_emb=audio,
                              kv_cache=self.kv_cache, crossattn_cache=self.cross_cache,
                              current_start=start * FRAME_SEQ, cache_start=None).permute(0, 2, 1, 3, 4)
        return self._x0(flow.flatten(0, 1), x.flatten(0, 1), timestep.flatten(0, 1)).unflatten(0, flow.shape[:2])

    def block(self, n, window, prompt):
        with self.torch.no_grad():
            return self._block(n, window, prompt)

    def _block(self, n, window, prompt):
        torch = self.torch
        t0 = time.perf_counter()
        if prompt != self.prompt:
            self.cond = self.encode_prompt(prompt)
            self.prompt = prompt
        audio = self.audio_embedding(window, n)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        start = n * BLOCK
        y = self.y_first if n == 0 else self.y_rest
        h, w = self.height // 8, self.width // 8
        x = torch.randn([1, BLOCK, 16, h, w], device=self.dev, dtype=self.dt, generator=self.gen)
        for i, t in enumerate(self.steps):
            if n == 0:
                x[:, :1] = self.ref
            timestep = torch.ones([1, BLOCK], device=self.dev, dtype=torch.int64) * t
            pred = self._forward(x, self.cond, y, audio, timestep, start)
            if i < len(self.steps) - 1:
                nxt = self.steps[i + 1] * torch.ones([BLOCK], device=self.dev, dtype=torch.long)
                noise = torch.randn(pred.flatten(0, 1).shape, device=self.dev, dtype=pred.dtype, generator=self.gen)
                x = self.scheduler.add_noise(pred.flatten(0, 1), noise, nxt).unflatten(0, pred.shape[:2])
        if n == 0:
            pred[:, :1] = self.ref
        self._forward(pred, self.cond, y, audio, torch.ones_like(timestep) * 0, start)   # clean context into the cache
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        video = self.vae.decode_to_pixel(pred, use_cache=True)[0]               # [T, 3, H, W] in [-1, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        frames = (video.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
        t3 = time.perf_counter()
        self.timing = {"audio_ms": round((t1 - t0) * 1000, 1), "dit_ms": round((t2 - t1) * 1000, 1),
                       "vae_ms": round((t3 - t2) * 1000, 1)}
        return frames

    def warmup(self, blocks):
        from PIL import Image
        t0 = time.time()
        xs = np.linspace(0, 255, 512, dtype=np.uint8)
        arr = np.stack([np.tile(xs, (512, 1)), np.tile(xs[::-1], (512, 1)), np.full((512, 512), 128, np.uint8)], -1)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "PNG")
        self.start(buf.getvalue(), DEFAULT_PROMPT, 0)
        rng = np.random.default_rng(0)
        for n in range(blocks):
            t = time.time()
            win = (rng.standard_normal(int(SR * self.args.audio_window)) * 0.05).astype(np.float32)
            fr = self.block(n, win, DEFAULT_PROMPT if n < 3 else DEFAULT_PROMPT + " The person waves the right hand.")
            log(f"warmup block {n}: {time.time() - t:.2f}s frames={len(fr)} {self.timing}")
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
        self.track = None
        self.out: asyncio.Queue = asyncio.Queue()
        self.stats = {"blocks": [], "events": [], "frames": 0, "reanchors": 0}
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

    def render(self, init):
        import cv2
        eng = self.eng
        try:
            with self.server.gpu_lock:
                eng.start(init["image"], init["prompt"], init["seed"])
                with self.lock:
                    self.t_ready = time.monotonic()
                self.loop.call_soon_threadsafe(self.ready_evt.set)
                n = 0
                wlen = int(SR * self.args.audio_window)
                while not self.stop.is_set():
                    first = 0 if n == 0 else 12 * n - 3                   # first output frame of block n
                    last = 8 if n == 0 else 12 * n + 8
                    end = (last + 1) * SPF                                # audio needed up to here
                    tl = self.window(end)
                    if self.stop.is_set():
                        break
                    win = tl[max(0, end - wlen):end]
                    if len(win) < wlen:
                        win = np.concatenate([np.zeros(wlen - len(win), np.float32), win])
                    prompt = init["prompt"] if not self.control_prompt else f"{init['prompt']} {self.control_prompt}"
                    t0 = time.monotonic()
                    frames = eng.block(n, win, prompt)
                    t1 = time.monotonic()
                    self.stats["blocks"].append({"n": n, "first_frame": first, "frames": len(frames),
                                                 "audio_end_s": round(end / SR, 3), "start_s": round(t0 - self.t_ready, 3),
                                                 "done_s": round(t1 - self.t_ready, 3),
                                                 "compute_ms": round((t1 - t0) * 1000, 1), **eng.timing,
                                                 "prompt_changed": prompt != init["prompt"]})
                    for j, fr in enumerate(frames):
                        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                                               [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
                        k = first + j
                        self.loop.call_soon_threadsafe(self.out.put_nowait,
                                                       (jpg.tobytes(), to48k(tl[k * SPF:(k + 1) * SPF])))
                    n += 1
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
        await self.conn.ready(MODEL, self.server.config["repo_commit"] or "unknown", self.eng.width, self.eng.height, FPS,
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
                    if self.track and self.track.startswith("digital_human") and kind == "text":
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
                    await self.conn.ack(msg.get("event_id"), "unsupported", "LiveTalk has no reference-image editing")
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
        eng = engine
        self.config = {
            "model": "LiveTalk-1.3B-V0.1", "repo": "https://github.com/GAIR-NLP/LiveTalk", "repo_commit": commit,
            "hf_weights": "GAIR/LiveTalk-1.3B-V0.1@be14ffdf356e73e328db88595ccf62e7e1374b74 + Wan-AI/Wan2.1-T2V-1.3B "
                          "(T5, VAE) + facebook/wav2vec2-base-960h", "weights": weights,
            "width": eng.width, "height": eng.height, "fps": FPS, "steps": len(eng.steps),
            "denoising_step_list": [round(float(t), 2) for t in eng.steps], "shift": 5,
            "block_latents": BLOCK, "block_frames": 4 * BLOCK, "local_attn_latents": WINDOW, "sink_latents": SINK,
            "rope_positions": ROPE_POSITIONS, "audio_window_s": self.args.audio_window,
            "vae": "Wan2.1 VAE, causal cached decode per block", "dtype": "bfloat16", "gpu": eng.gpu, "gpu_count": 1,
            "control_mapping": CONTROL_MAPPING_VERSION,
            "prompt": "avatar_performance: init.prompt (rendering_context, benchmark boilerplate stripped) or the "
                      "upstream example prompt, + latest control; digital_human: upstream example prompt",
            "torch": eng.torch.__version__, "load_s": round(eng.load_s, 1)}

    def write_stats(self, sess):
        s = dict(sess.stats)
        b = s["blocks"]
        if b:
            frames = sum(x["frames"] for x in b)
            s["summary"] = {"blocks": len(b), "frames_generated": frames, "frames_sent": s["frames"],
                            "mean_compute_ms": round(float(np.mean([x["compute_ms"] for x in b[2:] or b])), 1),
                            "generation_fps": round(frames / max(1e-6, sum(x["compute_ms"] for x in b) / 1000), 2),
                            "realtime_budget_ms": round(4 * BLOCK / FPS * 1000, 1),
                            "max_lag_s": round(max(x["done_s"] - x["audio_end_s"] for x in b), 3),
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
    ap.add_argument("--port", type=int, default=18811)
    ap.add_argument("--repo", default=str(MODEL_DIR / "repo"))
    ap.add_argument("--weights", default=str(MODEL_DIR / "weights"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audio_window", type=float, default=5.0, help="s of past audio for each block's wav2vec pass")
    ap.add_argument("--grace", type=float, default=0.06)
    ap.add_argument("--warmup_blocks", type=int, default=8)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--stats_dir", default=str(MODEL_DIR / "logs/sessions"))
    args = ap.parse_args()
    engine = Engine(args)
    engine.warmup(args.warmup_blocks)
    server = Server(engine, args)
    log(f"listening on ws://{args.host}:{args.port}/session")
    sp.serve(server.handler, args.host, args.port, health=server.health)


if __name__ == "__main__":
    main()
