#!/usr/bin/env python3
"""Avatar-Forever (LTX-2.3 22B) self-hosted model server (docs/protocol.md).

    python server.py --port 18871 [--size 640*640]

One GPU. The server drives upstream's own chunk-wise autoregressive sampler (leeruibin/avatarforever,
``ltx_pipelines.utils.autoregressive.autoregressive_euler_denoising_loop``) one AR chunk at a time
(``start_chunk_idx=k, end_chunk_idx=k+1``) with upstream's inference.py defaults: one-stage distilled sampling,
sigmas (1.0, 0.98125, 0.909375, 0.421875, 0.0) = 4 steps, AR chunk 4 latent frames, 1 history chunk, the first chunk
kept as a sink, relative positions, no ForeverCache, gated first-frame channel condition from the portrait (current
chunk only), the portrait as the frame-0 latent (strength 1, crf 0), no CFG / STG, 25 fps.

Tracks
  * avatar_performance (native): supplied speech arrives as ``audio_in`` and drives the lips through the LTX audio
    latents (the model's clean audio-conditioning stream); visual controls (``text`` user_prompt / ``control``) are
    put into the text prompt of the next AR chunk (Gemma 3 text encoder; control map ``avatarforever-control-map/1``,
    same vocabulary as the other self-hosted avatar models); controls without body / face content are unsupported.
  * digital_human_audio / digital_human_text as a composition renderer: the client's shared Qwen ASR -> LLM -> TTS
    stack sends the TTS speech as ``audio_in``; dialogue ``text`` events are acked unsupported (handled client side).

Real time, no look-ahead
  AR chunk k covers latent frames 4k..4k+3 = video frames 0..24 (k = 0) or 32k-7..32k+24, i.e. 1.28 s of video per
  chunk at 25 fps. Upstream maps audio latents (25 per second) to chunks proportionally: chunk k is conditioned on
  audio latent steps [a_k, b_k) with b_k = ceil(4(k+1) * A / L) ~ 32(k+1), i.e. audio up to ~(32k+32)/25 s, 0.28 s
  past the chunk's last frame (upstream's alignment, a model property; disclosed). The chunk starts only once the
  session clock has passed b_k / 25 s; audio not yet received then is silence. Offline, upstream VAE-encodes the
  whole audio file at once; live, each chunk's audio latents come from the LTX audio VAE over a trailing window
  (``--audio_window_s``, default 10 s) that ends at b_k, so nothing later than b_k is ever read. Frames are paced out
  at 25 fps with the matching 48 kHz slice of the received speech (``audio_out``), lip-aligned at a constant delay
  (one chunk + compute); when generation is slower than real time, frames arrive late (never re-timed or dropped).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("AF_ROOT", os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/avatar_forever"))
sys.path[:0] = [str(HERE.parents[2])]
from streameval.runtime import protocol as sp  # noqa: E402

for _k, _v in {"TORCHINDUCTOR_CACHE_DIR": str(ROOT / "inductor_cache"), "TRITON_CACHE_DIR": str(ROOT / "triton_cache"),
               "PYTORCH_ALLOC_CONF": "expandable_segments:True", "HF_HUB_OFFLINE": "1"}.items():
    os.environ.setdefault(_k, _v)

MODEL_NAME = "Avatar-Forever (LTX-2.3 22B)"
CONTROL_MAPPING_VERSION = "avatarforever-control-map/1"
IN_RATE, OUT_RATE = sp.AUDIO_IN_RATE, sp.AUDIO_OUT_RATE
FPS = 25
SIGMAS = (1.0, 0.98125, 0.909375, 0.421875, 0.0)          # upstream inference.py --stage1-sigmas default
AR_CHUNK, AR_HISTORY = 4, 1                                # upstream defaults
SAMPLES_PER_AUDIO_LATENT = IN_RATE // 25                  # 16 kHz / hop 160 / downsample 4 -> 25 latents/s
# upstream inference.py DEFAULT_PROMPT (two paragraphs: motion, then temporal stability)
MOTION_PROMPT = (
    "Natural audio-driven speaking motion with accurate lip synchronization, smooth and continuous facial "
    "animation, subtle head movement, natural blinking, gentle breathing, and relaxed upper-body motion. "
    "Expressions and gestures should respond naturally to the rhythm, tone, and emotion of the speech while "
    "remaining restrained and realistic.")
STABILITY_PROMPT = (
    "Maintain strong temporal consistency across all frames. Keep facial appearance, identity, pose, body "
    "structure, clothing details, lighting, and background stable throughout the video. Avoid sudden motion "
    "changes, excessive gestures, unnatural expression shifts, frame-to-frame appearance variation, flickering, "
    "jitter, ghosting, texture instability, temporal artifacts, or deformation. All motion should be coherent, "
    "fluid, stable, and naturally driven by the audio.")
DEFAULT_PROMPT = MOTION_PROMPT + "\n\n" + STABILITY_PROMPT

# ---------------------------------------------------------------------------------------------------------------
# Control mapping: benchmark visual control (free English text) -> the text prompt of the next AR chunk.
# Same vocabulary as selfhost/soulx_liveact (body / gaze / face content applied; camera, scene, appearance and speech
# content unsupported).
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
    """-> {"status": applied|unsupported, "channels": [...], "prompt": str|None, "detail": str}"""
    clean = text or ""
    for p in BOILERPLATE:
        clean = p.sub("", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    action, gaze, emotion = _hits(ACTION_WORDS, clean), _hits(GAZE_WORDS, clean), _hits(EMOTION_WORDS, clean)
    other = sorted({m.group(0).lower() for p in UNSUPPORTED_PATTERNS for m in re.finditer(p, clean, re.I)})
    channels = (["body"] if action else []) + (["gaze"] if gaze else []) + (["face"] if emotion else [])
    if not channels:
        why = f"no body/face content (matched {other})" if other else "no body/face content"
        return {"status": "unsupported", "channels": [], "prompt": None,
                "detail": f"{CONTROL_MAPPING_VERSION}: {why}; Avatar-Forever is steered only by its text prompt"}
    detail = f"{CONTROL_MAPPING_VERSION}: text prompt of the next AR chunk ({'+'.join(channels)})"
    if other:
        detail += f"; parts the portrait-anchored model cannot change ignored ({', '.join(other)})"
    return {"status": "applied", "channels": channels, "prompt": clean, "detail": detail}


def compose_prompt(scene: str, control: str | None) -> str:
    """Upstream prompt guidance: a concrete scene description first, then the motion and stability prompt. With an
    active performance control the control replaces the generic 'restrained' motion paragraph."""
    parts = [scene.strip()] if scene and scene.strip() else []
    if control:
        parts += [control.strip(), STABILITY_PROMPT]
    else:
        parts += [DEFAULT_PROMPT]
    return "\n\n".join(parts)


class _NoBar:
    def __init__(self, *a, **k):
        pass

    def update(self, *a):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, args):
        import torch
        self.torch, self.args = torch, args
        self.width, self.height = [int(v) for v in args.size.split("*")]
        self.fps = FPS
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_grad_enabled(False)
        sys.path.insert(0, str(ROOT / "repo"))
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import encode_audio
        from ltx_core.model.video_vae import decode_video
        from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
        from ltx_pipelines import ARA2VidDistilledPipeline
        from ltx_pipelines.utils import autoregressive as ar
        from ltx_pipelines.utils import helpers as hp
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.media_io import load_image_conditioning
        ar.tqdm = _NoBar
        self.ar, self.hp = ar, hp
        self.EulerDiffusionStep, self.GaussianNoiser = EulerDiffusionStep, GaussianNoiser
        self.encode_audio, self.decode_video, self.Audio = encode_audio, decode_video, Audio
        self.AudioLatentShape, self.VideoPixelShape = AudioLatentShape, VideoPixelShape
        self.ImageConditioningInput, self.load_image_conditioning = ImageConditioningInput, load_image_conditioning
        t0 = time.time()
        # upstream inference.py omits the constructor's required `loras` (open issue #5); none are used
        self.pipe = ARA2VidDistilledPipeline(distilled_checkpoint_path=args.checkpoint, gemma_root=args.gemma_root,
                                             spatial_upsampler_path=None, loras=())
        self.device, self.dtype = self.pipe.device, self.pipe.dtype
        # transformer with the first-frame channel-condition projection (gated, zero init), as inference.py builds it
        self.extra_config = self.pipe._channel_condition_extra_config(
            first_frame_channel_condition_latent=None, stage2_first_frame_channel_condition_latent=None,
            init="zero", mode="gated", fallback_channel_dim=self.pipe.pipeline_components.video_latent_channels)
        self.mods = self.pipe._get_fast_modules(stage_mode="one-stage", transformer_extra_config=self.extra_config)
        self.channel_condition_weights = self._channel_condition_norms()
        self.sigmas = torch.tensor(SIGMAS, device=self.device, dtype=torch.float32)
        self.output_shape = VideoPixelShape(batch=1, frames=args.max_frames, width=self.width, height=self.height,
                                            fps=float(FPS))
        self.total_latent = self.pipe._total_video_latent_frames(self.output_shape)
        self.audio_shape = AudioLatentShape.from_video_pixel_shape(self.output_shape)
        self.video_chunks = ar.build_progressive_chunk_ranges(self.total_latent, AR_CHUNK)
        self.audio_chunks = ar.build_aligned_audio_chunk_ranges(self.video_chunks, total_video_frames=self.total_latent,
                                                                total_audio_steps=self.audio_shape.frames)
        self.prompt_cache = {}
        self.load_s = time.time() - t0
        self.log(f"loaded in {self.load_s:.0f}s size={self.width}x{self.height} max_frames={args.max_frames} "
                 f"latent_frames={self.total_latent} audio_latents={self.audio_shape.frames} "
                 f"chunks={len(self.video_chunks)} mem={torch.cuda.memory_allocated() / 2**30:.1f}GiB "
                 f"channel_condition_weight_norms={self.channel_condition_weights}")

    def log(self, msg):
        print(f"[avatar_forever] {msg}", flush=True)

    def _channel_condition_norms(self):
        tr = getattr(self.mods.transformer, "velocity_model", self.mods.transformer)
        out = {}
        for name, p in tr.named_parameters():
            if "channel_condition" in name:
                out[name] = round(float(p.float().norm()), 6)
        return out

    # ---- per-chunk geometry --------------------------------------------------------------------------------
    def chunk_frames(self, k):
        """(first pixel frame, number of frames) of AR chunk k."""
        return (0, 1 + 8 * (AR_CHUNK - 1)) if k == 0 else (8 * AR_CHUNK * k - 7, 8 * AR_CHUNK)

    def audio_needed_s(self, k):
        return self.audio_chunks[k][1] / 25.0

    # ---- prompt ------------------------------------------------------------------------------------------
    def context(self, prompt):
        if prompt not in self.prompt_cache:
            if len(self.prompt_cache) > 64:
                self.prompt_cache.clear()
            self.prompt_cache[prompt] = self.pipe._encode_prompt_context(
                prompt=prompt, images=[], enhance_prompt=False, text_encoder=self.mods.text_encoder,
                embeddings_processor=self.mods.embeddings_processor)
        return self.prompt_cache[prompt]

    # ---- session -----------------------------------------------------------------------------------------
    def start(self, image_bytes, prompt, seed):
        torch, hp = self.torch, self.hp
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, "PNG")
            path = f.name
        try:
            images = [self.ImageConditioningInput(path=path, frame_idx=0, strength=1.0, crf=0)]
            conditionings = self.pipe._build_video_conditionings(
                images=images, video_frame_conditionings=[], latent_conditionings=[], global_condition_images=[],
                global_latent_conditionings=[], global_condition_position="prepend",
                global_condition_repeat_to_target_frames=False, global_condition_position_mode="zero",
                height=self.height, width=self.width, video_encoder=self.mods.video_encoder, dtype=self.dtype)
            # inference.py encode_first_frame_channel_condition (image given -> not derived from chunk 0)
            pix = self.load_image_conditioning(image_path=path, height=self.height, width=self.width,
                                               dtype=self.dtype, device=self.device, crf=0)
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                ff_latent = self.mods.video_encoder(pix)[:1].contiguous()
        finally:
            os.unlink(path)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noiser = self.GaussianNoiser(generator=self.generator)
        self.video_state, self.video_tools = hp.noise_video_state(
            output_shape=self.output_shape, noiser=noiser, conditionings=conditionings,
            components=self.pipe.pipeline_components, dtype=self.dtype, device=self.device,
            noise_scale=float(SIGMAS[0]))
        cc = self.pipe._repeat_first_frame_channel_condition_latent(latent=ff_latent, output_shape=self.output_shape)
        self.video_state = hp.apply_video_channel_condition_to_state(
            video_state=self.video_state, video_tools=self.video_tools, channel_condition=cc)
        lat = self.video_state.latent
        self.tokens_per_frame = (lat.shape[1] - self.video_state.conditioning_prefix_token_count) // self.total_latent
        self.audio_grid = torch.zeros(self.audio_shape.to_torch_shape(), device=self.device, dtype=self.dtype)
        self.zero_gen = torch.Generator(device=self.device).manual_seed(0)
        self.context(prompt)
        self.stepper = self.EulerDiffusionStep()

    def _encode_audio_window(self, k, audio16k):
        """Write the audio latents of chunk k ([a_k, b_k)) from a trailing window that ends at b_k."""
        torch = self.torch
        a, b = self.audio_chunks[k]
        s0 = b - int(round(self.args.audio_window_s * 25))
        # fixed window length (constant conv shapes): before the session start the timeline is silence
        need = (b - s0) * SAMPLES_PER_AUDIO_LATENT
        audio16k = np.asarray(audio16k, dtype=np.float32)[-need:]
        if len(audio16k) < need:
            audio16k = np.concatenate([np.zeros(need - len(audio16k), np.float32), audio16k])
        wav = torch.from_numpy(np.ascontiguousarray(audio16k)).to(self.device)
        audio = self.Audio(waveform=wav[None, None].repeat(1, 2, 1), sampling_rate=IN_RATE)   # stereo (upstream)
        lat = self.encode_audio(audio, self.mods.audio_encoder)                                 # [1, 8, T, 16]
        n = min(b - a, lat.shape[2] - (a - s0))
        if n > 0:
            self.audio_grid[:, :, a:a + n] = lat[:, :, a - s0:a - s0 + n].to(self.dtype)
        return s0, int(lat.shape[2]), n

    def chunk(self, k, audio16k, prompt):
        """Generate AR chunk k; audio16k = session speech [s0 * 640, b_k * 640) for _encode_audio_window."""
        torch, hp, ar = self.torch, self.hp, self.ar
        marks = [time.perf_counter()]

        def mark():
            torch.cuda.synchronize()
            marks.append(time.perf_counter())
        s0, n_lat, n_set = self._encode_audio_window(k, audio16k)
        v_ctx, a_ctx = self.context(prompt)
        mark()
        audio_state, _ = hp.noise_audio_state(
            output_shape=self.output_shape, noiser=self.GaussianNoiser(generator=self.zero_gen), conditionings=[],
            components=self.pipe.pipeline_components, dtype=self.dtype, device=self.device, noise_scale=0.0,
            initial_latent=self.audio_grid)
        audio_state = replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))
        transformer = self.mods.transformer

        def builder():
            return hp.simple_denoising_func(video_context=v_ctx, audio_context=a_ctx, transformer=transformer)
        self.video_state, _ = ar.autoregressive_euler_denoising_loop(
            sigmas=self.sigmas, video_state=self.video_state, audio_state=audio_state, stepper=self.stepper,
            denoise_fn_builder=builder, total_video_latent_frames=self.total_latent, video_chunk_size=AR_CHUNK,
            history_chunk_count=AR_HISTORY, sink_first_chunk=True, relative_positions=True,
            history_feature_cache=False, derive_channel_condition_from_first_chunk=False,
            channel_condition_current_chunk_only=True, first_frame_prefix_condition=False,
            start_chunk_idx=k, end_chunk_idx=k + 1)
        mark()
        # decode: this chunk's 4 latents (k = 0: 25 frames); later chunks with one prefix latent, first frame dropped
        f0, f1 = self.video_chunks[k]
        p0 = f0 if k == 0 else f0 - self.args.decode_prefix
        pre = self.video_state.conditioning_prefix_token_count
        tok = self.video_state.latent[:, pre + p0 * self.tokens_per_frame: pre + f1 * self.tokens_per_frame]
        h, w = self.height // 32, self.width // 32
        grid = tok.reshape(1, f1 - p0, h, w, tok.shape[-1]).permute(0, 4, 1, 2, 3).contiguous()
        frames = torch.cat(list(self.decode_video(grid, self.mods.video_decoder, None, self.generator)), dim=0)
        _, n = self.chunk_frames(k)
        frames = frames[-n:]
        mark()
        out = frames.cpu().numpy()
        mark()
        self.timing = {key: round((b - a) * 1000, 1) for key, a, b in
                       zip(("audio_text_ms", "dit_ms", "vae_ms", "copy_ms"), marks, marks[1:])}
        self.timing.update({"audio_window_start": s0, "audio_window_latents": n_lat, "audio_latents_set": n_set})
        return out

    def warmup(self, chunks=4):
        rng = np.random.default_rng(0)
        buf = io.BytesIO()
        from PIL import Image
        x = np.linspace(0, 255, self.width, dtype=np.uint8)
        Image.fromarray(np.stack([np.tile(x, (self.height, 1))] * 3, -1)).save(buf, "PNG")
        self.start(buf.getvalue(), DEFAULT_PROMPT, 0)
        speech = (rng.standard_normal(IN_RATE * 12) * 0.05).astype(np.float32)
        for k in range(chunks):
            t0 = time.time()
            b = self.audio_chunks[k][1] * SAMPLES_PER_AUDIO_LATENT
            s0 = max(0, self.audio_chunks[k][1] - int(round(self.args.audio_window_s * 25))) * SAMPLES_PER_AUDIO_LATENT
            fr = self.chunk(k, speech[s0:b], DEFAULT_PROMPT if k < 2 else compose_prompt("", "Wave the right hand."))
            self.log(f"warmup chunk {k}: {time.time() - t0:.2f}s frames={fr.shape} {self.timing}")
        self.video_state = None
        self.torch.cuda.empty_cache()
        self.log(f"warmup done; peak mem {self.torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")


# ---------------------------------------------------------------------------------------------------------------
class Session:
    """The generation thread owns the GPU; the asyncio loop owns the socket."""

    def __init__(self, server, conn, loop):
        self.server, self.conn, self.loop = server, conn, loop
        self.engine = server.engine
        self.fps = FPS
        self.lock = threading.Lock()
        self.audio = np.zeros(0, dtype=np.float32)       # session-timeline speech, 16 kHz, index 0 = ready time
        self.anchor = None
        self.last_pts = -1
        self.control_prompt = None
        self.ended = threading.Event()
        self.t0 = None
        self.out = queue.Queue()
        self.stats = {"chunks": [], "events": []}
        self.error = None

    def add_audio(self, pts_ns, payload):
        pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        now = time.monotonic()
        with self.lock:
            if self.t0 is None:
                return
            if self.anchor is None or pts_ns < self.last_pts:
                self.anchor = int(round((now - self.t0) * IN_RATE)) - int(pts_ns * IN_RATE // 1_000_000_000)
            self.last_pts = pts_ns
            pos = max(0, self.anchor + int(pts_ns * IN_RATE // 1_000_000_000))
            end = pos + len(pcm)
            if end > len(self.audio):
                self.audio = np.concatenate([self.audio, np.zeros(end - len(self.audio), np.float32)])
            self.audio[pos:end] = pcm

    def audio_window(self, a, b):
        with self.lock:
            seg = self.audio[a:b]
        return np.concatenate([seg, np.zeros(max(0, b - a - len(seg)), np.float32)])

    def generate(self, init):
        eng = self.engine
        args = self.server.args
        scene = init["scene"]
        with eng.torch.inference_mode():
            self._generate(init, eng, args, scene)
        self.out.put(None)

    def _generate(self, init, eng, args, scene):
        eng.start(init["image"], compose_prompt(scene, None), init["seed"])
        self.t0 = time.monotonic()
        self.loop.call_soon_threadsafe(self.ready_evt.set)
        k = 0
        while not self.ended.is_set():
            if k >= len(eng.video_chunks):
                self.error = f"session longer than --max_frames {args.max_frames}"
                break
            need = eng.audio_needed_s(k)
            wait = self.t0 + need + args.audio_margin - time.monotonic()
            if wait > 0 and self.ended.wait(wait):
                break
            b = eng.audio_chunks[k][1] * SAMPLES_PER_AUDIO_LATENT
            s0 = max(0, eng.audio_chunks[k][1] - int(round(args.audio_window_s * 25))) * SAMPLES_PER_AUDIO_LATENT
            prompt = compose_prompt(scene, self.control_prompt)
            t_start = time.monotonic()
            frames = eng.chunk(k, self.audio_window(s0, b), prompt)
            t_done = time.monotonic()
            first, n = eng.chunk_frames(k)
            self.stats["chunks"].append({"k": k, "first_frame": first, "frames": int(frames.shape[0]),
                                         "audio_needed_s": round(need, 3), "start_s": round(t_start - self.t0, 3),
                                         "done_s": round(t_done - self.t0, 3),
                                         "compute_ms": round((t_done - t_start) * 1000, 1), **eng.timing,
                                         "control": self.control_prompt})
            self.out.put((first, frames, t_done))
            k += 1

    async def pump(self):
        """Pace frames at 25 fps with the matching speech slice: frame idx goes out at anchor + idx/fps. The anchor
        only moves earlier (min over chunks of done - first/fps + slack); frames it skips over go out at once."""
        loop = asyncio.get_running_loop()
        anchor = None
        spf = OUT_RATE // self.fps
        encode = self.server.encode
        while True:
            item = await loop.run_in_executor(None, self.out.get)
            if item is None:
                break
            first, frames, t_done = item
            cand = t_done - first / self.fps + (0 if anchor is None else self.server.args.pace_slack)
            anchor = cand if anchor is None else min(anchor, cand)
            jpegs = await loop.run_in_executor(None, lambda: [encode(fr) for fr in frames])
            for i, jpg in enumerate(jpegs):
                idx = first + i
                delay = anchor + idx / self.fps - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                a = int(IN_RATE * idx / self.fps)
                pcm16 = self.audio_window(a, a + IN_RATE // self.fps)
                pcm48 = np.interp(np.arange(spf) / 3.0, np.arange(len(pcm16)), pcm16)
                now = time.monotonic_ns()
                await self.conn.frame(jpg, now)
                await self.conn.audio((np.clip(pcm48, -1, 1) * 32767).astype("<i2").tobytes(), now)
                self.stats.setdefault("frame_sent_s", []).append(round(time.monotonic() - self.t0, 4))


class Server:
    def __init__(self, engine, args):
        self.engine, self.args = engine, args
        self.busy = False
        self.gpu_lock = threading.Lock()
        import cv2
        self.cv2 = cv2
        self.config = self._config()

    def encode(self, frame):
        ok, buf = self.cv2.imencode(".jpg", frame[:, :, ::-1], [self.cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
        return buf.tobytes()

    def _config(self):
        eng, args = self.engine, self.args
        torch = eng.torch
        p = ROOT / "weights_hash.json"
        weights = json.loads(p.read_text()) if p.exists() else {}
        commit = (ROOT / "repo_commit.txt").read_text().strip() if (ROOT / "repo_commit.txt").exists() else None
        return {"model": MODEL_NAME, "repo": "https://github.com/leeruibin/avatarforever", "repo_commit": commit,
                "hf_weights": "LetsThink/AvatarForever@622ea82482c09e7fb2d8e29cb7ff1d009c820f05 "
                              "(avatarforever-ltx-2.3-22b.safetensors) + google/gemma-3-12b-it-qat-q4_0-unquantized",
                "weights": weights,
                "license": "LTX-2 Community License Agreement (repo LICENSE; LTX-2.3 derivative; HF repo has no "
                           "card); Gemma 3: Gemma Terms of Use",
                "width": eng.width, "height": eng.height, "fps": FPS, "steps": len(SIGMAS) - 1, "sigmas": SIGMAS,
                "ar_chunk_latent_frames": AR_CHUNK, "ar_chunk_frames": 8 * AR_CHUNK, "ar_history_chunks": AR_HISTORY,
                "ar_sink_first_chunk": True, "ar_relative_positions": True, "ar_history_feature_cache": False,
                "first_frame": "portrait as latent frame 0 (strength 1, crf 0) + gated first-frame channel condition "
                               "(current chunk only)",
                "channel_condition_weight_norms": eng.channel_condition_weights,
                "cfg": "none (guidance 1.0, STG 0)", "max_frames": args.max_frames,
                "audio": f"LTX audio VAE over a trailing {args.audio_window_s} s window ending at the chunk's last "
                         "audio latent (silence before the session start; upstream encodes the whole file); chunk k "
                         "waits for audio up to b_k/25 s",
                "decode": f"per chunk, {args.decode_prefix} prefix latent(s), no tiling",
                "prompt": "scene (init.prompt = rendering_context) + upstream motion/stability prompt; a control "
                          "replaces the motion paragraph from the next AR chunk",
                "control_mapping": CONTROL_MAPPING_VERSION,
                "gpu": torch.cuda.get_device_name(0), "gpu_count": 1,
                "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "load_s": round(eng.load_s, 1)}

    async def handler(self, conn):
        for _ in range(1200):                    # the previous session may still be finishing its last chunk
            if not self.busy:
                break
            await asyncio.sleep(0.1)
        else:
            await conn.error("server busy: one session at a time")
            return
        self.busy = True
        loop = asyncio.get_running_loop()
        sess = Session(self, conn, loop)
        sess.ready_evt = asyncio.Event()
        gen_future = pump_task = None
        track = None
        try:
            while True:
                m = await conn.recv()
                if isinstance(m, bytes):
                    tag, pts, payload = sp.unpack(m)
                    if tag == sp.AUDIO_IN:
                        sess.add_audio(pts, payload)
                    continue
                kind = m.get("type")
                if kind == "init":
                    track = m.get("track")
                    if not m.get("image_b64"):
                        await conn.error("init.image_b64 (portrait) required")
                        break
                    seed = int(m.get("seed") if m.get("seed") is not None else self.args.seed)
                    scene = (m.get("prompt") or "").strip()
                    sess.stats.update({"case_id": m.get("case_id"), "track": track, "seed": seed, "scene": scene})
                    init = {"image": base64.b64decode(m["image_b64"]), "seed": seed, "scene": scene}
                    gen_future = loop.run_in_executor(None, self._generate_locked, sess, init)
                    await sess.ready_evt.wait()
                    if sess.error:
                        await conn.error(sess.error)
                        break
                    await conn.ready(MODEL_NAME, self.config["repo_commit"], self.engine.width, self.engine.height,
                                     FPS, protocol=sp.PROTOCOL_VERSION, config=self.config)
                    pump_task = asyncio.create_task(sess.pump())
                elif kind in ("text", "control"):
                    eid = m.get("event_id")
                    text = m.get("text") or ""
                    if kind == "control":
                        params = m.get("params") or {}
                        text = " ".join(str(x) for x in (m.get("action"), params.get("text") or params.get("prompt"))
                                        if x)
                    if track and track.startswith("digital_human"):
                        res = {"status": "unsupported",
                               "detail": "renderer only: dialogue is handled by the client composition layer"}
                    else:
                        res = map_control(text)
                        if res["status"] == "applied":
                            sess.control_prompt = res["prompt"]
                    t = None if sess.t0 is None else round(time.monotonic() - sess.t0, 3)
                    sess.stats["events"].append({"event_id": eid, "at_s": t, **res})
                    await conn.ack(eid, res["status"], res["detail"])
                elif kind == "reference":
                    await conn.ack(m.get("event_id"), "unsupported", "Avatar-Forever has no reference-image editing")
                elif kind == "end":
                    break
        except sp.ConnectionClosed:
            pass
        finally:
            sess.ended.set()
            if pump_task is not None:
                pump_task.cancel()
            await conn.close()
            try:                                  # at once (the server may be stopped right after the client closes)
                path = self._write_stats(sess)
            except Exception:                     # noqa: BLE001 (stats racing the last chunk: written again below)
                path = None
            if gen_future is not None:
                await gen_future
            self._write_stats(sess, path)
            self.busy = False

    def _generate_locked(self, sess, init):
        with self.gpu_lock:
            try:
                sess.generate(init)
            except Exception as exc:                      # surface to the client instead of hanging
                import traceback
                traceback.print_exc()
                sess.error = f"{type(exc).__name__}: {exc}"
                sess.out.put(None)
                sess.loop.call_soon_threadsafe(sess.ready_evt.set)
            finally:
                self.engine.video_state = None

    def _write_stats(self, sess, path=None):
        s = sess.stats
        c = s["chunks"]
        if c:
            frames = sum(x["frames"] for x in c)
            s["summary"] = {
                "chunks": len(c), "frames": frames,
                "mean_chunk_compute_ms": round(float(np.mean([x["compute_ms"] for x in c[2:] or c])), 1),
                "generation_fps": round(frames / max(1e-6, sum(x["compute_ms"] for x in c) / 1000), 2),
                "realtime_budget_ms_per_chunk": round(8 * AR_CHUNK / FPS * 1000, 1),
                "max_start_lag_s": round(max(x["start_s"] - x["audio_needed_s"] for x in c), 3),
                "first_frame_s": c[0]["done_s"]}
            if path is not None:
                self.engine.log("session summary " + json.dumps(s["summary"]))
        d = Path(self.args.stats_dir)
        d.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(s.get("case_id") or "session"))
        path = path or d / f"{name}.{int(time.time())}.json"
        path.write_text(json.dumps({"config": self.config, **s}, indent=1))
        return path

    def health(self):
        return {"ok": True, "model": MODEL_NAME, "busy": self.busy}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18871)
    ap.add_argument("--checkpoint", default=str(ROOT / "weights/avatarforever-ltx-2.3-22b.safetensors"))
    ap.add_argument("--gemma_root", default=str(ROOT / "weights/gemma-3-12b-it-qat-q4_0-unquantized"))
    ap.add_argument("--size", default="640*640", help="W*H (multiples of 32); upstream default 768*512")
    ap.add_argument("--max_frames", type=int, default=12001, help="8n+1 frames allocated per session (480 s)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audio_window_s", type=float, default=10.0)
    ap.add_argument("--audio_margin", type=float, default=0.03, help="s to wait past the audio a chunk needs")
    ap.add_argument("--decode_prefix", type=int, default=1)
    ap.add_argument("--warmup_chunks", type=int, default=4)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--pace_slack", type=float, default=0.15, help="s of output delay kept for compute jitter")
    ap.add_argument("--stats_dir", default=str(ROOT / "logs/sessions"))
    args = ap.parse_args()
    assert (args.max_frames - 1) % 8 == 0
    engine = Engine(args)
    with engine.torch.inference_mode():
        engine.warmup(args.warmup_chunks)
    server = Server(engine, args)
    engine.log(f"listening on ws://{args.host}:{args.port}/session")
    sp.serve(server.handler, args.host, args.port, health=server.health)


if __name__ == "__main__":
    main()
