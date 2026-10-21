#!/usr/bin/env python3
"""Alibaba-Quark LiveAvatar 14B self-hosted model server (docs/protocol.md).

LiveAvatar (Wan2.2-S2V-14B + the LiveAvatar DMD LoRA, 4 denoising steps) generates an audio-driven avatar video from
one portrait, block by block (3 latent frames = 12 video frames at 25 fps), with a rolling KV cache, a reference
"attention sink" and upstream's timestep-forcing pipeline parallelism (TPP): every denoising step keeps its own KV
cache, so the four steps can run on four GPUs as a pipeline, followed by a streaming VAE decoder on a fifth.

Launch (rank 0 serves the WebSocket; every rank holds the model parts of its pipeline stage):

    torchrun --nproc_per_node=5 server.py --port 8765          # TPP as upstream: 4 DiT ranks (1 step each) + 1 VAE rank
    torchrun --nproc_per_node=3 server.py --dit_ranks 2 ...     # 2 DiT ranks (2 steps each) + 1 VAE rank
    python server.py --port 8765                                # one GPU: 4 steps + VAE in one process

Tracks
  * avatar_performance (native): supplied speech arrives as ``audio_in`` (16 kHz PCM, 10 ms chunks) and drives the
    lips; visual controls arrive as ``text`` / ``control`` events and are appended to the text prompt of the next
    block (see README "Control mapping"); controls without body / face content are acked unsupported.
  * digital_human_audio / digital_human_text as a composition renderer: the client's shared Qwen ASR -> LLM -> TTS
    stack sends the TTS speech as ``audio_in``; the server only renders it. Dialogue ``text`` events are acked
    unsupported here (the composition layer consumes them client-side).

Real time, no look-ahead
  Block n covers video frames 12n .. 12n+11 and is conditioned on the wav2vec2 features of exactly those frames, so it
  starts only once the session clock has passed the end of its last frame plus ``--lookahead`` (0.1 s of right
  context for wav2vec2); audio that has not arrived by then is silence. A control event changes the prompt of the next
  block that starts after it arrives. Frames are paced out at 25 fps together with the matching 48 kHz slice of the
  received speech (``audio_out``), so the capture is lip-aligned with a constant delay of about one block + compute
  (see ``Session.pump``); when generation is slower than real time frames simply arrive late (never re-timed).
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
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("QUARK_LIVEAVATAR_ROOT") or (
    os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/quark_liveavatar"))
REPO = Path(os.environ.get("LIVEAVATAR_REPO", MODEL_DIR / "repo"))
sys.path[:0] = [str(REPO), str(HERE.parents[2])]
from streameval.runtime import protocol as sp  # noqa: E402

for _k, _v in {"TORCHINDUCTOR_CACHE_DIR": str(MODEL_DIR / "inductor_cache"), "TRITON_CACHE_DIR": str(MODEL_DIR / "triton_cache"),
               "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ.setdefault(_k, _v)
# Use the ptxas build installed with the configured Triton toolchain.
_PTXAS = MODEL_DIR / "env/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas"
if _PTXAS.exists():
    os.environ["TRITON_PTXAS_PATH"] = str(_PTXAS)

MODEL_NAME = "Quark Live Avatar 14B"
CONTROL_MAPPING_VERSION = "liveavatar-control-map/1"
IN_RATE, OUT_RATE = sp.AUDIO_IN_RATE, sp.AUDIO_OUT_RATE
FPS = 25                                   # Wan2.2-S2V sample_fps (also the audio-embedding frame rate)
BLOCK_LAT = 3                              # latent frames per block (num_frames_per_block)
BLOCK_FRAMES = 4 * BLOCK_LAT               # 12 video frames per block
CLIP_BLOCKS = 4                            # infer_frames 48 = one rolling-KV window of 12 latent frames
MOTION_FRAMES, LAT_MOTION = 73, 19         # framepack motion context (the portrait repeated), as upstream
FIRST_DROP = 3                             # upstream drops the first 3 decoded frames of the stream
STEPS = 4
REFRESH_BLOCK = CLIP_BLOCKS                # upstream re-anchors the sink on the first generated frame at clip 1
LORA = {"rank": 128, "alpha": 64.0, "targets": "q,k,v,o,ffn.0,ffn.2"}   # liveavatar/configs/s2v_causal_sft.yaml
FP8_IGNORE = ["text_embedding", "time_embedding", "time_projection", "head.head",
              "casual_audio_encoder.encoder.final_linear"]

# ---------------------------------------------------------------------------------------------------------------
# Control mapping: benchmark visual control (free English text) -> LiveAvatar's text prompt.
# LiveAvatar is conditioned on one free-text prompt (umT5 cross-attention; upstream: "prompts steer upper-body
# motion"). A control is applied when it asks for a visible movement / pose of the performer's body or head, a gaze
# direction or a facial expression: its text is appended to the session prompt. Camera moves, scene / lighting /
# clothing / style changes and speech content have no counterpart (the camera, look and setting come from the
# reference image; the lips follow the supplied audio only) and are acked "unsupported".
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
    """-> {"status": applied|unsupported, "content": [...], "prompt": str|None, "detail": str}"""
    clean = text or ""
    for p in BOILERPLATE:
        clean = p.sub("", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    body, gaze, face = _hits(ACTION_WORDS, clean), _hits(GAZE_WORDS, clean), _hits(EMOTION_WORDS, clean)
    other = sorted({m.group(0).lower() for p in UNSUPPORTED_PATTERNS for m in re.finditer(p, clean, re.I)})
    content = (["body"] if body else []) + (["gaze"] if gaze else []) + (["expression"] if face else [])
    if not content:
        why = f"no body/face content (matched {other})" if other else "no body/face content"
        return {"status": "unsupported", "content": [], "prompt": None,
                "detail": f"{CONTROL_MAPPING_VERSION}: {why}; LiveAvatar's text prompt steers only the performer's "
                          "motion and expression (camera, setting and look come from the reference image)"}
    detail = f"{CONTROL_MAPPING_VERSION}: appended to the text prompt ({'+'.join(content)})"
    if other:
        detail += f"; parts without a LiveAvatar control ignored ({', '.join(other)})"
    return {"status": "applied", "content": content, "prompt": clean, "detail": detail}


def compose_prompt(base: str, control: str | None) -> str:
    return base if not control else f"{base} {control}"


def gradient_png(w, h):
    from PIL import Image
    x = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.stack([np.tile(x, (h, 1)), np.tile(x[::-1], (h, 1)), np.full((h, w), 128, np.uint8)], -1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------------------------
class Engine:
    """One pipeline stage per rank. Ranks 0..D-1 run the DiT (4/D denoising steps each, every step with its own KV
    cache as in upstream TPP), rank V runs the streaming VAE decoder (V = D, or D-1 when there is no extra rank).
    Rank 0 also owns the session: portrait VAE encode, umT5 prompt encoder, wav2vec2 audio features, WebSocket.
    Block headers travel over a gloo group, tensors over NCCL point-to-point; decoded frames (JPEG) return to rank 0
    over a second gloo group."""

    def __init__(self, args):
        import torch
        import torch.distributed as dist
        self.torch, self.dist, self.args = torch, dist, args
        self.rank = int(os.getenv("RANK", 0))
        self.world = int(os.getenv("WORLD_SIZE", 1))
        self.local = int(os.getenv("LOCAL_RANK", 0))
        self.width, self.height = [int(v) for v in args.size.split("*")]
        self.h8, self.w8 = self.height // 8, self.width // 8
        self.fsl = (self.h8 // 2) * (self.w8 // 2)          # tokens per latent frame
        D = args.dit_ranks or (self.world - 1 if self.world > 1 else 1)
        if STEPS % D or self.world not in (D, D + 1):
            raise SystemExit(f"--dit_ranks {D} with world {self.world}: need D | 4 and world = D (VAE on the last DiT "
                             "rank) or D + 1 (separate VAE rank)")
        self.D, self.V = D, (D if self.world == D + 1 else D - 1)
        per = STEPS // D
        self.steps = list(range(self.rank * per, (self.rank + 1) * per)) if self.rank < D else []
        self.is_dit, self.is_vae, self.is_owner = self.rank < D, self.rank == self.V, self.rank == 0
        self.next = (self.rank + 1 if self.rank < D - 1 else (self.V if self.V != self.rank else None)) \
            if self.is_dit else None
        self.prev = None if self.rank == 0 else self.rank - 1
        torch.cuda.set_device(self.local)
        self.dev = torch.device("cuda", self.local)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if self.world > 1:
            from datetime import timedelta
            dist.init_process_group("nccl", rank=self.rank, world_size=self.world, timeout=timedelta(days=7),
                                    device_id=self.dev)
            self.hdr = dist.new_group(backend="gloo", timeout=timedelta(days=7))
            # results VAE rank -> rank 0; a full-world group (a sub-group blocks its non-members in new_group)
            self.ret = dist.new_group(backend="gloo", timeout=timedelta(days=7))
        self.timing = {}
        self._load()
        if self.world > 1:
            dist.barrier()                 # every rank loaded; also brings up the NCCL communicator on all ranks

    def log(self, msg, all_ranks=False):
        if self.rank == 0 or all_ranks:
            print(f"[liveavatar r{self.rank}] {msg}", flush=True)

    # ---- loading -------------------------------------------------------------------------------------------
    def _load(self):
        torch, args = self.torch, self.args
        os.environ["ENABLE_COMPILE"] = "true" if args.compile else "false"
        t0 = time.time()
        ckpt = args.ckpt_dir
        if self.is_dit:
            from liveavatar.models.wan.causal_model_s2v import CausalWanModel_S2V
            from diffusers import FlowMatchEulerDiscreteScheduler
            model = CausalWanModel_S2V.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map=self.dev)
            model.eval().requires_grad_(False)
            model.to(torch.bfloat16)
            model.num_frame_per_block = BLOCK_LAT
            model = self._merge_lora(model)
            self.fp8 = args.fp8 == "on"
            if self.fp8:
                from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8
                replace_linear_with_scaled_fp8(model, ignore_keys=FP8_IGNORE)
            self.model = model
            self.sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3)
            self.sched.set_timesteps(STEPS, device=self.dev)
            self.timesteps = [self.sched.timesteps[i] for i in range(STEPS)]
            self._alloc_caches()
            self.cond_zero = torch.zeros(1, 16, BLOCK_LAT, self.h8, self.w8, device=self.dev, dtype=torch.bfloat16)
            self.audio_dummy = torch.zeros(1, 25, 1024, BLOCK_FRAMES, device=self.dev, dtype=torch.bfloat16)
        if self.is_vae or self.is_owner:
            from liveavatar.models.wan.wan_2_2.modules.vae_streaming import WanVAE
            self.vae = WanVAE(vae_pth=os.path.join(ckpt, "Wan2.1_VAE.pth"), device=self.dev, dtype=torch.bfloat16)
        if self.is_owner:
            from liveavatar.models.wan.wan_2_2.modules.t5 import T5EncoderModel
            from liveavatar.models.wan.wan_2_2.modules.s2v.audio_encoder import AudioEncoder
            self.t5 = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=self.dev,
                                     checkpoint_path=os.path.join(ckpt, "models_t5_umt5-xxl-enc-bf16.pth"),
                                     tokenizer_path=os.path.join(ckpt, "google/umt5-xxl"))
            self.aenc = AudioEncoder(model_id=os.path.join(ckpt, "wav2vec2-large-xlsr-53-english"))
            self.aenc.model.to(self.dev, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.prompt_cache = {}
        self.load_s = time.time() - t0
        self.log(f"loaded in {self.load_s:.1f}s  size={self.width}x{self.height} world={self.world} dit_ranks={self.D} "
                 f"vae_rank={self.V} steps={self.steps} fp8={getattr(self, 'fp8', None)} compile={args.compile}",
                 all_ranks=True)

    def _merge_lora(self, model):
        """upstream WanS2V.add_lora_to_model: peft LoRA (rank 128, alpha 64) on q,k,v,o,ffn.0,ffn.2, load every tensor
        of liveavatar.safetensors, merge and unload."""
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        model = get_peft_model(model, LoraConfig(r=LORA["rank"], lora_alpha=LORA["alpha"], init_lora_weights=True,
                                                 target_modules=LORA["targets"].split(",")))
        sd = load_file(self.args.lora, device="cpu")
        if not next(iter(sd)).startswith("base_model.model."):
            sd = {f"base_model.model.{k}": v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_all = len(list(model.named_parameters()))
        self.lora_loaded = {"tensors": len(sd), "updated": n_all - len(missing), "unexpected": len(unexpected)}
        self.log(f"LoRA {self.lora_loaded}")
        return model.merge_and_unload()

    def _alloc_caches(self):
        torch = self.torch
        kv_len = CLIP_BLOCKS * BLOCK_LAT * self.fsl
        bf = torch.bfloat16
        n_layers = self.model.num_layers
        self.cond = [{"cond_k": torch.zeros([1, 3000, 40, 128], dtype=bf, device=self.dev),
                      "cond_v": torch.zeros([1, 3000, 40, 128], dtype=bf, device=self.dev),
                      "cond_end": torch.tensor([0], dtype=torch.long, device=self.dev)} for _ in range(n_layers)]
        # one rolling self-attention cache per denoising step (timestep forcing); the clean sink (cond) part is the
        # same for every step on a rank, so it is shared (as upstream's single-GPU pipeline does)
        self.kv = {i: [{"k": torch.zeros([1, kv_len, 40, 128], dtype=bf, device=self.dev),
                        "v": torch.zeros([1, kv_len, 40, 128], dtype=bf, device=self.dev), **self.cond[l]}
                       for l in range(n_layers)] for i in self.steps}
        self.xcache = [{"k": torch.zeros([1, 0, 40, 128], dtype=bf, device=self.dev),
                        "v": torch.zeros([1, 0, 40, 128], dtype=bf, device=self.dev), "is_init": False}
                       for _ in range(n_layers)]

    # ---- rank-0 encoders -----------------------------------------------------------------------------------
    def encode_prompt(self, text):
        """umT5 context, zero-padded to the model's 512 tokens (same as the model's own padding; keeps shapes static)."""
        torch = self.torch
        if text not in self.prompt_cache:
            if len(self.prompt_cache) > 256:
                self.prompt_cache.clear()
            with torch.no_grad():
                ctx = self.t5([text], self.dev)[0].to(torch.bfloat16)
            pad = torch.zeros(512, ctx.shape[1], dtype=ctx.dtype, device=self.dev)
            pad[:ctx.shape[0]] = ctx
            self.prompt_cache[text] = pad
        return self.prompt_cache[text]

    def encode_portrait(self, image_bytes):
        """upstream TPP: resize the short side + center crop, ref latent = VAE(1 frame), motion latents = VAE(portrait
        repeated 73 frames) (19 latent frames)."""
        torch = self.torch
        from PIL import Image
        from torchvision import transforms
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = transforms.CenterCrop((self.height, self.width))(transforms.Resize(min(self.height, self.width))(img))
        px = transforms.ToTensor()(img).unsqueeze(1).unsqueeze(0) * 2 - 1.0          # [1,3,1,H,W]
        px = px.to(self.dev, torch.bfloat16)
        with torch.no_grad():
            ref = torch.stack(self.vae.encode(px))                                     # [1,16,1,h,w] fp32
            motion = torch.stack(self.vae.encode(px.repeat(1, 1, MOTION_FRAMES, 1, 1)))  # [1,16,19,h,w] fp32
        return ref, motion

    def audio_features(self, pcm, s0, frames):
        """wav2vec2-large-xlsr-53 hidden states (25 layers) of a 16 kHz window starting at sample s0 (a multiple of
        320), sampled at the given video frames as upstream does for a whole file (50 fps features, linear
        interpolation to 30 fps, frame f -> 30 fps index round(1.2 f)). -> [1, 25, 1024, len(frames)] bf16"""
        torch = self.torch
        iv = self.aenc.processor(pcm, sampling_rate=IN_RATE, return_tensors="pt").input_values
        with torch.no_grad():
            hs = self.aenc.model(iv.to(self.dev, torch.bfloat16), output_hidden_states=True).hidden_states
        feat = torch.cat(hs).float()                                                   # [25, T, 1024]
        T = feat.shape[1]
        pos = torch.tensor([round(f * 30 / FPS) * 50 / 30 - s0 / 320 for f in frames], device=self.dev)
        pos = pos.clamp(0, T - 1)
        lo = pos.floor().long()
        hi = (lo + 1).clamp(max=T - 1)
        w = (pos - lo.float())[None, :, None]
        emb = feat[:, lo] * (1 - w) + feat[:, hi] * w                                  # [25, F, 1024]
        return emb.permute(0, 2, 1).unsqueeze(0).to(torch.bfloat16)

    # ---- DiT / VAE stage work ------------------------------------------------------------------------------
    def start(self, ref, motion, ctx, seed):
        """New session on this rank: reset caches, prefill the sink (DiT) / prime the stream decoder (VAE)."""
        torch = self.torch
        import random
        random.seed(seed + 1000 * self.rank)                # upstream draws the sink RoPE offset with `random`
        torch.manual_seed(seed)
        if self.is_dit:
            self.ref, self.motion, self.ctx_base = ref, motion, ctx
            for caches in self.kv.values():
                for layer in caches:
                    layer["k"].zero_()
                    layer["v"].zero_()
            for layer in self.cond:
                layer["cond_end"].zero_()
            self.prefill(ref)
        if self.is_vae:
            m = self.vae.model
            m.first_decode = True
            m.clear_cache_decode()
            with torch.no_grad():
                self.vae.stream_decode(motion[:, :, :7].to(torch.bfloat16))   # upstream primes with 7 motion latents

    def prefill(self, ref):
        torch = self.torch
        t = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            self.model([self.cond_zero[0]], t=torch.zeros(1, BLOCK_LAT, device=self.dev), context=[self.ctx_base],
                       seq_len=None, cond_states=self.cond_zero, motion_latents=self.motion, ref_latents=ref,
                       audio_input=self.audio_dummy, motion_frames=[MOTION_FRAMES, LAT_MOTION],
                       drop_motion_frames=False, sink_flag=True, kv_cache=self.kv[self.steps[0]],
                       crossattn_cache=self.xcache, current_start=0, current_end=BLOCK_LAT * self.fsl)
        torch.cuda.synchronize(self.dev)
        self.timing["prefill_ms"] = round((time.perf_counter() - t) * 1000, 1)

    def run_steps(self, latent, n, audio, ctx):
        torch = self.torch
        start = n * BLOCK_LAT * self.fsl
        t = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in self.steps:
                ts = self.timesteps[i]
                pred = self.model([latent], t=torch.full([1, BLOCK_LAT], float(ts), device=self.dev), context=[ctx],
                                  seq_len=None, cond_states=self.cond_zero, motion_latents=self.motion,
                                  ref_latents=self.ref, audio_input=audio, motion_frames=[MOTION_FRAMES, LAT_MOTION],
                                  drop_motion_frames=False, kv_cache=self.kv[i], crossattn_cache=self.xcache,
                                  current_start=start, current_end=start + BLOCK_LAT * self.fsl, mask=None)[0]
                self.sched._step_index, self.sched._begin_index = i, 0
                latent = self.sched.step(pred.unsqueeze(0), ts, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)
        torch.cuda.synchronize(self.dev)
        self.timing[f"dit_r{self.rank}_ms"] = round((time.perf_counter() - t) * 1000, 1)
        return latent

    def decode(self, latent, n):
        """-> uint8 frames [T, H, W, 3] (12 per block, 9 for the first block)"""
        torch = self.torch
        t = time.perf_counter()
        with torch.no_grad():
            img = torch.stack(self.vae.stream_decode(latent.unsqueeze(0)))[:, :, -BLOCK_FRAMES:]
            if n == 0:
                img = img[:, :, FIRST_DROP:]
            frames = ((img[0].float().clamp(-1, 1) + 1) * 127.5 + 0.5).clamp(0, 255).to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).contiguous().cpu().numpy()
        self.timing["vae_ms"] = round((time.perf_counter() - t) * 1000, 1)
        return frames

    # ---- transport -----------------------------------------------------------------------------------------
    def send(self, dst, hdr, tensors):
        """header (gloo) then tensors (NCCL), in order; hdr["tensors"] describes them for the receiver."""
        hdr = dict(hdr, tensors=[(name, tuple(t.shape), str(t.dtype).replace("torch.", "")) for name, t in tensors])
        self.dist.send_object_list([hdr], dst=dst, group=self.hdr)
        for _, t in tensors:
            self.dist.send(t.contiguous(), dst=dst)

    def recv(self, src):
        box = [None]
        self.dist.recv_object_list(box, src=src, group=self.hdr)
        hdr = box[0]
        out = {}
        for name, shape, dtype in hdr.get("tensors", []):
            buf = self.torch.empty(shape, dtype=getattr(self.torch, dtype), device=self.dev)
            self.dist.recv(buf, src=src)
            out[name] = buf
        return hdr, out

    def to_owner(self, obj):
        """VAE rank -> rank 0 (gloo). Rank 0 handles its own results directly."""
        self.dist.send_object_list([obj], dst=0, group=self.ret)

    def jpegs(self, frames):
        import cv2
        q = [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality]
        return [cv2.imencode(".jpg", fr[:, :, ::-1], q)[1].tobytes() for fr in frames]

    def stage(self, hdr, ts, deliver):
        """Run this rank's part of one message and pass it on. `deliver(obj)` hands results to rank 0's session."""
        op = hdr["op"]
        if op == "start":
            if self.next is not None:          # downstream ranks prefill in parallel
                self.send(self.next, {k: v for k, v in hdr.items() if k != "tensors"},
                          [(k, ts[k]) for k in (("ref", "motion", "ctx") if self.next < self.D else ("motion",))])
            self.start(ts.get("ref"), ts["motion"], ts.get("ctx"), hdr["seed"])
            if self.is_vae:
                deliver({"op": "started", "sid": hdr["sid"], "timing": dict(self.timing)})
            return
        if op != "block":
            return
        n = hdr["n"]
        timing = dict(hdr.get("timing") or {})
        latent = ts["latent"]
        if self.is_dit:
            if "ctx" in ts:
                self.ctx = ts["ctx"]
            if "ref" in ts:                    # clip 1: re-anchor the sink on the first generated latent frame
                self.ref = ts["ref"]
                self.prefill(self.ref)
                timing[f"refresh_r{self.rank}_ms"] = self.timing["prefill_ms"]
            latent = self.run_steps(latent, n, ts["audio"], self.ctx)
            timing[f"dit_r{self.rank}_ms"] = self.timing[f"dit_r{self.rank}_ms"]
            if self.next is not None:
                h = {k: v for k, v in hdr.items() if k != "tensors"}
                h["timing"] = timing
                if self.next < self.D:
                    fw = [("latent", latent), ("audio", ts["audio"])] + [(k, ts[k]) for k in ("ctx", "ref") if k in ts]
                else:
                    fw = [("latent", latent)]
                self.send(self.next, h, fw)
        if self.is_vae:
            frames = self.decode(latent, n)
            timing["vae_ms"] = self.timing["vae_ms"]
            t = time.perf_counter()
            jp = self.jpegs(frames)
            timing["jpeg_ms"] = round((time.perf_counter() - t) * 1000, 1)
            res = {"op": "frames", "sid": hdr["sid"], "n": n,
                   "first_index": FIRST_DROP if n == 0 else BLOCK_FRAMES * n, "jpegs": jp, "timing": timing,
                   "done_mono": time.monotonic()}
            if n == 0:
                res["ref"] = latent.unsqueeze(0)[:, :, 0:1].cpu()
            deliver(res)

    def worker_loop(self):
        """Ranks >= 1: receive from the previous rank, work, pass on; forever."""
        deliver = self.to_owner
        while True:
            hdr, ts = self.recv(self.prev)
            if hdr["op"] == "quit":
                return
            self.stage(hdr, ts, deliver)

    def return_loop(self, sink):
        """Rank 0 (separate VAE rank): receive results from the VAE rank, forever."""
        while True:
            box = [None]
            self.dist.recv_object_list(box, src=self.V, group=self.ret)
            sink(box[0])


# ---------------------------------------------------------------------------------------------------------------
class Session:
    """Rank-0 session state. The generation thread runs rank 0's stage; the asyncio loop owns the socket."""

    _next_sid = 0

    def __init__(self, server, conn, loop, realtime=True):
        Session._next_sid += 1
        self.sid = Session._next_sid
        self.server, self.conn, self.loop, self.realtime = server, conn, loop, realtime
        self.engine = server.engine
        self.lock = threading.Lock()
        self.audio = np.zeros(0, dtype=np.float32)       # session-timeline speech, 16 kHz, index 0 = ready time
        self.anchor = None                                # session sample offset of stream pts 0
        self.last_pts = -1
        self.control_prompt = None
        self.ended = threading.Event()
        self.started = threading.Event()
        self.ref_refresh = queue.Queue()
        self.t0 = None
        self.out = queue.Queue()
        self.stats = {"blocks": [], "events": []}
        self.error = None
        self.ready_evt = None
        self.max_blocks = None
        self.pump_task = None

    # audio_in: place each 10 ms chunk on the session timeline by its stream pts (re-anchored on arrival if the
    # client restarts its stream clock). Chunks arriving after the block that needed them was generated are late.
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
            seg = self.audio[max(0, a):b]
        return np.concatenate([np.zeros(max(0, -a), np.float32), seg,
                               np.zeros(max(0, b - max(0, a) - len(seg)), np.float32)])

    def on_result(self, res):
        """Results for this session (from the VAE rank or rank 0 itself)."""
        if res["op"] == "started":
            self.stats["start_timing"] = res.get("timing")
            self.started.set()
        elif res["op"] == "frames":
            if res["n"] == 0 and "ref" in res:
                self.ref_refresh.put(res["ref"])
            self.out.put(res)

    def generate(self, init):
        """Generation thread (rank 0)."""
        eng, args, torch = self.engine, self.server.args, self.engine.torch
        torch.cuda.set_device(eng.local)
        seed = int(init["seed"])
        base = init["prompt"]
        t = time.perf_counter()
        ref, motion = eng.encode_portrait(init["image"])
        ctx_base = eng.encode_prompt(base)
        self.stats["encode_ms"] = round((time.perf_counter() - t) * 1000, 1)
        hdr = {"op": "start", "sid": self.sid, "seed": seed}
        ts = {"ref": ref, "motion": motion, "ctx": ctx_base}
        eng.stage(hdr, ts, self.on_result)
        if not eng.is_vae and not self.started.wait(600):
            raise RuntimeError("pipeline did not acknowledge the session start")
        self.t0 = time.monotonic()
        self.loop.call_soon_threadsafe(self.ready_evt.set)
        gen = torch.Generator(device=eng.dev)
        last_ctx = None
        n = 0
        while not self.ended.is_set() and (self.max_blocks is None or n < self.max_blocks):
            first = BLOCK_FRAMES * n
            need = (first + BLOCK_FRAMES) / FPS + args.lookahead            # session seconds the block needs
            if self.realtime:
                wait = self.t0 + need - time.monotonic()
                if wait > 0 and self.ended.wait(wait):
                    break
            r, b = divmod(n, CLIP_BLOCKS)
            if b == 0:                                                       # upstream: noise per clip, seed + r
                gen.manual_seed(seed + r)
                clip_noise = torch.randn(16, CLIP_BLOCKS * BLOCK_LAT, eng.h8, eng.w8, dtype=torch.bfloat16,
                                         device=eng.dev, generator=gen)
            latent = clip_noise[:, b * BLOCK_LAT:(b + 1) * BLOCK_LAT].contiguous()
            s1 = int(round(need * IN_RATE))
            s0 = s1 - int(args.audio_context * IN_RATE) // 320 * 320   # fixed-length window; < 0 = silence before t0
            t_a = time.perf_counter()
            audio = eng.audio_features(self.audio_window(s0, s1), s0, range(first, first + BLOCK_FRAMES))
            prompt = compose_prompt(base, self.control_prompt)
            ctx = eng.encode_prompt(prompt)
            pre_ms = round((time.perf_counter() - t_a) * 1000, 1)
            ts = {"latent": latent, "audio": audio}
            if ctx is not last_ctx:
                ts["ctx"], last_ctx = ctx, ctx
            if n == REFRESH_BLOCK:
                try:
                    ts["ref"] = self.ref_refresh.get(timeout=600).to(eng.dev)
                except queue.Empty:
                    raise RuntimeError("first block never came back from the VAE rank")
            t_start = time.monotonic()
            stat = {"n": n, "first_frame": first, "audio_needed_s": round(need, 3),
                    "start_s": round(t_start - self.t0, 3), "prompt": prompt}
            self.stats["blocks"].append(stat)
            eng.stage({"op": "block", "sid": self.sid, "n": n, "timing": {"pre_ms": pre_ms}}, ts, self.on_result)
            stat["rank0_done_s"] = round(time.monotonic() - self.t0, 3)
            n += 1
        self.out.put(None)

    async def pump(self):
        """Pace frames at 25 fps with the matching speech slice: frame idx goes out at anchor + idx/fps. The anchor
        is the earliest (over blocks) of done - 12n/fps + slack (block n nominally starts at frame 12n; the first
        block drops its first 3 frames but needs the same audio), i.e. the steady-state output delay; frames of a
        block that is later than that go out at once."""
        loop = asyncio.get_running_loop()
        anchor = None
        spf = OUT_RATE // FPS
        slack = self.server.args.pace_slack
        done_s = {}
        while True:
            item = await loop.run_in_executor(None, self.out.get)
            if item is None:
                break
            n, first_index, jpegs = item["n"], item["first_index"], item["jpegs"]
            t_done = item.get("done_mono", time.monotonic())
            cand = t_done - BLOCK_FRAMES * n / FPS + slack
            anchor = cand if anchor is None else min(anchor, cand)
            done_s[n] = round(t_done - self.t0, 3)
            if n < len(self.stats["blocks"]):
                self.stats["blocks"][n].update(done_s=done_s[n], **{k: v for k, v in item["timing"].items()})
            for k, jpg in enumerate(jpegs):
                idx = first_index + k
                delay = anchor + idx / FPS - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                a = int(IN_RATE * idx / FPS)
                pcm16 = self.audio_window(a, a + IN_RATE // FPS)
                pcm48 = np.interp(np.arange(spf) / 3.0, np.arange(len(pcm16)), pcm16)
                now = time.monotonic_ns()
                if self.conn is not None:
                    await self.conn.frame(jpg, now)
                    await self.conn.audio((np.clip(pcm48, -1, 1) * 32767).astype("<i2").tobytes(), now)
                self.stats.setdefault("frame_sent_s", []).append(round(time.monotonic() - self.t0, 4))
        if self.error and self.conn is not None:
            await self.conn.error(self.error)


class Server:
    def __init__(self, engine, args):
        self.engine, self.args = engine, args
        self.busy = False
        self.gpu_lock = threading.Lock()
        # every rank-0 session (and the warm-up) runs on this one thread, so per-thread CUDA library state set up
        # during the warm-up is reused by every session
        from concurrent.futures import ThreadPoolExecutor
        self.gen_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rank0-gen")
        self.current = None
        self.config = self._config()

    def _config(self):
        eng, args = self.engine, self.args
        torch = eng.torch
        weights = {}
        p = MODEL_DIR / "weights_hash.json"
        if p.exists():
            weights = json.loads(p.read_text())
        commit = (MODEL_DIR / "repo_commit.txt").read_text().strip() if (MODEL_DIR / "repo_commit.txt").exists() else None
        return {"model": MODEL_NAME, "repo": "https://github.com/Alibaba-Quark/LiveAvatar", "repo_commit": commit,
                "hf_weights": "Wan-AI/Wan2.2-S2V-14B + Quark-Vision/Live-Avatar (LoRA merged)", "weights": weights,
                "width": eng.width, "height": eng.height, "fps": FPS, "steps": STEPS, "solver": "euler, shift 3",
                "block_frames": BLOCK_FRAMES, "kv_window_latent_frames": CLIP_BLOCKS * BLOCK_LAT,
                "gpu": torch.cuda.get_device_name(eng.dev), "gpu_count": eng.world,
                "parallel": (f"timestep-forcing pipeline: {eng.D} DiT rank(s) x {STEPS // eng.D} step(s), VAE on rank "
                             f"{eng.V}") if eng.world > 1 else "single GPU (4 steps + VAE sequential)",
                "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(eng.dev))),
                "fp8": args.fp8 == "on", "torch_compile": args.compile, "guidance_scale": 0,
                "audio": "wav2vec2-large-xlsr-53-english, 25 layers, %.1f s left context, %.2f s look-ahead" % (
                    args.audio_context, args.lookahead),
                "prompt": "init.prompt (rendering_context) or --base_prompt, + latest control; sink prefill uses "
                          "the session prompt",
                "control_mapping": CONTROL_MAPPING_VERSION, "lora": getattr(eng, "lora_loaded", None),
                "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "load_s": round(eng.load_s, 1)}

    def on_result(self, res):
        s = self.current
        if s is not None and res.get("sid") == s.sid:
            s.on_result(res)

    async def handler(self, conn):
        for _ in range(1200):                    # the previous session may still be finishing its last block
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
        gen_future = ready_task = None
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
                    prompt = (m.get("prompt") or "").strip() or self.args.base_prompt
                    sess.stats.update({"case_id": m.get("case_id"), "track": track, "seed": seed, "prompt": prompt})
                    init = {"image": base64.b64decode(m["image_b64"]), "seed": seed, "prompt": prompt}
                    self.current = sess
                    gen_future = loop.run_in_executor(self.gen_pool, self._generate_locked, sess, init)
                    # keep reading the socket while the session starts: audio_in sent before `ready` must be
                    # dropped as it arrives, not queue up and shift the audio timeline
                    ready_task = asyncio.create_task(self._announce(conn, sess))
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
                    await conn.ack(m.get("event_id"), "unsupported", "LiveAvatar has no reference-image editing")
                elif kind == "end":
                    break
        except sp.ConnectionClosed:
            pass
        finally:
            # close the socket at once; rank 0's running block finishes in the background before the next session
            sess.ended.set()
            for task in (ready_task, sess.pump_task):
                if task is not None:
                    task.cancel()
            await conn.close()
            if gen_future is not None:
                await gen_future
            self._write_stats(sess)
            self.busy = False

    async def _announce(self, conn, sess):
        await sess.ready_evt.wait()
        if sess.error:
            await conn.error(sess.error)
            return
        await conn.ready(MODEL_NAME, self.config["repo_commit"], self.engine.width, self.engine.height, FPS,
                         protocol=sp.PROTOCOL_VERSION, config=self.config)
        sess.pump_task = asyncio.create_task(sess.pump())

    def _generate_locked(self, sess, init):
        with self.gpu_lock:
            try:
                sess.generate(init)
            except Exception as exc:                      # surface to the client instead of hanging
                import traceback
                traceback.print_exc()
                sess.error = f"{type(exc).__name__}: {exc}"
                sess.out.put(None)
                if sess.ready_evt is not None:
                    sess.loop.call_soon_threadsafe(sess.ready_evt.set)

    def _write_stats(self, sess):
        s = sess.stats
        b = [x for x in s["blocks"] if "done_s" in x]
        if b:
            frames = sum(BLOCK_FRAMES if x["n"] else BLOCK_FRAMES - FIRST_DROP for x in b)
            span = b[-1]["done_s"] - b[0]["start_s"]
            keys = sorted({k for x in b for k in x if k.endswith("_ms")})
            s["summary"] = {
                "blocks": len(b), "frames": frames,
                "latency_s_mean": round(float(np.mean([x["done_s"] - x["start_s"] for x in b[2:] or b])), 3),
                "wall_fps": round(frames / max(1e-6, span), 2) if span > 0 else None,
                "stage_ms_mean": {k: round(float(np.mean([x[k] for x in b[2:] or b if k in x])), 1) for k in keys},
                "realtime_budget_ms_per_block": round(BLOCK_FRAMES / FPS * 1000, 1),
                "max_start_lag_s": round(max(x["start_s"] - x["audio_needed_s"] for x in s["blocks"]), 3),
                "first_frame_s": b[0]["done_s"]}
            self.engine.log("session summary " + json.dumps(s["summary"]))
        d = Path(self.args.stats_dir)
        d.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(s.get("case_id") or "session"))
        (d / f"{name}.{int(time.time())}.json").write_text(json.dumps({"config": self.config, **s}, indent=1))

    def health(self):
        return {"ok": True, "model": MODEL_NAME, "busy": self.busy}

    def warmup(self, blocks):
        """Compile / autotune every stage before /health: synthetic sessions at full speed (noise audio). The second,
        short session compiles whatever a session restart re-traces, so the first real session starts as fast as
        every later one."""
        eng = self.engine

        class _Loop:
            def call_soon_threadsafe(self, fn, *a):
                fn(*a)

        t0 = time.time()
        for count in (blocks, REFRESH_BLOCK + 2):
            sess = Session(self, None, _Loop(), realtime=False)
            sess.ready_evt = threading.Event()
            sess.max_blocks = count
            sess.audio = (np.random.default_rng(0).standard_normal(IN_RATE * 60) * 0.05).astype(np.float32)
            self.current = sess
            ts = time.time()
            th = self.gen_pool.submit(self._generate_locked, sess, {
                "image": gradient_png(eng.width, eng.height), "seed": 0,
                "prompt": "A person is talking naturally to the camera."})
            sess.ready_evt.wait()
            eng.log(f"warmup session ready after {time.time() - ts:.2f}s")
            got, last = 0, time.time()
            while got < count:
                item = sess.out.get(timeout=3600)
                if item is None:
                    if sess.error:
                        raise RuntimeError(f"warmup failed: {sess.error}")
                    continue
                got += 1
                now = time.time()
                eng.log(f"warmup block {item['n']}: +{now - last:.2f}s (t={now - t0:.1f}s) {item['timing']}")
                last = now
            th.result()
        self.current = None
        eng.log(f"warmup done in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ckpt_dir", default=str(MODEL_DIR / "weights/Wan2.2-S2V-14B"))
    ap.add_argument("--lora", default=str(MODEL_DIR / "weights/LiveAvatar/liveavatar.safetensors"))
    ap.add_argument("--size", default="512*512", help="W*H (benchmark portraits are square; upstream's 720*400 "
                                                      "area gives 512x512 for a square portrait)")
    ap.add_argument("--dit_ranks", type=int, default=0, help="DiT pipeline ranks (default world-1, or 1)")
    ap.add_argument("--fp8", choices=["on", "off"], default="off", help="upstream's scaled-FP8 linear layers")
    ap.add_argument("--no_compile", dest="compile", action="store_false")
    ap.add_argument("--seed", type=int, default=420)
    ap.add_argument("--base_prompt", default="A person is talking naturally to the camera.")
    ap.add_argument("--lookahead", type=float, default=0.1, help="s of audio past a block's last frame (wav2vec2 "
                                                                 "right context) before the block starts")
    ap.add_argument("--audio_context", type=float, default=4.0, help="s of wav2vec2 left context")
    ap.add_argument("--warmup_blocks", type=int, default=10)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--pace_slack", type=float, default=0.15, help="s of output delay kept for compute jitter")
    ap.add_argument("--stats_dir", default=str(MODEL_DIR / "logs/sessions"))
    ap.add_argument("--bench", action="store_true", help="warm up, print stage timings, exit")
    args = ap.parse_args()
    os.environ["ENABLE_COMPILE"] = "true" if args.compile else "false"

    engine = Engine(args)
    if engine.rank != 0:
        engine.worker_loop()
        return
    server = Server(engine, args)
    if engine.world > 1 and engine.V != 0:
        threading.Thread(target=engine.return_loop, args=(server.on_result,), daemon=True).start()
    server.warmup(args.warmup_blocks)
    if args.bench:
        engine.log("bench done")
        os._exit(0)
    engine.log(f"listening on ws://{args.host}:{args.port}/session")
    sp.serve(server.handler, args.host, args.port, health=server.health)


if __name__ == "__main__":
    main()
