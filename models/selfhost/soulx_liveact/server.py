#!/usr/bin/env python3
"""SoulX-LiveAct 18B self-hosted model server (docs/protocol.md).

Launch (2 GPUs, Ulysses sequence parallel as upstream; rank 0 serves the WebSocket, all ranks run the model):

    torchrun --nproc_per_node=2 --master_port=29641 server.py --port 8765 --size 512*512 --fps 20

Tracks
  * avatar_performance (native): supplied speech arrives as ``audio_in`` (16 kHz PCM, 10 ms chunks) and drives the
    lips; visual controls arrive as ``text`` / ``control`` events and are mapped onto LiveAct's text
    action/emotion prompt (see README "Control mapping"); controls outside that vocabulary are acked unsupported.
  * digital_human_audio / digital_human_text as a composition renderer: the client's shared Qwen ASR -> LLM -> TTS
    stack sends the TTS speech as ``audio_in``; the server only renders it. Dialogue ``text`` events are acked
    unsupported here (the composition layer consumes them client-side).

Real time, no look-ahead
  LiveAct generates blocks of 32 frames (the first block 21) from a sliding 53-frame audio window (upstream
  ``generate.py --steam_audio``). Block n needs speech up to frame 32(n-1)+55 (n>=1; 55 for n=0), so it starts
  only once that much session time has elapsed since ``ready``; audio that has not arrived by then is silence.
  A control event changes the text prompt of the next block that starts after it arrives. Frames are paced out
  at the model fps together with the matching 48 kHz slice of the received speech (``audio_out``), so the capture
  is lip-aligned, with a constant delay of about one block + compute (see ``Session.pump``); when generation is
  slower than real time frames simply arrive late (never re-timed).
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
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("SOULX_LIVEACT_ROOT") or (
    os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/soulx_liveact"))
REPO = Path(os.environ.get("LIVEACT_REPO", MODEL_DIR / "repo"))
sys.path[:0] = [str(REPO), str(HERE.parents[2])]
from streameval.runtime import protocol as sp  # noqa: E402

for _k, _v in {"USE_CHANNELS_LAST_3D": "1", "TORCHINDUCTOR_CACHE_DIR": str(MODEL_DIR / "inductor_cache"),
               "TRITON_CACHE_DIR": str(MODEL_DIR / "triton_cache")}.items():
    os.environ.setdefault(_k, _v)
# triton 3.4 (torch 2.8) rejects a CUDA 13 ptxas, and its bundled 12.8 ptxas lacks sm_103 (compute capability 10.3);
# setup.sh installs nvidia-cuda-nvcc-cu12 12.9 into the env for this.
_PTXAS = MODEL_DIR / "env/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas"
if _PTXAS.exists():
    os.environ["TRITON_PTXAS_PATH"] = str(_PTXAS)

MODEL_NAME = "SoulX-LiveAct 18B"
CONTROL_MAPPING_VERSION = "liveact-control-map/1"
IN_RATE, OUT_RATE = sp.AUDIO_IN_RATE, sp.AUDIO_OUT_RATE
BLOCK_LAT = [6, 8]              # latent frames of the first / later blocks (upstream blksz_lst)
VAE_T = 4
FRAME_NUM = (sum(BLOCK_LAT) - 1) * VAE_T + 1          # 53-frame audio/context window
TIMESTEPS = [1000.0, 937.5, 833.33333333, 0.0]       # 3 denoising steps (upstream)

# ---------------------------------------------------------------------------------------------------------------
# Control mapping: benchmark visual control (free English text) -> LiveAct text prompt channel.
# LiveAct's editable controls are an action prompt and an emotion prompt fed through the same umT5 text encoder
# (upstream examples: bow, nod, wave hand, wide eyes, wry smile, puzzled face). A control is applied when it asks
# for a visible movement / pose of the performer's body (action) or a facial expression (emotion); gaze and eye
# direction are eye actions. Everything else (camera, scene, lighting, props, clothing, style, speaking content)
# has no counterpart and is acked "unsupported".
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
# camera / scene / appearance operations (a gaze target such as "look at the camera" is not a camera operation)
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
    channels = (["action"] if action or gaze else []) + (["emotion"] if emotion else [])
    if not channels:
        why = f"no action/emotion content (matched {other})" if other else "no action/emotion content"
        return {"status": "unsupported", "channels": [], "prompt": None,
                "detail": f"{CONTROL_MAPPING_VERSION}: {why}; LiveAct exposes only action/emotion prompts"}
    detail = f"{CONTROL_MAPPING_VERSION}: mapped to {'+'.join(channels)} prompt"
    if other:
        detail += f"; parts without a LiveAct control ignored ({', '.join(other)})"
    return {"status": "applied", "channels": channels, "prompt": clean, "detail": detail}


def compose_prompt(base: str, control: str | None) -> str:
    return base if not control else f"{base} {control}"


# ---------------------------------------------------------------------------------------------------------------
def sox_tempo(x: np.ndarray, rate: float) -> np.ndarray:
    """Upstream resample_audio(): sox tempo 25/fps then x3 gain (the input is already 16 kHz)."""
    if len(x) == 0:
        return x
    try:
        import torch
        import torchaudio
        y, _ = torchaudio.sox_effects.apply_effects_tensor(torch.from_numpy(x)[None], IN_RATE,
                                                           [["tempo", f"{rate}"]])
        y = y[0].numpy()
    except Exception:
        proc = subprocess.run(["sox", "-q", "-t", "f32", "-r", str(IN_RATE), "-c", "1", "-", "-t", "f32", "-",
                               "tempo", f"{rate}"], input=x.astype(np.float32).tobytes(),
                              stdout=subprocess.PIPE, check=True)
        y = np.frombuffer(proc.stdout, dtype=np.float32)
    return (y * 3.0).astype(np.float32)


class Engine:
    """Model state shared by all ranks; every GPU call happens in lockstep driven by rank 0's commands."""

    def __init__(self, args):
        import torch
        import torch.distributed as dist
        self.torch, self.dist, self.args = torch, dist, args
        self.rank = int(os.getenv("RANK", 0))
        self.world = int(os.getenv("WORLD_SIZE", 1))
        self.device = int(os.getenv("LOCAL_RANK", 0))
        self.width, self.height = [int(v) for v in args.size.split("*")]
        self.fps = args.fps
        torch.cuda.set_device(self.device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if self.world > 1:
            from datetime import timedelta
            dist.init_process_group("nccl", rank=self.rank, world_size=self.world, timeout=timedelta(hours=6))
            self.ctrl = dist.new_group(backend="gloo", timeout=timedelta(hours=6))
            from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
            init_distributed_environment(rank=self.rank, world_size=self.world)
            initialize_model_parallel(sequence_parallel_degree=self.world, ring_degree=1, ulysses_degree=self.world)
            import model_liveact.model_memory_sp as mm
            self._patch_attention(mm)
        else:
            self.ctrl = None
            import model_liveact.model_memory as mm
        self._load(mm)

    def _patch_attention(self, mm):
        """Upstream hard-codes yunchang's SAGE_FP8_SM90 kernel (Hopper only). Elsewhere: SageAttention's generic FP8
        kernel on sm89/sm120 (same algorithm, INT8 QK + FP8 PV); on Blackwell (sm_10x, where SageAttention 2.2 has
        no kernel) exact cuDNN SDPA (needs cuDNN >= 9.1x, see setup.sh), else flash-attn 2. The choice follows the
        compute capability reported by PyTorch."""
        cap = self.torch.cuda.get_device_capability(self.device)
        name = self.args.attn
        if name == "auto":
            name = {(9, 0): "SAGE_FP8_SM90", (8, 9): "SAGE_FP8", (12, 0): "SAGE_FP8"}.get(
                cap, "TORCH_CUDNN" if cap[0] == 10 else "FA")
        self.attn_kernel = name
        if not name.startswith("SAGE"):
            mm.USE_SAGEATTN = False          # cross-attention: sdpa instead of sageattn (no sm_103 kernel)
        if name != "SAGE_FP8_SM90":
            real = mm.AttnType

            class _Shim:
                def __getattr__(self, attr):
                    return getattr(real, attr)
            shim = _Shim()
            shim.SAGE_FP8_SM90 = getattr(real, name)
            mm.AttnType = shim

    def _load(self, mm):
        torch, args = self.torch, self.args
        from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
        from wan.modules.clip import CLIPModel
        from wan.modules.t5 import T5EncoderModel
        from transformers import Wav2Vec2FeatureExtractor
        from src.audio_analysis.wav2vec2 import Wav2Vec2Model
        from torchvision import transforms
        from util_liveact import center_rescale_crop_keep_ratio, get_embedding, get_msk, get_audio_emb
        from fp8_gemm import FP8GemmOptions, enable_fp8_gemm
        self.get_embedding, self.get_msk, self.get_audio_emb = get_embedding, get_msk, get_audio_emb
        t0 = time.time()
        dev = self.device
        self.frame_len = (self.height // 16) * (self.width // 16)
        tokens = self.frame_len * sum(BLOCK_LAT) // self.world
        kv_dtype = torch.float8_e4m3fn if args.fp8_kv_cache else torch.bfloat16

        def cache():
            return {i: {l: {"k": torch.zeros([1, tokens, 40, 128], dtype=kv_dtype, device=dev),
                            "v": torch.zeros([1, tokens, 40, 128], dtype=kv_dtype, device=dev),
                            "k_scale": torch.ones((1, tokens, 40, 1), dtype=torch.float32, device=dev)
                            if args.fp8_kv_cache else None,
                            "v_scale": torch.ones((1, tokens, 40, 1), dtype=torch.float32, device=dev)
                            if args.fp8_kv_cache else None,
                            "mean_memory": False, "offload_cache": False, "fp8_kv_cache": args.fp8_kv_cache}
                        for l in range(40)} for i in range(len(TIMESTEPS) - 1)}
        self.kv_cache = cache()
        model = mm.WanModel.from_pretrained(args.ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
        model = model.to(dtype=torch.bfloat16)
        for n in range(40):
            model.blocks[n].self_attn.init_kvidx(self.frame_len, self.world)
        if args.fp8_gemm == "auto":       # vLLM's cutlass FP8 GEMM fails ("Error Internal") on sm_103
            args.fp8_gemm = "on" if torch.cuda.get_device_capability(dev) in ((8, 9), (9, 0), (12, 0)) else "off"
        if args.fp8_gemm == "on":
            enable_fp8_gemm(model, options=FP8GemmOptions())
        model = model.to(dev)
        model.freqs = model.freqs.to(dev)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = torch.compile(model) if args.compile else model
        self.vae = LightVAE(vae_path=os.path.join(args.ckpt_dir, "Wan2.1_VAE.pth"), dtype=torch.bfloat16,
                            device=dev, use_lightvae=False, parallel=(self.world > 1))
        self.vae.model.eval()
        self.clip = CLIPModel(
            checkpoint_path=os.path.join(args.ckpt_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            tokenizer_path=os.path.join(args.ckpt_dir, "xlm-roberta-large"), dtype=torch.bfloat16, device=dev)
        self.clip.model = self.clip.model.to(dev, dtype=torch.bfloat16)
        self.t5 = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=dev,
                                 checkpoint_path=os.path.join(args.ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                                 tokenizer_path=os.path.join(args.ckpt_dir, "google/umt5-xxl"))
        self.audio_encoder = Wav2Vec2Model.from_pretrained(args.wav2vec_dir, local_files_only=True,
                                                           torch_dtype=torch.bfloat16).to(dev).eval()
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.w2v_fe = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_dir, local_files_only=True)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda im: center_rescale_crop_keep_ratio(im, (self.height, self.width))),
            transforms.ToTensor(), transforms.Resize((self.height, self.width)),
            transforms.Normalize([0.5] * 3, [0.5] * 3)])
        self.timesteps = [torch.tensor([t], device=dev, dtype=torch.float32) for t in TIMESTEPS]
        self.ref_masks = torch.ones(3, self.height // 8, self.width // 8, device=dev, dtype=torch.bfloat16)
        self.prompt_cache = {}
        self.load_s = time.time() - t0
        self.log(f"loaded in {self.load_s:.1f}s  size={self.width}x{self.height} world={self.world}")

    def log(self, msg):
        if self.rank == 0:
            print(f"[liveact] {msg}", flush=True)

    # ---- lockstep commands ----------------------------------------------------------------------------------
    def encode_prompt(self, text):
        if text not in self.prompt_cache:
            if len(self.prompt_cache) > 256:
                self.prompt_cache.clear()
            with self.torch.no_grad():
                self.prompt_cache[text] = self.t5(texts=text, device=self.device)[0].to(self.device,
                                                                                         self.torch.bfloat16)
        return [self.prompt_cache[text]]

    def start(self, cmd):
        """New session: portrait -> CLIP features and the VAE-encoded conditioning latent; seed the RNG."""
        torch = self.torch
        from PIL import Image
        image = Image.open(io.BytesIO(cmd["image"])).convert("RGB")
        cond = self.transform(image).unsqueeze(1).unsqueeze(0).to(self.device, torch.bfloat16)
        with torch.no_grad():
            self.clip_ctx = self.clip.visual(cond)
            msk = self.get_msk(FRAME_NUM, cond, (4, 8, 8), self.device)
            pad = torch.zeros(1, 3, FRAME_NUM - 1, self.height, self.width, device=self.device, dtype=torch.bfloat16)
            y = self.vae.encode(torch.concat([cond, pad], dim=2)).to(self.device).unsqueeze(0)
            self.y = torch.concat([msk, y], dim=1)[:, :, :FRAME_NUM // 4 + 1]
        self.encode_prompt(cmd["prompt"])
        torch.manual_seed(int(cmd["seed"]))
        self.pre_latent = None
        self.block_index = 0

    def block(self, cmd):
        """One autoregressive block; returns uint8 frames [T,H,W,3] on rank 0."""
        torch = self.torch
        n = cmd["n"]
        f = 0 if n == 0 else 1
        marks = [time.perf_counter()]

        def mark():
            torch.cuda.synchronize(self.device)
            marks.append(time.perf_counter())
        audio = sox_tempo(np.asarray(cmd["audio"], dtype=np.float32), 25 / self.fps)
        with torch.no_grad():
            emb = self.get_embedding(audio, self.w2v_fe, self.audio_encoder, device=self.device)
            audio_embs = self.get_audio_emb(emb, 0, FRAME_NUM, self.device)
            context = self.encode_prompt(cmd["prompt"])
            mark()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = torch.randn(16, BLOCK_LAT[f], self.height // 8, self.width // 8, dtype=torch.bfloat16,
                                     device=self.device)
                s0, s1 = sum(BLOCK_LAT[:f]), sum(BLOCK_LAT[:f + 1])
                for i in range(len(TIMESTEPS) - 1):
                    t = self.timesteps[i]
                    pred = self.model([latent], t=t, kv_cache=self.kv_cache[i], skip_audio=i not in (1, 2),
                                      context=context, clip_fea=self.clip_ctx, ref_target_masks=self.ref_masks,
                                      audio=audio_embs, y=self.y[:, :, s0:s1], start_idx=s0 * self.frame_len,
                                      end_idx=s1 * self.frame_len, update_cache=n > 1)[0]
                    x0 = latent + (-pred) * (self.timesteps[i][0] / 1000)
                    nxt = self.timesteps[i + 1][0] / 1000
                    latent = (1 - nxt) * x0 + torch.randn_like(x0) * nxt
                mark()
                if f == 0:
                    video = self.vae.decode(latent)
                else:
                    video = self.vae.decode(torch.concat([self.pre_latent[:, -3:], latent], dim=1))[:, :, 9:]
                self.pre_latent = latent
                mark()
            if self.rank != 0:
                return None
            frames = ((video[0].permute(1, 2, 3, 0).float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
        mark()
        self.timing = {k: round((b - a) * 1000, 1) for k, a, b in zip(("audio_text_ms", "dit_ms", "vae_ms", "copy_ms"),
                                                                     marks, marks[1:])}
        return frames.numpy()

    def warmup(self, blocks=4):
        rng = np.random.default_rng(0)
        img = Image_bytes_gradient(self.width, self.height)
        self.start({"image": img, "prompt": "A person is talking naturally.", "seed": 0})
        for n in range(blocks):
            t0 = time.time()
            self.block({"n": n, "audio": (rng.standard_normal(int(IN_RATE * 55 / self.fps)) * 0.05).astype(np.float32),
                        "prompt": "A person is talking naturally." if n < 2 else "A person waves the right hand."})
            self.torch.cuda.synchronize()
            self.log(f"warmup block {n}: {time.time() - t0:.2f}s {getattr(self, 'timing', '')}")

    # ---- command bus ----------------------------------------------------------------------------------------
    def bcast(self, cmd):
        if self.world > 1:
            box = [cmd]
            self.dist.broadcast_object_list(box, src=0, group=self.ctrl)
            return box[0]
        return cmd

    def run(self, cmd):
        op = cmd["op"]
        if op == "start":
            return self.start(cmd)
        if op == "block":
            return self.block(cmd)
        return None

    def worker_loop(self):
        while True:
            cmd = self.bcast(None)
            if cmd is None or cmd["op"] == "idle":
                continue
            if cmd["op"] == "quit":
                return
            self.run(cmd)


def Image_bytes_gradient(w, h):
    from PIL import Image
    x = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.stack([np.tile(x, (h, 1)), np.tile(x[::-1], (h, 1)), np.full((h, w), 128, np.uint8)], -1)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------------------------
class Session:
    """Rank-0 session state. The generation thread owns the GPU; the asyncio loop owns the socket."""

    def __init__(self, server, conn, loop):
        self.server, self.conn, self.loop = server, conn, loop
        self.engine = server.engine
        self.fps = self.engine.fps
        self.lock = threading.Lock()
        self.audio = np.zeros(0, dtype=np.float32)       # session-timeline speech, 16 kHz, index 0 = ready time
        self.anchor = None                                # session sample offset of stream pts 0
        self.last_pts = -1
        self.control_prompt = None
        self.ended = threading.Event()
        self.t0 = None
        self.out = queue.Queue()
        self.stats = {"blocks": [], "events": []}
        self.error = None

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
            seg = self.audio[a:b]
        return np.concatenate([seg, np.zeros(max(0, b - a - len(seg)), np.float32)])

    def generate(self, init):
        """Generation thread (rank 0): lockstep start + blocks, pacing by audio availability."""
        eng = self.engine
        fps = self.fps
        base = init["prompt"]
        eng.run(eng.bcast({"op": "start", "image": init["image"], "prompt": base, "seed": init["seed"]}))
        self.t0 = time.monotonic()
        self.loop.call_soon_threadsafe(self.ready_evt.set)
        n = 0
        while not self.ended.is_set():
            start = 0 if n == 0 else 32 * (n - 1)
            need_until = (start + FRAME_NUM + 2) / fps                 # seconds of session time the block needs
            wait = self.t0 + need_until + self.server.args.audio_margin - time.monotonic()
            if wait > 0 and self.ended.wait(wait):
                break
            a = int(IN_RATE * start / fps)
            b = int(IN_RATE * (start + FRAME_NUM + 2) / fps)
            prompt = compose_prompt(base, self.control_prompt)
            t_start = time.monotonic()
            frames = eng.run(eng.bcast({"op": "block", "n": n, "audio": self.audio_window(a, b),
                                        "prompt": prompt}))
            t_done = time.monotonic()
            first = 0 if n == 0 else 21 + 32 * (n - 1)
            self.stats["blocks"].append({"n": n, "first_frame": first, "frames": int(frames.shape[0]),
                                         "audio_needed_s": round(need_until, 3),
                                         "start_s": round(t_start - self.t0, 3), "done_s": round(t_done - self.t0, 3),
                                         "compute_ms": round((t_done - t_start) * 1000, 1), **eng.timing,
                                         "prompt": prompt})
            self.out.put((first, frames, t_done))
            n += 1
        self.out.put(None)

    async def pump(self):
        """Pace frames at the model fps with the matching speech slice: frame idx goes out at anchor + idx/fps. The
        first block waits for a 57-frame audio window but holds only 21 frames, so it sets a delay that later blocks
        beat by ~1 s; the anchor only moves earlier (min over blocks of done - first/fps + slack), and the frames it
        skips over go out at once. That catch-up happens in the pre-activity readiness wait (>= 4 s after the first
        frame), so the timed activity sees the steady-state delay (~ one block + compute)."""
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
            for k, jpg in enumerate(jpegs):
                idx = first + k
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
        self.config = self._config()
        import cv2
        self.cv2 = cv2

    def encode(self, frame):
        ok, buf = self.cv2.imencode(".jpg", frame[:, :, ::-1], [self.cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
        return buf.tobytes()

    def _config(self):
        eng, args = self.engine, self.args
        torch = eng.torch
        weights = {}
        p = MODEL_DIR / "weights_hash.json"
        if p.exists():
            weights = json.loads(p.read_text())
        commit = (MODEL_DIR / "repo_commit.txt").read_text().strip() if (MODEL_DIR / "repo_commit.txt").exists() else None
        return {"model": MODEL_NAME, "repo": "https://github.com/Soul-AILab/SoulX-LiveAct", "repo_commit": commit,
                "hf_weights": "Soul-AILab/LiveAct + TencentGameMate/chinese-wav2vec2-base", "weights": weights,
                "width": eng.width, "height": eng.height, "fps": eng.fps, "steps": len(TIMESTEPS) - 1,
                "block_frames": 32, "gpu": torch.cuda.get_device_name(eng.device), "gpu_count": eng.world,
                "parallel": f"ulysses sequence parallel x{eng.world}",
                "attention_kernel": getattr(eng, "attn_kernel", "FA (world 1)"),
                "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(eng.device))),
                "fp8_gemm": args.fp8_gemm, "fp8_kv_cache": args.fp8_kv_cache, "torch_compile": args.compile,
                "audio_cfg": 1.0, "prompt": "init.prompt (rendering_context) or --base_prompt, + latest control",
                "control_mapping": CONTROL_MAPPING_VERSION,
                "torch": torch.__version__, "load_s": round(eng.load_s, 1)}

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
                    prompt = (m.get("prompt") or "").strip() or self.args.base_prompt
                    sess.stats.update({"case_id": m.get("case_id"), "track": track, "seed": seed, "prompt": prompt})
                    init = {"image": base64.b64decode(m["image_b64"]), "seed": seed, "prompt": prompt}
                    gen_future = loop.run_in_executor(None, self._generate_locked, sess, init)
                    await sess.ready_evt.wait()
                    if sess.error:
                        await conn.error(sess.error)
                        break
                    await conn.ready(MODEL_NAME, self.config["repo_commit"], self.engine.width, self.engine.height,
                                     self.engine.fps, protocol=sp.PROTOCOL_VERSION, config=self.config)
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
                    await conn.ack(m.get("event_id"), "unsupported", "LiveAct has no reference-image editing")
                elif kind == "end":
                    break
        except sp.ConnectionClosed:
            pass
        finally:
            # close the socket at once; the running block (if any) finishes in the background before the next session
            sess.ended.set()
            if pump_task is not None:
                pump_task.cancel()
            await conn.close()
            if gen_future is not None:
                await gen_future
            self._write_stats(sess)
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

    def _write_stats(self, sess):
        s = sess.stats
        b = s["blocks"]
        if b:
            frames = sum(x["frames"] for x in b)
            gen_span = b[-1]["done_s"] - b[0]["start_s"]
            s["summary"] = {
                "blocks": len(b), "frames": frames,
                "mean_block_compute_ms": round(float(np.mean([x["compute_ms"] for x in b[2:] or b])), 1),
                "generation_fps": round(frames / max(1e-6, sum(x["compute_ms"] for x in b) / 1000), 2),
                "wall_fps": round(frames / max(1e-6, gen_span), 2) if gen_span > 0 else None,
                "realtime_budget_ms_per_block": round(32 / self.engine.fps * 1000, 1),
                "max_start_lag_s": round(max(x["start_s"] - x["audio_needed_s"] for x in b), 3),
                "first_frame_s": b[0]["done_s"]}
            self.engine.log("session summary " + json.dumps(s["summary"]))
        d = Path(self.args.stats_dir)
        d.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(s.get("case_id") or "session"))
        (d / f"{name}.{int(time.time())}.json").write_text(json.dumps({"config": self.config, **s}, indent=1))

    def health(self):
        return {"ok": True, "model": MODEL_NAME, "busy": self.busy}


def idle_ticker(server):
    """Keep the gloo/nccl control channel alive between sessions (as upstream demo.py does)."""
    while True:
        time.sleep(1.0)
        if server.gpu_lock.acquire(blocking=False):
            try:
                server.engine.bcast({"op": "idle"})
            finally:
                server.gpu_lock.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ckpt_dir", default=str(MODEL_DIR / "weights/LiveAct"))
    ap.add_argument("--wav2vec_dir", default=str(MODEL_DIR / "weights/chinese-wav2vec2-base"))
    ap.add_argument("--size", default="512*512", help="W*H; upstream documents 720*416 / 416*720 (20 fps on "
                                                      "2xH100) and 512*512 (action/emotion editing mode)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base_prompt", default="A person is talking naturally to the camera.")
    ap.add_argument("--audio_margin", type=float, default=0.03, help="s to wait past the audio a block needs")
    ap.add_argument("--fp8_gemm", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--attn", default="auto", help="yunchang AttnType for the SP self-attention (auto: by GPU)")
    ap.add_argument("--fp8_kv_cache", action="store_true")
    ap.add_argument("--no_compile", dest="compile", action="store_false")
    ap.add_argument("--warmup_blocks", type=int, default=6)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--pace_slack", type=float, default=0.15, help="s of output delay kept for compute jitter")
    ap.add_argument("--stats_dir", default=str(MODEL_DIR / "logs/sessions"))
    args = ap.parse_args()

    engine = Engine(args)
    engine.warmup(args.warmup_blocks)
    if engine.rank != 0:
        engine.worker_loop()
        return
    server = Server(engine, args)
    if engine.world > 1:
        threading.Thread(target=idle_ticker, args=(server,), daemon=True).start()
    engine.log(f"listening on ws://{args.host}:{args.port}/session")
    sp.serve(server.handler, args.host, args.port, health=server.health)


if __name__ == "__main__":
    main()
