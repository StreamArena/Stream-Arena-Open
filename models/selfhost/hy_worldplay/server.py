"""HY-World 1.5 / HunyuanWorld WorldPlay (AR distilled, 4-step) self-hosted server, docs/protocol.md.

One WebSocket connection = one session, generated live chunk by chunk (upstream `_ar_rollout_inner`, unrolled):
  init (prompt + image_b64) -> image center-cropped to 832x480, VAE-encoded as the i2v condition, SigLIP + Qwen2.5-VL
                               + ByT5 prompt encoding, text KV cache, then `ready`. A worker thread generates one
                               chunk = 4 latents (16 frames @ 24 fps; the first chunk 13) per step: reconstituted
                               context memory -> 4 denoising steps -> causal VAE decode of just these latents (decoder
                               feature cache kept across chunks) -> JPEG `frame`s. The worker never runs more than one
                               chunk ahead of the wall clock, so events are applied live.
  text user_prompt/text_control -> prompt switch at the next chunk: "<init prompt>\n<latest instruction>" re-encoded
                               and the text KV cache rebuilt (ack "applied" when that chunk starts).
  text camera_control       -> keyword map onto the model's native action space (translate forward/back/left/right,
                               yaw, pitch; orbit/arc = strafe + counter-yaw; zoom-in ~ dolly-in; hold/settle = stop), held
                               until the next camera event; ack "applied" with the mapping, "unsupported" if nothing
                               maps (crane/pedestal, "return to front view", ...).
  reference op=set          -> the new image replaces the SigLIP semantic condition and its instruction joins the
                               prompt (text KV cache rebuilt); the pixel anchor (first latent) is unchanged.
  reference other ops / control -> ack "unsupported".
  init without image_b64    -> `error` unless --t2v: the released checkpoint is 480p I2V only (upstream run.sh:
                               "we only provide the i2v model"); --t2v runs it with a zero image condition.

Multi-GPU: `python -m torch.distributed.run --nproc_per_node N server.py ...` runs the DiT with upstream sequence
parallelism (sp = N); rank 0 serves the socket, decodes and broadcasts (session inputs, per-chunk events / stop) to the
follower ranks over a gloo group, so every rank runs the same chunk in lockstep.

Upstream code: Tencent-Hunyuan/HY-WorldPlay (see setup.sh), imported unchanged from $HYWP_REPO.
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
import socket
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repository root -> streameval
from streameval.runtime import protocol as proto  # noqa: E402

MODEL_DIR = Path(os.path.expanduser(os.environ.get("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")) + "/hy_worldplay")
REPO = Path(os.environ.get("HYWP_REPO", MODEL_DIR / "repo"))
MODEL = "hy_worldplay"
FPS = 24
W, H = 832, 480
CHUNK = 4  # latents per AR chunk
FWD, YAW = 0.08, 3.0  # upstream pose-string speeds per latent: 0.08 units, 3 degrees


class SessionEnd(Exception):
    pass


def log(*a):
    print(time.strftime("%H:%M:%S"), "[hywp]", *a, flush=True)


# ---------------------------------------------------------------------------------------------- camera language

_DIR = r"(?:\s+\w+){0,3}?\s*(?:to(?:ward)?s?\s+the\s+)?"


def camera_motion(text: str):
    """Free-text camera instruction -> (per-latent motion dict for generate_camera_trajectory_local, label) or None."""
    t = text.lower()
    deg = 3.14159265 / 180 * YAW
    if re.search(r"\b(orbit\w*|arc|arcs|arcing|circl\w+|revolv\w+)\b", t):
        ccw = re.search(r"counter-?clockwise|anti-?clockwise|rightward|\bright\b", t) and "clockwise" not in \
            re.sub(r"(counter|anti)-?clockwise", "", t)
        if re.search(r"counter-?clockwise|anti-?clockwise", t) or (ccw and "arc" in t):
            return {"right": FWD, "yaw": -deg}, "orbit counter-clockwise (strafe right + yaw left)"
        return {"right": -FWD, "yaw": deg}, "orbit clockwise (strafe left + yaw right)"
    m = re.search(r"\b(pan|turn|rotate|yaw|swing|sweep|look|swivel)\w*" + _DIR + r"(left|right)\b", t)
    if m:
        return ({"yaw": -deg}, "yaw left") if m.group(2) == "left" else ({"yaw": deg}, "yaw right")
    m = re.search(r"\b(tilt|look|crane|pitch|angle)\w*" + _DIR + r"(up|upward|down|downward)\b", t)
    if m:
        up = m.group(2).startswith("up")
        return ({"pitch": deg}, "pitch up") if up else ({"pitch": -deg}, "pitch down")
    m = re.search(r"\b(truck|strafe|slide|move|dolly|step|drift|shift)\w*" + _DIR + r"(left|right)\b", t)
    if m:
        return ({"right": -FWD}, "strafe left") if m.group(2) == "left" else ({"right": FWD}, "strafe right")
    if re.search(r"\b(pull\s+(the\s+camera\s+)?back|dolly\s+(out|back)|move\s+back|retreat|back\s+away|zoom\s+out"
                 r"|backward)", t):
        return {"forward": -FWD}, "move backward"
    if re.search(r"\b(zoom\s+in|close\s+in|focus\s+tightly|tighten)", t):
        return {"forward": FWD}, "move forward (zoom approximated by dolly-in)"
    if re.search(r"\b(push|dolly|advance|approach|forward|fly|glide|move\s+(in|closer|toward)|through|into|enter"
                 r"|follow|track|tracking|chase|walk|continue\s+(along|down|past))", t):
        return {"forward": FWD}, "move forward"
    if re.search(r"\bpan\b", t):
        return {"yaw": deg}, "yaw right (pan without direction)"
    if re.search(r"^\s*(hold|settle|stop|freeze|pause|keep\s+(the\s+)?(camera|frame|shot)\s+(still|steady|static|locked))"
                 r"|\b(static|locked-off|stationary)\s+(camera|shot)", t):
        return {}, "hold (no motion)"
    return None


# ---------------------------------------------------------------------------------------------- model

class Engine:
    def __init__(self, args):
        self.args = args
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        for k, v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1"}.items():
            os.environ.setdefault(k, v)
        if "MASTER_PORT" not in os.environ:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        sys.path.insert(0, str(REPO))
        import torch
        import torch.distributed as dist
        from datetime import timedelta
        self.rank, self.world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=self.rank, world_size=self.world)
        # rank 0 serves the socket and drives followers chunk by chunk over a CPU (gloo) control group
        self.ctrl = dist.new_group(backend="gloo", timeout=timedelta(days=7)) if self.world > 1 else None
        self.dist = dist
        from hyvideo.commons.infer_state import initialize_infer_state
        initialize_infer_state(argparse.Namespace(
            sage_blocks_range="0-53", use_sageattn=args.sageattn, enable_torch_compile=args.compile,
            use_fp8_gemm=False, quant_type="fp8-per-block", include_patterns="double_blocks", use_vae_parallel=False))
        import hyvideo.generate as gen  # module import sets the parallel state: sp = WORLD_SIZE
        from hyvideo.commons import PIPELINE_CONFIGS
        from hyvideo.models.autoencoders import hunyuanvideo_15_vae_w_cache as vae_mod
        from hyvideo.models.transformers.worldplay_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer as DiT
        from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline as Pipe
        from hyvideo.pipelines.pipeline_utils import retrieve_timesteps
        from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
        from hyvideo.utils.retrieval_context import generate_points_in_sphere, select_aligned_memory_frames
        from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        self.torch, self.gen, self.retrieve_timesteps = torch, gen, retrieve_timesteps
        self.select_memory, self.trajectory = select_aligned_memory_frames, generate_camera_trajectory_local
        base = Path(args.weights) / "HunyuanVideo-1.5"
        self.ckpt = Path(args.weights) / "HY-WorldPlay/ar_distilled_action_model/model.safetensors"
        dev = torch.device("cuda")
        t0 = time.time()
        # DiT from config (no 33 GB base download): the HY-World checkpoint holds every weight (strict load)
        with init_empty_weights():
            dit = DiT.from_config(DiT.load_config(base / "transformer/480p_i2v"))
            dit.add_action_parameters()
        sd = {k: v.to(torch.bfloat16) for k, v in load_file(str(self.ckpt)).items()}
        dit.load_state_dict(sd, strict=True, assign=True)
        del sd
        dit = dit.to(torch.bfloat16).to(dev).eval().requires_grad_(False)
        vae_cfg = Pipe.get_vae_inference_config()
        vae = vae_mod.AutoencoderKLConv3D.from_pretrained(base / "vae", torch_dtype=vae_cfg["dtype"]).to(dev)
        byt5, prompt_format = Pipe._load_byt5(str(base), True, 256, device=dev)
        text_encoder, _ = Pipe._load_text_encoders(str(base), device=dev)
        vision_encoder = Pipe._load_vision_encoder(str(base), device=dev)
        self.pipe = Pipe(vae=vae, text_encoder=text_encoder, transformer=dit,
                         scheduler=FlowMatchDiscreteScheduler.from_pretrained(base / "scheduler"),
                         text_encoder_2=None, progress_bar_config={"disable": True},
                         byt5_model=byt5["byt5_model"], byt5_tokenizer=byt5["byt5_tokenizer"],
                         byt5_max_length=byt5["byt5_max_length"], prompt_format=prompt_format,
                         execution_device="cuda", vision_encoder=vision_encoder, enable_offloading=False,
                         **PIPELINE_CONFIGS["480p_i2v"])
        p = self.pipe
        p._guidance_scale, p._guidance_rescale, p._clip_skip = 1.0, 0.0, None  # few-step: no CFG (upstream)
        self.points_local = generate_points_in_sphere(50000, 8.0).to(dev)
        self.gpu = torch.cuda.get_device_name(0)
        self.weights_sha = self._weights_hash()
        log(f"rank {self.rank}/{self.world}: model loaded in {time.time() - t0:.0f}s on {self.gpu}; attn_mode={dit.config.attn_mode} "
            f"vae={vae_cfg['dtype']} mem={torch.cuda.memory_allocated() / 2**30:.1f}GiB")
        self._warmup()

    def bcast(self, obj=None):
        if self.world == 1:
            return obj
        box = [obj]
        self.dist.broadcast_object_list(box, src=0, group=self.ctrl)
        return box[0]

    def follow(self):
        """Ranks > 0: mirror every session rank 0 starts (same inputs, same chunk events, no decode/output)."""
        from PIL import Image
        while True:
            m = self.bcast()
            s = Session(self, m["init"], None, None)
            s.prepare(Image.open(io.BytesIO(m["image"])) if m["image"] else None)
            s.run()

    def _weights_hash(self):
        f = Path(self.args.weights) / "weights.sha256"
        return f.read_text().split()[0] if f.exists() else "unknown"

    def _warmup(self):
        from PIL import Image
        t0 = time.time()
        s = Session(self, {"prompt": "warmup", "seed": 0}, None, None)
        s.prepare(Image.new("RGB", (W, H), (120, 140, 160)))
        s.warmup = True
        s.max_chunks = 2
        s.run()
        log(f"warmup {time.time() - t0:.1f}s; chunk stats {json.dumps(s.stats)}")

    def system_config(self):
        a = self.args
        return {"model_version": "HY-World 1.5 WorldPlay AR distilled (HY-World1.5-Autoregressive-480P-I2V-distill)",
                "weights_sha256": self.weights_sha, "width": W, "height": H, "fps": FPS, "steps": 4,
                "chunk_latents": CHUNK, "chunk_frames": CHUNK * 4, "guidance_scale": 1.0, "dtype": "bf16",
                "memory": "reconstituted context (memory_frames=20, temporal_context=12)", "gpu": self.gpu,
                "gpus": self.world, "sequence_parallel": self.world, "sageattn": bool(a.sageattn), "torch_compile": bool(a.compile), "t2v_zero_image": a.t2v}


class Session:
    """One live AR rollout; the worker thread talks to the event loop through thread-safe queues."""

    def __init__(self, engine: Engine, init: dict, loop, out_q):
        self.e, self.loop, self.out_q = engine, loop, out_q
        self.base_prompt = (init.get("prompt") or "").strip()
        self.seed = int(init.get("seed") or 0)
        self.pending = queue.Queue()  # ("prompt"|"camera"|"reference", event_id, payload)
        self.stop = threading.Event()
        self.t_ready = None
        self.warmup = False
        self.max_chunks = int(engine.args.max_chunks)
        self.stats = {"chunks": 0, "gen_s": [], "decode_s": []}

    def _emit(self, item):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.out_q.put_nowait, item)

    # ---- conditioning -------------------------------------------------------------------------------------
    def _encode_text(self, prompt: str):
        p, dev = self.e.pipe, self.e.torch.device("cuda")
        emb, _, mask, _ = p.encode_prompt(prompt, dev, 1, False, None, clip_skip=None, data_type="video")
        return emb, mask, p._prepare_byt5_embeddings(prompt, dev)

    def _vision(self, image):
        torch = self.e.torch
        if image is None:
            p = self.e.pipe
            return torch.zeros(1, p.config.vision_num_semantic_tokens, p.config.vision_states_dim,
                               device="cuda", dtype=p.target_dtype)
        return self.e.pipe._prepare_vision_states(image, "480p", self.lat_proto, torch.device("cuda"))

    def _text_cache(self):
        """Upstream first stage of _ar_rollout_inner: text (+ SigLIP) KV cache for all vision blocks."""
        torch, p = self.e.torch, self.e.pipe
        p.init_kv_cache()
        emb, mask, byt5 = self.text
        with torch.autocast(device_type="cuda", dtype=p.target_dtype, enabled=True):
            p._kv_cache = p.transformer(
                bi_inference=False, ar_txt_inference=True, ar_vision_inference=False,
                timestep_txt=torch.tensor([0], device="cuda", dtype=torch.bfloat16),
                text_states=emb[0, None], encoder_attention_mask=mask[0, None],
                vision_states=self.vision[0, None], mask_type=self.task,
                extra_kwargs={"byt5_text_states": byt5["byt5_text_states"][0, None],
                              "byt5_text_mask": byt5["byt5_text_mask"][0, None]},
                kv_cache=p._kv_cache, cache_txt=True)
        self.txt_cache = [{"k_txt": c["k_txt"], "v_txt": c["v_txt"]} for c in p._kv_cache]

    def prepare(self, image):
        """image: PIL (center-cropped to 832x480 here) or None (t2v)."""
        torch, p = self.e.torch, self.e.pipe
        from PIL import Image, ImageOps
        self.task = "t2v" if image is None else "i2v"
        if image is not None:
            image = ImageOps.fit(image.convert("RGB"), (W, H), Image.LANCZOS)
        self.lh, self.lw = H // 16, W // 16
        c = p.transformer.config.in_channels
        self.lat_proto = torch.zeros(1, c, 1, self.lh, self.lw, device="cuda")
        self.gen = torch.Generator(device="cuda").manual_seed(self.seed)
        self.text = self._encode_text(self.base_prompt or "a scene")
        self.vision = self._vision(image)
        with torch.no_grad():
            cond0 = p.get_image_condition_latents(self.task, image, H, W) if image is not None else None
        mask = p.get_task_mask(self.task, 1)
        self.cond0 = p._prepare_cond_latents(self.task, cond0, self.lat_proto, mask)            # [1, C+1, 1, h, w]
        self.cond_rest = p._prepare_cond_latents("t2v", None, self.lat_proto, torch.zeros(1))    # zero latent, mask 0
        self._text_cache()
        self.latents = None   # clean latents so far [1, C, T, h, w]
        self.cond = None
        self.motions = []     # one per latent after the first
        self.motion, self.motion_label = {}, "hold"
        p.vae.clear_cache()
        self.first_decode = True

    def _sync(self):
        """Rank 0 drains the event queue; with sequence parallelism every rank gets the same (stop, events)."""
        events = []
        while self.e.rank == 0:
            try:
                events.append(self.pending.get_nowait())
            except queue.Empty:
                break
        return self.e.bcast({"stop": self.stop.is_set(), "events": events} if self.e.rank == 0 else None)

    def _apply_pending(self, chunk_i, events):
        prompt_ev, ref_ev, cam_ev = [], [], []
        for kind, eid, payload in events:
            {"prompt": prompt_ev, "reference": ref_ev, "camera": cam_ev}[kind].append((eid, payload))
        frame = 0 if chunk_i == 0 else 13 + (chunk_i - 1) * 16
        for eid, (motion, label) in cam_ev:
            self.motion, self.motion_label = motion, label
            self._emit(("ack", eid, "applied", f"camera: {label}; from chunk {chunk_i} (frame {frame})"))
        if not (prompt_ev or ref_ev):
            return
        from PIL import Image
        text = None
        for eid, payload in ref_ev:
            img = Image.open(io.BytesIO(payload["image"])).convert("RGB")
            from PIL import ImageOps
            self.vision = self._vision(ImageOps.fit(img, (W, H), Image.LANCZOS))
            text = payload.get("text") or text
        if prompt_ev:
            text = prompt_ev[-1][1]
        if text:
            self.text = self._encode_text(f"{self.base_prompt}\n{text}" if self.base_prompt else text)
        self._text_cache()
        for eid, _ in prompt_ev:
            self._emit(("ack", eid, "applied", f"prompt switched at chunk {chunk_i} (frame {frame})"))
        for eid, _ in ref_ev:
            self._emit(("ack", eid, "applied", f"reference re-encoded as SigLIP condition at chunk {chunk_i} "
                                                 f"(frame {frame}); pixel anchor unchanged"))

    def _poses(self):
        """Upstream pose_to_input over the whole trajectory so far (viewmats, Ks, action labels per latent)."""
        K = [[969.6969696969696, 0.0, 960.0], [0.0, 969.6969696969696, 540.0], [0.0, 0.0, 1.0]]
        poses = self.e.trajectory(self.motions)
        pose_json = {str(i): {"extrinsic": q.tolist(), "K": K} for i, q in enumerate(poses)}
        vm, ks, act = self.e.gen.pose_to_input(pose_json, len(poses))
        return vm.unsqueeze(0), ks.unsqueeze(0), act.unsqueeze(0)

    # ---- one chunk (upstream _ar_rollout_inner body, per chunk) ----------------------------------------------
    def _chunk(self, chunk_i):
        torch, p, e = self.e.torch, self.e.pipe, self.e
        dev, dt = torch.device("cuda"), p.target_dtype
        start = chunk_i * CHUNK
        self.motions += [dict(self.motion) for _ in range(CHUNK - (chunk_i == 0))]  # latent 0 = identity pose
        viewmats, Ks, action = self._poses()
        noise = torch.randn((1, self.lat_proto.shape[1], CHUNK, self.lh, self.lw), generator=self.gen,
                            device=dev, dtype=torch.float32).to(dt)
        cond_new = torch.cat([self.cond0] + [self.cond_rest] * (CHUNK - 1), 2) if chunk_i == 0 else \
            torch.cat([self.cond_rest] * CHUNK, 2)
        self.cond = cond_new if self.cond is None else torch.cat([self.cond, cond_new], 2)
        n_tokens = CHUNK * self.lh * self.lw
        timesteps, steps = e.retrieve_timesteps(p.scheduler, 4, dev, n_tokens=n_tokens) if chunk_i == 0 else \
            (self.timesteps, 4)
        if chunk_i == 0:
            self.timesteps = timesteps
        p.init_kv_cache()
        for c, t in zip(p._kv_cache, self.txt_cache):
            c.update(t)
        selected = []
        if chunk_i > 0:
            for s0 in range(start, start + CHUNK, 4):
                selected += e.select_memory(viewmats[0].cpu().detach().numpy(), s0, memory_frames=20,
                                            temporal_context_size=12, pred_latent_size=4,
                                            points_local=e.points_local, device=dev)
            selected = [x for x in sorted(set(selected)) if not start <= x < start + CHUNK]
            ctx = torch.cat([self.latents[:, :, selected], self.cond[:, :, selected]], 1)
            with torch.autocast(device_type="cuda", dtype=dt, enabled=True):
                p._kv_cache = p.transformer(
                    bi_inference=False, ar_txt_inference=False, ar_vision_inference=True, hidden_states=ctx,
                    timestep=torch.full((len(selected),), 14, device=dev, dtype=timesteps.dtype), timestep_r=None,
                    mask_type=self.task, return_dict=False, viewmats=viewmats[:, selected].to(dev, dt),
                    Ks=Ks[:, selected].to(dev, dt), action=action[:, selected].to(dev, dt), kv_cache=p._kv_cache,
                    cache_vision=True, rope_temporal_size=ctx.shape[2], start_rope_start_idx=0)
            p.scheduler.set_timesteps(steps, device=dev)
        lat = noise
        cond_in = self.cond[:, :, start:start + CHUNK]
        vm_in, ks_in, act_in = (x[:, start:start + CHUNK].to(dev, dt) for x in (viewmats, Ks, action))
        for t in timesteps:
            x = p.scheduler.scale_model_input(torch.cat([lat, cond_in], 1), t)
            with torch.autocast(device_type="cuda", dtype=dt, enabled=True):
                pred = p.transformer(
                    bi_inference=False, ar_txt_inference=False, ar_vision_inference=True, hidden_states=x,
                    timestep=torch.full((CHUNK,), t, device=dev, dtype=timesteps.dtype), timestep_r=None,
                    mask_type=self.task, return_dict=False, viewmats=vm_in, Ks=ks_in, action=act_in,
                    kv_cache=p._kv_cache, cache_vision=False, rope_temporal_size=x.shape[2] + len(selected),
                    start_rope_start_idx=len(selected))[0]
            lat = p.scheduler.step(pred, t, lat, return_dict=False)[0][:, :, -CHUNK:]
        self.latents = lat if self.latents is None else torch.cat([self.latents, lat], 2)
        return lat

    def _decode(self, lat):
        """Causal VAE decode of the new latents only; the decoder feature cache carries across chunks."""
        torch, vae = self.e.torch, self.e.pipe.vae
        z = lat / vae.config.scaling_factor
        if getattr(vae.config, "shift_factor", None):
            z = z + vae.config.shift_factor
        outs = []
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            for i in range(z.shape[2]):
                outs.append(vae.decoder(z[:, :, i:i + 1].to(vae.dtype), feat_cache=vae._feat_map, feat_idx=[0],
                                        first_chunk=self.first_decode))
                self.first_decode = False
        v = (torch.cat(outs, 2)[0].float() / 2 + 0.5).clamp(0, 1)                 # C T H W
        return (v.permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu().numpy()  # T H W C

    def run(self):
        torch = self.e.torch
        torch.manual_seed(self.seed)
        frames_out, t_start = 0, time.monotonic()
        try:
            with torch.no_grad():
                for i in range(self.max_chunks):
                    if self.t_ready is not None:  # real time: at most one chunk of video buffered ahead
                        while not self.stop.is_set() and \
                                self.t_ready + frames_out / FPS - time.monotonic() > CHUNK * 4 / FPS:
                            time.sleep(0.02)
                    msg = self._sync()
                    if msg["stop"]:
                        raise SessionEnd()
                    self._apply_pending(i, msg["events"])
                    t0 = time.monotonic()
                    lat = self._chunk(i)
                    torch.cuda.synchronize()
                    t1 = time.monotonic()
                    frames = self._decode(lat) if self.e.rank == 0 else []
                    t2 = time.monotonic()
                    self.stats["chunks"] += 1
                    self.stats["gen_s"].append(round(t1 - t0, 3))
                    self.stats["decode_s"].append(round(t2 - t1, 3))
                    frames_out += len(frames)
                    self._emit(("frames", time.monotonic_ns(), frames))
                    if not self.warmup and self.e.rank == 0:
                        log(f"chunk {i}: {len(frames)} frames gen {t1 - t0:.2f}s decode {t2 - t1:.2f}s "
                            f"({len(frames) / (t2 - t0):.1f} fps) motion={self.motion_label}")
            self._emit(("done", "max_chunks reached"))
        except SessionEnd:
            self._emit(("done", "ended"))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._emit(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            self.stats["wall_s"] = round(time.monotonic() - t_start, 2)
            self.e.pipe.vae.clear_cache()
            self.latents = self.cond = None
            torch.cuda.empty_cache()


def _jpeg(frame, quality):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------------------------- protocol

class Server:
    def __init__(self, engine: Engine, jpeg_quality: int):
        self.e, self.q = engine, jpeg_quality
        self.busy = asyncio.Lock()

    def health(self):
        return {"ok": True, "model": MODEL, "busy": self.busy.locked(), **self.e.system_config()}

    async def handle(self, conn):
        async with self.busy:  # one GPU session at a time
            await self._session(conn)

    async def _session(self, conn):
        msg = await conn.recv()
        if not isinstance(msg, dict) or msg.get("type") != "init":
            await conn.error("first message must be init")
            return
        if not msg.get("image_b64") and not self.e.args.t2v:
            await conn.error("unsupported: HY-World 1.5 WorldPlay is released as a 480p image-to-video model only "
                             "(upstream: 'we only provide the i2v model'); text-only start is not supported")
            return
        loop = asyncio.get_running_loop()
        out_q: asyncio.Queue = asyncio.Queue()
        sess = Session(self.e, msg, loop, out_q)
        raw = base64.b64decode(msg["image_b64"]) if msg.get("image_b64") else None

        def prepare():
            from PIL import Image
            self.e.bcast({"init": {"prompt": msg.get("prompt"), "seed": msg.get("seed")}, "image": raw})
            sess.prepare(Image.open(io.BytesIO(raw)) if raw else None)

        t0 = time.monotonic()
        try:
            await loop.run_in_executor(None, prepare)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            await conn.error(f"prepare failed: {type(exc).__name__}: {exc}")
            return
        log(f"session {msg.get('case_id')} prepared in {time.monotonic() - t0:.2f}s task={sess.task}")
        await conn.ready(MODEL, "HY-World1.5-AR-480P-I2V-distill", W, H, FPS, protocol=proto.PROTOCOL_VERSION,
                         system_config={**self.e.system_config(), "seed": sess.seed})
        sess.t_ready = time.monotonic()
        worker = threading.Thread(target=sess.run, daemon=True)
        worker.start()
        sender = asyncio.create_task(self._pump(conn, out_q))
        try:
            while True:
                m = await conn.recv()
                if isinstance(m, bytes):
                    continue
                t = m.get("type")
                if t == "end":
                    break
                eid = m.get("event_id", "")
                if t == "text":
                    kind = m.get("kind") or "user_prompt"
                    if kind in ("user_prompt", "text_control"):
                        sess.pending.put(("prompt", eid, m.get("text", "")))
                    elif kind == "camera_control":
                        mapped = camera_motion(m.get("text", ""))
                        if mapped is None:
                            await conn.ack(eid, "unsupported", "camera_control: no mapping onto the model's action "
                                                               "space (translate / yaw / pitch)")
                        else:
                            sess.pending.put(("camera", eid, mapped))
                    else:
                        await conn.ack(eid, "unsupported", f"text kind {kind} not supported")
                elif t == "reference":
                    slots = m.get("slots") or []
                    img = slots[0].get("image_b64") if slots else None
                    if m.get("op", "set") == "set" and img:
                        sess.pending.put(("reference", eid, {"image": base64.b64decode(img), "text": m.get("text")}))
                    else:
                        await conn.ack(eid, "unsupported", f"reference op {m.get('op')} not supported")
                elif t == "control":
                    await conn.ack(eid, "unsupported", "structured control events are not supported")
                elif eid:
                    await conn.ack(eid, "unsupported", f"message type {t} not supported")
        except proto.ConnectionClosed:
            pass
        finally:
            sess.stop.set()
            await loop.run_in_executor(None, worker.join)
            sender.cancel()
            s = sess.stats
            log("session stats", json.dumps({"chunks": s["chunks"], "mean_gen_s": _mean(s["gen_s"]),
                                             "mean_decode_s": _mean(s["decode_s"]), "wall_s": s.get("wall_s")}))

    async def _pump(self, conn, out_q):
        interval, next_t = 1.0 / FPS, time.monotonic()
        while True:
            item = await out_q.get()
            kind = item[0]
            if kind == "ack":
                await conn.ack(item[1], item[2], item[3])
            elif kind == "frames":
                for f in item[2]:
                    now = time.monotonic()
                    if next_t > now:
                        await asyncio.sleep(next_t - now)
                    await conn.frame(_jpeg(f, self.q), time.monotonic_ns())
                    next_t = max(next_t, time.monotonic() - interval) + interval
            elif kind == "error":
                await conn.error(item[1])
                await conn.close()
                return
            elif kind == "done":
                return


def _mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--weights", type=Path, default=MODEL_DIR / "weights")
    ap.add_argument("--max-chunks", type=int, default=60, help="rollout cap: 60 chunks = 40 s")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--t2v", action="store_true", help="accept text-only init (zero image condition; not upstream)")
    ap.add_argument("--sageattn", action="store_true")
    ap.add_argument("--compile", action="store_true", help="upstream torch.compile wrapper on the DiT blocks")
    args = ap.parse_args()
    engine = Engine(args)
    if engine.rank > 0:
        return engine.follow()
    server = Server(engine, args.jpeg_quality)
    log(f"serving ws://{args.host}:{args.port}/session")
    proto.serve(server.handle, args.host, args.port, health=server.health)


if __name__ == "__main__":
    main()
