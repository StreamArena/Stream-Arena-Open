"""Shared streaming loop for self-hosted instruction-edit servers (selfhost/<model>/server.py).

Protocol: docs/protocol.md, served with streameval.runtime.protocol (standard library only).
A model supplies an engine with:

    name, version                      strings for `ready`
    system_config() -> dict            recorded in `ready.config`
    out_size(init) -> (width, height)  output resolution for this session
    fps                                model frame rate (source frames are subsampled to it)
    run(session)                       GPU loop; calls session.take(...) / emit(...) / applied(...)

EditSession handles everything else: source-frame intake subsampled to the model fps (when the GPU falls behind
the newest frames win and older ones are dropped, so latency stays bounded), pass-through of the source until the
first instruction (edit servers must emit frames before any instruction), paced output (a chunk of frames is
spread over its duration instead of arriving as one burst), and acks: a text instruction is acked `applied` when
the first frames generated with it are emitted, `error` if a newer instruction replaced it before it took effect;
control events are acked `unsupported`, reference events too unless the engine accepts them.

Reference editing (optional): an engine with `reference_prompt(target) -> str` accepts `reference` events with
one reference slot (same rule as selfhost/joyai_video_edit: set replaces, patch merges, remove / reset drop; a
resulting state of exactly one slot is applied, 2+ slots or an empty state are acked `unsupported` and the
previous state is kept). An applied reference sets the session's reference image and the prompt
reference_prompt(<slot scope target>); like an instruction it is acked `applied` with the first frames generated
with it. The GPU loop reads the reference taken with the frames from `session.taken_ref` = (image | None,
generation), the generation increasing with every applied reference event.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
from pathlib import Path
import sys
import threading
import time
from collections import deque

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from streameval.runtime import protocol as sp  # noqa: E402

LOG = logging.getLogger("edit_stream")
GPU = threading.Lock()                         # one session on the GPU at a time (a retry may overlap)


def fit_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop to the target aspect ratio, then resize (no letterbox, no distortion)."""
    w, h = img.size
    if w / h > width / height:
        nw = round(h * width / height)
        img = img.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    elif w / h < width / height:
        nh = round(w * height / width)
        img = img.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return img.resize((width, height), Image.BICUBIC)


def decode_image(b64: str) -> Image.Image:
    if "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def sha256_files(files: list[Path], root: Path) -> dict:
    """{path relative to root: sha256}; cached in root/sha256_cache.json by (path, size, mtime)."""
    cache_path = root / "sha256_cache.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    out = {}
    for p in files:
        p = Path(p).resolve()
        if not p.exists():
            continue
        st = p.stat()
        key = f"{p}:{st.st_size}:{int(st.st_mtime)}"
        if key not in cache:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):
                    h.update(block)
            cache[key] = h.hexdigest()
        out[str(p.relative_to(root.resolve()))] = cache[key]
    try:
        cache_path.write_text(json.dumps(cache, indent=1))
    except OSError:
        pass
    return out


class EditSession:
    def __init__(self, engine, conn: sp.ServerConnection, loop, init: dict, jpeg_quality: int = 90):
        self.e, self.conn, self.loop, self.init = engine, conn, loop, init
        self.seed = int(init.get("seed") or 0)
        self.w, self.h = engine.out_size(init)
        self.fps = float(engine.fps)
        self.iv = int(1e9 / self.fps)
        self.jpeg_quality = jpeg_quality
        self.next_pts = None
        self.last_pts = None
        self.cv = threading.Condition()
        self.pending = deque()                 # kept source frames (JPEG)
        self.prompt = (init.get("prompt") or "").strip() or None
        self.event = None                      # event id of an instruction not yet taken by the GPU loop
        self.slots = {}                        # reference slot state {slot: {"scope", "image"}} (reference engines)
        self.ref, self.ref_gen = None, 0       # current reference image (PIL) and its generation
        self.taken_ref = (None, 0)             # reference state taken by the GPU loop with the last frames
        self.out = deque()                     # frames to send: (uint8 HxWx3 | PIL image)
        self.stop = False
        self.stats = {"frames_in": 0, "frames_kept": 0, "frames_dropped": 0, "frames_passthrough": 0,
                      "frames_out": 0, "chunks": 0, "chunk_ms": []}
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.sender = threading.Thread(target=self._send_loop, daemon=True)

    # ---- event loop side ------------------------------------------------------------------------------
    def start(self):
        self.worker.start()
        self.sender.start()

    def on_frame(self, pts_ns: int, jpeg: bytes):
        self.stats["frames_in"] += 1
        held = pts_ns == self.last_pts and self.stats["frames_out"] == 0 and not self.out
        self.last_pts = pts_ns
        if self.next_pts is None:
            self.next_pts = pts_ns
        if pts_ns + self.iv // 4 < self.next_pts and not held:
            return                                   # temporal subsampling to the model fps
        # until the first output frame the client repeats the held opening frame (same pts): keep the repeats, so
        # an engine that edits from the start (initial reference) can fill its first chunk
        self.next_pts = max(self.next_pts + self.iv, pts_ns) if not held else self.next_pts
        with self.cv:
            self.pending.append(jpeg)
            self.stats["frames_kept"] += 1
            self.cv.notify_all()

    def on_text(self, text: str, event_id):
        with self.cv:
            if self.event is not None:
                self._ack(self.event, "error", f"superseded by {event_id} before taking effect")
            self.prompt, self.event = text, event_id
            self.cv.notify_all()

    def on_reference(self, msg: dict):
        """Single-slot reference state machine. Returns (status, detail) to ack now, or None when the event is
        applied (acked by the GPU loop when its first frames are emitted)."""
        op, state = msg.get("op"), dict(self.slots)
        if op == "set":
            state = {str(s["slot"]): dict(s) for s in msg.get("slots") or []}
        elif op == "patch":
            for s in msg.get("slots") or []:
                prev = state.get(str(s["slot"]), {})
                state[str(s["slot"])] = {**prev, **{k: v for k, v in s.items() if v is not None}}
        elif op == "remove":
            for s in msg.get("slots") or []:
                state.pop(str(s["slot"] if isinstance(s, dict) else s), None)
        elif op == "reset":
            state = {}
        else:
            return "unsupported", f"reference op {op!r}"
        if len(state) != 1:
            return "unsupported", ("multiple_independent_reference_slots_not_supported" if state
                                   else "empty_state_restore_has_no_expressible_input")
        (slot, spec), = state.items()
        image = spec.get("image")
        if spec.get("image_b64"):
            image = decode_image(spec["image_b64"])
        if image is None:
            return "error", f"slot {slot} has no image"
        target = str(spec.get("scope") or "main subject").split(". Apply")[0].strip().rstrip(".")
        with self.cv:
            if self.event is not None:
                self._ack(self.event, "error", f"superseded by {msg.get('event_id')} before taking effect")
            self.slots = {slot: {"scope": spec.get("scope", ""), "image": image}}
            self.ref, self.ref_gen = image, self.ref_gen + 1
            self.prompt, self.event = self.e.reference_prompt(target), msg.get("event_id")
            self.cv.notify_all()
        return None

    def close(self):
        with self.cv:
            self.stop = True
            self.cv.notify_all()

    # ---- GPU thread side ------------------------------------------------------------------------------
    def _ack(self, event_id, status, detail=None):
        asyncio.run_coroutine_threadsafe(self.conn.ack(event_id, status, detail), self.loop)

    def applied(self, event_id, detail: str):
        if event_id is not None:
            self._ack(event_id, "applied", detail)

    def take(self, count):
        """Block until count(prompt) source frames are kept; return (frames, prompt, event_id) with the newest
        frames as float32 HxWx3 in [-1, 1] and the instruction they are to be edited with (event_id: the
        instruction event taking effect now, else None). Passes frames through while there is no prompt.
        Returns None when the session ends."""
        while True:
            passthrough = []
            with self.cv:
                if self.stop:
                    return None
                n = count(self.prompt) if self.prompt else None
                if n is None:
                    passthrough, _ = list(self.pending), self.pending.clear()
                elif len(self.pending) >= n:
                    drop = len(self.pending) - n
                    for _ in range(drop):
                        self.pending.popleft()
                    self.stats["frames_dropped"] += drop
                    items = [self.pending.popleft() for _ in range(n)]
                    prompt, event, self.event = self.prompt, self.event, None
                    self.taken_ref = (self.ref, self.ref_gen)
                    break
                if not passthrough:
                    self.cv.wait(0.02)
                    continue
            for jpg in passthrough:
                self.emit([fit_crop(Image.open(io.BytesIO(jpg)).convert("RGB"), self.w, self.h)])
                self.stats["frames_passthrough"] += 1
        frames = [np.asarray(fit_crop(Image.open(io.BytesIO(j)).convert("RGB"), self.w, self.h), np.float32)
                  / 127.5 - 1.0 for j in items]
        return frames, prompt, event

    def emit(self, frames):
        """Queue output frames (uint8 HxWx3 arrays or PIL images) for paced sending."""
        with self.cv:
            self.out.extend(frames)
            self.cv.notify_all()

    def _work(self):
        try:
            with GPU:
                self.e.run(self)
        except Exception as exc:  # noqa: BLE001 - reported to the client as a fatal error
            LOG.exception("session failed")
            asyncio.run_coroutine_threadsafe(self.conn.error(f"{type(exc).__name__}: {exc}"), self.loop)
        finally:
            self.close()

    def _send_loop(self):
        """Send queued frames one model frame interval apart; never lag: a backlog above one second is flushed."""
        due = 0
        while True:
            with self.cv:
                while not self.out and not (self.stop and not self.worker.is_alive()):
                    self.cv.wait(0.05)
                if not self.out:
                    return
                frame, backlog = self.out.popleft(), len(self.out)
            now = time.monotonic_ns()
            if backlog < self.fps and due > now:
                time.sleep((due - now) / 1e9)
            due = max(due, now) + self.iv
            image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
            buf = io.BytesIO()
            image.save(buf, "JPEG", quality=self.jpeg_quality)
            try:
                asyncio.run_coroutine_threadsafe(self.conn.frame(buf.getvalue()), self.loop).result(10)
            except Exception:  # noqa: BLE001 - connection gone
                return
            self.stats["frames_out"] += 1

    def summary(self) -> dict:
        s = dict(self.stats)
        ms = s.pop("chunk_ms")
        if ms:
            s.update(chunk_ms_mean=round(float(np.mean(ms)), 1), chunk_ms_p95=round(float(np.percentile(ms, 95)), 1))
        return s


def serve(engine, host: str, port: int, jpeg_quality: int = 90):
    async def handler(conn: sp.ServerConnection):
        loop = asyncio.get_running_loop()
        sess = None
        try:
            while True:
                m = await conn.recv()
                if isinstance(m, bytes):
                    tag, pts, data = sp.unpack(m)
                    if tag == sp.SOURCE_FRAME and sess is not None:
                        sess.on_frame(pts, data)
                    continue
                typ, eid = m.get("type"), m.get("event_id")
                if typ == "init" and sess is None:
                    sess = EditSession(engine, conn, loop, m, jpeg_quality)
                    await conn.ready(engine.name, engine.version, sess.w, sess.h, sess.fps,
                                     protocol=sp.PROTOCOL_VERSION, config=engine.system_config())
                    sess.start()
                elif typ == "text":
                    text = (m.get("text") or "").strip()
                    if sess is None or not text:
                        await conn.ack(eid, "error", "empty text" if sess else "text before init")
                    else:
                        sess.on_text(text, eid)
                elif typ == "reference" and hasattr(engine, "reference_prompt"):
                    if sess is None:
                        await conn.ack(eid, "error", "reference before init")
                        continue
                    try:
                        ack = sess.on_reference(m)
                    except Exception as exc:  # noqa: BLE001 - e.g. an undecodable image
                        ack = ("error", repr(exc)[:300])
                    if ack is not None:
                        await conn.ack(eid, *ack)
                elif typ in ("reference", "control"):
                    await conn.ack(eid, "unsupported", f"{engine.name} has no {typ} input (text instructions only)")
                elif typ == "end":
                    break
        except sp.ConnectionClosed:
            pass
        finally:
            if sess is not None:
                sess.close()
                await loop.run_in_executor(None, sess.worker.join, 60)
                await loop.run_in_executor(None, sess.sender.join, 10)
                if sess.event is not None and not conn.closed:
                    await conn.ack(sess.event, "error", "session ended before the instruction took effect")
                LOG.info("session end %s stats=%s", sess.init.get("case_id"), json.dumps(sess.summary()))

    LOG.info("listening ws://%s:%d/session", host, port)
    sp.serve(handler, host, port, health=lambda: {"ok": True, "model": engine.name, "version": engine.version})
