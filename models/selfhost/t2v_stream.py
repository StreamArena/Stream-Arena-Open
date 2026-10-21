"""Shared live loop for self-hosted text-conditioned autoregressive video generators (selfhost/<model>/server.py).

Protocol: docs/protocol.md, served with streameval.runtime.protocol (standard library only).
A model supplies an engine with:

    name, version, width, height, fps   strings / numbers for `ready`
    system_config() -> dict             recorded in `ready.system_config`
    run(session)                        GPU loop: one causal block per iteration, see GenSession

GenSession does everything else:

  * prompt state. The text condition of a block is "<init prompt>\n<latest instruction>" (the init prompt alone
    until the first instruction). `user_prompt` and `camera_control` texts are both instructions: these models have
    no input besides text, so a camera instruction is expressed as prompt text. Nothing is rewritten and earlier
    instructions are not accumulated (same policy as selfhost/hy_worldplay and selfhost/matrix_game_35).
  * live scheduling. `session.next_block()` blocks until the next block is due, then returns the prompt to use and
    the events that take effect with it. A block is due when the frames still waiting to be shown would run out
    before the block could be finished (measured block time + 15 %), so a model that is faster than real time is
    never more than about one block ahead of the wall clock and a slower one generates back to back. No event is
    read ahead: an instruction can only reach a block that starts after it arrived.
  * acks. An event is acked `applied` when the first block conditioned on it starts (detail: block index and first
    output frame), `error` if the session ends first. `reference` / `control` are `unsupported`; an init image is
    refused unless the engine sets `image_start = True` (it then reads `session.init["image_b64"]` itself).
  * paced output. Frames are sent one frame interval apart, never dropped.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from collections import deque

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from streameval.runtime import protocol as sp  # noqa: E402

LOG = logging.getLogger("t2v_stream")
GPU = threading.Lock()                         # one session on the GPU at a time (a retry may overlap)
PROMPT_POLICY = ("block prompt = '<init prompt>\\n<latest instruction>' (init prompt alone before the first "
                 "instruction); user_prompt and camera_control texts are instructions, verbatim, not accumulated; "
                 "an instruction takes effect at the first block that starts after it arrives")


def sha256_files(files: list[Path], root: Path) -> dict:
    """{path relative to root: sha256}; symlinked files (weights shared with another model directory) are hashed
    through the link and keep their name under root. Cached in root/sha256_cache.json by (real path, size, mtime)."""
    cache_path = Path(root) / "sha256_cache.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    out = {}
    for p in files:
        real = Path(os.path.realpath(p))
        if not real.exists():
            continue
        st = real.stat()
        key = f"{real}:{st.st_size}:{int(st.st_mtime)}"
        if key not in cache:
            h = hashlib.sha256()
            with open(real, "rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):
                    h.update(block)
            cache[key] = h.hexdigest()
        out[os.path.relpath(p, root)] = cache[key]
    try:
        cache_path.write_text(json.dumps(cache, indent=1))
    except OSError:
        pass
    return out


class GenSession:
    def __init__(self, engine, conn: sp.ServerConnection, loop, init: dict, jpeg_quality: int = 90):
        self.e, self.conn, self.loop, self.init = engine, conn, loop, init
        self.seed = int(init.get("seed") or 0)
        self.base_prompt = (init.get("prompt") or "").strip()
        self.fps = float(engine.fps)
        self.iv = int(1e9 / self.fps)
        self.jpeg_quality = jpeg_quality
        self.cv = threading.Condition()
        self.events = deque()                  # (event_id, kind, text) not yet taken by the GPU loop
        self.instruction = None
        self.out = deque()                     # uint8 HxWx3 frames waiting to be sent
        self.stop = False
        self.block_s = None                    # measured time of the last block (generation + decode)
        self.blocks = 0
        self.frames_made = 0
        self.stats = {"blocks": 0, "frames_out": 0, "block_s": [], "prompt_switches": 0, "waited_s": 0.0}
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.sender = threading.Thread(target=self._send_loop, daemon=True)

    # ---- event loop side ------------------------------------------------------------------------------
    def start(self):
        self.worker.start()
        self.sender.start()

    def on_text(self, text: str, event_id, kind: str):
        with self.cv:
            self.events.append((event_id, kind, text))
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.stop = True
            self.cv.notify_all()

    # ---- GPU thread side ------------------------------------------------------------------------------
    def _ack(self, event_id, status, detail=None):
        asyncio.run_coroutine_threadsafe(self.conn.ack(event_id, status, detail), self.loop)

    def prompt(self) -> str:
        if not self.instruction:
            return self.base_prompt
        return f"{self.base_prompt}\n{self.instruction}" if self.base_prompt else self.instruction

    def next_block(self):
        """Wait until the next block is due. Returns (prompt, changed, events) or None when the session ended;
        `events` = [(event_id, kind)] taking effect with this block (ack them with `applied(...)` once the block
        is set up), `changed` = the prompt differs from the previous block's."""
        t0 = time.monotonic()
        with self.cv:
            while not self.stop:
                lead = 0.0 if self.block_s is None else self.block_s * 1.15
                if len(self.out) / self.fps <= lead:
                    break
                self.cv.wait(0.01)
            if self.stop:
                return None
            events, before = list(self.events), self.prompt()
            self.events.clear()
            if events:
                self.instruction = events[-1][2]
            prompt = self.prompt()
        self.stats["waited_s"] += time.monotonic() - t0
        self._t_block = time.monotonic()
        changed = prompt != before
        self.stats["prompt_switches"] += bool(changed)
        return prompt, changed, [(eid, kind) for eid, kind, _ in events]

    def applied(self, events, detail: str = ""):
        for i, (eid, kind) in enumerate(events):
            note = f"{kind} as prompt text from block {self.blocks} (output frame {self.frames_made})"
            if i < len(events) - 1:
                note += "; replaced by a later instruction released before this block started"
            self._ack(eid, "applied", note + (f"; {detail}" if detail else ""))

    def emit(self, frames):
        """Queue the frames of the block just generated (uint8 HxWx3 arrays) for paced sending."""
        took = time.monotonic() - self._t_block
        self.block_s = took
        self.blocks += 1
        self.frames_made += len(frames)
        self.stats["blocks"] += 1
        self.stats["block_s"].append(round(took, 3))
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
        due = 0
        while True:
            with self.cv:
                while not self.out and not self.stop:
                    self.cv.wait(0.05)
                if self.stop:
                    return
                frame = self.out.popleft()
                self.cv.notify_all()
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, "JPEG", quality=self.jpeg_quality)
            now = time.monotonic_ns()
            if due > now:
                time.sleep((due - now) / 1e9)
            due = max(due, time.monotonic_ns()) + self.iv
            try:
                asyncio.run_coroutine_threadsafe(self.conn.frame(buf.getvalue()), self.loop).result(10)
            except Exception:  # noqa: BLE001 - connection gone
                return
            self.stats["frames_out"] += 1

    def summary(self) -> dict:
        s = dict(self.stats)
        t = s.pop("block_s")
        s["waited_s"] = round(s["waited_s"], 2)
        if t:
            steady = sorted(t[1:] or t)
            s.update(block_s_first=t[0], block_s_median=steady[len(steady) // 2], block_s_max=steady[-1])
        return s


def serve(engine, host: str, port: int, jpeg_quality: int = 90):
    async def handler(conn: sp.ServerConnection):
        loop = asyncio.get_running_loop()
        sess = None
        try:
            while True:
                m = await conn.recv()
                if isinstance(m, bytes):
                    continue
                typ, eid = m.get("type"), m.get("event_id")
                if typ == "init" and sess is None:
                    if m.get("image_b64") and not getattr(engine, "image_start", False):
                        await conn.error(f"unsupported: {engine.name} is text-to-video only (no image input)")
                        return
                    sess = GenSession(engine, conn, loop, m, jpeg_quality)
                    await conn.ready(engine.name, engine.version, engine.width, engine.height, engine.fps,
                                     protocol=sp.PROTOCOL_VERSION,
                                     system_config={**engine.system_config(), "seed": sess.seed})
                    sess.start()
                elif typ == "text":
                    text, kind = (m.get("text") or "").strip(), m.get("kind") or "user_prompt"
                    if sess is None or not text:
                        await conn.ack(eid, "error", "empty text" if sess else "text before init")
                    elif kind not in ("user_prompt", "camera_control", "text_control"):
                        await conn.ack(eid, "unsupported", f"text kind {kind} not supported")
                    else:
                        sess.on_text(text, eid, kind)
                elif typ in ("reference", "control"):
                    await conn.ack(eid, "unsupported", f"{engine.name} has no {typ} input (text prompts only)")
                elif typ == "end":
                    break
                elif eid:
                    await conn.ack(eid, "unsupported", f"message type {typ} not supported")
        except sp.ConnectionClosed:
            pass
        finally:
            if sess is not None:
                sess.close()
                await loop.run_in_executor(None, sess.worker.join, 120)
                await loop.run_in_executor(None, sess.sender.join, 10)
                if not conn.closed:
                    for eid, _, _ in list(sess.events):
                        await conn.ack(eid, "error", "session ended before the instruction took effect")
                LOG.info("session end %s stats=%s", sess.init.get("case_id"), json.dumps(sess.summary()))

    LOG.info("listening ws://%s:%d/session", host, port)
    sp.serve(handler, host, port, health=lambda: {"ok": True, "model": engine.name, "version": engine.version})
