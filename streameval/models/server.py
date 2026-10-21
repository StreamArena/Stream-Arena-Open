"""Serve a Python StreamingModel over the WebSocket protocol (one session per connection).

    streameval serve --model my_pkg.my_model:MyModel [--kwargs '{"steps": 4}'] --port 8765

Useful when the model must run on another machine (a GPU node) than the harness, or to check that a model behaves
the same in-process and behind the wire. A new model instance is built per session.
"""
from __future__ import annotations

import asyncio
import base64
import json

from ..runtime import protocol as sp
from .base import ActionEvent, OutputSink, ReferenceEvent, ReferenceSlot, SessionInit, TextEvent


class _WireSink(OutputSink):
    """Sink methods are called from model threads; they schedule sends on the server's event loop."""

    def __init__(self, conn: sp.ServerConnection, loop: asyncio.AbstractEventLoop, failed: asyncio.Event):
        self.conn, self.loop, self.failed = conn, loop, failed
        self.acked: set[str] = set()

    def _call(self, coro):
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def ready(self, **meta):
        fields = {"model": "python", "version": "", "width": 0, "height": 0, "fps": 0, **meta}
        self._call(self.conn.ready(protocol=sp.PROTOCOL_VERSION, **fields))

    def video_frame(self, jpeg, pts_ns=None):
        self._call(self.conn.frame(jpeg, pts_ns))

    def video_rgb(self, width, height, rgb, pts_ns=None):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.frombytes("RGB", (width, height), bytes(rgb() if callable(rgb) else rgb)).save(buf, "JPEG", quality=90)
        self.video_frame(buf.getvalue(), pts_ns)

    def audio(self, pcm48k, pts_ns=None):
        self._call(self.conn.audio(pcm48k, pts_ns))

    def ack(self, event_id, status="applied", detail=None):
        if event_id not in self.acked:
            self.acked.add(event_id)
            self._call(self.conn.ack(event_id, status, detail))

    def fatal(self, message):
        self._call(self.conn.error(message))
        self.loop.call_soon_threadsafe(self.failed.set)


def _image(b64: str | None) -> bytes | None:
    return base64.b64decode(b64) if b64 else None


def _event(message: dict) -> TextEvent | ReferenceEvent | ActionEvent:
    if message["type"] == "control":
        params = message.get("params") or {}
        return ActionEvent(message["event_id"], message.get("action") or "", params.get("phase") or "",
                           int(params.get("duration_ms") or 0))
    if message["type"] == "text":
        return TextEvent(message["event_id"], message.get("kind") or "user_prompt", message.get("text") or "")
    slots = tuple(ReferenceSlot(s["slot"], _image(s.get("image_b64")), s.get("scope"), s.get("asset_id"))
                  for s in message.get("slots") or [])
    return ReferenceEvent(message["event_id"], message.get("op", "set"), slots, message.get("scope"),
                          message.get("text"))


def _init(message: dict) -> SessionInit:
    return SessionInit(track=message["track"], case_id=message["case_id"], width=message["width"],
                       height=message["height"], fps=message["fps"], seed=message["seed"],
                       output=message.get("output") or {}, prompt=message.get("prompt") or "",
                       persona=message.get("persona"), image=_image(message.get("image_b64")))


def serve_model(factory, host: str = "127.0.0.1", port: int = 8765, name: str = "python"):
    """Blocking: serve ws://host:port/session, building a model with factory() for every connection's init message
    and closing it when the connection ends. name is reported by GET /health."""
    async def handler(conn: sp.ServerConnection):
        loop = asyncio.get_running_loop()
        failed = asyncio.Event()
        sink, model = _WireSink(conn, loop, failed), None
        try:
            while not failed.is_set():
                message = await conn.recv()
                if isinstance(message, bytes):
                    tag, pts, data = sp.unpack(message)
                    if model is None:
                        continue
                    if tag == sp.AUDIO_IN:
                        model.push_audio(pts, data)
                    elif tag == sp.SOURCE_FRAME:
                        model.push_source_frame(pts, data)
                    continue
                kind = message.get("type")
                if kind == "init" and model is None:
                    model = factory()
                    await loop.run_in_executor(None, model.open, _init(message), sink)
                elif kind in ("text", "reference", "control") and model is not None:
                    model.send_event(_event(message))
                elif kind == "end":
                    break
        except sp.ConnectionClosed:
            pass
        finally:
            if model is not None:
                await loop.run_in_executor(None, model.close)

    print(f"serving {name} on ws://{host}:{port}/session", flush=True)
    sp.serve(handler, host, port, health=lambda: {"ok": True, "model": name})


def main_serve(spec: str, kwargs: str | None, host: str, port: int):
    from .registry import load_class
    cls = load_class(spec)
    params = json.loads(kwargs) if kwargs else {}
    serve_model(lambda: cls(**params), host, port, spec)
