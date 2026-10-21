"""A model behind the WebSocket protocol (docs/protocol.md): a server in any language, one connection per session.

The protocol is the StreamingModel interface on the wire: init -> ready, SF/AI binary media in, VF/AO binary media
out, text/reference events -> ack, end. Python servers can reuse streameval.runtime.protocol, or expose a
StreamingModel class directly with ``streameval serve --model module:Class``.
"""
from __future__ import annotations

import threading
import time

from ..runtime import protocol as sp
from .base import ActionEvent, OutputSink, ReferenceEvent, SessionInit, StreamingModel, TextEvent

CONNECT_TIMEOUT_S = 30
END_WAIT_S = 5


def connect(url: str, timeout_s: float = CONNECT_TIMEOUT_S) -> sp.ClientConnection:
    end, last = time.monotonic() + timeout_s, None
    while time.monotonic() < end:
        try:
            return sp.ClientConnection(url)
        except (OSError, sp.ConnectionClosed) as exc:
            last = exc
            time.sleep(0.5)
    raise ConnectionError(f"cannot connect to {url}: {last}")


class WebSocketModel(StreamingModel):
    def __init__(self, url: str, connect_timeout_s: float = CONNECT_TIMEOUT_S, end_wait_s: float = END_WAIT_S):
        self.url, self.connect_timeout_s, self.end_wait_s = url, connect_timeout_s, end_wait_s
        self.conn = self.sink = None
        self.closing = False
        self.server_closed = threading.Event()
        self.unknown: list[str] = []

    def describe(self) -> dict:
        return {"transport": "WebSocket " + sp.PROTOCOL_VERSION, "url": self.url}

    def open(self, init: SessionInit, sink: OutputSink) -> None:
        self.sink, self.closing = sink, False
        self.server_closed.clear()
        self.conn = connect(self.url, self.connect_timeout_s)
        threading.Thread(target=self._receive, name="arena-ws-receive", daemon=True).start()
        self.conn.send_json(init.message())

    def _receive(self):
        sink = self.sink
        while True:
            try:
                message = self.conn.recv()
                if isinstance(message, bytes):
                    tag, pts, data = sp.unpack(message)
                    if tag == sp.FRAME:
                        sink.video_frame(data, pts)
                    elif tag == sp.AUDIO_OUT:
                        if len(data) % 2:
                            raise ValueError("audio payload must contain complete int16 samples")
                        if data:
                            sink.audio(data, pts)
                    else:
                        raise ValueError("unexpected server media tag")
                    continue
                if not isinstance(message, dict):
                    raise ValueError("server JSON must be an object")
                kind = message.get("type")
                if kind == "ready":
                    sink.ready(**{k: v for k, v in message.items() if k != "type"})
                elif kind == "ack":
                    if not isinstance(message.get("event_id"), str) or message.get("status") not in (
                            "applied", "unsupported", "error"):
                        raise ValueError("invalid event acknowledgement")
                    sink.ack(message["event_id"], message["status"], message.get("detail"))
                elif kind == "error":
                    sink.fatal("server error: " + str(message.get("message"))[:500])
                else:
                    self.unknown.append(str(kind))
            except (sp.ConnectionClosed, OSError) as exc:
                self.server_closed.set()
                if not self.closing:
                    sink.fatal("server_closed: " + str(exc))
                return
            except Exception as exc:  # malformed media/JSON and sink failures must reach the harness
                if not self.closing:
                    sink.fatal("protocol_error: " + str(exc)[:500])
                self.conn.close()
                return

    def send_event(self, event: TextEvent | ReferenceEvent | ActionEvent) -> None:
        self.conn.send_json(event.message())

    def push_audio(self, pts_ns: int, pcm16k: bytes) -> None:
        self.conn.send_binary(sp.pack(sp.AUDIO_IN, pts_ns, pcm16k))

    def push_source_frame(self, pts_ns: int, jpeg: bytes) -> None:
        self.conn.send_binary(sp.pack(sp.SOURCE_FRAME, pts_ns, jpeg))

    def close(self) -> bool:
        if self.conn is None:
            return False
        self.closing = True
        confirmed = False
        try:
            self.conn.send_json({"type": "end"})
            self.server_closed.wait(self.end_wait_s)
            # read before our own close(), which marks the connection closed whatever the server did
            confirmed = self.server_closed.is_set() or self.conn.closed
        finally:
            self.conn.close()
        return confirmed
