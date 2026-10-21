"""Wire format of the WebSocket model protocol (docs/protocol.md), standard library only.

A model server written in Python may import this module (``from streameval.runtime import protocol``) or copy it
(it has no dependencies); servers in other languages reimplement the byte layout below, which is the contract:

  binary media message = 2-byte ASCII tag + uint64 LITTLE-ENDIAN pts_ns + payload
      b"SF" source_frame  client -> server  JPEG, pts_ns = source media time (frame_index * 1e9 / fps)
      b"AI" audio_in      client -> server  s16le 16 kHz mono, 10 ms (160 samples), pts_ns = stream time of
                                             the first sample since the audio stream started
      b"VF" frame         server -> client  JPEG, pts_ns = server monotonic clock when generated
      b"AO" audio_out     server -> client  s16le 48 kHz mono, any block size, pts_ns = server monotonic clock
  JSON control messages are WebSocket text frames with a "type" field.

Besides the helpers, this file carries a small dependency-free WebSocket implementation:
  * ``serve(handler, host, port)``: asyncio server; ``GET /health`` answers 200, ``/session`` upgrades to a
    WebSocket and calls ``await handler(conn)`` with a ``ServerConnection`` (one connection = one session);
  * ``ClientConnection``: blocking client used by streameval.models.websocket.
Both implement RFC 6455 framing (masking, fragmentation, ping/pong, close) without extensions.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import socket
import struct
import threading
from urllib.parse import urlparse

PROTOCOL_VERSION = "streameval-selfhost/1"
SOURCE_FRAME, AUDIO_IN, FRAME, AUDIO_OUT = b"SF", b"AI", b"VF", b"AO"
AUDIO_IN_RATE, AUDIO_OUT_RATE = 16000, 48000
AUDIO_IN_CHUNK = AUDIO_IN_RATE // 100          # 10 ms
ACK_STATUS = ("applied", "unsupported", "error")
_HEADER = struct.Struct("<2sQ")
_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def pack(tag: bytes, pts_ns: int, payload: bytes) -> bytes:
    if tag not in (SOURCE_FRAME, AUDIO_IN, FRAME, AUDIO_OUT):
        raise ValueError("unknown media tag")
    return _HEADER.pack(tag, int(pts_ns)) + bytes(payload)


def unpack(message: bytes) -> tuple[bytes, int, bytes]:
    if len(message) < _HEADER.size:
        raise ValueError("binary message shorter than its header")
    tag, pts = _HEADER.unpack_from(message)
    return tag, pts, bytes(message[_HEADER.size:])


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ---- framing ------------------------------------------------------------------------

OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


def _frame(opcode: int, payload: bytes, mask: bool) -> bytes:
    head = bytearray([0x80 | opcode])
    n = len(payload)
    bit = 0x80 if mask else 0
    if n < 126:
        head.append(bit | n)
    elif n < 1 << 16:
        head.append(bit | 126)
        head += struct.pack(">H", n)
    else:
        head.append(bit | 127)
        head += struct.pack(">Q", n)
    if not mask:
        return bytes(head) + payload
    key = os.urandom(4)
    return bytes(head) + key + _xor(payload, key)


def _xor(data: bytes, key: bytes) -> bytes:
    if not data:
        return b""
    n = len(data)
    k = int.from_bytes((key * (n // 4 + 1))[:n], "little")
    return (int.from_bytes(data, "little") ^ k).to_bytes(n, "little")


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()


class ConnectionClosed(Exception):
    pass


# ---- asyncio server -------------------------------------------------------------------

class ServerConnection:
    """One WebSocket session on the server side. recv() returns dict (text/JSON) or bytes (binary)."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, path: str):
        self.reader, self.writer, self.path = reader, writer, path
        self.lock = asyncio.Lock()
        self.closed = False

    async def _read_frame(self):
        b1, b2 = await self.reader.readexactly(2)
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack(">H", await self.reader.readexactly(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", await self.reader.readexactly(8))[0]
        key = await self.reader.readexactly(4) if b2 & 0x80 else None
        data = await self.reader.readexactly(n)
        return b1 & 0x80, b1 & 0x0F, _xor(data, key) if key else data

    async def recv(self):
        """Next message: dict for a JSON text frame, bytes for a binary frame; ConnectionClosed at the end."""
        parts, kind = [], None
        while True:
            try:
                fin, op, data = await self._read_frame()
            except (asyncio.IncompleteReadError, ConnectionError) as exc:
                self.closed = True
                raise ConnectionClosed(str(exc)) from exc
            if op == OP_PING:
                await self._send(OP_PONG, data)
                continue
            if op == OP_PONG:
                continue
            if op == OP_CLOSE:
                if not self.closed:
                    await self._send(OP_CLOSE, data[:2])
                self.closed = True
                raise ConnectionClosed("closed by peer")
            if op in (OP_TEXT, OP_BINARY):
                kind, parts = op, [data]
            else:
                parts.append(data)
            if fin:
                payload = b"".join(parts)
                return json.loads(payload.decode()) if kind == OP_TEXT else payload

    async def _send(self, opcode: int, payload: bytes):
        async with self.lock:
            self.writer.write(_frame(opcode, payload, mask=False))
            await self.writer.drain()

    async def send_json(self, message: dict):
        await self._send(OP_TEXT, json.dumps(message).encode())

    async def send_binary(self, data: bytes):
        await self._send(OP_BINARY, data)

    # protocol conveniences
    async def ready(self, model: str, version: str, width: int, height: int, fps: float, **extra):
        await self.send_json({"type": "ready", "model": model, "version": version, "width": width, "height": height,
                              "fps": fps, **extra})

    async def ack(self, event_id: str, status: str = "applied", detail: str | None = None):
        if status not in ACK_STATUS:
            raise ValueError("ack status must be one of " + ", ".join(ACK_STATUS))
        await self.send_json({"type": "ack", "event_id": event_id, "status": status,
                              **({"detail": detail} if detail else {})})

    async def frame(self, jpeg: bytes, pts_ns: int | None = None):
        await self.send_binary(pack(FRAME, pts_ns if pts_ns is not None else _mono(), jpeg))

    async def audio(self, pcm48k: bytes, pts_ns: int | None = None):
        await self.send_binary(pack(AUDIO_OUT, pts_ns if pts_ns is not None else _mono(), pcm48k))

    async def error(self, message: str):
        await self.send_json({"type": "error", "message": message})

    async def close(self):
        if not self.closed:
            self.closed = True
            try:
                await self._send(OP_CLOSE, struct.pack(">H", 1000))
            except ConnectionError:
                pass
        self.writer.close()


def _mono() -> int:
    import time
    return time.monotonic_ns()


async def _dispatch(handler, health, reader, writer):
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        writer.close()
        return
    lines = head.decode("latin-1").split("\r\n")
    try:
        method, target, _ = lines[0].split(" ", 2)
    except ValueError:
        writer.close()
        return
    headers = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:] if l)}
    path = urlparse(target).path
    if headers.get("upgrade", "").lower() != "websocket":
        if method == "GET" and path == "/health":
            body = json.dumps(health() if health else {"ok": True}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
        else:
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    if path != "/session":
        writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Accept: {accept_key(headers['sec-websocket-key'])}\r\n\r\n").encode())
    await writer.drain()
    conn = ServerConnection(reader, writer, path)
    try:
        await handler(conn)
    except ConnectionClosed:
        pass
    finally:
        await conn.close()


async def serve_forever(handler, host: str = "127.0.0.1", port: int = 8765, health=None):
    server = await asyncio.start_server(lambda r, w: _dispatch(handler, health, r, w), host, port,
                                        limit=1 << 20)
    async with server:
        await server.serve_forever()


def serve(handler, host: str = "127.0.0.1", port: int = 8765, health=None):
    """Blocking: run the asyncio server. handler(conn) is an async function; health() -> dict for GET /health."""
    asyncio.run(serve_forever(handler, host, port, health))


# ---- blocking client ------------------------------------------------------------------

class ClientConnection:
    """Blocking WebSocket client. send_* are thread-safe; recv() from one reader thread only."""

    def __init__(self, url: str, timeout: float = 10.0):
        u = urlparse(url)
        if u.scheme != "ws":
            raise ValueError("only ws:// URLs are supported")
        self.sock = socket.create_connection((u.hostname, u.port or 80), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path or '/'} HTTP/1.1\r\nHost: {u.hostname}:{u.port or 80}\r\n"
                           "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionClosed("handshake aborted")
            head += chunk
        head, self.buffer = head.split(b"\r\n\r\n", 1)
        lines = head.decode("latin-1").split("\r\n")
        if " 101 " not in lines[0] + " ":
            raise ConnectionClosed("handshake refused: " + lines[0])
        headers = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:] if l)}
        if headers.get("sec-websocket-accept") != accept_key(key):
            raise ConnectionClosed("bad handshake accept key")
        self.sock.settimeout(None)
        self.lock = threading.Lock()
        self.closed = False

    def _exact(self, n: int) -> bytes:
        while len(self.buffer) < n:
            chunk = self.sock.recv(max(65536, n - len(self.buffer)))
            if not chunk:
                self.closed = True
                raise ConnectionClosed("connection closed")
            self.buffer += chunk
        data, self.buffer = self.buffer[:n], self.buffer[n:]
        return data

    def _send(self, opcode: int, payload: bytes):
        with self.lock:
            if self.closed:
                raise ConnectionClosed("connection closed")
            self.sock.sendall(_frame(opcode, payload, mask=True))

    def send_json(self, message: dict):
        self._send(OP_TEXT, json.dumps(message).encode())

    def send_binary(self, data: bytes):
        self._send(OP_BINARY, data)

    def recv(self):
        """Next message: dict (JSON text) or bytes (binary). Raises ConnectionClosed."""
        parts, kind = [], None
        while True:
            try:
                b1, b2 = self._exact(2)
                n = b2 & 0x7F
                if n == 126:
                    n = struct.unpack(">H", self._exact(2))[0]
                elif n == 127:
                    n = struct.unpack(">Q", self._exact(8))[0]
                key = self._exact(4) if b2 & 0x80 else None
                data = self._exact(n)
            except OSError as exc:
                self.closed = True
                raise ConnectionClosed(str(exc)) from exc
            data = _xor(data, key) if key else data
            op = b1 & 0x0F
            if op == OP_PING:
                self._send(OP_PONG, data)
                continue
            if op == OP_PONG:
                continue
            if op == OP_CLOSE:
                self.closed = True
                raise ConnectionClosed("closed by server")
            if op in (OP_TEXT, OP_BINARY):
                kind, parts = op, [data]
            else:
                parts.append(data)
            if b1 & 0x80:
                payload = b"".join(parts)
                return json.loads(payload.decode()) if kind == OP_TEXT else payload

    def close(self):
        try:
            if not self.closed:
                self._send(OP_CLOSE, struct.pack(">H", 1000))
        except (OSError, ConnectionClosed):
            pass
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
