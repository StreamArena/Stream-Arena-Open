#!/usr/bin/env python3
"""Template of a model server (docs/protocol.md): the same toy instruction editor as model.py, over WebSocket.

Use this shape when the model runs in another process, environment or machine (a model in another language
implements the same messages). Register it with "server" (the harness starts it) or "url" (already running):

    python examples/my_model/server.py --port 8765
    streameval run --system my_model_ws_instruction --case sa.v1.001.instruction_edit --launch

The protocol code (streameval/runtime/protocol.py) is standard library only; copy it next to the server if the
model's environment should not install streameval.
"""
from __future__ import annotations

import argparse
import asyncio
import io
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from streameval.runtime import protocol as sp  # noqa: E402


def edit(image: Image.Image, instruction: str | None) -> Image.Image:
    """Replace with the model (see model.py)."""
    if instruction and any(k in instruction.lower() for k in ("black and white", "grayscale", "greyscale")):
        return image.convert("L").convert("RGB")
    return image


class Session:
    """One connection = one session. Receiving never waits for the model: source frames only replace the newest
    pending frame, and a separate task edits it and sends the result."""

    def __init__(self, conn: sp.ServerConnection):
        self.conn = conn
        self.instruction: str | None = None
        self.source: bytes | None = None
        self.new_frame = asyncio.Event()

    async def run(self):
        worker = None
        try:
            while True:
                try:
                    message = await self.conn.recv()
                except sp.ConnectionClosed:
                    break
                if isinstance(message, bytes):
                    tag, _, payload = sp.unpack(message)
                    if tag == sp.SOURCE_FRAME:
                        self.source = payload
                        self.new_frame.set()
                elif message["type"] == "init":
                    if message["track"] != "instruction_edit":
                        await self.conn.error("this server only runs instruction_edit")
                        break
                    await self.conn.ready("my_model", "0.1", message["width"], message["height"], message["fps"],
                                          protocol=sp.PROTOCOL_VERSION, config={"seed": message.get("seed")})
                    worker = asyncio.create_task(self.edit_loop())
                elif message["type"] == "text" and message.get("kind") == "text_control":
                    self.instruction = message["text"]
                    await self.conn.ack(message["event_id"], "applied")
                elif message["type"] in ("text", "reference", "control"):
                    await self.conn.ack(message["event_id"], "unsupported", "text instructions only")
                elif message["type"] == "end":
                    break
        finally:
            if worker is not None:
                worker.cancel()

    async def edit_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            await self.new_frame.wait()
            self.new_frame.clear()
            jpeg, self.source = self.source, None
            # the model runs off the event loop so that receiving (and acks) never wait for it
            out = await loop.run_in_executor(None, self.render, jpeg, self.instruction)
            await self.conn.frame(out)

    @staticmethod
    def render(jpeg: bytes, instruction: str | None) -> bytes:
        image = edit(Image.open(io.BytesIO(jpeg)).convert("RGB"), instruction)
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=90)
        return buf.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    async def handler(conn):
        await Session(conn).run()

    # load the model here, before serving: /health answers once serve() runs, and the harness waits for it
    print(f"my_model listening on ws://{args.host}:{args.port}/session", flush=True)
    sp.serve(handler, args.host, args.port, health=lambda: {"ok": True, "model": "my_model"})


if __name__ == "__main__":
    main()
