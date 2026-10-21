"""MockModel: a CPU stand-in for a streaming model, for tests, smoke runs and as a reading example.

No model, no GPU, no network:

* ready ``ready_delay`` s after open, then frames at the init fps (JPEG);
* editing tracks: echoes the latest source frame; after every applied event the frame carries a tint and a border
  in a colour derived from the event id (a visible "edit");
* avatar / generation tracks: the init image (or a gradient) with a moving bar and the same per-event overlay;
* avatar tracks stream 48 kHz audio continuously: silence, or with ``echo_audio`` the received 16 kHz input
  resampled (a stand-in for a renderer that speaks what it is given);
* every event is acked; operations listed in ``unsupported`` (e.g. ``reference:remove``) are acked "unsupported".

Expose it over the WebSocket protocol with ``streameval serve --model streameval.models.mock:MockModel``.
"""
from __future__ import annotations

import hashlib
import io
import threading
import time

import numpy as np

from .base import ActionEvent, OutputSink, ReferenceEvent, SessionInit, StreamingModel, TextEvent

AVATAR_TRACKS = {"digital_human_audio", "digital_human_text", "avatar_performance"}
EDIT_TRACKS = {"instruction_edit", "reference_edit"}


def color_of(event_id: str) -> tuple:
    h = hashlib.sha256(event_id.encode()).digest()
    return (h[0], h[1], h[2])


def _jpeg(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=80)
    return buf.getvalue()


class MockModel(StreamingModel):
    """The CPU stand-in described above, run in-process by the harness or served with ``streameval serve``.

    echo_audio: speak the received input audio instead of silence. unsupported: operations acked "unsupported".
    ready_delay: seconds from open() to ready. max_fps: cap on the output frame rate."""

    def __init__(self, echo_audio: bool = False, unsupported=(), ready_delay: float = 0.3,
                 max_fps: float | None = None):
        self.echo_audio, self.unsupported = echo_audio, set(unsupported)
        self.ready_delay, self.max_fps = ready_delay, max_fps
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.threads: list[threading.Thread] = []

    def describe(self) -> dict:
        return {"transport": "in-process", "model": "mock", "echo_audio": self.echo_audio,
                "unsupported": sorted(self.unsupported)}

    # ---- session ----------------------------------------------------------------------------------
    def open(self, init: SessionInit, sink: OutputSink) -> None:
        from PIL import Image
        self.init, self.sink = init, sink
        self.source = None
        self.overlay = None
        self.audio = bytearray()
        self.base = Image.open(io.BytesIO(init.image)).convert("RGB") if init.image else None
        self.stopped.clear()
        self.threads = [threading.Thread(target=self._start, name="mock-model", daemon=True)]
        self.threads[0].start()

    def _start(self):
        init = self.init
        width, height = int(init.width or 320), int(init.height or 240)
        fps = float(init.fps or 10)
        if self.max_fps:
            fps = min(fps, self.max_fps)
        if self.stopped.wait(self.ready_delay):
            return
        self.sink.ready(model="mock", version="1", width=width, height=height, fps=fps,
                        config={"seed": init.seed, "gpu": "none (CPU mock)"})
        if init.track in AVATAR_TRACKS:
            speaker = threading.Thread(target=self._speaker, name="mock-speaker", daemon=True)
            self.threads.append(speaker)
            speaker.start()
        self._video(width, height, fps)

    def _picture(self, width: int, height: int, k: int):
        from PIL import Image, ImageDraw
        if self.init.track in EDIT_TRACKS:
            if self.source is None:
                return None
            image = Image.open(io.BytesIO(self.source)).convert("RGB")
        else:
            if self.base is not None:
                image = self.base.resize((width, height))
            else:
                x = np.linspace(0, 255, width, dtype=np.uint8)
                image = Image.fromarray(np.stack([np.tile(x, (height, 1))] * 3, axis=-1))
            draw = ImageDraw.Draw(image)
            y = (k * 4) % max(1, image.height)
            draw.rectangle([0, y, image.width, y + 6], fill=(255, 255, 255))
        overlay = self.overlay
        if overlay is not None:
            color, label = overlay
            image = Image.blend(image, Image.new("RGB", image.size, color), 0.35)
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, image.width - 1, image.height - 1], outline=color, width=8)
            draw.text((12, 12), label, fill=(255, 255, 255))
        return image

    def _video(self, width, height, fps):
        t0, k = time.monotonic(), 0
        while not self.stopped.is_set():
            image = self._picture(width, height, k)
            if image is not None:
                self.sink.video_frame(_jpeg(image), time.monotonic_ns())
            k += 1
            self.stopped.wait(max(0.0, t0 + k / fps - time.monotonic()))

    def _speaker(self):
        """20 ms blocks of 48 kHz audio on an absolute clock: echoed speech or silence."""
        block = 960
        t0, k = time.monotonic(), 0
        while not self.stopped.is_set():
            with self.lock:
                take = bytes(self.audio[:block * 2])
                del self.audio[:block * 2]
            self.sink.audio(take + b"\x00" * (block * 2 - len(take)), time.monotonic_ns())
            k += 1
            self.stopped.wait(max(0.0, t0 + k * 0.02 - time.monotonic()))

    # ---- inputs ----------------------------------------------------------------------------------
    def send_event(self, event: TextEvent | ReferenceEvent | ActionEvent) -> None:
        op = ("reference:" + event.op if isinstance(event, ReferenceEvent) else
              "action:" + event.action if isinstance(event, ActionEvent) else "text:" + event.kind)
        if op in self.unsupported or op.split(":", 1)[-1] in self.unsupported:
            self.sink.ack(event.event_id, "unsupported", "mock: " + op + " not supported")
            return
        self.overlay = (color_of(event.event_id), event.event_id)
        self.sink.ack(event.event_id, "applied")

    def push_source_frame(self, pts_ns: int, jpeg: bytes) -> None:
        self.source = jpeg

    def push_audio(self, pts_ns: int, pcm16k: bytes) -> None:
        if self.echo_audio:
            pcm = np.frombuffer(pcm16k, dtype=np.int16)
            with self.lock:
                self.audio += np.repeat(pcm, 3).astype(np.int16).tobytes()   # 16 kHz -> 48 kHz (sample hold)

    def close(self) -> bool:
        self.stopped.set()
        for t in self.threads:
            t.join(2)
        return not any(t.is_alive() for t in self.threads)
