"""Capture of a model's output on the host clock: the receipts the judge reads.

Written under <attempt>/browser/ (the folder is named after the browser-based providers, whose capture page writes
the same files):

  frame_samples.jsonl + frames/s_NNNNNN.jpg        first received frame of every activity bin (content independent)
  remote_video_frames.jsonl                        every received frame: host_ns, server pts, size
  remote_audio.s16le + remote_audio_blocks.jsonl   48 kHz mono int16 on a gapless sample clock
  vad_online.jsonl                                 online voice activity of the received audio
  reference_frame.jpg                              last frame received before the activity epoch

The page clock is the host clock (page_ms = time.monotonic_ns() / 1e6), so result.json records
clock_offset_ns = 0. Receive time stands in for presentation time (there is no display).
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import io
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np

RATE = 48000
# the AudioWorklet segmenter of the browser-based providers (providers/js/vidu_worklet.js) uses the same constants
WINDOW_S, VOICED_DBFS, MIN_VOICED_S, MAX_GAP_S = 0.02, -45.0, 0.3, 0.25


class OnlineVAD:
    """Energy segmenter: emits ('onset', start, previous_end, detected) and ('offset', end) in sample frames."""

    def __init__(self, emit, rate: int = RATE):
        self.emit, self.rate = emit, rate
        self.win = round(rate * WINDOW_S)
        self.pending = np.zeros(0, dtype=np.float64)
        self.frame = 0            # index of the first pending sample
        self.seg_start = self.last_voiced_end = self.last_qualified_end = None
        self.seg_voiced, self.reported, self.speaking = 0, False, False

    def feed(self, pcm: np.ndarray):
        data = np.concatenate([self.pending, pcm.astype(np.float64) / 32768.0])
        n = len(data) // self.win
        for i in range(n):
            chunk = data[i * self.win:(i + 1) * self.win]
            self.window(self.frame + i * self.win, float(np.sqrt(np.mean(chunk * chunk))))
        self.pending = data[n * self.win:]
        self.frame += n * self.win

    def window(self, frame: int, rms: float):
        db = 20 * np.log10(max(rms, 1e-9))
        gap_limit = MAX_GAP_S * self.rate
        if db <= VOICED_DBFS:
            if self.speaking and frame - self.last_voiced_end > gap_limit:
                self.speaking = False
                self.emit("offset", end=self.last_voiced_end)
            return
        gap = float("inf") if self.last_voiced_end is None else frame - self.last_voiced_end
        if self.seg_start is None or gap > gap_limit:
            self.seg_start, self.seg_voiced, self.reported = frame, 0, False
        self.seg_voiced += self.win
        self.last_voiced_end = frame + self.win
        if self.seg_voiced >= MIN_VOICED_S * self.rate:
            self.speaking = True
            if not self.reported:
                self.reported = True
                self.emit("onset", start=self.seg_start, previous_end=self.last_qualified_end,
                          detected=self.last_voiced_end)
            self.last_qualified_end = self.last_voiced_end


class MediaCapture:
    """Thread-safe sink for a model's output; lifecycle events go to .events for the harness."""

    def __init__(self, directory: Path, sample_ms: int, clock=time.monotonic_ns):
        self.dir = Path(directory)
        (self.dir / "frames").mkdir(parents=True, mode=0o700)
        self.sample_ms, self.clock = sample_ms, clock
        self.events: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.zero_ms = self.bin_limit = None
        self.next_bin = 0
        self.frames = self.audio_blocks = self.samples = 0
        self.last_frame = None      # ("jpeg", bytes) or ("raw", width, height, data, mode)
        self.received = []          # (host_ns, server_pts_ns, bytes)
        self.last_page_ms = None
        self.block_starts: list[int] = []
        self.block_pages: list[float] = []
        self.audio_attached = False
        self.ended = False
        self.vad = OnlineVAD(self._vad)
        self.reference = None

    def page_ms(self) -> float:
        return self.clock() / 1e6

    def _line(self, name: str, row: dict):
        with (self.dir / name).open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def emit(self, event: str, **fields):
        self.events.put({"event": event, **fields, "host_ns": self.clock()})

    # ---- video --------------------------------------------------------------------
    def _receive(self, last, size: int, pts_ns):
        """Book one received frame; returns (page_ms, case_ms, first, bin or None)."""
        with self.lock:
            if self.ended:
                return None
            now = self.clock()
            page = now / 1e6
            self.frames += 1
            first = self.frames == 1
            self.last_frame = last
            self.received.append((now, pts_ns, size))
            case_ms = None if self.zero_ms is None else page - self.zero_ms
            b = None
            if case_ms is not None:
                k = int(case_ms // self.sample_ms)
                if case_ms >= 0 and k >= self.next_bin and k < self.bin_limit:
                    self.next_bin, b = k + 1, k
        return page, case_ms, first, b

    def _sample(self, b: int, raw: bytes, case_ms: float, page: float, width: int, height: int, pts_ns):
        relative = f"frames/s_{b:06d}.jpg"
        with (self.dir / relative).open("xb") as handle:
            handle.write(raw)
        self._line("frame_samples.jsonl", {
            "sample_bin": b, "scheduled_bin_start_ms": b * self.sample_ms, "case_ms": case_ms,
            "presentation_ms": page, "presentation_basis": "receive_callback", "width": width, "height": height,
            "selection_basis": "content_independent", "path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
            **({"server_pts_ns": pts_ns} if pts_ns is not None else {})})

    def video_jpeg(self, data: bytes, pts_ns: int | None = None):
        """A JPEG frame as received; a kept frame is stored without re-encoding."""
        from PIL import Image
        booked = self._receive(("jpeg", data), len(data), pts_ns)
        if booked is None:
            return
        page, case_ms, first, b = booked
        if not first and b is None:
            return
        width, height = Image.open(io.BytesIO(data)).size
        if first:
            self.emit("first_remote_frame", presentation_ms=page, width=width, height=height)
        if b is not None:
            self._sample(b, data, case_ms, page, width, height, pts_ns)

    def video(self, width: int, height: int, data, mode: str = "RGB", pts_ns: int | None = None):
        """A raw frame: its bytes in PIL `mode`, or a callable returning them (converted only if the frame is kept)."""
        data = data if callable(data) else bytes(data)
        booked = self._receive(("raw", width, height, data, mode), width * height, pts_ns)
        if booked is None:
            return
        page, case_ms, first, b = booked
        if first:
            self.emit("first_remote_frame", presentation_ms=page, width=width, height=height)
        if b is not None:
            self._sample(b, _jpeg(width, height, data, mode, 90), case_ms, page, width, height, pts_ns)

    # ---- audio --------------------------------------------------------------------
    def attach_audio(self):
        if not self.audio_attached:
            self.audio_attached = True
            self.emit("remote_audio_attached", sample_rate=RATE)

    def audio(self, pcm: bytes, rate: int = RATE, channels: int = 1):
        if rate != RATE:
            raise ValueError("capture expects 48 kHz audio")
        samples = np.frombuffer(pcm, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
        if not len(samples):
            return
        with self.lock:
            if self.ended:
                return
            # The first sample becomes available at this callback, regardless of the block size.
            page = self.page_ms()
            if self.last_page_ms is not None:
                page = max(page, self.last_page_ms)
            self.last_page_ms = page
            first = self.samples
            self.block_starts.append(first)
            self.block_pages.append(page)
            self.samples += len(samples)
            self.audio_blocks += 1
            with (self.dir / "remote_audio.s16le").open("ab") as handle:
                handle.write(samples.tobytes())
            self._line("remote_audio_blocks.jsonl", {"first_frame": first, "samples": len(samples), "page_ms": page})
            self.vad.feed(samples)

    def frame_page_ms(self, frame: int | None):
        if frame is None or not self.block_starts:
            return None
        i = max(0, bisect_right(self.block_starts, frame) - 1)
        return self.block_pages[i] + (frame - self.block_starts[i]) * 1000 / RATE

    def _vad(self, kind: str, **data):
        if kind == "onset":
            row = {"start_frame": data["start"], "previous_end_frame": data["previous_end"],
                   "detected_frame": data["detected"], "start_page_ms": self.frame_page_ms(data["start"]),
                   "previous_end_page_ms": self.frame_page_ms(data["previous_end"]),
                   "detected_page_ms": self.frame_page_ms(data["detected"])}
            self._line("vad_online.jsonl", row)
            self.emit("speech_onset", **row)
        else:
            row = {"end_frame": data["end"], "end_page_ms": self.frame_page_ms(data["end"])}
            self._line("vad_online.jsonl", row)
            self.emit("speech_offset", **row)

    # ---- epoch / finish -----------------------------------------------------------
    def start(self, zero_ns: int, duration_ms: int) -> dict:
        with self.lock:
            self.zero_ms, self.bin_limit, self.next_bin = zero_ns / 1e6, -(-duration_ms // self.sample_ms), 0
            last = self.last_frame
        if last is not None:
            raw = last[1] if last[0] == "jpeg" else _jpeg(*last[1:], 95)
            (self.dir / "reference_frame.jpg").write_bytes(raw)
            self.reference = {"path": "reference_frame.jpg", "sha256": hashlib.sha256(raw).hexdigest(),
                              "basis": "last frame received before the activity epoch"}
        return {"zero_page_ms": self.zero_ms, "reference_frame": self.reference}

    def finish(self) -> dict:
        with self.lock:
            self.ended = True
            rows = list(self.received)
            summary = {"frames": self.frames, "samples": self.next_bin, "audio_blocks": self.audio_blocks,
                       "audio_samples": self.samples, "reference_frame": self.reference}
        with (self.dir / "remote_video_frames.jsonl").open("w") as handle:
            for host_ns, pts, size in rows:
                handle.write(json.dumps({"host_ns": host_ns, "server_pts_ns": pts, "bytes": size}) + "\n")
        return summary


def _jpeg(width, height, data, mode, quality) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.frombytes(mode, (width, height), bytes(data() if callable(data) else data)).convert("RGB").save(
        buf, "JPEG", quality=quality)
    return buf.getvalue()
