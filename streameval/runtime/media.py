"""Media plumbing of the harness: ffmpeg, audio loading, the 10 ms audio-input pacer and the real-time source pusher.

Both pacers run on the host clock and hand each chunk/frame to a ``send(pts_ns, payload)`` callable, which the
harness binds to ``StreamingModel.push_audio`` / ``push_source_frame``.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
import wave

import numpy as np

AUDIO_IN_RATE = 16000
AUDIO_IN_CHUNK = AUDIO_IN_RATE // 100          # 10 ms


# ---- helpers -------------------------------------------------------------------------------------------

def ffmpeg() -> str:
    """ffmpeg on PATH, else the binary bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`") from exc


def probe_video(path: Path) -> dict:
    """width, height and average fps of the first video stream (ffprobe if present, else `ffmpeg -i`)."""
    probe = shutil.which("ffprobe") or (Path(ffmpeg()).with_name("ffprobe") if Path(ffmpeg()).with_name(
        "ffprobe").exists() else None)
    if probe:
        out = subprocess.run([str(probe), "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,avg_frame_rate", "-of", "json", str(path)],
                             capture_output=True, text=True, check=True)
        s = json.loads(out.stdout)["streams"][0]
        num, den = s["avg_frame_rate"].split("/")
        return {"width": s["width"], "height": s["height"], "fps": float(num) / float(den)}
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    line = next((l for l in out.stderr.splitlines() if "Video:" in l), None)
    size = line and re.search(r"(?<![0-9])(\d{2,5})x(\d{2,5})(?![0-9])", line)
    fps = line and re.search(r"([0-9.]+) fps", line)
    if not (size and fps):
        raise RuntimeError(f"cannot read the video stream of {path}")
    return {"width": int(size[1]), "height": int(size[2]), "fps": float(fps[1])}


def push_spec(info: dict, policy: dict) -> dict:
    """Source resolution scaled down to policy.max_side (never upscaled), even sizes; fps capped at max_fps."""
    scale = min(1.0, policy["max_side"] / max(info["width"], info["height"]))
    even = lambda v: max(2, int(v) - int(v) % 2)
    return {"width": even(info["width"] * scale), "height": even(info["height"] * scale),
            "fps": min(policy["max_fps"], max(1, round(info["fps"]))), "source": info, "scaled": scale < 1.0,
            "upscaled": False}


def resample(pcm, src: int, dst: int) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.float64)
    if src != dst and len(pcm):
        n = round(len(pcm) * dst / src)
        pcm = np.interp(np.arange(n) * src / dst, np.arange(len(pcm)), pcm)
    return np.clip(np.round(pcm), -32768, 32767).astype(np.int16)


def load_pcm(path: Path, rate: int) -> np.ndarray:
    """A PCM16 WAV as mono int16 at `rate` (linear resampling)."""
    with wave.open(str(path)) as w:
        src, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("expected PCM16 audio")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return resample(pcm, src, rate)


def wav_ns(path: Path) -> int:
    """Measured duration of a WAV in nanoseconds."""
    with wave.open(str(path)) as w:
        return round(w.getnframes() * 1_000_000_000 / w.getframerate())


# ---- audio-input pacer ---------------------------------------------------------------------------------

class AudioInPacer:
    """Continuous 16 kHz mono input stream in 10 ms chunks on an absolute clock.

    Silence unless a clip is playing; clips queue back to back (never overlap, never stretch the timeline). A chunk
    later than MAX_LATE_NS stops the pacer with a fatal event. send(pts_ns, pcm) gets every chunk with pts = stream
    time of its first sample; on_chunk(pcm) sees every chunk sent (composition systems whose renderer does not play
    audio capture the pushed speech from it)."""
    CHUNK = AUDIO_IN_CHUNK
    MAX_LATE_NS = 200_000_000

    def __init__(self, send, events: queue.Queue, on_chunk=None, clock=time.monotonic_ns, sleep=time.sleep):
        self.send, self.events, self.on_chunk, self.clock, self.sleep = send, events, on_chunk, clock, sleep
        self.lock = threading.Lock()
        self.clips = deque()
        self.k = self.tail_k = 0
        self.t0 = None
        self.stopped = threading.Event()
        self.error = None
        self.thread = None

    def due(self, k: int) -> int:
        return self.t0 + k * 10_000_000

    def start(self):
        self.t0 = self.clock()
        self.thread = threading.Thread(target=self._run, name="arena-audio-in", daemon=True)
        self.thread.start()

    def step(self) -> bool:
        wait = self.due(self.k) - self.clock()
        if wait > 0:
            self.sleep(wait / 1e9)
        late = self.clock() - self.due(self.k)
        if late > self.MAX_LATE_NS:
            self.error = f"audio_pacing_deadline_exceeded: {late / 1e9:.3f}s"
            self.events.put({"event": "fatal", "error": self.error, "host_ns": self.clock()})
            return False
        with self.lock:
            chunk = np.zeros(self.CHUNK, dtype=np.int16)
            if self.clips and self.k >= self.clips[0]["start_k"]:
                clip = self.clips[0]
                offset = (self.k - clip["start_k"]) * self.CHUNK
                part = clip["pcm"][offset:offset + self.CHUNK]
                chunk[:len(part)] = part
                if offset + self.CHUNK >= len(clip["pcm"]):
                    self.clips.popleft()
            k = self.k
            self.k += 1
        data = chunk.tobytes()
        try:
            self.send(k * 10_000_000, data)
        except Exception as exc:
            self.error = "audio_in send failed: " + type(exc).__name__
            self.events.put({"event": "fatal", "error": self.error, "host_ns": self.clock()})
            return False
        if self.on_chunk is not None:
            self.on_chunk(data)
        return True

    def _run(self):
        while not self.stopped.is_set() and self.step():
            pass

    def play(self, event_id: str, pcm: np.ndarray) -> dict:
        """Queue a 16 kHz clip from the next chunk (or right after the queued clips); returns its schedule."""
        with self.lock:
            start_k = max(self.k + 1, self.tail_k)
            self.clips.append({"event_id": event_id, "pcm": pcm, "start_k": start_k})
            self.tail_k = start_k + -(-len(pcm) // self.CHUNK)
        start = self.due(start_k)
        end = start + round(len(pcm) * 1_000_000_000 / AUDIO_IN_RATE)
        return {"event_id": event_id, "duration_s": len(pcm) / AUDIO_IN_RATE, "page_start_ms": start / 1e6,
                "page_end_ms": end / 1e6, "chunk_index": start_k,
                "basis": "absolute 10 ms audio_in pacer, 16 kHz mono (page clock = host monotonic)"}

    def stop(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(2)


# ---- source pusher (editing) ---------------------------------------------------------------------------

class SourcePusher:
    """Decodes the case source video with ffmpeg at the push size/fps and sends one JPEG per frame.

    Before the epoch the first frame is held (sent at fps with media time 0); after go_live(epoch) frame k is sent at
    epoch + k / fps with pts = media time k / fps. A frame more than one period late is dropped (and counted) so the
    source never drifts from the case clock; after the last decoded frame the last one is held. Every sent frame is
    logged to source_frames.jsonl (presentation_ms = send time, media_time in s), which the judge uses to pair
    output frames with source frames."""

    def __init__(self, path: Path, push: dict, send, log_path: Path, events: queue.Queue, clock=time.monotonic_ns):
        self.path, self.push, self.send, self.events, self.clock = path, push, send, events, clock
        self.fps = push["fps"]
        self.log = log_path.open("w")
        self.frames: queue.Queue = queue.Queue(maxsize=max(8, 2 * int(self.fps)))
        self.epoch = None
        self.stopped = threading.Event()
        self.first = None
        self.stats = {"sent": 0, "held": 0, "dropped_late": 0, "decoded": 0, "exhausted_at_frame": None}
        self.proc = subprocess.Popen(
            [ffmpeg(), "-v", "error", "-nostdin", "-i", str(path), "-vf",
             f"fps={self.fps},scale={push['width']}:{push['height']}", "-f", "image2pipe", "-c:v", "mjpeg",
             "-q:v", "3", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.decoder = threading.Thread(target=self._decode, name="arena-source-decode", daemon=True)
        self.decoder.start()
        self.sender = None

    def _decode(self):
        try:
            self._decode_frames()
        except (ValueError, OSError):   # stdout closed by stop()
            pass

    def _decode_frames(self):
        buf = b""
        for chunk in iter(lambda: self.proc.stdout.read(1 << 16), b""):
            buf += chunk
            while True:
                end = buf.find(b"\xff\xd9")
                if end < 0:
                    break
                frame, buf = buf[:end + 2], buf[end + 2:]
                start = frame.find(b"\xff\xd8")
                self.stats["decoded"] += 1
                if not self._put(frame[start:] if start >= 0 else frame):
                    return
        self._put(None)

    def _put(self, item) -> bool:
        while not self.stopped.is_set():
            try:
                self.frames.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def start(self, timeout: float = 60):
        self.first = self.frames.get(timeout=timeout)
        if self.first is None:
            raise RuntimeError("source video has no frames")
        self.sender = threading.Thread(target=self._run, name="arena-source-push", daemon=True)
        self.sender.start()

    def go_live(self, epoch_ns: int):
        self.epoch = epoch_ns

    def _emit(self, data: bytes, index: int, media_s: float, due_ns: int, phase: str):
        self.send(round(media_s * 1e9), data)
        self.log.write(json.dumps({"frame": index, "media_time": media_s, "presentation_ms": self.clock() / 1e6,
                                   "due_ms": due_ns / 1e6, "phase": phase, "bytes": len(data)}) + "\n")

    def _wait(self, due: int) -> bool:
        wait = due - self.clock()
        if wait > 0:
            time.sleep(min(wait / 1e9, 0.02))
            return False
        return True

    def _run(self):
        period = 1e9 / self.fps
        try:
            t0, h = self.clock(), 0
            while self.epoch is None and not self.stopped.is_set():
                due = t0 + round(h * period)
                if self._wait(due) and self.epoch is None:
                    self._emit(self.first, -1, 0.0, due, "held_before_epoch")
                    self.stats["held"] += 1
                    h += 1
            k, current, exhausted = 0, self.first, False
            while not self.stopped.is_set():
                due = self.epoch + round(k * period)
                if not self._wait(due):
                    continue
                if k > 0 and not exhausted:
                    nxt = self.frames.get()
                    if nxt is None:
                        exhausted = True
                        self.stats["exhausted_at_frame"] = k
                    else:
                        current = nxt
                if k > 0 and self.clock() - due > period:
                    self.stats["dropped_late"] += 1
                else:
                    self._emit(current, k, k / self.fps, due, "live")
                    self.stats["sent"] += 1
                k += 1
        except Exception as exc:
            if not self.stopped.is_set():
                self.events.put({"event": "fatal", "host_ns": self.clock(),
                                 "error": "source push failed: " + type(exc).__name__ + ": " + str(exc)[:300]})

    def stop(self) -> dict:
        self.stopped.set()
        if self.sender is not None:
            self.sender.join(2)
        self.proc.kill()
        self.proc.wait()
        self.proc.stdout.close()
        self.log.close()
        return dict(self.stats)
