"""RtcAvatar: a vendor avatar reached through its REST API plus a WebRTC room joined in-process (A layer).

The vendor adapters of models/providers/{runway,liveavatar,tavus} are StreamingModels run by the ordinary harness
(runtime.harness), so readiness, the release clock, capture, VAD and result.json are exactly those of every other
avatar system. A subclass supplies the vendor calls:

    create(init)            REST calls up to a joinable session (the request body is recorded, never a key)
    join()                  connect the room client (providers/livekit.py or providers/daily.py) to self.sink
    send_text(eid, text)    a released user_prompt (conversation_text); returns facts for the dispatch record
    end()                   stop the vendor session and leave the room; True once termination is confirmed

open() returns at once; create + join run on a thread and sink.ready() follows once the avatar's audio track is
subscribed (the harness then waits for the first frame and the greeting/idle rule). A session the vendor ends
during the case is reported with sink.fatal and is a started failure (provider_session_ended_early).

Input audio: the harness streams 16 kHz mono in 10 ms chunks; the room's microphone track takes 48 kHz, so every
chunk is upsampled linearly (``Upsampler``; continuous across chunks, one input sample = 62.5 us of delay).
"""
from __future__ import annotations

import hashlib
import threading
import time

import numpy as np

from ..models.base import StreamingModel, TextEvent
from ..util import scrub
from .account import Account

JOIN_MAX_S = 90


class Upsampler:
    """s16le mono 16 kHz -> 48 kHz, linear interpolation continuous across chunk boundaries."""

    def __init__(self):
        self.last = 0.0

    def __call__(self, pcm16k: bytes) -> bytes:
        x = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float64)
        if not len(x):
            return b""
        s = np.concatenate(([self.last], x))
        y = np.interp(np.arange(3 * len(x)) / 3, np.arange(len(s)), s)
        self.last = x[-1]
        return np.clip(np.round(y), -32768, 32767).astype(np.int16).tobytes()


class RtcAvatar(Account, StreamingModel):
    """Base class of the vendor avatars reached over REST + a WebRTC room. A subclass implements create, join,
    end (and send_text if the vendor takes runtime text) and sets self.room; the harness runs it like any
    StreamingModel."""

    client_transport = "in-process WebRTC client"
    audio_transport = "WebRTC microphone track (48 kHz mono, 10 ms frames)"
    text_transport = None   # how send_text reaches the vendor; None: the vendor has no runtime text input

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sink = self.init = self.room = self.thread = None
        self.closing = False
        self.publish_audio = True
        self.record = {"status": "prepared", "termination_confirmed": False}
        self.notes = []
        self.upsample = Upsampler()

    # ---- vendor hooks --------------------------------------------------------------------------------
    def create(self, init) -> dict:
        raise NotImplementedError

    def join(self) -> None:
        raise NotImplementedError

    def send_text(self, event_id: str, text: str) -> dict:
        raise ValueError(f"{type(self).__name__} has no runtime user-text input")

    def end(self) -> bool:
        raise NotImplementedError

    # ---- helpers for the hooks -----------------------------------------------------------------------
    def note(self, kind: str, **details):
        self.notes.append({"kind": kind, "host_ns": time.monotonic_ns(), **details})

    def ended(self, reason: str):
        """Called by the room client when the vendor side goes away; a failure unless we are closing."""
        self.note("provider_ended", reason=reason)
        if not self.closing and self.sink is not None:
            self.sink.fatal("provider ended the session: " + reason)

    # ---- StreamingModel ------------------------------------------------------------------------------
    def open(self, init, sink) -> None:
        missing = self.missing_credentials()
        if missing:
            raise RuntimeError("credentials missing: " + ", ".join(missing))
        self.init, self.sink = init, sink
        self.publish_audio = init.track != "digital_human_text"   # no microphone track on the text track
        self.thread = threading.Thread(target=self._start, name="rtc-avatar-start", daemon=True)
        self.thread.start()

    def _start(self):
        try:
            self.record["create"] = self.create(self.init)
            self.note("session_created")
            self.join()
            self.note("media_joined")
            if not self.room.audio_ready.wait(JOIN_MAX_S):
                raise RuntimeError(f"no avatar audio track within {JOIN_MAX_S} s of joining")
            if self.closing:
                return
            self.sink.ready(model=type(self).__name__, session=self.record.get("create"))
        except Exception as exc:
            if self.record["status"] == "submission_unknown":
                self.note("create_outcome_unknown")
            if not self.closing:
                self.sink.fatal("session start failed: " + type(exc).__name__ + ": " + scrub(str(exc))[:400])

    def send_event(self, event) -> None:
        if not isinstance(event, TextEvent) or event.kind != "user_prompt" or self.text_transport is None:
            self.sink.ack(event.event_id, "unsupported", f"{type(self).__name__} takes user_prompt text only"
                          if self.text_transport else f"{type(self).__name__} has no runtime text input")
            return
        try:
            detail = self.send_text(event.event_id, event.text)
            self.note("text_sent", event_id=event.event_id, transport=self.text_transport,
                      text_sha256=hashlib.sha256(event.text.encode()).hexdigest(), **(detail or {}))
            self.sink.ack(event.event_id, "applied")
        except Exception as exc:
            self.sink.ack(event.event_id, "error", type(exc).__name__ + ": " + scrub(str(exc))[:300])

    def push_audio(self, pts_ns: int, pcm16k: bytes) -> None:
        room = self.room
        if room is not None and self.publish_audio:
            room.audio_sink(self.upsample(pcm16k))

    def close(self) -> bool:
        self.closing = True
        if self.thread is not None:
            self.thread.join(JOIN_MAX_S + 120)
        terminated = bool(self.end())
        self.record["termination_confirmed"] = terminated
        return terminated

    def system_config(self, system_id: str, spec: dict) -> dict:
        text = spec["track"] == "digital_human_text"
        return {"client": {
            "transport": self.client_transport,
            "capture": "avatar participant only; first received frame per activity bin (receive time, no display); "
                       "remote audio 48 kHz mono int16 with online VAD",
            "audio_in": None if text else "16 kHz mono s16le 10 ms chunks on an absolute clock, silence between "
                                          "clips, upsampled to 48 kHz (linear, continuous across chunks) into the "
                                          + self.audio_transport,
            "text_in": self.text_transport if text else None,
            "start": f"vendor REST create + room join on a thread; ready once the avatar's audio track is "
                     f"subscribed (at most {JOIN_MAX_S} s after joining)"}}

    def describe(self) -> dict:
        return {"transport": type(self).__name__ + ": vendor REST + " + self.client_transport,
                "audio_transport": self.audio_transport if self.publish_audio else None,
                "text_transport": self.text_transport}

    def receipt(self) -> dict:
        return {**self.record, "notes": self.notes}
