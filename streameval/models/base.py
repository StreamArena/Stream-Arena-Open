"""The model interface: what a streaming model implements to be run by the harness.

A model is one live session at a time. The harness opens it with the case's initial conditions, waits until it is
ready, then streams inputs on the case clock and records everything the model outputs:

    harness                                           model
    -------                                           -----
    open(SessionInit, sink)                    ->     start a session; sink.ready(...) once it can take inputs
    send_event(ReferenceEvent "initial")       ->     (reference_edit only) the preloaded reference A
                                               <-     sink.video_frame(jpeg) ... continuously, from the start
                                               <-     sink.audio(pcm 48 kHz)  ... avatar tracks
    -- readiness: ready + first frame (avatars: plus the greeting/idle rule); then the activity epoch --
    push_source_frame(pts_ns, jpeg)            ->     editing tracks: the live source video, in real time
    push_audio(pts_ns, pcm 16 kHz)             ->     audio tracks: a continuous 10 ms stream (silence between clips)
    send_event(Text|Reference|ActionEvent)     ->     a released event; reply with sink.ack(event_id, status)
    close()                                    ->     end the session at the case deadline (or on failure)

Rules the harness enforces, whatever the model does:

* a model only ever sees inputs whose release time has passed; it never sees the oracle or a future event;
* an event whose operation the model does not declare is not sent; it is recorded ``unsupported`` and scored as a
  failure. ``sink.ack(event_id, "unsupported")`` has the same effect; ``"error"`` records a failed dispatch;
* frames and audio are timed on the harness's clock when they reach the sink. Deliver them as soon as they exist;
  buffering output delays it on the record (latency is measured from the release instant to the sampled output);
* every sink method is thread-safe and non-blocking; call them from any thread.

Operations (the ``ops`` of a registry entry) per track are listed in streameval.benchmark.tracks.TRACK_OPS:
``user_audio`` / ``speech_audio`` arrive via push_audio; ``user_prompt`` / ``text_control`` / ``camera_control`` as
TextEvent; ``reference_update`` as ReferenceEvent(op "set", slot "subject"); ``reference:<op>`` as ReferenceEvent;
``action`` (opt-in) as ActionEvent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass, field

PROTOCOL_VERSION = "streameval-selfhost/1"
ACK_STATUS = ("applied", "unsupported", "error")


def _mime(data: bytes) -> str:
    return "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/webp" if data[8:12] == b"WEBP" else "image/jpeg"


@dataclass(frozen=True)
class SessionInit:
    """Initial conditions of one session.

    width/height/fps: the requested output (editing: the pushed source size and rate). prompt: the opening prompt
    (generation), the rendering context (avatar performance) or "". persona: digital human persona + known facts.
    image: the portrait (avatars with benchmark_portrait), the first reference image (image-to-video) or None.
    duration_ms: the case's activity length after readiness (public in the roster; lets a vendor adapter size its
    session); not part of the protocol's init message.
    """
    track: str
    case_id: str
    width: int
    height: int
    fps: float
    seed: int
    output: dict
    prompt: str = ""
    persona: str | None = None
    image: bytes | None = None
    duration_ms: int | None = None

    def message(self) -> dict:
        """The init message of the WebSocket protocol (docs/protocol.md)."""
        msg = {"type": "init", "protocol": PROTOCOL_VERSION, "track": self.track, "case_id": self.case_id,
               "width": self.width, "height": self.height, "fps": self.fps, "prompt": self.prompt,
               "seed": self.seed, "output": self.output}
        if self.persona is not None:
            msg["persona"] = self.persona
        if self.image is not None:
            msg.update(image_b64=base64.b64encode(self.image).decode(), image_mime=_mime(self.image))
        return msg


@dataclass(frozen=True)
class TextEvent:
    """kind: user_prompt | text_control | camera_control, or dialogue_reply (composition systems with text input:
    the reply the shared dialogue stack wrote for the avatar to speak)."""
    event_id: str
    kind: str
    text: str

    def message(self) -> dict:
        return {"type": "text", "event_id": self.event_id, "text": self.text, "kind": self.kind}


@dataclass(frozen=True)
class ReferenceSlot:
    """One target slot. image None = restore the slot to the live source (remove)."""
    slot: str
    image: bytes | None = None
    scope: str | None = None
    asset_id: str | None = None

    def message(self) -> dict:
        if self.image is None:
            return {"slot": self.slot, "image_b64": None, "scope": None}
        return {"slot": self.slot, "image_b64": base64.b64encode(self.image).decode(), "image_mime": _mime(self.image),
                "scope": self.scope, "asset_id": self.asset_id}


@dataclass(frozen=True)
class ReferenceEvent:
    """op: set (replace the active slots) | patch (add or replace the listed ones) | remove (restore the listed
    slots) | reset (restore all). scope: the operation's protection clause. text: the instruction that came with a
    reference_update (image-to-video), else None."""
    event_id: str
    op: str
    slots: tuple[ReferenceSlot, ...] = field(default_factory=tuple)
    scope: str | None = None
    text: str | None = None

    def message(self) -> dict:
        msg = {"type": "reference", "event_id": self.event_id, "op": self.op,
               "slots": [s.message() for s in self.slots]}
        if self.text is not None:
            msg["text"] = self.text
        else:
            msg["scope"] = self.scope
        return msg


@dataclass(frozen=True)
class ActionEvent:
    """A structured game control. phase: start (hold `action` until its stop, at most duration_ms) | stop (release
    the held action; duration_ms 0) | pulse (perform it once for duration_ms)."""
    event_id: str
    action: str
    phase: str
    duration_ms: int

    def message(self) -> dict:
        return {"type": "control", "event_id": self.event_id, "action": self.action,
                "params": {"phase": self.phase, "duration_ms": self.duration_ms}}


class OutputSink(ABC):
    """Where a model delivers its output. Implemented by the harness."""

    @abstractmethod
    def ready(self, **meta) -> None:
        """The session accepts inputs. meta (model, version, width, height, fps, config ...) is recorded."""

    @abstractmethod
    def video_frame(self, jpeg: bytes, pts_ns: int | None = None) -> None:
        """One output frame as JPEG. pts_ns: the model's own clock, recorded for diagnostics only."""

    @abstractmethod
    def video_rgb(self, width: int, height: int, rgb: bytes, pts_ns: int | None = None) -> None:
        """One output frame as packed RGB24 (converted to JPEG only if the frame is sampled). rgb may also be a
        zero-argument callable returning those bytes, called only for a sampled frame (saves converting every
        frame of a client whose decoder delivers another layout)."""

    @abstractmethod
    def audio(self, pcm48k: bytes, pts_ns: int | None = None) -> None:
        """Output speech: s16le mono 48 kHz, any block size, gapless (send silence rather than nothing)."""

    @abstractmethod
    def ack(self, event_id: str, status: str = "applied", detail: str | None = None) -> None:
        """Outcome of send_event: applied | unsupported | error (one per event; later ones are ignored)."""

    @abstractmethod
    def fatal(self, message: str) -> None:
        """The session cannot continue; the attempt fails (a failure in the denominator once it started)."""


class StreamingModel(ABC):
    """One model. The harness calls open/close once per session and the push/send methods in between, from its own
    threads (push_audio and push_source_frame each from a dedicated pacer thread, send_event from the dispatch
    loop); implementations must not block them for long."""

    @abstractmethod
    def open(self, init: SessionInit, sink: OutputSink) -> None:
        """Start a session. Return promptly; report readiness with sink.ready() (from any thread)."""

    @abstractmethod
    def send_event(self, event: TextEvent | ReferenceEvent | ActionEvent) -> None:
        """A released event; report its outcome with sink.ack(event.event_id, ...)."""

    def push_audio(self, pts_ns: int, pcm16k: bytes) -> None:
        """10 ms of input audio (s16le mono 16 kHz, 160 samples); pts_ns = stream time of its first sample.
        Optional; the default ignores the input."""

    def push_source_frame(self, pts_ns: int, jpeg: bytes) -> None:
        """One frame of the live source video (editing tracks); pts_ns = its media time.
        Optional; the default ignores the input."""

    @abstractmethod
    def close(self) -> bool:
        """End the session and release it. Return True once the session is known to have ended."""

    def describe(self) -> dict:
        """Optional facts recorded in binding.json (e.g. the server URL); no credentials."""
        return {}

    def receipt(self) -> dict:
        """Optional facts about the session as it ran, recorded in result.json under provider.receipt after close()
        (a vendor's session id, its final status, how termination was confirmed, client notes); no credentials."""
        return {}
