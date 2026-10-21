"""Daily call client for the RTC avatars (providers/rtc.py), with a virtual microphone.

The avatar participant's video goes to sink.video_rgb (the renderer delivers RGBA; converted to RGB only for a
sampled frame) and its audio to sink.audio (48 kHz mono, 20 ms callbacks). The user's audio is written to the
virtual microphone "streameval-mic" (48 kHz mono, non-blocking). Needs the ``rtc`` extra (daily-python);
imported lazily, so planning and dry runs never need it.
"""
from __future__ import annotations

import threading
import time

import numpy as np

MIC = "streameval-mic"


def _rgb(width: int, height: int, rgba) -> bytes:
    return np.frombuffer(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].tobytes()


class DailyCall:
    """One Daily call that forwards the remote participant's media to `sink`; created by an RtcAvatar subclass in
    join(). note/ended report call events, app messages are kept in .messages, .audio_ready is set once a remote
    participant's renderers are attached."""

    def __init__(self, sink, *, note=lambda kind, **kw: None, ended=lambda reason: None, publish_audio=True):
        from daily import CallClient, Daily, EventHandler
        self.sink, self.note, self.ended, self.publish_audio = sink, note, ended, publish_audio
        self.audio_ready = threading.Event()
        self.remote = set()
        self.renderers = []   # kept referenced for the native client
        self.messages = []
        Daily.init()
        self.mic = Daily.create_microphone_device(MIC, sample_rate=48000, channels=1,
                                                  non_blocking=True) if publish_audio else None
        call = self

        class Handler(EventHandler):
            def on_call_state_updated(self, state):
                call.note("call_state", state=str(state))
                if state == "left":
                    call.ended("call_left")

            def on_error(self, message):
                call.note("daily_error", message=str(message)[:300])

            def on_participant_joined(self, participant):
                call.attach(participant)

            def on_participant_updated(self, participant):
                call.attach(participant)

            def on_participant_left(self, participant, reason):
                if participant.get("id") in call.remote:
                    call.ended("avatar_participant_left: " + str(reason))

            def on_app_message(self, message, sender):
                if isinstance(message, dict):
                    props = message.get("properties") or {}
                    call.messages.append({"host_ns": time.monotonic_ns(), "event_type": message.get("event_type"),
                                          "turn_idx": message.get("turn_idx"),
                                          "role": props.get("role") if isinstance(props, dict) else None})

        self.client = CallClient(event_handler=Handler())
        self.client.set_user_name("StreamEval evaluator")

    def attach(self, participant):
        pid = participant.get("id")
        if not isinstance(pid, str) or participant.get("local") or pid in self.remote:
            return
        self.remote.add(pid)
        sink = self.sink
        video = lambda _, frame, __: sink.video_rgb(frame.width, frame.height,
                                                    lambda f=frame: _rgb(f.width, f.height, f.buffer))
        audio = lambda _, data, __: sink.audio(bytes(data.audio_frames))
        self.renderers += [video, audio]
        self.client.set_video_renderer(pid, video, color_format="RGBA")
        self.client.set_audio_renderer(pid, audio, sample_rate=48000, callback_interval_ms=20)
        self.audio_ready.set()

    def join(self, url: str, timeout: float = 30):
        done, errors = threading.Event(), []
        settings = {"inputs": {"camera": False, "microphone": (
            {"isEnabled": True, "settings": {"deviceId": MIC}} if self.publish_audio else False)}}
        self.client.join(url, client_settings=settings,
                         completion=lambda data, error: (errors.append(str(error)[:300]) if error else None, done.set()))
        if not done.wait(timeout) or errors:
            raise RuntimeError("Daily join failed or timed out: " + "; ".join(errors))

    def audio_sink(self, pcm48k: bytes):
        if self.mic is not None:
            self.mic.write_frames(pcm48k)

    def send(self, message: dict):
        self.client.send_app_message(message)

    def close(self):
        """Leave, then let in-flight renderer and completion callbacks drain before releasing the native client
        (releasing right after the last audio block can crash Daily's call-manager thread under heavy load)."""
        from daily import Daily
        left = threading.Event()
        self._leave_done = lambda error: left.set()   # referenced until the native completion has fired
        try:
            self.client.leave(completion=self._leave_done)
            left.wait(30)
            time.sleep(1.0)
        finally:
            self.client.release()
            time.sleep(0.5)
            Daily.deinit()
