"""LiveKit room client for the RTC avatars (providers/rtc.py), on its own asyncio loop thread.

The avatar participant's video goes to sink.video_rgb (RGB24, converted to JPEG only for sampled frames) and its
audio to sink.audio (48 kHz mono, 20 ms frames). The local microphone track (48 kHz mono) is fed by audio_sink
without ever blocking the harness's pacer thread. Needs the ``rtc`` extra (``pip install -e .[rtc]``); imported
lazily, so planning and dry runs never need it.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time


class LiveKitRoom:
    """One LiveKit room connection that forwards the avatar's media to `sink`; created by an RtcAvatar subclass
    in join(). accept(identity) selects the avatar participant, note/ended report room events,
    data_topics are the data-channel topics kept in .data, .audio_ready is set once avatar audio is subscribed."""

    def __init__(self, sink, *, accept=lambda identity: True, note=lambda kind, **kw: None,
                 ended=lambda reason: None, data_topics=(), publish_audio=True):
        self.sink, self.accept, self.note, self.ended = sink, accept, note, ended
        self.data_topics, self.publish_audio = set(data_topics), publish_audio
        self.audio_ready = threading.Event()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name="livekit-room", daemon=True)
        self.thread.start()
        self.room = self.source = None
        self.tasks = []
        self.data = []

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coro, timeout):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def connect(self, url: str, token: str, timeout: float = 30):
        self.call(self._connect(url, token, timeout), timeout + 5)

    async def _connect(self, url, token, timeout):
        from livekit import rtc
        self.room = rtc.Room(loop=self.loop)

        @self.room.on("track_subscribed")
        def subscribed(track, publication, participant):
            self.note("track_subscribed", track_kind=str(track.kind), identity=participant.identity)
            if not self.accept(participant.identity):
                return
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self.tasks.append(self.loop.create_task(self._video(track)))
            elif track.kind == rtc.TrackKind.KIND_AUDIO:
                self.audio_ready.set()
                self.tasks.append(self.loop.create_task(self._audio(track)))

        @self.room.on("disconnected")
        def disconnected(reason):
            self.note("room_disconnected", reason=str(reason))
            self.ended("room disconnected: " + str(reason))

        @self.room.on("participant_disconnected")
        def left(participant):
            self.note("participant_disconnected", identity=participant.identity)
            if self.accept(participant.identity):
                self.ended("avatar_participant_left")

        @self.room.on("data_received")
        def data(packet):
            if packet.topic in self.data_topics:
                try:
                    payload = json.loads(packet.data)
                except (ValueError, TypeError):
                    payload = {"unparsed_bytes": len(packet.data)}
                self.data.append({"host_ns": time.monotonic_ns(), "topic": packet.topic, "payload": payload})

        await asyncio.wait_for(self.room.connect(url, token), timeout=timeout)
        if self.publish_audio:
            self.source = rtc.AudioSource(48000, 1, queue_size_ms=1000, loop=self.loop)
            track = rtc.LocalAudioTrack.create_audio_track("streameval-user-audio", self.source)
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            await self.room.local_participant.publish_track(track, options)

    async def _video(self, track):
        from livekit import rtc
        stream = rtc.VideoStream(track, capacity=8, format=rtc.VideoBufferType.RGB24)
        try:
            async for packet in stream:
                frame = packet.frame
                self.sink.video_rgb(frame.width, frame.height, lambda f=frame: bytes(f.data))
        finally:
            await stream.aclose()

    async def _audio(self, track):
        from livekit import rtc
        stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1, frame_size_ms=20)
        try:
            async for packet in stream:
                self.sink.audio(bytes(packet.frame.data))
        finally:
            await stream.aclose()

    def audio_sink(self, pcm48k: bytes):
        """Called from the harness's pacer thread; never blocks it."""
        if self.source is None:
            return
        from livekit import rtc
        frame = rtc.AudioFrame(pcm48k, 48000, 1, len(pcm48k) // 2)
        asyncio.run_coroutine_threadsafe(self.source.capture_frame(frame), self.loop)

    def publish_data(self, payload: dict, topic: str):
        self.call(self.room.local_participant.publish_data(json.dumps(payload), reliable=True, topic=topic), 5)

    def remote_identities(self) -> list[str]:
        return [] if self.room is None else [p.identity for p in self.room.remote_participants.values()]

    def close(self):
        async def shutdown():
            if self.room is not None:
                try:
                    await asyncio.wait_for(self.room.disconnect(), timeout=10)
                except Exception as exc:
                    self.note("disconnect_error", error=type(exc).__name__)
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            if self.source is not None:
                await self.source.aclose()
        try:
            self.call(shutdown(), 30)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(5)
