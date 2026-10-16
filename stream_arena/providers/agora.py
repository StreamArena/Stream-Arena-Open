"""Agora RTC channel client for the released Vivix adapter (Agora Python Server SDK), lazy SDK import for offline installation checks.

Remote video (I420) and audio (48 kHz mono before mixing) of the provider's publisher uid go to a
a sink bridge; the local audio track is fed 10 ms 48 kHz frames by the shared harness
through audio_sink. SDK callbacks only copy buffers into a queue (the SDK forbids work in its
callbacks); one worker thread per media kind converts and hands them to the capture, so the capture's
receive time is the worker's dequeue time (queueing delay is recorded per run as worker_lag_ms).
Imports the SDK lazily so planning and tests never need it.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

RATE = 48000


def i420_to_rgb(width: int, height: int, y: bytes, u: bytes, v: bytes, ys: int, us: int, vs: int) -> bytes:
    """BT.601 limited-range I420 (the WebRTC decoder output) to packed RGB24."""
    Y = np.frombuffer(y, dtype=np.uint8)[:ys * height].reshape(height, ys)[:, :width].astype(np.float32)
    h2, w2 = (height + 1) // 2, (width + 1) // 2
    U = np.frombuffer(u, dtype=np.uint8)[:us * h2].reshape(h2, us)[:, :w2].astype(np.float32)
    V = np.frombuffer(v, dtype=np.uint8)[:vs * h2].reshape(h2, vs)[:, :w2].astype(np.float32)
    U = U.repeat(2, 0).repeat(2, 1)[:height, :width] - 128.0
    V = V.repeat(2, 0).repeat(2, 1)[:height, :width] - 128.0
    C = (Y - 16.0) * 1.164
    rgb = np.stack([C + 1.596 * V, C - 0.392 * U - 0.813 * V, C + 2.017 * U], axis=-1)
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8).tobytes()


class AgoraRoom:
    def __init__(self, capture, *, publisher: str, note=lambda kind, **kw: None, publish_audio: bool = True,
                 log_dir: str | None = None):
        self.capture, self.publisher, self.note = capture, str(publisher), note
        self.publish_audio, self.log_dir = publish_audio, log_dir
        self.disconnected = None
        self.service = self.conn = None
        self.queues = {"video": queue.Queue(maxsize=200), "audio": queue.Queue(maxsize=2000)}
        self.stop = threading.Event()
        self.workers = [threading.Thread(target=self._work, args=(kind,), name="v1-agora-" + kind, daemon=True)
                        for kind in self.queues]
        self.lag_ms = {"video": [], "audio": []}
        self.dropped = 0
        self.audio_attached = False
        self.connected = threading.Event()
        self.push_errors = 0
        self.format_noted = False
        self.outbox: queue.Queue = queue.Queue()
        self.push_lag_ms = []
        self.sender = threading.Thread(target=self._send, name="v1-agora-push", daemon=True)

    # ---- SDK callbacks (copy only) --------------------------------------------------------
    def _put(self, item):
        try:
            self.queues[item[0]].put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _work(self, kind: str):
        source = self.queues[kind]
        while not self.stop.is_set():
            try:
                item = source.get(timeout=0.1)
            except queue.Empty:
                continue
            self.lag_ms[kind].append((time.monotonic_ns() - item[1]) / 1e6)
            if item[0] == "video":
                _, _, w, h, y, u, v, ys, us, vs = item
                # converted only for the frames the capture keeps (2 per second), not all 24
                self.capture.video(w, h, lambda: i420_to_rgb(w, h, y, u, v, ys, us, vs), "RGB")
            else:
                if not self.audio_attached:
                    self.audio_attached = True
                    self.capture.attach_audio()
                self.capture.audio(item[2])

    def connect(self, app_id: str, channel: str, token: str | None, uid: str, timeout: float = 30):
        from agora.rtc.agora_base import (AgoraServiceConfig, AudioPublishType, AudioScenarioType,
                                          AudioSubscriptionOptions, ChannelProfileType, ClientRoleType,
                                          RTCConnConfig, RtcConnectionPublishConfig, VideoPublishType)
        from agora.rtc.agora_service import AgoraService
        from agora.rtc.audio_frame_observer import IAudioFrameObserver
        from agora.rtc.local_user_observer import IRTCLocalUserObserver
        from agora.rtc.rtc_connection_observer import IRTCConnectionObserver
        from agora.rtc.video_frame_observer import IVideoFrameObserver
        room = self

        class Conn(IRTCConnectionObserver):
            def on_connected(self, agora_rtc_conn, conn_info, reason):
                room.note("agora_connected", reason=int(reason))
                room.connected.set()

            def on_disconnected(self, agora_rtc_conn, conn_info, reason):
                room.note("agora_disconnected", reason=int(reason))
                room.disconnected = room.disconnected or f"agora_disconnected_{int(reason)}"

            def on_connection_lost(self, agora_rtc_conn, conn_info):
                room.note("agora_connection_lost")

            def on_connection_failure(self, agora_rtc_conn, info, reason):
                room.note("agora_connection_failure", reason=int(reason))
                room.disconnected = room.disconnected or f"agora_connection_failure_{int(reason)}"

            def on_user_joined(self, agora_rtc_conn, user_id):
                room.note("participant_joined", identity=str(user_id))

            def on_user_left(self, agora_rtc_conn, user_id, reason):
                room.note("participant_left", identity=str(user_id), reason=int(reason))
                if str(user_id) == room.publisher:
                    room.disconnected = room.disconnected or "avatar_participant_left"

        class User(IRTCLocalUserObserver):
            def on_user_video_track_subscribed(self, agora_local_user, user_id, info, agora_remote_video_track):
                room.note("track_subscribed", track_kind="video", identity=str(user_id))

            def on_user_audio_track_subscribed(self, agora_local_user, user_id, agora_remote_audio_track):
                room.note("track_subscribed", track_kind="audio", identity=str(user_id))
                if str(user_id) == room.publisher and not room.audio_attached:
                    room.audio_attached = True
                    room.capture.attach_audio()   # a queue put only

        class Video(IVideoFrameObserver):
            def on_frame(self, channel_id, remote_uid, frame):
                if str(remote_uid) == room.publisher:
                    h2 = (frame.height + 1) // 2
                    room._put(("video", time.monotonic_ns(), frame.width, frame.height,
                               bytes(frame.y_buffer[:frame.y_stride * frame.height]),
                               bytes(frame.u_buffer[:frame.u_stride * h2]), bytes(frame.v_buffer[:frame.v_stride * h2]),
                               frame.y_stride, frame.u_stride, frame.v_stride))
                return 1

        class Audio(IAudioFrameObserver):
            def on_playback_audio_frame_before_mixing(self, agora_local_user, channelId, uid, audio_frame,
                                                      vad_result_state, vad_result_bytearray):
                if str(uid) == room.publisher:
                    if audio_frame.samples_per_sec != RATE or audio_frame.channels != 1:
                        if not room.format_noted:
                            room.format_noted = True
                            room.note("unexpected_audio_format", rate=audio_frame.samples_per_sec,
                                      channels=audio_frame.channels)
                    else:
                        room._put(("audio", time.monotonic_ns(), bytes(audio_frame.buffer)))
                return 1

        self._observers = (Conn(), User(), Video(), Audio())   # keep references alive for ctypes
        config = AgoraServiceConfig()
        config.appid = app_id
        config.enable_video = 1
        config.audio_scenario = AudioScenarioType.AUDIO_SCENARIO_DEFAULT
        if self.log_dir:
            config.log_path = self.log_dir + "/agorasdk.log"
            config.data_dir = config.config_dir = self.log_dir
        self.service = AgoraService()
        if self.service.initialize(config) != 0:
            raise RuntimeError("agora service initialize failed")
        conn_config = RTCConnConfig(
            auto_subscribe_audio=1, auto_subscribe_video=1, client_role_type=ClientRoleType.CLIENT_ROLE_BROADCASTER,
            channel_profile=ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
            audio_subs_options=AudioSubscriptionOptions(packet_only=0, pcm_data_only=1, bytes_per_sample=2,
                                                        number_of_channels=1, sample_rate_hz=RATE))
        publish = RtcConnectionPublishConfig(audio_scenario=AudioScenarioType.AUDIO_SCENARIO_DEFAULT,
                                             is_publish_audio=self.publish_audio, is_publish_video=False,
                                             audio_publish_type=AudioPublishType.AUDIO_PUBLISH_TYPE_PCM,
                                             video_publish_type=VideoPublishType.VIDEO_PUBLISH_TYPE_NONE)
        self.conn = self.service.create_rtc_connection(conn_config, publish)
        self.conn.register_observer(self._observers[0])
        self.conn.register_local_user_observer(self._observers[1])
        self.conn.get_local_user().set_playback_audio_frame_before_mixing_parameters(1, RATE)
        self.conn.register_audio_frame_observer(self._observers[3], 0, None)
        self.conn.register_video_frame_observer(self._observers[2])
        for worker in self.workers:
            worker.start()
        ret = self.conn.connect(token or "", channel, str(uid))
        if ret != 0:
            raise RuntimeError(f"agora connect returned {ret}")
        if not self.connected.wait(timeout):
            raise RuntimeError("agora connect timeout")
        if self.publish_audio:
            self.conn.publish_audio()
            self.sender.start()

    def audio_sink(self, pcm: bytes):
        """Called by the pacer thread with one 10 ms 48 kHz mono frame; never blocks it. The SDK's push call can
        block while the SDK itself stalls (observed up to 1.6 s, together with a gap in its received audio), so a
        sender thread makes the push; its delay behind the pacer clock is recorded (stats().audio_push_lag_ms)."""
        if self.conn is not None and self.publish_audio:
            self.outbox.put((time.monotonic_ns(), pcm))

    def _send(self):
        while True:
            item = self.outbox.get()
            if item is None:
                return
            queued, pcm = item
            conn = self.conn
            if conn is None:
                continue
            # the SDK wraps the buffer with ctypes from_buffer, which needs a writable bytearray
            if conn.push_audio_pcm_data(bytearray(pcm), RATE, 1) != 0:
                self.push_errors += 1
            self.push_lag_ms.append((time.monotonic_ns() - queued) / 1e6)

    def stats(self) -> dict:
        def summary(values):
            lag = sorted(values)
            pick = (lambda q: round(lag[min(len(lag) - 1, int(q * len(lag)))], 2)) if lag else (lambda q: None)
            return {"p50": pick(0.5), "p99": pick(0.99), "max": pick(1.0), "items": len(lag)}
        return {"worker_lag_ms": {k: summary(v) for k, v in self.lag_ms.items()}, "dropped": self.dropped,
                "audio_push_errors": self.push_errors, "audio_push_lag_ms": summary(self.push_lag_ms),
                "audio_push_late_100ms": sum(x > 100 for x in self.push_lag_ms)}

    def close(self):
        if self.sender.is_alive():
            self.outbox.put(None)
            self.sender.join(5)
        try:
            if self.conn is not None:
                if self.publish_audio:
                    self.conn.unpublish_audio()
                self.conn.disconnect()
                self.conn.release()
        finally:
            self.stop.set()
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(5)
            if self.service is not None:
                self.service.release()
            self.conn = self.service = None
