"""Vendor adapters without a network: the Account facts, the RTC avatar base (providers/rtc.py) on a fake room, and the
dry run of every registered vendor system. Nothing here starts a vendor session."""
import threading

import numpy as np
import pytest

from streameval.models import registry
from streameval.models.base import SessionInit, TextEvent
from streameval.providers.rtc import RtcAvatar, Upsampler

VENDOR_ENV = ("RUNWAYML_API_SECRET", "LIVEAVATAR_API_KEY", "TAVUS_API_KEY", "TAVUS_PAL_ID", "XMAX_API_KEY",
              "VIDU_S2_API_KEY", "DASHSCOPE_API_KEY")


@pytest.fixture
def no_keys(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)   # no ./.env
    for key in VENDOR_ENV:
        monkeypatch.setenv(key, "")


def test_upsampler_is_continuous():
    up = Upsampler()
    ramp = np.arange(320, dtype=np.int16) * 10
    out = np.frombuffer(up(ramp[:160].tobytes()) + up(ramp[160:].tobytes()), dtype=np.int16)
    assert len(out) == 960
    # one input sample of delay, then a straight line through every input sample (no seam at the chunk boundary)
    assert out[3] == 0 and out[6] == 10 and np.all(np.diff(out[3:]) >= 3) and np.all(np.diff(out[3:]) <= 4)
    assert np.all(np.frombuffer(Upsampler()(np.full(160, 7, np.int16).tobytes()), np.int16)[3:] == 7)


class Sink:
    def __init__(self):
        self.ready_meta, self.acks, self.fatal_msg, self.audio_in = None, {}, None, []
        self.ready_evt = threading.Event()

    def ready(self, **meta):
        self.ready_meta = meta
        self.ready_evt.set()

    def ack(self, event_id, status="applied", detail=None):
        self.acks[event_id] = status

    def fatal(self, message):
        self.fatal_msg = message

    def video_frame(self, *a, **k): pass
    def video_rgb(self, *a, **k): pass
    def audio(self, *a, **k): pass


class FakeRoom:
    def __init__(self):
        self.audio_ready = threading.Event()
        self.audio_ready.set()
        self.sent = []

    def audio_sink(self, pcm):
        self.sent.append(len(pcm))


class FakeAvatar(RtcAvatar):
    credentials_env = {"api_key": "FAKE_AVATAR_KEY"}
    text_transport = "fake data channel"

    def create(self, init):
        self.record["status"] = "created"
        return {"session_id": "s1", "seconds": init.duration_ms // 1000}

    def join(self):
        self.room = FakeRoom()

    def send_text(self, event_id, text):
        return {"chars": len(text)}

    def end(self):
        self.note("ended")
        return True


def init(track="digital_human_audio"):
    return SessionInit(track=track, case_id="c1", width=512, height=512, fps=25, seed=1, output={}, persona="p",
                       duration_ms=15000)


def test_rtc_avatar_lifecycle(monkeypatch):
    monkeypatch.setenv("FAKE_AVATAR_KEY", "sentinel-fake-key")
    model, sink = FakeAvatar(), Sink()
    model.open(init(), sink)
    assert sink.ready_evt.wait(5) and sink.ready_meta["session"] == {"session_id": "s1", "seconds": 15}
    model.push_audio(0, b"\0\0" * 160)
    assert model.room.sent == [960]
    model.send_event(TextEvent("e1", "user_prompt", "hello"))
    model.send_event(TextEvent("e2", "text_control", "zoom"))
    assert sink.acks == {"e1": "applied", "e2": "unsupported"}
    model.ended("avatar_participant_left")
    assert sink.fatal_msg == "provider ended the session: avatar_participant_left"
    assert model.close() is True
    model.ended("room disconnected")   # while closing: not a failure
    assert sink.fatal_msg == "provider ended the session: avatar_participant_left"
    receipt = model.receipt()
    assert receipt["termination_confirmed"] and [n["kind"] for n in receipt["notes"]][:2] == ["session_created",
                                                                                             "media_joined"]
    assert "sentinel" not in str(receipt)


def test_rtc_avatar_text_track_has_no_microphone(monkeypatch):
    monkeypatch.setenv("FAKE_AVATAR_KEY", "sentinel-fake-key")
    model, sink = FakeAvatar(), Sink()
    model.open(init("digital_human_text"), sink)
    assert sink.ready_evt.wait(5)
    model.push_audio(0, b"\0\0" * 160)
    assert model.room.sent == []
    model.close()


def test_rtc_avatar_refuses_without_credentials(no_keys):
    with pytest.raises(RuntimeError, match="FAKE_AVATAR_KEY"):
        FakeAvatar().open(init(), Sink())


def test_vendor_accounts_and_estimates(no_keys):
    table = registry.systems()
    case = {"duration_ms": 15000}
    expected = {"runway_characters_audio": (["RUNWAYML_API_SECRET"], 0.22, 300),
                "liveavatar_full_audio": (["LIVEAVATAR_API_KEY"], 4, 1200),
                "tavus_cvi_full_audio": (["TAVUS_API_KEY", "TAVUS_PAL_ID"], 60 / 60 * 0.37, None)}
    for system_id, (missing, cost, cap) in expected.items():
        vendor = registry.account(table[system_id]["entry"])
        assert vendor.missing_credentials() == missing
        assert vendor.estimate(case) == pytest.approx(cost)
        assert vendor.session_cap_s == cap
        config = registry.system_config(system_id, table[system_id])
        assert config["client"]["transport"] == vendor.client_transport and config["session_cap_ms"] == (cap and cap * 1000)
    assert registry.account(table["runway_characters_audio"]["entry"]).exceeds_session_cap({"duration_ms": 295000},
                                                                                            "avatar")


def test_tavus_pal_has_no_account_ids():
    import json
    from pathlib import Path
    pal = json.loads((Path(registry.REPO) / "models/providers/tavus/pal_v1.json").read_text())
    assert "pal_id" not in pal and "persona_id" not in pal["persona"]
    assert set(pal["pinned_fields"]) <= set(pal["persona"])


def test_credentials_never_enter_the_configuration(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for i, key in enumerate(VENDOR_ENV):
        monkeypatch.setenv(key, f"sentinel-secret-{i}")
    for system_id, spec in registry.systems().items():
        if registry.account(spec["entry"]) is not None:
            assert "sentinel-secret" not in str(registry.system_config(system_id, spec)), system_id
