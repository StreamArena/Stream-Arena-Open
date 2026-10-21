"""Session harness, model interface and run contract, on the synthetic benchmark set (tests/synthetic.py).

Sessions run for the full case duration (15 s), so they are marked slow; the unit tests run without a session.
"""
import json
import shutil
import socket
import threading
import time

import pytest

from streameval.models import registry
from streameval.models.base import ActionEvent, ReferenceEvent, ReferenceSlot, SessionInit, TextEvent
from streameval.models.server import _event
from streameval.models.mock import MockModel
from streameval.runtime import batch, protocol
from streameval.runtime.contract import validate_run
from streameval.util import read_json, write_json

import synthetic

FAILING = '''
from streameval.models.base import StreamingModel

class NeverReady(StreamingModel):
    def open(self, init, sink):
        sink.fatal("cannot start")
    def send_event(self, event):
        pass
    def close(self):
        return True
'''


@pytest.fixture(scope="session")
def bench(tmp_path_factory):
    return synthetic.build(tmp_path_factory.mktemp("synthetic"))


@pytest.fixture(scope="session")
def mock_registry(tmp_path_factory):
    root = tmp_path_factory.mktemp("registry")
    (root / "mock").mkdir()
    write_json(root / "mock/system.json", {
        "system_id": "mock", "model_id": "mock", "name": "Mock", "provider": "test", "system_kind": "native",
        "tracks": ["digital_human_audio", "avatar_performance", "general_interaction", "instruction_edit"],
        "portrait_policy": "benchmark_portrait", "python": "streameval.models.mock:MockModel",
        "kwargs": {"echo_audio": True}, "unsupported_ops": ["camera_control"]})
    (root / "broken").mkdir()
    (root / "broken/model.py").write_text(FAILING)
    write_json(root / "broken/system.json", {
        "system_id": "broken", "model_id": "broken", "name": "Broken", "provider": "test", "system_kind": "native",
        "tracks": ["general_interaction"], "portrait_policy": None, "python": "model.py:NeverReady"})
    return root


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- no session ----------------------------------------------------------------------------------------

def test_wire_messages():
    init = SessionInit("instruction_edit", "c", 320, 240, 25.0, 7, {"width": 512}, image=b"\x89PNG\r\n\x1a\nxx")
    msg = init.message()
    assert list(msg)[:4] == ["type", "protocol", "track", "case_id"] and msg["image_mime"] == "image/png"
    assert TextEvent("e1", "text_control", "hi").message() == {"type": "text", "event_id": "e1", "text": "hi",
                                                              "kind": "text_control"}
    ref = ReferenceEvent("e2", "patch", (ReferenceSlot("clothing", b"\xff\xd8x", "top", "a"), ReferenceSlot("hair")),
                         scope="keep the rest").message()
    assert ref["slots"][0]["image_mime"] == "image/jpeg" and ref["slots"][1] == {"slot": "hair", "image_b64": None,
                                                                                 "scope": None}
    assert ref["scope"] == "keep the rest" and "text" not in ref
    act = ActionEvent("e3", "move_forward", "start", 10000)
    assert act.message() == {"type": "control", "event_id": "e3", "action": "move_forward",
                             "params": {"phase": "start", "duration_ms": 10000}}
    assert _event(act.message()) == act
    tag, pts, payload = protocol.unpack(protocol.pack(protocol.SOURCE_FRAME, 123, b"abc"))
    assert (tag, pts, payload) == (protocol.SOURCE_FRAME, 123, b"abc")


def test_registry_entries(mock_registry):
    table = registry.systems([mock_registry])
    assert {"mock_audio", "mock_performance", "mock_t2v", "mock_instruction", "broken_t2v"} <= set(table)
    config = registry.system_config("mock_t2v", table["mock_t2v"])
    assert config["ops"] == ["user_prompt"] and config["sampling"]["frame_sample_ms"] == 250
    # structured actions are opt-in: only an entry listing the op receives them
    assert "action" not in registry.ops({}, "general_interaction")
    assert registry.ops({"ops": ["user_prompt", "action"]}, "general_interaction") == ["user_prompt", "action"]
    with pytest.raises(ValueError):
        registry.validate({"system_id": "x", "model_id": "x", "name": "x", "provider": "x", "system_kind": "native",
                           "tracks": ["instruction_edit"], "portrait_policy": None})   # no python/server/url
    sid, spec = registry.adhoc("reference_edit", url="ws://127.0.0.1:1/session")
    assert sid == "adhoc_reference" and spec["group"] == "reference_edit"


def test_admission_flags_unsupported_events(bench, mock_registry):
    spec = registry.spec_of("mock_t2v", [mock_registry])
    adm = batch.admission("mock_t2v", spec, bench)
    assert [r["case_id"] for r in adm["cases"]] == [synthetic.CASE["general_interaction"]]
    assert adm["cases"][0]["unsupported_events"][0]["operation"] == "camera_control"


def test_rerun_rule_states(tmp_path):
    order = [{"job_id": "a.r01"}, {"job_id": "b.r01"}, {"job_id": "c.r01"}]
    write_json(tmp_path / "a.r01/result.json", {"scheduler": {"started_at_ns": None}, "status": "failed"})
    write_json(tmp_path / "b.r01/result.json", {"scheduler": {"started_at_ns": 5}, "status": "failed"})
    assert [p["attempt"] for p in batch.pending_attempts(tmp_path, order, False)] == ["c.r01"]
    assert [p["attempt"] for p in batch.pending_attempts(tmp_path, order, True)] == ["a.r01.retry1"]
    # Without a final result, an interrupted attempt stays unknown and cannot be retried.
    (tmp_path / "c.r01").mkdir()
    write_json(tmp_path / "c.r01/runner.json", {"pid": 1})
    assert batch.started(tmp_path / "c.r01") is None
    assert [p["attempt"] for p in batch.pending_attempts(tmp_path, order, True)] == ["a.r01.retry1"]


def test_non_starts_are_retried(bench, mock_registry, tmp_path):
    spec = registry.spec_of("broken_t2v", [mock_registry])
    report = batch.run_batch("broken_t2v", spec, bench, tmp_path, population="development", repetitions=1,
                             registry=(mock_registry,), stagger_s=0, log=lambda line: None)
    folder = tmp_path / "broken_t2v/development"
    job = synthetic.CASE["general_interaction"] + ".r01"
    assert sorted(p.name for p in folder.iterdir() if p.is_dir()) == [job] + [f"{job}.retry{k}" for k in (1, 2, 3)]
    assert report["not_started"] == 1 and report["completed"] == 0
    result = read_json(folder / job / "result.json")
    assert result["failure"]["type"] == "SessionEnded" and result["scheduler"]["started_at_ns"] is None
    assert validate_run(folder / job, bench.cases[synthetic.CASE["general_interaction"]])["ok"]


# ---- sessions -------------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def edit_attempt(bench, mock_registry, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs") / "edit"
    spec = registry.spec_of("mock_instruction", [mock_registry])
    batch.run_attempt("mock_instruction", spec, bench, synthetic.CASE["instruction_edit"], out, smoke=True)
    return out


@pytest.mark.slow
def test_in_process_edit_session(bench, edit_attempt):
    case = bench.cases[synthetic.CASE["instruction_edit"]]
    report = validate_run(edit_attempt, case)
    assert report["ok"], report["errors"]
    assert report["counts"]["frame_samples"] >= 55 and report["counts"]["source_frames"] > 300
    result = read_json(edit_attempt / "result.json")
    assert result["activity_execution_complete"] and result["termination_confirmed"]
    assert all(d["ack"]["status"] == "applied" for d in result["dispatch"].values())


@pytest.mark.slow
def test_contract_catches_tampering(bench, edit_attempt, tmp_path):
    case = bench.cases[synthetic.CASE["instruction_edit"]]
    copy = tmp_path / "copy"
    shutil.copytree(edit_attempt, copy)
    (copy / "browser/frames/s_000010.jpg").write_bytes(b"not the frame")
    assert any("s_000010.jpg sha256 mismatch" in e for e in validate_run(copy, case)["errors"])
    other = {**case, "duration_ms": 30000}
    assert any("session_sha256" in e for e in validate_run(edit_attempt, other)["errors"])


@pytest.mark.slow
def test_websocket_reference_session(bench, tmp_path):
    port = _free_port()
    from streameval.models.server import serve_model
    threading.Thread(target=serve_model, args=(lambda: MockModel(unsupported=("reference:reset",)), "127.0.0.1",
                                               port, "mock"), daemon=True).start()
    registry.wait_health(port, 10)
    sid, spec = registry.adhoc("reference_edit", url=f"ws://127.0.0.1:{port}/session")
    case_id = synthetic.CASE["reference_edit"]
    result = batch.run_attempt(sid, spec, bench, case_id, tmp_path / "ref", smoke=True)
    assert validate_run(tmp_path / "ref", bench.cases[case_id])["ok"]
    assert result["provider"]["initial_reference_ack"]["status"] == "applied"
    assert result["dispatch"]["e000"]["status"] == "sent"
    assert result["dispatch"]["e001"]["status"] == "unsupported"     # acked unsupported by the model
    assert result["termination_confirmed"]


@pytest.mark.slow
def test_avatar_performance_session(bench, mock_registry, tmp_path):
    case_id = synthetic.CASE["avatar_performance"]
    spec = registry.spec_of("mock_performance", [mock_registry])
    started = time.monotonic()
    result = batch.run_attempt("mock_performance", spec, bench, case_id, tmp_path / "ap", smoke=True)
    assert time.monotonic() - started >= 15 + 4      # case duration plus the avatar idle-readiness rule
    report = validate_run(tmp_path / "ap", bench.cases[case_id])
    assert report["ok"], report["errors"]
    assert report["counts"]["audio_seconds"] > 15
    assert result["readiness"]["idle"] is True
    assert any(s["event"] == "speech_onset" for s in result["speech_events"])   # the echoed supplied speech
    assert result["dispatch"]["e001"]["transport"] == "audio_in"
    assert json.loads((tmp_path / "ap/binding.json").read_text())["portrait_used_as_avatar_input"]


@pytest.mark.slow
def test_digital_human_session(bench, mock_registry, tmp_path):
    case_id = synthetic.CASE["digital_human_audio"]
    spec = registry.spec_of("mock_audio", [mock_registry])
    result = batch.run_attempt("mock_audio", spec, bench, case_id, tmp_path / "dh", smoke=True)
    assert validate_run(tmp_path / "dh", bench.cases[case_id])["ok"]
    assert result["status"] == "completed" and result["dispatch"]["u1"]["transport"] == "audio_in"
    assert any(s["event"] == "speech_onset" for s in result["speech_events"])


@pytest.mark.slow
def test_batch_freezes_its_plan(bench, mock_registry, tmp_path):
    def run(spec):
        return batch.run_batch("mock_t2v", spec, bench, tmp_path, population="development", repetitions=1,
                               registry=(mock_registry,), stagger_s=0, log=lambda line: None)

    spec = registry.spec_of("mock_t2v", [mock_registry])
    assert {k: run(spec)[k] for k in ("jobs", "completed")} == {"jobs": 1, "completed": 1}
    folder = tmp_path / "mock_t2v/development"
    job = folder / (synthetic.CASE["general_interaction"] + ".r01")
    assert read_json(job / "result.json")["dispatch"]["e001"]["status"] == "unsupported"   # camera_control
    assert validate_run(job, bench.cases[synthetic.CASE["general_interaction"]])["counts"]["unsupported"] == 1
    assert run(spec)["completed"] == 1 and not (folder / (job.name + ".retry1")).exists()   # nothing left to run
    with pytest.raises(SystemExit, match="was frozen with a different"):
        run({**spec, "entry": {**spec["entry"], "seed": 7}})        # a changed system needs a new --out
