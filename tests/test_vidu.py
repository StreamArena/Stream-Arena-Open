"""Vidu S2 backend: declarations, the edit mapping and the refusals, all without a network request."""
import sys
from pathlib import Path

import pytest

from streameval.benchmark.dataset import Benchmark
from streameval.models import registry

SYSTEMS = ("vidu_s2_avatar_audio", "vidu_s2_avatar_text", "vidu_s2_avatar_performance_text", "vidu_s2_editing_all",
           "vidu_s2_editing_t2i")
TEMPLATE = ("Change only the scarf draped over her shoulders to: a red silk scarf. Keep everything else, the current "
            "action and the camera unchanged.")


@pytest.fixture
def no_keys(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)   # no ./.env
    for key in ("VIDU_S2_API_KEY", "VIDU_S2_API_BASE", "DASHSCOPE_API_KEY"):
        monkeypatch.setenv(key, "")


def spec(system_id):
    return registry.systems()[system_id]


def backend(system_id):
    return registry.account(spec(system_id)["entry"])


def module(system_id):
    return sys.modules[type(backend(system_id)).__module__]


def test_registered_systems_and_configuration(no_keys):
    table = registry.systems()
    assert set(SYSTEMS) <= set(table)
    for system_id in SYSTEMS:
        config = registry.system_config(system_id, table[system_id])
        assert config["client"]["transport"].startswith("AliRTC Web SDK 7.1.9")
        assert config["client"]["sdk"]["sha256"].startswith("8100338351ce")
        assert config["sampling"]["frame_sample_ms"] == (500 if system_id.startswith("vidu_s2_avatar_") and
                                                         "performance" not in system_id else 250)
        assert config["api_origin"] == "https://api.vidu.cn"
    t2i = registry.system_config("vidu_s2_editing_t2i", table["vidu_s2_editing_t2i"])
    assert t2i["system_kind"] == "composition" and [c["model"] for c in t2i["components"]] == ["qwen-image", "vidu-s2"]
    assert registry.system_config("vidu_s2_editing_all", table["vidu_s2_editing_all"])["components"] == []
    perf = registry.system_config("vidu_s2_avatar_performance_text", table["vidu_s2_avatar_performance_text"])
    assert perf["avatar"]["input"].startswith("WS type 99 'LINE")


def test_missing_credentials_and_no_estimate(no_keys):
    assert backend("vidu_s2_avatar_audio").missing_credentials() == ["VIDU_S2_API_KEY"]
    assert backend("vidu_s2_editing_t2i").missing_credentials() == ["VIDU_S2_API_KEY", "DASHSCOPE_API_KEY"]
    assert backend("vidu_s2_editing_all").estimate({"duration_ms": 15000}) is None


def test_unofficial_origin_is_refused(no_keys, monkeypatch):
    monkeypatch.setenv("VIDU_S2_API_BASE", "https://example.com")
    with pytest.raises(ValueError, match="VIDU_S2_API_BASE"):
        backend("vidu_s2_editing_all").base()
    s2 = module("vidu_s2_editing_all").s2
    with pytest.raises(ValueError, match="official"):
        s2.S2HTTP(base="https://example.com", key="vda_x", http=object(), ws_factory=object())
    with pytest.raises(ValueError, match="malformed"):
        s2.S2HTTP(base="https://api.vidu.cn", key="sk-x", http=object(), ws_factory=object())


def test_run_is_refused_before_any_request(no_keys, tmp_path):
    bench = Benchmark(Path(__file__).resolve().parents[1] / "benchmark/v1/development")
    case = next(c for c in bench.cases.values() if c["input_profile"] == "conversation_text")
    mod = module("vidu_s2_avatar_text")
    job = mod.ViduJob(backend("vidu_s2_avatar_text"), "vidu_s2_avatar_text", spec("vidu_s2_avatar_text"), case,
                      bench.case_assets(case["case_id"]), tmp_path / "run", repeat=1, oracle=None, smoke=True,
                      admission=None, verify=False)
    with pytest.raises(RuntimeError, match="VIDU_S2_API_KEY"):
        job.preflight()


def test_dispositions(no_keys):
    bench = Benchmark(Path(__file__).resolve().parents[1] / "benchmark/v1/development")
    t2i, ref = backend("vidu_s2_editing_t2i"), backend("vidu_s2_editing_all")
    for case in bench.cases.values():
        if case["group"] == "instruction_edit":
            rows = t2i.event_dispositions("vidu_s2_editing_t2i", spec("vidu_s2_editing_t2i"), case)
            # the development roster's free-form instructions are outside the six v1 templates
            assert rows and all(r["reason"] == "instruction_outside_v1_templates" for r in rows)
        elif case["group"] == "reference_edit":
            rows = ref.event_dispositions("vidu_s2_editing_all", spec("vidu_s2_editing_all"), case)
            assert all(r["supported"] for r in rows)
    case = {"events": [{"event_id": "e000", "payload": {"type": "text_control", "text": TEMPLATE}},
                       {"event_id": "e001", "payload": {"type": "text_control", "text": "make it blue"}}]}
    rows = t2i.event_dispositions("vidu_s2_editing_t2i", spec("vidu_s2_editing_t2i"), case)
    assert [r["supported"] for r in rows] == [True, False]


def test_edit_mapping():
    mapping = module("vidu_s2_editing_all").mapping
    assert mapping.garment("the jacket worn by the man") and mapping.garment("a scarf draped over her shoulders")
    assert not mapping.garment("the mug on the desk")
    parsed = mapping.parse_instruction(TEMPLATE)
    assert parsed == {"op": "set", "target": "the scarf draped over her shoulders", "look": "a red silk scarf"}
    assert mapping.parse_instruction("Cancel the edit: restore the mug to its current original appearance in the "
                                     "live source, without replaying earlier frames.")["op"] == "restore"
    with pytest.raises(ValueError, match="v1 templates"):
        mapping.parse_instruction("Turn the mug blue.")

    state = mapping.EditState()
    mapping.apply_instruction(state, parsed)
    mapping.apply_instruction(state, {"op": "patch", "target": "the mug", "look": "blue"})
    assert state.newest() == ("the mug", "blue", False)
    mapping.apply_instruction(state, {"op": "restore", "target": "the mug"})
    target, _, worn = state.newest()
    assert worn and mapping.editing_type(worn) == "virtual_tryon"

    state = mapping.EditState()
    mapping.apply_reference(state, {"operation": "set", "references": [
        {"target_slot": "a", "asset_id": "img.a", "scope": "the coat worn by the woman"}]})
    mapping.apply_reference(state, {"operation": "patch", "references": [
        {"target_slot": "b", "asset_id": "img.b", "scope": "the lamp"}]})
    assert state.newest() == ("b", "img.b", False) and state.basis() == "newest_active_target:b"
    mapping.apply_reference(state, {"operation": "reset"})
    assert state.newest() is None and state.basis() == mapping.NO_TARGET
