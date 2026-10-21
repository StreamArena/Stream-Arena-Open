"""The path an outside user takes: the template model in examples/my_model, and the development roster."""
import json
import socket

import pytest

from streameval import REPO
from streameval.benchmark.roster import Roster, from_preparation
from streameval.judge.units import attempts
from streameval.models import registry
from streameval.runtime import batch
from streameval.runtime.contract import validate_run
from streameval.scoring.plan import prepare_files, write_preparation
from streameval.util import read_json

import synthetic

DEV = REPO / "benchmark/v1/development"


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    return synthetic.build(tmp_path_factory.mktemp("synthetic"))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- no session ----------------------------------------------------------------------------------------

def test_examples_are_registered():
    table = registry.systems()
    assert {"my_model_instruction", "my_model_ws_instruction"} <= set(table)
    config = registry.system_config("my_model_instruction", table["my_model_instruction"])
    assert config["ops"] == ["text_control"] and config["source_push"]["max_side"] == 640
    assert config["client"]["transport"].startswith("in-process Python")


def test_development_roster_is_a_roster(tmp_path):
    summary, plans, jobs = prepare_files(DEV / "cases.jsonl", DEV / "oracles.jsonl", population="development",
                                         repetitions=3)
    write_preparation(tmp_path / "development", summary, plans, jobs)
    roster = Roster(tmp_path / "development")
    assert len(roster.sessions) == 120 and roster.split == "public_dev"
    assert {c["tier"] for c in roster.sessions.values()} == {"development"}
    assert roster.group_core_rosters == {}
    assert len(roster.plans("development")) == 120 and len(jobs) == 120 * 3
    counts = {}
    for c in roster.sessions.values():
        counts[c["group"]] = counts.get(c["group"], 0) + 1
    assert counts == {"instruction_edit": 20, "reference_edit": 20, "digital_human": 40, "general_interaction": 20,
                      "avatar_performance": 20}


def test_official_population_names_are_reserved(tmp_path):
    summary, plans, jobs = prepare_files(DEV / "cases.jsonl", DEV / "oracles.jsonl", population="core",
                                         repetitions=1)
    write_preparation(tmp_path / "core", summary, plans, jobs)
    with pytest.raises(ValueError, match="reserved"):
        from_preparation(tmp_path / "core")


def test_one_sample_per_system_and_job(tmp_path):
    """Two systems in one run root sharing a job id are judged separately."""
    job = "sa.v1.001.instruction_edit.r01"
    for system in ("sys_a", "sys_b"):
        attempt = tmp_path / system / "development" / job
        attempt.mkdir(parents=True)
        (attempt / "binding.json").write_text(json.dumps({"system_id": system, "job_id": job}))
        (attempt / "result.json").write_text(json.dumps({"scheduler": {"started_at_ns": 1}}))
    assert sorted(a.parent.parent.name for a in attempts(tmp_path, None, None)) == ["sys_a", "sys_b"]
    assert [a.parent.parent.name for a in attempts(tmp_path, "sys_b", None)] == ["sys_b"]


# ---- sessions -------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_template_model_session(bench, tmp_path):
    case_id = synthetic.CASE["instruction_edit"]
    spec = registry.spec_of("my_model_instruction")
    result = batch.run_attempt("my_model_instruction", spec, bench, case_id, tmp_path / "a", smoke=True)
    report = validate_run(tmp_path / "a", bench.cases[case_id])
    assert report["ok"], report["errors"]
    assert result["status"] == "completed" and result["termination_confirmed"]
    assert all(d["ack"]["status"] == "applied" for d in result["dispatch"].values())


@pytest.mark.slow
def test_template_server_launched(bench, tmp_path):
    case_id = synthetic.CASE["instruction_edit"]
    spec = registry.spec_of("my_model_ws_instruction")
    server = registry.LaunchedServer(spec["entry"], _free_port(), log=tmp_path / "server.log")
    try:
        result = batch.run_attempt("my_model_ws_instruction", spec, bench, case_id, tmp_path / "a", url=server.url,
                                   smoke=True)
    finally:
        server.stop()
    report = validate_run(tmp_path / "a", bench.cases[case_id])
    assert report["ok"], report["errors"]
    assert result["status"] == "completed"
    assert not any(d["status"] == "unsupported" for d in result["dispatch"].values())
    assert read_json(tmp_path / "a/binding.json")["system_config"]["registry_entry"]["model"] == "my_model"
