"""Offline scoring of one sampling job from its judged observations (observations.json) against a frozen
preparation (streameval.scoring.plan): one score report per job, the input of the leaderboard.

This verifies snapshot and text-evidence integrity, not reviewer reliability,
provider execution, admission or media semantics. It never opens media assets,
calls a model, or converts a preparation plan into an official result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from ..util import digest
from .accounting import _artifact, _index
from .calculation import summarize_units

PREPARATION_VERSION = "streameval.evaluation-preparation/1"
REPORT_VERSION = "streameval.prepared-run-score/1"
# the latency calculation name frozen in every v1 metric plan (now streameval.scoring.accounting.account_session)
LATENCY_CALCULATION = "arena_runtime.metrics.account_session"
HASH = re.compile(r"[a-f0-9]{64}\Z")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Nonfinite JSON number")


def _read_text_json(path, *, lines=False):
    path = Path(path).resolve()
    if path.suffix.lower() != (".jsonl" if lines else ".json"):
        raise ValueError("Scoring reads only explicit JSON/JSONL text artifacts, never media")
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    parse = lambda value: json.loads(value, object_pairs_hook=_object, parse_constant=_invalid_constant)
    value = [parse(line) for line in text.splitlines() if line.strip()] if lines else parse(text)
    if not isinstance(value, (dict, list)):
        raise ValueError("Evidence and preparation files must contain JSON objects or arrays")
    return value, hashlib.sha256(raw).hexdigest()


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Artifact references must be relative paths inside their controlled root")
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Artifact is missing or escapes its controlled root")
    return target


def load_preparation(directory):
    """Verify the frozen snapshot without reopening a later checkout's sources.

    Old revisions remain valid study snapshots. The manifest's source hashes
    are recorded bindings, not a request to read today's files at those paths.
    Hash consistency alone cannot authenticate the manifest's author.
    """
    root = Path(directory).resolve()
    manifest, manifest_sha = _read_text_json(_inside(root, "manifest.json"))
    plans, _ = _read_text_json(_inside(root, "metric_plans.jsonl"), lines=True)
    jobs, _ = _read_text_json(_inside(root, "sampling_jobs.jsonl"), lines=True)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != PREPARATION_VERSION
            or manifest.get("official_evaluation") is not False or manifest.get("formal_scores") != 0
            or manifest.get("plans_sha256") != digest(plans) or manifest.get("jobs_sha256") != digest(jobs)):
        raise ValueError("Preparation manifest or frozen plans/jobs hashes disagree")
    sources = manifest.get("source_sha256")
    if (not isinstance(sources, dict) or not sources or
            any(not isinstance(path, str) or not path or not isinstance(value, str) or not HASH.fullmatch(value)
                for path, value in sources.items())):
        raise ValueError("Preparation requires frozen source-file hash bindings")
    repetitions = manifest.get("repetitions")
    if (type(repetitions) is not int or repetitions < 1
            or type(manifest.get("conditions")) is not int or manifest["conditions"] != len(plans)
            or manifest.get("planned_jobs_per_system_before_admission_filter") != len(jobs)):
        raise ValueError("Preparation condition/repetition counts disagree")
    indexed_plans = _index(plans, "case_id", "metric plans")
    indexed_jobs = _index(jobs, "job_id", "sampling jobs")
    spec_hashes = set()
    for plan in plans:
        if (plan.get("official_evaluation") is not False or not isinstance(plan.get("metrics"), dict)
                or any(not isinstance(plan.get(key), str) or not HASH.fullmatch(plan[key])
                       for key in ("session_sha256", "oracle_sha256", "metric_spec_sha256"))):
            raise ValueError("Invalid frozen metric plan or its hash bindings")
        spec_hashes.add(plan["metric_spec_sha256"])
        for name, metric in plan["metrics"].items():
            if name == "latency":
                if (not isinstance(metric, dict) or metric.get("calculation") != LATENCY_CALCULATION
                        or type(metric.get("denominator")) is not int or metric["denominator"] < 0):
                    raise ValueError("Latency must use its separate frozen opportunity accounting")
                continue
            if not isinstance(metric, dict) or not isinstance(metric.get("units"), list):
                raise ValueError("A metric requires frozen observation units")
            empty = summarize_units(metric["units"], [])
            if (metric.get("denominator") != empty["denominator_units"]
                    or metric.get("not_applicable") != empty["not_applicable_units"]):
                raise ValueError("Frozen metric denominators disagree with their unit roster")
    if len(spec_hashes) != 1:
        raise ValueError("Do not mix metric specifications within one preparation")
    expected_ids = {f"{case_id}.r{repeat:02d}" for case_id in indexed_plans for repeat in range(1, repetitions + 1)}
    if set(indexed_jobs) != expected_ids:
        raise ValueError("The frozen sampling roster omits or adds expected repeated jobs")
    for job in jobs:
        plan = indexed_plans.get(job.get("case_id"))
        repeat = job.get("repeat")
        if (plan is None or type(repeat) is not int or not 1 <= repeat <= repetitions
                or job["job_id"] != f"{job['case_id']}.r{repeat:02d}"
                or any(job.get(key) != plan.get(key) for key in
                       ("group", "input_profile", "split", "session_sha256", "oracle_sha256"))):
            raise ValueError("A sampling job does not bind its exact frozen condition")
    return manifest, indexed_plans, indexed_jobs, manifest_sha


def conjunctive_edit_success(plan, metric_decisions):
    """scoring/2: an edit event passes only if its preservation unit also passes.

    A preservation fail turns the edit event into a fail; an undetermined
    preservation keeps a passing edit event undetermined. Returns None when
    the plan does not declare the conjunction (v0.2 preparations)."""
    metric = plan["metrics"].get("edit_success")
    if not metric or not any(u.get("requires_units") for u in metric["units"]):
        return None
    edit = {d["unit_id"]: d for d in metric_decisions.get("edit_success", [])}
    keep = {d["unit_id"]: d for d in metric_decisions.get("preservation", [])}
    combined = []
    for unit in metric["units"]:
        if not unit["applicable"] or unit["unit_id"] not in edit:
            continue
        decision = dict(edit[unit["unit_id"]])
        guards = [keep.get(u, {"status": "undetermined"})["status"] for u in unit.get("requires_units", [])]
        if decision["status"] == "pass":
            if "fail" in guards or "output_missing" in guards:
                decision["status"] = "fail"
            elif "undetermined" in guards:
                decision["status"] = "undetermined"
        combined.append(decision)
    return summarize_units(metric["units"], combined)


def strict_session(decisions, units):
    """Secondary report: 100 if every planned turn passes, 0 if any fails."""
    statuses = {d["unit_id"]: d["status"] for d in decisions}
    planned = [statuses.get(u["unit_id"], "undetermined") for u in units if u["applicable"]]
    if any(s in {"fail", "output_missing"} for s in planned):
        return {"status": "complete", "score": 0, "lower": 0, "upper": 0}
    if all(s == "pass" for s in planned):
        return {"status": "complete", "score": 100, "lower": 100, "upper": 100}
    return {"status": "undetermined", "score": None, "lower": 0, "upper": 100}


def score_prepared_run(preparation, observations, *, evidence_root):
    manifest, plans, jobs, manifest_sha = load_preparation(preparation)
    bundle, observations_sha = _read_text_json(observations)
    required = {"job_id", "system_id", "system_config_sha256", "metric_decisions"}
    if (not isinstance(bundle, dict) or set(bundle) != required
            or not isinstance(bundle["job_id"], str) or bundle["job_id"] not in jobs
            or not isinstance(bundle["system_id"], str) or not bundle["system_id"].strip()
            or not isinstance(bundle["system_config_sha256"], str) or not HASH.fullmatch(bundle["system_config_sha256"])
            or not isinstance(bundle["metric_decisions"], dict)):
        raise ValueError("Observations require one frozen job, explicit system/configuration and metric decisions")
    job = jobs[bundle["job_id"]]
    plan = plans[job["case_id"]]
    if job.get("system_id") is not None and job["system_id"] != bundle["system_id"]:
        raise ValueError("Observed system does not match the system-bound frozen job")
    if job.get("system_config_sha256") is not None and job["system_config_sha256"] != bundle["system_config_sha256"]:
        raise ValueError("Observed system configuration does not match the frozen job")
    metric_decisions = bundle["metric_decisions"]
    if "latency" in metric_decisions:
        raise ValueError("Latency requires the opportunity accounting and independent presentation receipts")
    if metric_decisions.keys() - (plan["metrics"].keys() - {"latency"}):
        raise ValueError("Observations include a metric outside this frozen condition")
    controlled_root = Path(evidence_root).resolve()
    if not controlled_root.is_dir():
        raise ValueError("The explicit evidence root must exist")
    artifacts, scores = {}, {}
    for name, metric in plan["metrics"].items():
        if name == "latency":
            continue
        decisions = metric_decisions.get(name, [])
        if not isinstance(decisions, list):
            raise ValueError("Metric decisions must be an array")
        for decision in decisions:
            if (not isinstance(decision, dict) or not {"unit_id", "status", "evidence"} <= decision.keys()
                    or decision.keys() - {"unit_id", "status", "evidence", "value", "lower", "upper"}):
                raise ValueError("Unexpected metric decision fields")
            evidence = decision["evidence"]
            _artifact(evidence)
            path = _inside(controlled_root, evidence["artifact_ref"])
            if path.suffix.lower() not in {".json", ".jsonl"}:
                raise ValueError("Decision evidence must be a reviewed JSON/JSONL artifact, never raw media")
            if path not in artifacts:
                _, artifact_sha = _read_text_json(path, lines=path.suffix.lower() == ".jsonl")
                artifacts[path] = {"artifact_ref": str(path.relative_to(controlled_root)), "sha256": artifact_sha}
            if artifacts[path]["sha256"] != evidence["sha256"]:
                raise ValueError("Decision evidence bytes do not match their declared SHA256")
        scores[name] = summarize_units(metric["units"], decisions)
    conjoined = conjunctive_edit_success(plan, metric_decisions)
    if conjoined is not None:
        scores["edit_success"] = conjoined
    if plan["metrics"].get("dialogue", {}).get("units", [{}])[0].get("unit_id", "").startswith("dialogue.") \
            and plan["metrics"]["dialogue"]["units"][0]["unit_id"] != "dialogue.session":
        scores["dialogue_session_strict"] = strict_session(metric_decisions.get("dialogue", []),
                                                           plan["metrics"]["dialogue"]["units"])
    return {
        "schema_version": REPORT_VERSION, "kind": "prepared_run_score_diagnostic",
        "job_id": job["job_id"], "case_id": job["case_id"], "repeat": job["repeat"],
        "system_id": bundle["system_id"], "system_config_sha256": bundle["system_config_sha256"],
        **{key: plan[key] for key in ("group", "input_profile", "split", "duration_ms", "source_family_id",
                                      "session_sha256", "oracle_sha256", "metric_spec_sha256")},
        "preparation_manifest_sha256": manifest_sha,
        "evaluation_policy": manifest.get("evaluation_policy"),
        "metric_plan_sha256": digest(plan), "sampling_job_sha256": digest(job),
        "observations_sha256": observations_sha, "frozen_source_sha256": manifest["source_sha256"],
        "source_binding_validation": "frozen_snapshot_integrity; current_checkout_not_reopened",
        "source_files_revalidated": False, "metric_spec_status": manifest.get("metric_spec_status"),
        "metrics": scores,
        "latency": {"status": "requires_separate_accounting", "calculation": LATENCY_CALCULATION,
                    "denominator": plan["metrics"].get("latency", {}).get("denominator"),
                    "reason": "requires_final_scheduler_ledger_and_independent_client_presentation_receipts"},
        "verified_evidence_artifacts": sorted(artifacts.values(), key=lambda item: item["artifact_ref"]),
        "run_execution_authenticated": False, "reviewer_reliability_validated": False,
        "benchmark_eligible": False, "official_evaluation": False, "formal_scores": 0,
        "network_requests": 0, "media_read": False, "media_decoded": False,
    }
