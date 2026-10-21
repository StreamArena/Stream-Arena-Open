"""Freeze a preparation: one metric plan per condition and the condition x repeat sampling jobs.

A metric plan fixes, before any output exists, every observation unit of every metric of the condition's track
(scoring_v2.json group_columns), its weight and whether it applies; the per-job score report and the
leaderboard aggregate only over these units. The output directory has exactly the format
streameval.scoring.report.load_preparation reads:

  manifest.json         streameval.evaluation-preparation/1
  metric_plans.jsonl    one frozen metric plan per condition
  sampling_jobs.jsonl   condition x repeat jobs, state planned_not_started

Preparations and their sample bindings stay in the local run directory.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .. import REPO
from ..evaluation import POLICY_PATH, SCORING_PATH, policy_binding
from ..util import digest, sha256_file, write_json, write_jsonl
from .accounting import opportunity_plan
from .report import LATENCY_CALCULATION, PREPARATION_VERSION

SPEC = SCORING_PATH


def metric_plan(session, oracle, spec):
    """Freeze eligibility and search windows before participant outputs exist."""
    accounting = opportunity_plan(session, oracle)
    group, duration = session["group"], session["duration_ms"]
    by_event = {e["event_id"]: e for e in session["events"]}
    metrics = {}
    clip_ms = spec["sampling"]["uniform_clip_ms"]
    clips = [[start, min(start + clip_ms, duration)] for start in range(0, duration, clip_ms)]
    for name in spec["group_columns"][group]:
        if name == "latency":
            metrics[name] = {"calculation": LATENCY_CALCULATION,
                             "denominator": accounting["latency_denominator"], "units": accounting["opportunities"]}
            continue
        kind = spec["metrics"][name]["kind"]
        units = []
        if name in {"visual_quality", "identity", "stability", "continuity"}:
            for n, window in enumerate(clips):
                units.append({"unit_id": f"{name}.clip{n:03d}", "kind": kind,
                              "weight": window[1] - window[0], "applicable": True,
                              "window_ms": window, "timeline": "actual_client_presentation_activity",
                              "context": "current_released_state_and_initial_identity; include_prior_context_for_reappearance"})
        elif name == "dialogue" and spec["metrics"]["dialogue"]["unit"] == "dialogue_turn_opportunity":
            # one unit per planned turn, equal weight; the strict whole-session pass is derived from the same
            # decisions for reporting
            for cp in oracle["checkpoints"]:
                units.append({"unit_id": f"dialogue.{cp['event_id']}", "event_id": cp["event_id"], "kind": "binary",
                              "weight": 1, "applicable": True, "release": by_event[cp["event_id"]]["release"],
                              "checkpoint_sha256": digest(cp), "response_policy": cp.get("response_policy"),
                              "rule": "expected_outcome_of_this_turn; wait_and_backchannel_are_outcomes"})
        elif name == "dialogue":
            units.append({"unit_id": "dialogue.session", "kind": "binary", "weight": 1, "applicable": True,
                          "checkpoint_ids": [c["event_id"] for c in oracle["checkpoints"]],
                          "rule": "all_critical_semantic_and_turn_coordination_checkpoints; preserve_wait_and_backchannel"})
        else:
            reference_active = bool(session["initialization"].get("reference_asset_ids"))
            for cp in oracle["checkpoints"]:
                event, eid = by_event[cp["event_id"]], cp["event_id"]
                speech = event["payload"]["type"] == "speech_audio"
                applicable, exclusion = True, None
                if name == "av_sync":
                    applicable = speech if group == "avatar_performance" else cp.get("response_policy") == "substantive_response"
                    exclusion = None if applicable else "no_new_speech_response_expected; audit_silence_and_backchannel_separately"
                elif name == "following" and group == "avatar_performance":
                    applicable = not speech
                    exclusion = None if applicable else "supplied_speech_is_not_visual_control"
                elif name == "reference":
                    if group == "reference_edit":
                        applicable = bool(cp.get("active_state"))
                        exclusion = None if applicable else "all_reference_slots_reset; evaluate_current_source_restore_in_edit_success"
                    else:
                        reference_active |= event["payload"]["type"] == "reference_update"
                        applicable = reference_active
                        exclusion = None if applicable else "no_visual_reference_in_current_condition"
                conjunctive = (name == "edit_success"
                               and spec["metrics"]["edit_success"].get("protection_is_conjunctive") is True)
                units.append({"unit_id": f"{name}.{eid}", "event_id": eid, "kind": kind,
                              **({"requires_units": [f"preservation.{eid}"],
                                  "rule": "target_state_and_all_declared_protected_content_must_hold"} if conjunctive else {}),
                              "weight": 1, "applicable": applicable, "exclusion_reason": exclusion,
                              "release": event["release"], "checkpoint_sha256": digest(cp),
                              "check_window_ms": cp.get("check_window_ms"),
                              "response_policy": cp.get("response_policy", "control_effect"),
                              "timing_origin": cp.get("timing_origin", "actual_input_release"),
                              "source_visibility": "bind_pre_output_source_visibility; output_occlusion_cannot_exclude_a_failure",
                              **({"subwindow_resolution": "actual_voiced_and_silent_intervals; equal_weight_per_required_speech_event",
                                  "alignment_calibration_required": True} if name == "av_sync" else {})})
        metrics[name] = {"denominator": sum(u["applicable"] for u in units),
                         "not_applicable": sum(not u["applicable"] for u in units), "units": units}
    return {**{k: session[k] for k in ("case_id", "base_id", "group", "input_profile", "split", "source_family_id", "duration_ms")},
            "session_sha256": digest(session), "oracle_sha256": digest(oracle),
            "metric_spec_sha256": digest(spec), "metrics": metrics, "opportunity_plan": accounting,
            "official_evaluation": False}


def prepare(sessions: list[dict], oracles: dict, spec: dict, *, sources: dict, population: str,
            repetitions: int = 3):
    """Plans and jobs for the given conditions. sources: {path: sha256} the preparation binds (cases, oracles, spec)."""
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("Use 1..10 explicit repetitions; the default is 3")
    missing = [s["case_id"] for s in sessions if s["case_id"] not in oracles]
    if missing:
        raise ValueError(f"conditions without an oracle: {missing}")
    plans = [metric_plan(s, oracles[s["case_id"]], spec) for s in sessions]
    jobs = [{"job_id": f"{p['case_id']}.r{r:02d}", "case_id": p["case_id"], "repeat": r, "group": p["group"],
             "input_profile": p["input_profile"], "split": p["split"], "session_sha256": p["session_sha256"],
             "oracle_sha256": p["oracle_sha256"], "input_assets_complete": True, "system_id": None,
             "state": "planned_not_started", "may_start_inference": False,
             "requires": ["frozen_system_configuration", "profile_and_full_duration_admission",
                          "client_presentation_receipts", "automated_judge_per_calibration_report"]}
            for p in plans for r in range(1, repetitions + 1)]
    summary = {"schema_version": PREPARATION_VERSION, "dataset_version": "streameval_v1.0",
               "evaluation_policy": policy_binding(),
               "population": population, "source_sha256": dict(sources),
               "metric_spec_status": spec["status"], "conditions": len(plans),
               "conditions_by_group_profile": dict(Counter(p["group"] + "/" + p["input_profile"] for p in plans)),
               "behavior_opportunities": sum(p["opportunity_plan"]["behavior_denominator"] for p in plans),
               "latency_opportunities": sum(p["opportunity_plan"]["latency_denominator"] for p in plans),
               "repetitions": repetitions, "planned_jobs_per_system_before_admission_filter": len(jobs),
               "plans_sha256": digest(plans), "jobs_sha256": digest(jobs), "review_mode": "automated_only",
               "network_requests": 0, "media_decoded": False, "official_evaluation": False, "formal_scores": 0}
    return summary, plans, jobs


def write_preparation(out: Path, summary: dict, plans: list, jobs: list) -> None:
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; a preparation is frozen once written")
    out.mkdir(parents=True)
    write_json(out / "manifest.json", summary)
    write_jsonl(out / "metric_plans.jsonl", plans)
    write_jsonl(out / "sampling_jobs.jsonl", jobs)


def prepare_files(cases: Path, oracles: Path, *, population: str, spec_path: Path = SPEC, repetitions: int = 3,
                  case_ids=None):
    """Preparation of the conditions in a cases.jsonl / oracles.jsonl pair (e.g. benchmark/v1/development)."""
    from ..util import read_json, read_jsonl
    sessions = [s for s in read_jsonl(cases) if case_ids is None or s["case_id"] in case_ids]
    sources = {}
    for p in (cases, oracles, spec_path, POLICY_PATH):
        p = Path(p).resolve()
        sources[str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)] = sha256_file(p)
    return prepare(sorted(sessions, key=lambda s: s["case_id"]), {o["case_id"]: o for o in read_jsonl(oracles)},
                   read_json(spec_path), sources=sources, population=population, repetitions=repetitions)
