"""Per-session opportunity accounting (behaviour verdicts and latency); no media inference or ranking.

opportunity_plan freezes one behaviour opportunity per checkpoint of a condition (including wait/continue turns)
before any output exists; account_session accounts one finished run against the scheduler ledger and the judged
checkpoint decisions, keeping a missing decision undetermined (never a pass or an invented zero latency).
Artifact bindings document the trust boundary; hashes alone do not authenticate a reviewer or prove that the
evidence supports the claim.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import re

from ..benchmark import protocols as input_contract
from ..benchmark import session as v1_session
from ..util import digest

MS = 1_000_000

VERSION = "streameval.opportunity-accounting.v1"
AVATAR_PERFORMANCE_VERSION = "streameval.avatar-performance-accounting.v1"
FAILURES = {"timeout", "no_effect", "unsupported_during_run", "delivery_failed",
            "semantic_failure"}
UNCERTAIN = {"not_adjudicated", "insufficient_evidence", "unobservable"}
RELEASE_FAILURES = {"trigger_not_fired", "not_released", "release_expired"}


def _integer(value):
    return type(value) is int and value >= 0


def _index(rows, key, label):
    if not isinstance(rows, list):
        raise ValueError(label + " must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(label + " has an invalid identifier")
        if row[key] in result:
            raise ValueError(label + " has duplicate identifiers")
        result[row[key]] = row
    return result


def _performance_timing(session, events, checkpoints):
    """Validate authored speech windows and visual state/action checkpoints."""
    budgets = {}
    controls = [e for e in events.values() if e["payload"]["type"] == "user_prompt"]
    ends = {e["event_id"]: controls[i + 1]["release"]["at_ms"] if i + 1 < len(controls)
            else session["duration_ms"] for i, e in enumerate(controls)}
    for eid, event in events.items():
        cp, at = checkpoints[eid], event["release"]["at_ms"]
        if event["payload"]["type"] == "speech_audio":
            budget = cp.get("audio_budget_ms")
            if (cp.get("response_policy") != "speech_render" or cp.get("latency_opportunity") is not False or
                    cp.get("effect_kind") != "supplied_speech_alignment" or not _integer(budget) or budget == 0 or
                    cp.get("audio_asset_id") != event["payload"]["asset_id"] or
                    cp.get("planned_speaking_interval_ms") != [at, at + budget] or
                    cp.get("check_window_ms") != [at, at + budget] or
                    any(key in cp for key in ("response_deadline_ms", "effect_deadline_ms", "stable_effect_latency_target"))):
                raise ValueError("speech rendering checkpoint requires its own audio budget/alignment semantics, without response latency")
            budgets[eid] = budget
        else:
            until, deadline = cp.get("valid_until_ms"), cp.get("effect_deadline_ms")
            instant = cp.get("effect_kind") == "instant_action"
            stability = cp.get("completion_state_stable_window_ms") if instant else cp.get("stable_window_ms")
            if (cp.get("response_policy") != "control_effect" or cp.get("latency_opportunity") is not True or
                    not isinstance(cp.get("effect_kind"), str) or cp["effect_kind"] not in {"control_effect", "instant_action"} or
                    not _integer(until) or until != ends[eid] or not _integer(deadline) or deadline == 0 or
                    not _integer(stability) or stability == 0 or at + deadline + stability > until or
                    cp.get("check_window_ms") != [at + deadline, until] or "response_deadline_ms" in cp):
                raise ValueError("visual control needs a complete action/state observation window ending at the next visual control")
            if instant:
                if (cp.get("effect_deadline_target") != "first_observable_correct_action_response" or
                        cp.get("stable_effect_latency_target") != "action_completion" or
                        cp.get("action_observation_window_ms") != [at, until] or "stable_window_ms" in cp):
                    raise ValueError("instant performance actions require distinct onset/completion and post-completion stability")
            elif ("effect_deadline_target" in cp or "completion_state_stable_window_ms" in cp or
                    cp.get("stable_effect_latency_target", "stable_correct_performance_effect") != "stable_correct_performance_effect"):
                raise ValueError("state performance controls cannot reuse an instant-action timing target")
    errors = input_contract.validate_performance_audio_timing(session, budgets)
    if errors:
        raise ValueError("; ".join(errors))
    return budgets


def opportunity_plan(session, oracle):
    """Preserve one behavior opportunity per checkpoint, including wait/continue.

    This is an evaluator-only object. It must never enter a model adapter.
    Hashes use streameval.util.digest; the encoding is declared in the output.
    """
    v1_session.validate_session(session)
    profiles = {"digital_human": {"conversation_audio", "conversation_text"},
                "reference_edit": {v1_session.PROFILE_REFERENCE},
                "instruction_edit": {"edit_instruction"},
                "general_interaction": {"general_text_to_video", "general_image_text_to_video"},
                "avatar_performance": {"avatar_performance_audio"}}
    if session["input_profile"] not in profiles[session["group"]]:
        raise ValueError("accounting supports only current candidate roster profiles")
    if (oracle.get("case_id") != session["case_id"] or
            oracle.get("source_family_id") != session["source_family_id"] or
            oracle.get("failed_opportunity") != "keep_in_denominator"):
        raise ValueError("oracle identity or denominator policy mismatch")
    events = _index(session["events"], "event_id", "events")
    checkpoints = _index(oracle.get("checkpoints"), "event_id", "checkpoints")
    if events.keys() != checkpoints.keys():
        raise ValueError("every scheduled event requires exactly one checkpoint")
    text_dialogue = session["input_profile"] == "conversation_text"
    if text_dialogue and any(cp.get("timing_origin") != "actual_user_prompt_release_at_evaluator"
                             for cp in checkpoints.values()):
        raise ValueError("Text dialogue requires its separate text oracle and release-time origin")
    performance = session["group"] == "avatar_performance"
    if performance:
        _performance_timing(session, events, checkpoints)
    rows = []
    for event_id, event in events.items():
        checkpoint = checkpoints[event_id]
        avatar = session["group"] == "digital_human"
        policy = checkpoint.get("response_policy") if avatar or performance else "control_effect"
        if avatar and policy not in {"substantive_response", "no_response_expected", "continue_response"}:
            raise ValueError("unknown response policy")
        eligible = (policy == "control_effect" if performance else not avatar or policy == "substantive_response")
        target = ("first_substantive_response" if avatar else
                  checkpoint.get("stable_effect_latency_target", "stable_correct_performance_effect" if performance
                                 else "stable_correct_effect")) if eligible else None
        timing_keys = ("response_deadline_ms", "effect_deadline_ms", "effect_deadline_target",
                       "stable_window_ms", "completion_state_stable_window_ms", "valid_until_ms",
                       "review_window", "effect_kind", "audio_budget_ms", "planned_speaking_interval_ms",
                       "check_window_ms", "action_observation_window_ms")
        rows.append({"event_id": event_id, "checkpoint_sha256": digest(checkpoint),
                     "release": deepcopy(event["release"]), "denominator": 1,
                     "response_policy": policy, "latency_eligible": eligible,
                     "latency_origin": ("user_prompt_release" if text_dialogue else
                                        "end_of_presented_user_audio" if avatar else "input_release") if eligible else None,
                     "latency_target": target,
                     "timing_contract": {key: deepcopy(checkpoint[key]) for key in timing_keys if key in checkpoint},
                     **({"speech_alignment_eligible": policy == "speech_render"} if performance else {})})
    return {"schema_version": AVATAR_PERFORMANCE_VERSION if performance else VERSION, "kind": "design_opportunity_plan",
            "benchmark_eligible": False, "official_evaluation": False,
            "hash_encoding": "sha256:json-sort-keys-utf8-default-separators",
            **{key: session[key] for key in ("case_id", "base_id", "source_family_id", "group",
                                             "scenario", "input_profile", "split", "duration_ms")},
            "runner_session_sha256": digest(session), "oracle_sha256": digest(oracle),
            "behavior_denominator": len(rows),
            "latency_denominator": sum(row["latency_eligible"] for row in rows),
            **({"speech_alignment_denominator": sum(row["speech_alignment_eligible"] for row in rows)} if performance else {}),
            "opportunities": rows}


def _artifact(value):
    if (not isinstance(value, dict) or set(value) != {"artifact_ref", "sha256"} or
            not isinstance(value["artifact_ref"], str) or not value["artifact_ref"].strip() or
            not isinstance(value["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", value["sha256"])):
        raise ValueError("evidence requires an artifact reference and SHA256")


def _p95(values):
    return sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None


def account_session(session, oracle, snapshot, adjudications):
    """Account a finalized, attempted run without dropping missing decisions.

    A missing adjudication is undetermined, never a pass or an invented zero
    latency. Preflight exclusions do not have a started scheduler and are
    rejected here; they belong to admission, outside run success denominators.
    """
    plan = opportunity_plan(session, oracle)
    epoch, deadline = snapshot.get("started_at_ns"), snapshot.get("deadline_ns")
    finished = snapshot.get("finished_at_ns")
    if (snapshot.get("state") not in {"ended", "cancelled", "error"} or
            not _integer(epoch) or not _integer(finished) or finished < epoch or
            deadline != epoch + session["duration_ms"] * MS or
            snapshot.get("case_id") != session["case_id"] or
            snapshot.get("runner_session_sha256") != plan["runner_session_sha256"] or
            not isinstance(snapshot.get("scheduler_id"), str) or not snapshot["scheduler_id"] or
            snapshot.get("clock_domain_id") != "scheduler:" + snapshot["scheduler_id"] or
            snapshot.get("official_evaluation") is not False):
        raise ValueError("a started, final, session-bound scheduler ledger is required")
    rows = _index(snapshot.get("opportunities"), "event_id", "scheduler ledger")
    planned = _index(plan["opportunities"], "event_id", "opportunity plan")
    if list(rows) != list(planned) or type(snapshot.get("denominator")) is not int or snapshot["denominator"] != len(planned):
        raise ValueError("scheduler ledger omitted or added opportunities")
    required = {"schema_version", "runner_session_sha256", "oracle_sha256", "scheduler_sha256",
                "evidence_kind", "decisions"}
    if (not isinstance(adjudications, dict) or set(adjudications) != required or
            adjudications["schema_version"] != plan["schema_version"] or
            adjudications["runner_session_sha256"] != plan["runner_session_sha256"] or
            adjudications["oracle_sha256"] != plan["oracle_sha256"] or
            adjudications["scheduler_sha256"] != digest(snapshot) or
            adjudications["evidence_kind"] not in {"synthetic_fixture", "reviewed_run", "automated_judge_run"}):
        raise ValueError("adjudications must bind the exact session, oracle and final run ledger")
    decisions = _index(adjudications["decisions"], "event_id", "adjudications")
    if decisions.keys() - planned.keys():
        raise ValueError("adjudications reference unknown opportunities")
    performance = session["group"] == "avatar_performance"
    if performance:
        if (snapshot.get("speech_audio_timing_basis") != "caller_supplied_measured_lengths_not_actual_pcm_presentation" or
                snapshot.get("pcm_pacing_implemented") is not False):
            raise ValueError("performance accounting requires the explicit diagnostic speech timing boundary")
        speech_events = [e for e in session["events"] if e["payload"]["type"] == "speech_audio"]
        for index, event in enumerate(speech_events):
            eid = event["event_id"]
            row = rows[eid]
            measured = row.get("speech_audio_duration_ns")
            limit = epoch + (speech_events[index + 1]["release"]["at_ms"] if index + 1 < len(speech_events)
                             else session["duration_ms"]) * MS
            if (row.get("speech_audio_budget_ns") != planned[eid]["timing_contract"]["audio_budget_ms"] * MS or
                    not _integer(measured) or measured == 0 or measured > row["speech_audio_budget_ns"] or
                    row.get("input_end_at_ns") is not None or row.get("speech_audio_window_end_at_ns") != limit):
                raise ValueError("speech ledger must bind measured lengths, authored budgets and its nonoverlapping window")
            released, handed = row.get("released_at_ns"), row.get("handoff_at_ns")
            scheduled_end = row.get("scheduled_speech_audio_end_at_ns")
            if ((released is None and scheduled_end is not None) or
                    (released is not None and (not _integer(released) or scheduled_end != released + measured or scheduled_end > limit)) or
                    (handed is not None and (not _integer(handed) or handed + measured > limit))):
                raise ValueError("speech release/handoff must fit the available PCM window")
    accounted = []
    for event_id, expected in planned.items():
        row = rows[event_id]
        released, handed = row.get("released_at_ns"), row.get("handoff_at_ns")
        release_failed = row.get("status") in RELEASE_FAILURES
        if (type(row.get("denominator")) is not int or row["denominator"] != 1 or
                (not release_failed and (row.get("status") != "released" or
                 row.get("delivery_state") != "handed_off" or row.get("opportunity_failed") is not None)) or
                (release_failed and row.get("opportunity_failed") is not True)):
            raise ValueError("scheduler row is incomplete or contradictory")
        if released is not None and (not _integer(released) or not epoch <= released < deadline):
            raise ValueError("release timestamp is outside the activity")
        if handed is not None and (not _integer(handed) or released is None or not released <= handed < deadline):
            raise ValueError("handoff timestamp is outside the release/activity interval")
        if not release_failed and (released is None or handed is None):
            raise ValueError("released opportunity is missing its timestamps")
        decision = decisions.get(event_id)
        verdict, reason, latency, evidence, timing = "undetermined", "not_adjudicated", None, None, None
        if release_failed:
            verdict, reason = "fail", row["status"]
        if decision is not None:
            if (set(decision) != {"event_id", "checkpoint_sha256", "verdict", "reason", "evidence", "presentation"} or
                    decision["checkpoint_sha256"] != expected["checkpoint_sha256"] or
                    decision["verdict"] not in {"pass", "fail", "undetermined"}):
                raise ValueError("invalid or stale checkpoint decision")
            _artifact(decision["evidence"])
            if decision["verdict"] == "pass":
                if decision["reason"] != "criterion_satisfied" or release_failed:
                    raise ValueError("an unreleased/failed opportunity cannot pass")
                if (expected.get("speech_alignment_eligible") and
                        finished < handed + row["speech_audio_duration_ns"]):
                    raise ValueError("a full supplied-speech checkpoint cannot pass before its minimum paced delivery window ends")
            elif decision["verdict"] == "fail":
                if decision["reason"] not in FAILURES | ({row["status"]} if release_failed else set()):
                    raise ValueError("failure verdict requires a failure reason")
            elif decision["reason"] not in UNCERTAIN:
                raise ValueError("undetermined verdict requires an uncertainty reason")
            if release_failed and (decision["verdict"] != "fail" or decision["reason"] != row["status"]):
                raise ValueError("release failure cannot be replaced by a favorable decision")
            verdict, reason, evidence = decision["verdict"], decision["reason"], decision["evidence"]
            timing = decision["presentation"]
            if timing is not None:
                keys = {"basis", "clock_domain_id", "origin_kind", "origin_at_ns", "target",
                        "effect_started_at_ns", "first_valid_at_ns", "verified_at_ns", "evidence", "input_end"}
                if (verdict != "pass" or not expected["latency_eligible"] or not isinstance(timing, dict) or
                        set(timing) != keys or timing["basis"] != "client_presentation" or
                        timing["clock_domain_id"] != snapshot["clock_domain_id"] or
                        timing["origin_kind"] != expected["latency_origin"] or timing["target"] != expected["latency_target"]):
                    raise ValueError("latency requires a matching independent successful presentation receipt")
                avatar = session["group"] == "digital_human"
                audio_dialogue = avatar and session["input_profile"] == "conversation_audio"
                origin = timing["origin_at_ns"] if audio_dialogue else released
                first, verified = timing["first_valid_at_ns"], timing["verified_at_ns"]
                if (not _integer(origin) or not _integer(timing["origin_at_ns"]) or timing["origin_at_ns"] != origin or
                        not _integer(first) or not _integer(verified) or
                        not max(origin, handed) <= first <= verified <= min(deadline, finished)):
                    raise ValueError("invalid origin, clock ordering or presentation outside the activity")
                _artifact(timing["evidence"])
                contract = expected["timing_contract"]
                if audio_dialogue:
                    endpoint = timing["input_end"]
                    if (not isinstance(endpoint, dict) or
                            set(endpoint) != {"event_id", "clock_domain_id", "presented_end_at_ns", "evidence"} or
                            endpoint["event_id"] != event_id or endpoint["clock_domain_id"] != snapshot["clock_domain_id"] or
                            not _integer(endpoint["presented_end_at_ns"]) or endpoint["presented_end_at_ns"] != origin or
                            not _integer(row.get("input_end_at_ns")) or
                            origin < max(row["input_end_at_ns"], handed) or timing["effect_started_at_ns"] is not None):
                        raise ValueError("an independent, event-bound actual user input endpoint is required")
                    _artifact(endpoint["evidence"])
                elif timing["input_end"] is not None:
                    raise ValueError("Text/control timing cannot substitute an audio endpoint")
                if avatar:
                    if timing["effect_started_at_ns"] is not None:
                        raise ValueError("Dialogue timing cannot substitute a visual action onset")
                    if first > origin + contract["response_deadline_ms"] * MS:
                        raise ValueError("substantive response missed its checkpoint deadline")
                    release_order = sorted(((r["released_at_ns"], i, id_) for i, (id_, r) in enumerate(rows.items())
                                            if r.get("released_at_ns") is not None))
                    position = next(i for i, entry in enumerate(release_order) if entry[2] == event_id)
                    next_input = release_order[position + 1][0] if position + 1 < len(release_order) else None
                    if next_input is not None and verified > next_input:
                        raise ValueError("response evidence extends beyond the next released user turn")
                else:
                    if verified > epoch + contract["valid_until_ms"] * MS:
                        raise ValueError("effect evidence extends beyond this checkpoint's validity")
                    stability = contract.get("completion_state_stable_window_ms") or contract.get("stable_window_ms")
                    if not _integer(stability) or verified < first + stability * MS:
                        raise ValueError("the complete prescribed stability window is required")
                    if contract.get("effect_deadline_target") == "first_observable_correct_action_response":
                        onset = timing["effect_started_at_ns"]
                        if (not _integer(onset) or not max(origin, handed) <= onset <= first or
                                onset > origin + contract["effect_deadline_ms"] * MS):
                            raise ValueError("action onset must meet its own effect deadline")
                    elif (timing["effect_started_at_ns"] is not None or
                          first > origin + contract["effect_deadline_ms"] * MS):
                        raise ValueError("stable effect missed its checkpoint deadline")
                latency = first - origin
        accounted.append({"event_id": event_id, "denominator": 1,
                          "checkpoint_sha256": expected["checkpoint_sha256"],
                          "verdict": verdict, "reason": reason, "evidence": evidence,
                          "scheduler_failure_reason": row.get("failure_reason"),
                          "latency_eligible": expected["latency_eligible"],
                          "success_latency_ns": latency, "presentation": timing,
                          "released_at_ns": released,
                          "scheduled_input_end_at_ns": row.get("input_end_at_ns"),
                          **({"speech_alignment_eligible": expected["speech_alignment_eligible"],
                              "scheduled_speech_audio_end_at_ns": row.get("scheduled_speech_audio_end_at_ns")}
                             if performance else {}),
                          "phase": None if released is None else
                          ("early" if 3 * (released - epoch) < deadline - epoch else
                           "middle" if 3 * (released - epoch) < 2 * (deadline - epoch) else "late")})
    counts = Counter(row["verdict"] for row in accounted)
    eligible = [row for row in accounted if row["latency_eligible"]]
    timed = [row["success_latency_ns"] for row in eligible if row["success_latency_ns"] is not None]
    full = bool(eligible) and len(timed) == len(eligible)
    semantic_successes = sum(row["verdict"] == "pass" for row in eligible)
    # Conservative diagnostic only: do not invent timeout observations or
    # assume the relative ordering of unknown/failure tails.
    speech_rows = [row for row in accounted if row.get("speech_alignment_eligible")]
    return {"schema_version": plan["schema_version"], "kind": "session_accounting_diagnostic",
            "benchmark_eligible": False, "official_evaluation": False,
            "calibration_status": "not_calibrated", "evidence_kind": adjudications["evidence_kind"],
            "plan": plan, "scheduler_sha256": digest(snapshot), "adjudications_sha256": digest(adjudications),
            "behavior": {"denominator": len(accounted), "pass": counts["pass"], "fail": counts["fail"],
                         "undetermined": counts["undetermined"],
                         "confirmed_success_fraction": counts["pass"] / len(accounted),
                         "reasons": dict(sorted(Counter(row["reason"] for row in accounted).items())),
                         "is_session_success_metric": False},
            "latency": {"denominator": len(eligible), "semantically_successful": semantic_successes,
                        "measured_successes": len(timed), "unit": "ns",
                        "measurement_coverage_of_all_opportunities": len(timed) / len(eligible) if eligible else None,
                        "measurement_coverage_of_semantic_successes": len(timed) / semantic_successes if semantic_successes else None,
                        "subset_n_below_20_p95_is_sample_maximum": 0 < len(timed) < 20,
                        "confidence_interval": None,
                        "quantile_method": "nearest_rank",
                        "all_opportunity_p95": {"status": "measured" if full else "not_estimable" if eligible else "not_applicable",
                                                "value_ns": _p95(timed) if full else None,
                                                "reason": None if full else "incomplete_successful_presentation_evidence" if eligible else "no_response_opportunities"},
                        "measured_success_subset_p95_ns": _p95(timed),
                        "measured_success_subset_p50_ns": sorted(timed)[math.ceil(.5 * len(timed)) - 1] if timed else None,
                        "measured_success_latencies_ns": sorted(timed)},
            **({"speech_alignment": {"denominator": len(speech_rows),
                  "pass": sum(r["verdict"] == "pass" for r in speech_rows),
                  "fail": sum(r["verdict"] == "fail" for r in speech_rows),
                  "undetermined": sum(r["verdict"] == "undetermined" for r in speech_rows),
                  "basis": "reviewed_checkpoint_verdicts_not_automatic_audio_visual_alignment",
                  "measured_av_sync": None, "render_lag_per_audio_pts": None,
                  "included_in_control_response_speed": False}} if performance else {}),
            "opportunities": accounted}
