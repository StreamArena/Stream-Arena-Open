"""Pure score arithmetic for reviewed observations; no inference or admission.

Unit summaries (per job), roster aggregation over the duration strata (per system and metric) and the
family-cluster bootstrap interval.

The evaluator supplies a frozen unit roster and evidence-bound decisions. This
module cannot authenticate judgments or substitute for inspecting the media.
"""
from __future__ import annotations

from collections import defaultdict
import math
import random


def number(value):
    return type(value) in {int, float} and math.isfinite(value)


def summarize_units(units, decisions):
    """No omitted observation disappears from the frozen denominator.

    units: {unit_id, kind: binary|rubric|bounded_score, weight, applicable: bool}.
    decisions: {unit_id, status: observed|pass|fail|output_missing|undetermined,
    value?: 0..100, lower?: 0..100, upper?: 0..100,
    evidence: {artifact_ref, sha256}}. Partial bounded_score observations may
    supply lower/upper with status=undetermined; these never become point
    estimates or increase fully observed-unit coverage. N/A is a plan property,
    never a favorable runtime decision. Explicit missing output is a failure;
    a missing judgment is unknown. Observation bindings are caller-validated.
    """
    from .accounting import _artifact
    planned, actual = {}, {}
    for unit in units:
        if (not isinstance(unit.get("unit_id"), str) or not unit["unit_id"] or unit["unit_id"] in planned
                or unit.get("kind") not in {"binary", "rubric", "bounded_score"}
                or not number(unit.get("weight")) or unit["weight"] <= 0
                or type(unit.get("applicable")) is not bool):
            raise ValueError("Invalid or duplicate planned metric unit")
        planned[unit["unit_id"]] = unit
    for decision in decisions:
        key = decision.get("unit_id")
        if key not in planned or key in actual or not planned[key]["applicable"]:
            raise ValueError("Unknown, duplicate or pre-excluded observation")
        _artifact(decision.get("evidence"))
        status, kind, value = decision.get("status"), planned[key]["kind"], decision.get("value")
        has_bounds = "lower" in decision or "upper" in decision
        if has_bounds and (status != "undetermined" or kind != "bounded_score"):
            raise ValueError("Partial bounds belong only to undetermined bounded-score observations")
        if status == "observed":
            if kind not in {"rubric", "bounded_score"} or not number(value) or not 0 <= value <= 100:
                raise ValueError("Scored observations require a finite 0..100 value")
        elif status in {"pass", "fail"}:
            if kind != "binary" or value is not None:
                raise ValueError("Pass/fail is reserved for binary metrics without a numeric override")
            value = 100 if status == "pass" else 0
        elif status in {"output_missing", "undetermined"}:
            if value is not None:
                raise ValueError("Missing evidence must not carry an invented score")
            value = 0 if status == "output_missing" else None
        else:
            raise ValueError("Unknown observation status")
        lower, upper = (value, value) if value is not None else (0, 100)
        if has_bounds:
            lower, upper = decision.get("lower"), decision.get("upper")
            if not number(lower) or not number(upper) or not 0 <= lower <= upper <= 100:
                raise ValueError("Partial observations require finite ordered 0..100 bounds")
        actual[key] = {"value": value, "lower": lower, "upper": upper}
    required = [u for u in units if u["applicable"]]
    weight = sum(u["weight"] for u in required)
    known = [u for u in required if actual.get(u["unit_id"], {}).get("value") is not None]
    known_weight = sum(u["weight"] for u in known)
    numerator = sum(u["weight"] * actual[u["unit_id"]]["value"] for u in known)
    lower_sum = sum(u["weight"] * actual.get(u["unit_id"], {"lower": 0})["lower"] for u in required)
    upper_sum = sum(u["weight"] * actual.get(u["unit_id"], {"upper": 100})["upper"] for u in required)
    complete = bool(required) and len(known) == len(required)
    return {
        "status": "complete" if complete else "undetermined" if required else "not_applicable",
        "score": numerator / weight if complete else None,
        "lower": lower_sum / weight if weight else None,
        "upper": upper_sum / weight if weight else None,
        "observed_subset_mean": numerator / known_weight if known_weight else None,
        "coverage": known_weight / weight if weight else None,
        "denominator_units": len(required), "denominator_weight": weight,
        "observed_units": len(known), "undetermined_units": len(required) - len(known),
        "partially_bounded_units": sum(v["value"] is None and (v["lower"] > 0 or v["upper"] < 100)
                                       for v in actual.values()),
        "not_applicable_units": len(units) - len(required),
        "output_missing_units": sum(d["status"] == "output_missing" for d in decisions),
        "official_evaluation": False,
    }


def aggregate_roster(roster, results, *, duration_strata_ms, repetitions):
    """Equal repeats -> cases -> duration strata; excludes calibration by plan.

    A roster covers one model/system, metric, group and input profile. Missing
    expected runs remain unknown. It must already be an admitted study roster;
    this function neither admits models nor converts planning exclusions to
    failures. A not-started run is not an observed failure. A case's optional
    applicable bool defaults to True and must be frozen with the metric roster,
    never inferred from results. Pre-excluded cases contribute no run weight.
    If the entire ranking metric is pre-excluded, it is N/A; if only one duration
    is left without applicable cases, the five-stratum aggregate stays unknown.
    """
    if type(repetitions) is not int or repetitions <= 0 or not duration_strata_ms or len(set(duration_strata_ms)) != len(duration_strata_ms):
        raise ValueError("A unique duration roster and positive repeat count are required")
    indexed = {}
    boundary = set()
    families = {}
    for case in roster:
        cid = case["case_id"]
        if (cid in indexed or case["duration_ms"] not in duration_strata_ms
                or case["split"] not in {"public_dev", "calibration"}
                or type(case.get("applicable", True)) is not bool):
            raise ValueError("Invalid/duplicate case or duration/split outside the frozen study")
        boundary.add((case["group"], case["input_profile"]))
        if families.setdefault(case["source_family_id"], case["split"]) != case["split"]:
            raise ValueError("A source family crosses ranking and calibration")
        indexed[cid] = case
    if len(boundary) != 1:
        raise ValueError("Do not mix groups or input profiles")
    observed = {}
    for result in results:
        key = (result["case_id"], result["repeat"])
        if key[0] not in indexed or type(key[1]) is not int or not 1 <= key[1] <= repetitions or key in observed:
            raise ValueError("Unknown or repeated planned run")
        lower, upper, score = result.get("lower"), result.get("upper"), result.get("score")
        if not indexed[key[0]].get("applicable", True):
            if (result.get("status") != "not_applicable" or
                    any(result.get(field) is not None for field in ("score", "lower", "upper"))):
                raise ValueError("A pre-excluded case cannot receive a numeric observation")
        elif result.get("status") == "not_applicable":
            raise ValueError("Results cannot exclude a case from the frozen metric roster")
        elif (not number(lower) or not number(upper) or not 0 <= lower <= upper <= 100
                or (score is not None and (not number(score) or score != lower or lower != upper))):
            raise ValueError("Invalid score bounds")
        observed[key] = result
    bins = defaultdict(list)
    case_rows = []
    ranking = [c for c in indexed.values() if c["split"] == "public_dev"]
    excluded = [c for c in ranking if not c.get("applicable", True)]
    all_excluded = bool(ranking) and len(excluded) == len(ranking)
    for cid, case in indexed.items():
        if case["split"] == "calibration" or not case.get("applicable", True):
            continue
        runs = [observed.get((cid, r), {"lower": 0, "upper": 100, "score": None}) for r in range(1, repetitions + 1)]
        lower, upper = (sum(r[k] for r in runs) / repetitions for k in ("lower", "upper"))
        complete = all(r["score"] is not None for r in runs)
        row = {**case, "lower": lower, "upper": upper, "score": lower if complete else None}
        case_rows.append(row); bins[case["duration_ms"]].append(row)
    strata = []
    for duration in duration_strata_ms:
        cases = bins[duration]
        lower = sum(c["lower"] for c in cases) / len(cases) if cases else 0
        upper = sum(c["upper"] for c in cases) / len(cases) if cases else 100
        complete = bool(cases) and all(c["score"] is not None for c in cases)
        strata.append({"duration_ms": duration, "cases": len(cases),
                       "not_applicable_cases": sum(c["duration_ms"] == duration for c in excluded),
                       "status": "not_applicable" if all_excluded else "complete" if complete else "undetermined",
                       "lower": None if all_excluded else lower, "upper": None if all_excluded else upper,
                       "score": lower if complete else None})
    lower, upper = ((None, None) if all_excluded else
                    (sum(s[k] for s in strata) / len(strata) for k in ("lower", "upper")))
    complete = all(s["score"] is not None for s in strata)
    return {"status": "not_applicable" if all_excluded else "complete" if complete else "undetermined",
            "score": lower if complete else None, "lower": lower, "upper": upper,
            "case_results": case_rows, "strata": strata,
            "calibration_cases_excluded": sum(c["split"] == "calibration" for c in roster),
            "planned_ranking_runs_before_metric_exclusions": len(ranking) * repetitions,
            "not_applicable_ranking_cases": len(excluded),
            "not_applicable_case_ids": [c["case_id"] for c in excluded],
            "not_applicable_ranking_runs": len(excluded) * repetitions,
            "expected_ranking_runs": len(case_rows) * repetitions,
            "observed_ranking_runs": sum(indexed[cid]["split"] != "calibration" and indexed[cid].get("applicable", True)
                                         for cid, _ in observed),
            "official_evaluation": False}


def family_cluster_ci(aggregate, *, draws=2000, seed=20260923):
    """Percentile interval, resampling whole source families with replacement.

    Draws missing a duration stratum are rejected and counted. This interval
    reflects source-family sampling uncertainty, not judge reliability or an
    estimate of generalization from a small public pilot to all possible tasks.
    """
    if type(draws) is not int or draws < 100 or type(seed) is not int:
        raise ValueError("Use at least 100 bootstrap draws and an integer seed")
    rows = aggregate["case_results"]
    families = sorted({r["source_family_id"] for r in rows})
    if aggregate["score"] is None or len(families) < 5:
        return {"status": "not_estimable", "interval": None,
                "reason": "incomplete_measurements_or_fewer_than_five_source_families"}
    strata = [s["duration_ms"] for s in aggregate["strata"]]
    rng, samples, rejected = random.Random(seed), [], 0
    for _ in range(draws * 20):
        if len(samples) == draws:
            break
        weights = defaultdict(int)
        for family in rng.choices(families, k=len(families)):
            weights[family] += 1
        values = []
        for duration in strata:
            selected = [r for r in rows if r["duration_ms"] == duration]
            denominator = sum(weights[r["source_family_id"]] for r in selected)
            if not denominator:
                break
            values.append(sum(r["score"] * weights[r["source_family_id"]] for r in selected) / denominator)
        if len(values) == len(strata):
            samples.append(sum(values) / len(values))
        else:
            rejected += 1
    if len(samples) != draws:
        return {"status": "not_estimable", "interval": None, "reason": "insufficient_complete_cluster_draws"}
    samples.sort()
    return {"status": "estimated", "interval": [samples[math.ceil(.025 * draws) - 1], samples[math.ceil(.975 * draws) - 1]],
            "confidence": .95, "unit": "source_family", "families": len(families),
            "draws": draws, "rejected_empty_stratum_draws": rejected, "seed": seed}
