"""Aggregate per-job scores into the leaderboard file.

Input: a results root with one directory per system and population (streameval assemble writes it):
  <root>/<system_id>.<population>/system.json   {"system_id", "model_id", "name", "provider", "group",
                                                 "input_profile", "population", "portrait_policy", "system_kind",
                                                 "components", "system_config_sha256", ...}
      optional: "system_kind"     "native" | "composition" (default: "composition" when
                                  "components" is non-empty, else "native")
                "session_cap_ms"  the provider's hard session-length cap, for display only;
                                  runs it cuts short are scored as failures, never dropped
                "admitted_case_ids"  runtime admission metadata; never reduces the scoring denominator
      "portrait_policy" is "benchmark_portrait" when the provider animates the case
      portrait, a provider stock/fixed avatar id (e.g. "provider_stock_<name>")
      otherwise, and null for tracks without a portrait.
  <root>/<system>/scores/*.json       score reports (one per job, streameval.scoring.report)
  <root>/<system>/latency/*.json      opportunity accounting (one per job), optional
  <root>/<system>/av_sync_offset.json the system's A-V offset estimate (tooltip diagnostic), optional
Every job of the population's roster must be accounted for: a job with no score file stays in the denominator as
unknown (lower 0, upper 100), exactly as aggregate_roster requires.

    streameval leaderboard --results runs/results --roster runs/preparation/full --out FILE
"""
from __future__ import annotations

from collections import defaultdict
from .reporting_identity import reported_model_id, roster_summary
import hashlib
import json
import math
from pathlib import Path
import random

from .. import REPO
from ..benchmark.roster import Roster
from ..evaluation import SCORING_PATH, input_condition, policy_binding
from ..util import digest
from .calculation import aggregate_roster, family_cluster_ci

PROTOCOL = REPO / "evaluation/protocol"
STRATA = [15000, 30000, 60000, 120000, 300000]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_rank(values, q):
    values = sorted(values)
    return values[math.ceil(q * len(values)) - 1] if values else None


def population_ids(population: str, group: str, profile: str, roster: Roster) -> list[str]:
    """Use the frozen roster without provider- or outcome-dependent exclusions."""
    if population == "full":
        return sorted(c for c, s in roster.sessions.items()
                      if s["group"] == group and s["input_profile"] == profile)
    if population == "core" and group in roster.group_core_rosters:
        return list(roster.group_core_rosters[group])
    # text -> its core tier; extension / text_extension -> the extension tier; any other population (the
    # development roster) is its own tier
    tier = {"text": "core", "extension": "extension", "text_extension": "extension"}.get(population, population)
    if roster.doc.get("schema_version") == "streameval.preparation-roster/1":
        tier = population
    return sorted(c for c, s in roster.sessions.items() if s["group"] == group and s["input_profile"] == profile
                  and s["split"] == roster.split and s.get("tier") == tier)


def subset_score(agg: dict):
    """Diagnostic only: mean over the duration strata an admitted subset covers, when all its runs are observed."""
    covered = [s for s in agg["strata"] if s["cases"]]
    if not covered or agg["observed_ranking_runs"] != agg["expected_ranking_runs"] or any(s["score"] is None for s in covered):
        return None
    return sum(s["score"] for s in covered) / len(covered)


OVERALL_EXCLUDED_STATUSES = ("not_applicable",)
OVERALL_DEFINITION = ("Equal-weight mean of the track's applicable 0-100 metrics, excluding latency and only "
                      "predeclared not_applicable metrics (e.g. identity for a provider stock avatar). "
                      "Unavailable applicable measurements retain [0,100] bounds and their metric weight; "
                      "a point score exists only when every applicable metric is complete.")
AV_SYNC_RULE = PROTOCOL / "av_sync_rule_v2.json"
AV_SYNC_VALIDATION = PROTOCOL / "judge_av_sync_validation.json"
AV_SYNC_STRATUM = {"digital_human": "dialogue", "avatar_performance": "performance"}
def av_sync_meta(folder: Path, group: str, reports: list[dict]) -> dict:
    """A-V Sync metadata for an established av_sync metric: judge, frozen rule, system offset (tooltip diagnostic)
    and the provisional flag from the known-answer validation."""
    rule_bytes = AV_SYNC_RULE.read_bytes()
    rule = json.loads(rule_bytes)
    models = {r["av_sync_judge_model"] for r in reports if r.get("av_sync_judge_model")}
    if len(models) > 1:
        raise SystemExit(f"{folder.name}: reports mix A-V Sync judges")
    model = next(iter(models), rule["judge"]["model"])
    meta = {"judge": {"model": model,
                      "version": rule["judge_version"], "ruleSha256": hashlib.sha256(rule_bytes).hexdigest()}}
    offset = folder / "av_sync_offset.json"
    if offset.exists():
        o = json.loads(offset.read_text())
        meta["offset"] = {"tauMs": o.get("tau_ms"), "ci95Ms": o.get("ci95_ms"), "condition": o.get("condition"),
                          "sessions": o.get("sessions"), "toleranceMs": o.get("tolerance_ms")}
    if AV_SYNC_VALIDATION.exists():
        v = json.loads(AV_SYNC_VALIDATION.read_text())
        stratum = AV_SYNC_STRATUM.get(group)
        failing = sorted(k.split(":")[0].split(".", 1)[1] for k, c in v["classes"].items()
                         if k.startswith(f"{stratum}.") and not c["meets_criteria"])
        validated = (v.get("judge_version") == rule["judge_version"] and v.get("model") == model
                     and v.get("rule_sha256") == hashlib.sha256(rule_bytes).hexdigest())
        meta["provisional"] = bool(failing) or not validated
        if not validated:
            failing.append("configuration_not_validated")
        if failing:
            meta["provisionalClasses"] = failing
    else:
        meta["provisional"] = True
        meta["provisionalClasses"] = ["validation not run"]
    return meta


def content_validated(report: dict) -> bool:
    """Only a matching calibration with explicit admission validates the configured content judge."""
    extension = bool(report.get("extension_judge_version"))
    path = PROTOCOL / ("judge_ext_validation.json" if extension else "judge_v2_validation.json")
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    models = {k: v.get("model") for k, v in report.get("judge_configuration", {}).items()
              if k in ("primary", "second")}
    return (record.get("all_criteria_met") is True and record.get("models") == models
            and record.get("judge_version") == report.get("extension_judge_version" if extension else "judge_version")
            and record.get("prompts_sha256") == report.get("extension_prompts_sha256" if extension else "prompts_sha256"))


def overall_cluster_ci(aggregates: list[dict], *, draws: int = 2000, seed: int = 20260923) -> dict:
    """95% family-cluster bootstrap of the equal-weight mean of several roster aggregates.

    Each draw resamples whole source families once and applies the same family
    weights to every component metric, so the metrics' correlation within a
    family is preserved. With one aggregate this is exactly family_cluster_ci.
    """
    if type(draws) is not int or draws < 100 or type(seed) is not int:
        raise ValueError("Use at least 100 bootstrap draws and an integer seed")
    if not aggregates or any(a["score"] is None for a in aggregates):
        return {"status": "not_estimable", "interval": None, "reason": "incomplete_measurements"}
    families = sorted({r["source_family_id"] for a in aggregates for r in a["case_results"]})
    if len(families) < 5:
        return {"status": "not_estimable", "interval": None, "reason": "fewer_than_five_source_families"}
    layouts = [[(s["duration_ms"], [r for r in a["case_results"] if r["duration_ms"] == s["duration_ms"]])
                for s in a["strata"]] for a in aggregates]
    rng, samples = random.Random(seed), []
    for _ in range(draws * 20):
        if len(samples) == draws:
            break
        weights = defaultdict(int)
        for family in rng.choices(families, k=len(families)):
            weights[family] += 1
        means = []
        for layout in layouts:
            values = []
            for _duration, selected in layout:
                denominator = sum(weights[r["source_family_id"]] for r in selected)
                if not denominator:
                    break
                values.append(sum(r["score"] * weights[r["source_family_id"]] for r in selected) / denominator)
            if len(values) != len(layout):
                break
            means.append(sum(values) / len(values))
        if len(means) == len(layouts):
            samples.append(sum(means) / len(means))
    if len(samples) != draws:
        return {"status": "not_estimable", "interval": None, "reason": "insufficient_complete_cluster_draws"}
    samples.sort()
    return {"status": "estimated", "interval": [samples[math.ceil(.025 * draws) - 1], samples[math.ceil(.975 * draws) - 1]]}


def overall_metric(metrics: dict, aggregates: dict) -> dict:
    """Overall for one entry from its exported metrics and the roster aggregates behind them.

    Components: every content metric except predeclared not_applicable ones.
    Unavailable applicable measurements never reduce the denominator. The score exists only when every
    component is complete; bounds are the mean of the component bounds.
    """
    components = [n for n, m in metrics.items()
                  if n not in ("latency", "overall") and m["status"] not in OVERALL_EXCLUDED_STATUSES]
    if not components:
        return {"status": "not_established", "score": None, "lower": None, "upper": None, "ci95": None,
                "ci_status": "not_estimable", "components": [], "definition": OVERALL_DEFINITION}
    parts = [{**metrics[n], "lower": 0, "upper": 100} if metrics[n]["status"] == "not_established"
             else metrics[n] for n in components]
    mean = lambda values: sum(values) / len(values)  # noqa: E731
    complete = all(m["status"] == "complete" and m["score"] is not None for m in parts)
    bounded = all(m.get("lower") is not None and m.get("upper") is not None for m in parts)
    ci = (overall_cluster_ci([aggregates[n] for n in components])
          if complete and all(n in aggregates for n in components)
          else {"status": "not_estimable", "interval": None})
    return {"status": "complete" if complete else "undetermined",
            "score": mean([m["score"] for m in parts]) if complete else None,
            "lower": mean([m["lower"] for m in parts]) if bounded else None,
            "upper": mean([m["upper"] for m in parts]) if bounded else None,
            "ci95": ci["interval"], "ci_status": ci["status"], "components": components,
            "definition": OVERALL_DEFINITION}


SYSTEM_KINDS = ("native", "composition")


def system_kind(system: dict) -> str:
    """Native end-to-end system, or a renderer composed with shared external ASR/LLM/TTS."""
    kind = system.get("system_kind") or ("composition" if system.get("components") else "native")
    if kind not in SYSTEM_KINDS:
        raise SystemExit(f"{system.get('system_id')}: system_kind must be one of {SYSTEM_KINDS}, not {kind!r}")
    return kind


def system_entry(folder: Path, roster: Roster) -> dict:
    system = json.loads((folder / "system.json").read_text())
    group, profile = system["group"], system["input_profile"]
    population = system.get("population", "core")
    plans = roster.plans(population)
    full = population_ids(population, group, profile, roster)
    roster_ids = full
    if not roster_ids:
        raise SystemExit(f"{folder.name}: no admitted case in the {population} roster")
    sessions = roster.sessions
    # "public_dev" is aggregate_roster's marker for a ranking (non-calibration) condition
    cases = [{"case_id": c, "group": group, "input_profile": profile, "split": "public_dev",
              "source_family_id": sessions[c]["source_family_id"], "duration_ms": sessions[c]["duration_ms"]}
             for c in roster_ids]
    reports = [json.loads(p.read_text()) for p in sorted((folder / "scores").glob("*.json"))]
    if system.get("evaluation_policy") != policy_binding():
        raise SystemExit(f"{folder.name}: results do not bind the current evaluation policy")
    if roster.doc.get("evaluation_policy") != policy_binding():
        raise SystemExit("the roster does not bind the current evaluation policy")
    repetitions = roster.doc.get("repetitions", 3)
    if type(repetitions) is not int or repetitions < 1:
        raise SystemExit("roster requires a positive integer repetition count")
    if population in ("full", "core", "extension", "text", "text_extension") and repetitions != 3:
        raise SystemExit("paper-protocol populations require three repeats per condition")
    jobs = set()
    for r in reports:
        cid, repeat = r["case_id"], r["repeat"]
        if (cid not in roster_ids or r["system_config_sha256"] != system["system_config_sha256"]
                or r["system_id"] != system["system_id"]):
            raise SystemExit(f"{folder.name}: score {r['job_id']} is outside the roster or config")
        if (type(repeat) is not int or not 1 <= repeat <= repetitions
                or r["job_id"] != f"{cid}.r{repeat:02d}" or (cid, repeat) in jobs):
            raise SystemExit(f"{folder.name}: duplicate or invalid repeated job {r['job_id']}")
        if not r.get("unsupported") and r.get("evaluation_policy") != system["evaluation_policy"]:
            raise SystemExit(f"{folder.name}: inconsistent evaluation policy in {r['job_id']}")
        jobs.add((cid, repeat))
        if any(r.get(k) != plans[cid].get(k) for k in
               ("session_sha256", "oracle_sha256", "metric_plan_sha256")):
            raise SystemExit(f"{folder.name}: score {r['job_id']} does not match the frozen preparation")
        if r.get("metric_spec_sha256") != roster.doc["evaluation_protocol_sha256"]:
            raise SystemExit(f"{folder.name}: score {r['job_id']} uses another metric specification")
    # Task-specific extension units can differ between cases; model and policy settings must agree.
    judge_configs = {digest(r.get("judge_configuration")) for r in reports if not r.get("unsupported")}
    if len(judge_configs) > 1:
        raise SystemExit(f"{folder.name}: reports mix judge configurations")
    metrics, aggregates = {}, {}
    names = list(COLUMNS[group])
    spec = json.loads(SCORING_PATH.read_text())["metrics"]
    for name in names:
        if name == "av_sync" and not (AV_SYNC_RULE.exists() and AV_SYNC_VALIDATION.exists()):
            metrics[name] = {"status": "not_established", "score": None, "lower": None, "upper": None,
                             "reason": "the frozen A-V Sync rule has no published known-answer validation yet"}
            continue
        if (spec.get(name, {}).get("release_requires_calibration") and spec[name].get("alignment_model") is None
                and name != "av_sync"):
            metrics[name] = {"status": "not_established", "score": None, "lower": None, "upper": None,
                             "reason": "the protocol requires a calibrated alignment model, and none exists yet"}
            continue
        applicable_cases = [{**c, "applicable": name in plans[c["case_id"]].get(
            "applicable_metrics", plans[c["case_id"]]["metrics"])} for c in cases]
        if not any(c["applicable"] for c in applicable_cases):
            metrics[name] = {"status": "not_applicable", "score": None, "lower": None, "upper": None,
                             "reason": "no unit of this metric applies to the conditions of this roster"}
            continue
        results = []
        for r in reports:
            m = r["metrics"].get(name)
            if m is None:
                continue
            results.append({"case_id": r["case_id"], "repeat": r["repeat"], "status": m["status"], "score": m["score"],
                            "lower": m["lower"], "upper": m["upper"]})
        agg = aggregate_roster(applicable_cases, results, duration_strata_ms=STRATA, repetitions=repetitions)
        aggregates[name] = agg
        ci = family_cluster_ci(agg, draws=2000)
        metrics[name] = {"status": agg["status"], "score": agg["score"], "lower": agg["lower"], "upper": agg["upper"],
                         "ci95": ci["interval"], "ci_status": ci["status"], "observedRuns": agg["observed_ranking_runs"],
                         "expectedRuns": agg["expected_ranking_runs"],
                         "strata": [{k: s[k] for k in ("duration_ms", "cases", "status", "score", "lower", "upper")}
                                    for s in agg["strata"]]}
        if len(roster_ids) < len(full):
            metrics[name]["admittedSubsetScore"] = subset_score(agg)
        if name == "av_sync":
            metrics[name].update(av_sync_meta(folder, group, reports))
    for name, metric in metrics.items():
        if metric["status"] in ("not_applicable", "not_established") or name == "av_sync":
            continue
        current = [r for r in reports if not r.get("unsupported")]
        metric["provisional"] = not current or any(not content_validated(r) for r in current)
    metrics["overall"] = overall_metric(metrics, aggregates)
    lat_files = sorted((folder / "latency").glob("*.json")) if (folder / "latency").exists() else []
    lat_files = [p for p in lat_files if p.stem in {r["job_id"] for r in reports}]
    denominators = {cid: plans[cid].get("latency_denominator") for cid in roster_ids}
    if any(type(n) is not int or n < 0 for n in denominators.values()):
        raise SystemExit("roster plans require a frozen latency_denominator for every condition")
    timed, measured = [], 0
    eligible = sum(denominators.values()) * repetitions
    for p in lat_files:
        acct = json.loads(p.read_text())["latency"]
        cid = p.stem.rsplit(".r", 1)[0]
        values = acct.get("measured_success_latencies_ns", [])
        if (acct["denominator"] != denominators[cid] or acct["measured_successes"] != len(values)
                or not 0 <= len(values) <= denominators[cid]
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)):
            raise SystemExit(f"{p.name}: latency accounting does not match the frozen opportunity plan")
        measured += acct["measured_successes"]
        timed += values
    expected_jobs = len(roster_ids) * repetitions
    complete_latency = eligible > 0 and measured == eligible and len(lat_files) == expected_jobs
    metrics["latency"] = {"status": "measured" if complete_latency else "not_estimable" if eligible else "not_run",
                          "p95_ms": nearest_rank(timed, .95) / 1e6 if complete_latency and timed else None,
                          "success_subset_p50_ms": nearest_rank(timed, .50) / 1e6 if timed else None,
                          "success_subset_p95_ms": nearest_rank(timed, .95) / 1e6 if timed else None,
                          "measurement_coverage": measured / eligible if eligible else None}
    contract = system.get("input_contract", {"mode": "native", "transforms": []})
    condition = input_condition(contract)
    metrics["overall"]["provisional"] = any(m.get("provisional") for n, m in metrics.items() if n != "overall")
    entry = {"inputCondition": condition, "inputContract": contract,
             "evaluationPolicy": system.get("evaluation_policy"),
             "judgeConfigurationSha256": next(iter(judge_configs), None), "repetitions": repetitions,
             "systemId": system["system_id"], "modelId": system["model_id"], "name": system["name"],
             "provider": system["provider"], "group": group, "inputProfile": profile,
             "portraitPolicy": system.get("portrait_policy"), "components": system.get("components", []),
             "systemKind": system_kind(system), "sessionCapMs": system.get("session_cap_ms"),
             "systemConfigSha256": system["system_config_sha256"],
             "rosterSha256": roster.doc.get("group_core_rosters_sha256", {}).get(group),
             "population": population,
             "coverage": {"admittedCases": len(roster_ids), "rosterCases": len(full)},
             "rankable": False,
             "jobs": {"expected": expected_jobs, "scored": len(reports)}, "metrics": metrics}
    return entry


def build(results_root: Path, roster: Roster) -> dict:
    entries = []
    if results_root.exists():
        for folder in sorted(p for p in results_root.iterdir() if (p / "system.json").exists()):
            entries.append(system_entry(folder, roster))
    order = {"core": 0, "text": 1, "extension": 2, "text_extension": 3}
    entries.sort(key=lambda e: (e["group"], e["modelId"], order.get(e["population"], 9)))
    return {"schemaVersion": "streameval.v1-leaderboard/1", "version": roster.doc["dataset_version"],
            "releaseManifestSha256": roster.release_manifest_sha256,
            "evaluationProtocolSha256": roster.doc["evaluation_protocol_sha256"],
            "coreRosterSha256": roster.doc["group_core_rosters_sha256"],
            "evaluationPolicy": policy_binding(),
            "rankingRule": "Compare only the same dataset, population, task, input profile, input condition, "
                           "portrait policy, judge configuration, metric specification and repetition count. "
                           "Use identical failure accounting and display unresolved lower bounds. "
                           "No significance ranks are assigned. "
                           "Definitive rankings require independent human validation and matched observation endpoints. "
                           "No cross-task total.",
            "overallDefinition": OVERALL_DEFINITION,
            "entries": entries}


def encode(board: dict) -> bytes:
    return (json.dumps(public_board(board), indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


# ---- a readable table of a board file ---------------------------------------------------------------

COLUMNS = {"digital_human": ("visual_quality", "av_sync", "stability", "dialogue", "identity"),
           "avatar_performance": ("visual_quality", "av_sync", "stability", "following", "identity"),
           "general_interaction": ("visual_quality", "stability", "following", "reference", "continuity"),
           "instruction_edit": ("visual_quality", "stability", "following", "edit_success", "preservation"),
           "reference_edit": ("visual_quality", "stability", "reference", "edit_success", "preservation")}
TITLES = {"visual_quality": "Quality", "av_sync": "A–V Sync", "stability": "Stability", "dialogue": "Dialogue",
          "identity": "Identity", "following": "Following", "reference": "Reference", "continuity": "Continuity",
          "edit_success": "Edit Success", "preservation": "Preservation", "overall": "Overall"}
GROUP_TITLES = {"digital_human": "Digital-human dialogue", "avatar_performance": "Avatar performance",
                "general_interaction": "World generation", "instruction_edit": "Instruction editing",
                "reference_edit": "Reference editing"}
PUBLIC_SCHEMA = "streameval.leaderboard/2"


def displayed_value(metric: dict | None):
    """Paper convention: resolved score, otherwise the conservative lower endpoint."""
    if not metric or metric.get("status") in ("not_applicable", "not_established"):
        return None
    if "value" in metric:
        return metric["value"]
    if metric.get("score") is not None:
        return round(metric["score"], 2)
    lower = metric.get("lower")
    return math.floor(lower * 100 + 1e-8) / 100 if lower is not None else None


def public_board(board: dict) -> dict:
    """Export aggregate values only; sample bindings and execution evidence stay local.

    The schema deliberately has no per-case records, stratum diagnostics, endpoints,
    session counts or file hashes. Full bounds and bootstrap results remain in the
    evaluator's local score artifacts. Latency and coverage are local diagnostics.
    This formatter supports independently computed local evaluations of any
    registered system and population. No maintained evaluation results are bundled.
    """
    result = {"schemaVersion": PUBLIC_SCHEMA,
              "source": {k: v for k, v in board.get("source", {"kind": "local_evaluation"}).items()
                         if k in ("kind", "title", "section", "tables", "primaryPopulation", "coreConfigurations",
                                  "coreModelIdentifiers", "coreReportedModels", "coreConfigurationsByTask", "coreRuns",
                                  "fullConfigurations", "fullReportedModels", "fullConfigurationsByTask")},
              "display": {"content_decimals": 2,
                          "unresolved": "lower_bound", "order": "alphabetical_within_task"},
              "entries": []}
    keys = ("systemId", "modelId", "name", "provider", "group", "inputProfile", "portraitPolicy",
            "systemKind", "population", "inputRoute", "comparisonGroup")
    for entry in board["entries"]:
        row = {k: entry[k] for k in keys if k in entry}
        row["reportedModelId"] = reported_model_id(entry)
        row["comparisonGroup"] = _comparison_group(entry)
        if "inputRoute" not in row:
            if entry.get("systemKind") == "composition":
                row["inputRoute"] = "shared_dialogue_stack"
            elif entry.get("inputContract"):
                row["inputRoute"] = entry["inputContract"]["mode"]
        metrics = {}
        for name in (*COLUMNS[entry["group"]], "overall"):
            default = "not_applicable" if (entry["group"] == "general_interaction"
                                                   and entry.get("population") == "core" and name == "reference") else "not_established"
            cell = entry["metrics"].get(name, {"status": default})
            status = cell.get("status", "not_established")
            if status in ("undetermined", "incomplete"):
                status = "lower_bound"
            metrics[name] = {"status": status, "value": displayed_value(cell),
                             "provisional": bool(cell.get("provisional"))}
        row["metrics"] = metrics
        result["entries"].append(row)
    result["source"].update({k: v for k, v in roster_summary(result["entries"]).items()
                            if k.startswith(("core", "full"))})
    return result


def _comparison_group(entry: dict) -> str:
    if entry.get("comparisonGroup"):
        return entry["comparisonGroup"]
    if entry["group"] == "digital_human":
        stock = entry.get("portraitPolicy") not in (None, "benchmark_portrait", "original_portrait")
        return "Provider stock portrait" if stock else "Benchmark portrait"
    if entry["group"] == "avatar_performance":
        route = entry.get("inputRoute")
        derived = route == "text_derived" or entry.get("inputContract", {}).get("mode") == "derived"
        return "Text-derived line and action inputs" if derived else "Supplied audio"
    return ""


POPULATIONS = {"full": "complete benchmark", "core": "core conditions", "extension": "extension conditions",
               "text": "text input", "text_extension": "text input, extension conditions",
               "development": "development conditions"}


def _marker(entry: dict) -> str:
    """Paper notation: † shared dialogue stack, ‡ provider stock avatar, § separate input route."""
    if entry["group"] == "digital_human":
        if entry.get("systemKind") == "composition":
            return "†"
        return "‡" if entry.get("comparisonGroup") == "Provider stock portrait" else ""
    if entry.get("inputRoute") in ("text_derived", "instruction_to_reference"):
        return "§"
    return ""


NOTES = {"†": "† Renderer with the shared ASR, dialogue and TTS stack.",
         "‡": "‡ Provider-supplied avatar; Identity is N/A and Overall omits it.",
         "§": {"avatar_performance": "§ Text-derived line and action inputs; the other systems receive supplied audio.",
               "instruction_edit": "§ Instruction-to-reference composition."}}


def markdown(board: dict) -> str:
    """One table per population and task, in the layout of the paper's result tables."""
    grouped = {**board, "entries": [{**e, "comparisonGroup": _comparison_group(e)} for e in board["entries"]]}
    board = public_board(grouped)
    sets = {}
    for entry in board["entries"]:
        sets.setdefault((entry["population"], entry["group"]), []).append(entry)
    order = {p: i for i, p in enumerate(POPULATIONS)}
    groups = {g: i for i, g in enumerate(COLUMNS)}
    lines = ["# StreamEval results", "",
             "Scores range from 0 to 100; unresolved measurements show their lower bound. Overall is the equal-weight "
             "mean of the applicable content metrics. "
             "Timing and coverage diagnostics stay local. Rows are alphabetical; no "
             r"significance ranking is assigned. \* provisional metric; N/A not applicable. "
             "Definitions: [docs/evaluation.md](../docs/evaluation.md).", ""]
    for (population, group), entries in sorted(sets.items(), key=lambda i: (order.get(i[0][0], 9), groups[i[0][1]])):
        names = [n for n in (*COLUMNS[group], "overall")
                 if any(e["metrics"][n]["status"] != "not_applicable" for e in entries)]
        provisional = {n: any(e["metrics"][n]["provisional"] for e in entries) for n in names}
        headers = [TITLES[n] + (r"\*" if provisional[n] and n != "overall" else "")
                   for n in names]
        lines += [f"## {GROUP_TITLES[group]} · {POPULATIONS.get(population, population)}", "",
                  "| System | " + " | ".join(headers) + " |",
                  "|---|" + "---:|" * len(names)]
        marks = set()
        for entry in sorted(entries, key=lambda e: (e["name"].casefold(), e["systemId"])):
            cells = []
            for name in names:
                cell = entry["metrics"][name]
                value = cell["value"]
                cells.append(f"{value:.2f}" if value is not None else
                             "N/A" if cell["status"] == "not_applicable" else "–")
            mark = _marker(entry)
            marks.add(mark)
            lines.append("| " + (entry["name"] + mark).replace("|", r"\|") + " | " + " | ".join(cells) + " |")
        notes = [NOTES[m] if isinstance(NOTES[m], str) else NOTES[m].get(group, "") for m in ("†", "‡", "§")
                 if m in marks]
        lines += ["", " ".join(n for n in notes if n)] if any(notes) else []
        lines.append("")
    return "\n".join(lines).rstrip("\n")
