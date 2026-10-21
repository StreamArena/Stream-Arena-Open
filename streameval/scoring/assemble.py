"""Collect judged jobs into the results layout the leaderboard reads.

    RUNS/<system_id>/<population>/{batch_plan.json, admission.json}       (streameval batch)
    EVAL/<system_id>/<job_id>/{score.json, latency.json}                  (streameval judge)
    -> OUT/<system_id>.<population>/system.json
                                    scores/<job_id>.json
                                    latency/<job_id>.json
                                    av_sync_offset.json                   (avatar tracks)

system.json binds the frozen system configuration (from the batch plan), the population and the conditions the
system was admitted for (the batch's admission.json). Jobs that were never judged simply have no file and stay unknown
(lower 0, upper 100) in the leaderboard.

Avatar systems first get A-V Sync re-decided over their whole judged population (one offset per system,
streameval.judge.av_sync.decide_system); if that fails for a system its results are written but the call returns 4:
their av_sync decisions are the per-job ones at a partial system offset and must not be published.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil


AVATAR_GROUPS = ("digital_human", "avatar_performance")


def _ref(path: Path, root: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(root.resolve())) if path.is_relative_to(root.resolve()) else path.name


def assemble(runs: Path, eval_root: Path, out: Path, judge_set, root: Path, *,
             release_manifest_sha256: str | None = None, av_sync: bool = True, log=print) -> int:
    """Write OUT/<system>.<population>/ for every batch under RUNS; returns 0, or 4 when A-V Sync was not re-decided.

    judge_set: the streameval.judge.units.JudgeSet the jobs were judged with (plans, populations, preparations).
    release_manifest_sha256: optional dataset identity retained by callers."""
    runs, eval_root, out, root = (Path(x).resolve() for x in (runs, eval_root, out, root))
    plans = sorted(runs.glob("*/*/batch_plan.json"))
    av_failed = []
    if av_sync:
        from ..judge.av_sync import decide_system
        avatar = sorted({json.loads(p.read_text())["system_id"] for p in plans
                         if json.loads(p.read_text())["system_config"]["group"] in AVATAR_GROUPS})
        for system_id in avatar:
            try:
                summary = decide_system(runs, eval_root, system_id, judge_set, root)
                log(json.dumps({"av_sync": summary}))
                if summary.get("splice_errors") or summary.get("pending"):
                    av_failed.append(system_id)
            except Exception as exc:  # noqa: BLE001 - the other metrics are still assembled; 4 at the end
                log(json.dumps({"av_sync_not_redecided": system_id, "error": f"{type(exc).__name__}: {exc}"[:300]}))
                av_failed.append(system_id)
    written = {}
    for plan_path in plans:
        plan = json.loads(plan_path.read_text())
        config, system_id, pop = plan["system_config"], plan["system_id"], plan["population"]
        admission_path = plan_path.parent / "admission.json"
        rows = json.loads(admission_path.read_text())["cases"]
        in_pop = {r["case_id"] for r in rows if judge_set.population.get(r["case_id"]) == pop}
        admitted = {r["case_id"] for r in rows if r["admitted"]} & in_pop
        scores = []
        for score_path in sorted((eval_root / system_id).glob("*/score.json")):
            score = json.loads(score_path.read_text())
            if score["case_id"] in in_pop:
                scores.append((score_path, score))
        folder = out / f"{system_id}.{pop}"
        shutil.rmtree(folder, ignore_errors=True)
        (folder / "scores").mkdir(parents=True)
        (folder / "latency").mkdir()
        # conditions scored as unsupported failures (streameval judge --score-unsupported) join the roster
        unsupported = sorted({s["case_id"] for _, s in scores if s.get("unsupported")})
        ids = sorted(admitted | set(unsupported))
        system = {"system_id": system_id, "model_id": config["model_id"], "name": config["name"],
                  "provider": config["provider"], "group": config["group"], "input_profile": config["input_profile"],
                  "population": pop, "portrait_policy": config.get("portrait_policy"),
                  "components": config.get("components", []), "system_kind": config.get("system_kind"),
                  "session_cap_ms": config.get("session_cap_ms"),
                  "evaluation_policy": config.get("evaluation_policy"),
                  "input_contract": config.get("input_contract", {"mode": "native", "transforms": []}),
                  "input_condition": config.get("input_condition", "native"),
                  "system_config_sha256": plan["system_config_sha256"],
                  "admission_file": _ref(admission_path, root), "admission_sha256": plan.get("admission_sha256"),
                  "admitted_case_ids": ids}
        if unsupported:
            system["unsupported_case_ids"] = unsupported
        (folder / "system.json").write_text(json.dumps(system, indent=1) + "\n")
        offset = eval_root / system_id / "av_sync_offset.json"
        if offset.exists() and system_id not in av_failed:
            shutil.copyfile(offset, folder / "av_sync_offset.json")
        for score_path, score in scores:
            (folder / "scores" / f"{score['job_id']}.json").write_text(score_path.read_text())
            latency = score_path.parent / "latency.json"
            if latency.exists():
                (folder / "latency" / f"{score['job_id']}.json").write_text(latency.read_text())
        written[folder] = len(scores)
    for folder, n in written.items():
        log(json.dumps({"results": str(folder), "scored_jobs": n}))
    if av_failed:
        log(f"A-V Sync was not re-decided for {sorted(set(av_failed))}: their av_sync decisions are the per-job ones "
            "at a partial system offset and must not be published; fix the cause and rerun")
        return 4
    return 0
