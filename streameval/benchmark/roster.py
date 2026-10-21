"""Load an explicit local roster or derive one from a frozen preparation.

A roster folder (benchmark/v1/<population>) is prepared into a frozen
preparation by the batch runner; its roster is derived from that preparation.
"""
from __future__ import annotations

from pathlib import Path

from ..util import digest, read_json, read_jsonl, sha256_file

PREPARATION = {"core": "core", "extension": "extension", "text": "text", "text_extension": "text"}


def from_preparation(folder: Path) -> dict:
    """The roster document of a frozen preparation folder (manifest.json + metric_plans.jsonl)."""
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    plans = read_jsonl(folder / "metric_plans.jsonl")
    population = manifest["population"]
    if not plans:
        raise ValueError(f"{folder}: preparation has no conditions")
    splits = {p["split"] for p in plans}
    if len(splits) != 1 and population != "full":
        raise ValueError(f"{folder}: conditions of several splits {sorted(splits)}")
    if population in PREPARATION and "public_dev" in splits:
        raise ValueError(f"{folder}: population {population!r} is reserved for its complete task roster; "
                         "use development for the development roster")
    return {"schema_version": "streameval.preparation-roster/1", "dataset_version": manifest["dataset_version"],
            "note": f"Built from the preparation {folder.name} (population {population}); not a ranking roster.",
            "split": "full" if population == "full" else splits.pop(),
            "release_manifest_sha256": manifest["plans_sha256"],
            "evaluation_protocol_sha256": plans[0]["metric_spec_sha256"],
            "repetitions": manifest["repetitions"], "evaluation_policy": manifest.get("evaluation_policy"),
            "group_core_rosters": {}, "group_core_rosters_sha256": {},
            "cases": [{"case_id": p["case_id"], "group": p["group"], "input_profile": p["input_profile"],
                       "split": p["split"], "tier": population, "duration_ms": p["duration_ms"],
                       "source_family_id": p["source_family_id"]} for p in plans],
            "preparations": {population: {
                "manifest_sha256": sha256_file(folder / "manifest.json"),
                "plans": {p["case_id"]: {"metrics": sorted(p["metrics"]),
                                         "applicable_metrics": [n for n, m in p["metrics"].items() if m["denominator"]],
                                         "latency_denominator": p["metrics"]["latency"]["denominator"],
                                         "session_sha256": p["session_sha256"],
                                         "oracle_sha256": p["oracle_sha256"], "metric_plan_sha256": digest(p)}
                          for p in plans}}}}


class Roster:
    """A roster document loaded from a roster file or built from a preparation folder (from_preparation):
    .sessions by case id, .split and the per-population metric plans. Read by the scoring code."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.doc = from_preparation(self.path) if self.path.is_dir() else read_json(self.path)
        self.split = self.doc["split"]
        self.sessions = {c["case_id"]: c for c in self.doc["cases"]}

    @property
    def release_manifest_sha256(self) -> str:
        return self.doc["release_manifest_sha256"]

    @property
    def group_core_rosters(self) -> dict:
        return self.doc.get("group_core_rosters", {})

    def plans(self, population: str) -> dict:
        """{case_id: {"metrics": [...], "session_sha256", "oracle_sha256", "metric_plan_sha256"}}"""
        prep = population if population in self.doc["preparations"] else PREPARATION.get(population, population)
        if prep not in self.doc["preparations"]:
            raise ValueError(f"unknown population {population!r}")
        return self.doc["preparations"][prep]["plans"]
