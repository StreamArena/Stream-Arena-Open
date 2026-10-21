"""A benchmark set on disk: v1 sessions, their oracles and an asset manifest whose media live under the data root.

    benchmark/<set>/cases.jsonl     one v1 session per line (the timeline a model is run against)
    benchmark/<set>/oracles.jsonl   one oracle per case (what the judge checks; never shown to a model)
    benchmark/<set>/manifest.json   {"dataset_version", "population", "assets": {asset_id: {path, sha256, kind, bytes, ...}}}

An audio asset of the development conditions also carries ``script``, the authored text of the clip. The harness never
passes it to a model; only a session backend with a documented derived-text route reads it (models/providers/vidu,
avatar_performance: the supplied speech goes out as "LINE: <script>"), and its system_config says so.

Asset paths are relative to the data root (STREAMEVAL_DATA, default ./data). ``streameval validate`` checks that
every asset is present with the recorded sha256.
"""
from __future__ import annotations

from pathlib import Path

from .. import REPO
from ..config import data_root
from ..util import read_json, read_jsonl, sha256_file

DEFAULT = REPO / "benchmark/v1/full"


class Benchmark:
    """One benchmark set loaded from `root`: .cases and .oracles by case id, .assets from the manifest, media
    resolved under `data`. Used by the CLI and the runners (harness, batch) to look up cases and their files."""

    def __init__(self, root: Path = DEFAULT, data: Path | None = None):
        self.root = Path(root)
        self.data = Path(data) if data is not None else data_root()
        self.cases = {c["case_id"]: c for c in read_jsonl(self.root / "cases.jsonl")}
        oracles = self.root / "oracles.jsonl"
        self.oracles = {o["case_id"]: o for o in read_jsonl(oracles)} if oracles.exists() else {}
        self.manifest = read_json(self.root / "manifest.json")
        self.assets = self.manifest["assets"]

    def path(self, asset_id: str) -> Path:
        return self.data / self.assets[asset_id]["path"]

    def case_assets(self, case_id: str) -> dict[str, dict]:
        """asset_id -> {path (absolute), sha256[, script]} for one case; what the harness may read for it."""
        return {aid: {"path": self.path(aid), "sha256": self.assets[aid]["sha256"],
                      **({"script": self.assets[aid]["script"]} if "script" in self.assets[aid] else {})}
                for aid in self.cases[case_id]["asset_ids"]}

    def check(self, case_ids=None, *, hashes: bool = True) -> dict:
        """{asset_id: problem} for missing or altered media of the given cases (all by default)."""
        ids = sorted({a for c in (case_ids or self.cases) for a in self.cases[c]["asset_ids"]})
        problems = {}
        for aid in ids:
            path = self.path(aid)
            if not path.is_file():
                problems[aid] = "missing"
            elif hashes and sha256_file(path) != self.assets[aid]["sha256"]:
                problems[aid] = "sha256 mismatch"
        return problems
