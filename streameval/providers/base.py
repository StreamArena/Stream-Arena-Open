"""SessionBackend: a model that runs whole sessions with its own media stack.

Registered with ``"backend": "backend.py:Class"`` in a system.json (streameval/models/registry.py). The batch runner, the
admission, the dry run and the judge treat it like any other system; only the attempt itself is the backend's:

    run(...)  ->  writes <out>/{runner.json, binding.json, result.json, browser/...} (docs/protocol.md) and
                  returns result.json; ``streameval validate-run`` checks the folder, the judge reads nothing else

Credentials, cost and the session cap are an Account (providers/account.py): environment variables only, never
written to an artifact; a backend with a missing credential is refused before any request is made.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..benchmark.tracks import event_op
from .account import Account


class SessionBackend(Account, ABC):
    """Credentials, cost and session cap: providers.account.Account."""

    def event_dispositions(self, system_id: str, spec: dict, case: dict) -> list[dict]:
        """Per event of `case`: {event_id, operation, supported, reason}; decided before any output exists.
        Default: the registered ops of the system (models.registry.ops)."""
        from ..models.registry import ops
        supported = set(ops(spec["entry"], spec["track"]))
        return [{"event_id": e["event_id"], "operation": event_op(e["payload"]),
                 "supported": event_op(e["payload"]) in supported,
                 "reason": None if event_op(e["payload"]) in supported else "not_in_registered_ops"}
                for e in case["events"]]

    @abstractmethod
    def run(self, system_id: str, spec: dict, case: dict, assets: dict, out: Path, *, repeat: int = 1,
            oracle: dict | None = None, smoke: bool = False, admission: dict | None = None,
            verify: bool = True) -> dict:
        """Run one session of `case` into the new folder `out` and return result.json.

        assets: asset_id -> {path, sha256} of the case (benchmark.dataset.Benchmark.case_assets); verify=True checks
        every file read against its sha256. oracle: never shown to the model."""
