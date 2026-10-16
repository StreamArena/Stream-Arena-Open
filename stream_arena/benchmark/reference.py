"""Reference-control payloads of the reference_edit track (set / patch / remove / reset over named target slots).

A payload: {"type": "reference_control", "operation": "set"|"patch"|"remove"|"reset",
            "references": [{"asset_id": "sa.clean_reference_image....", "target_slot": str, "scope": str}],
            "remove": [target_slot, ...], "scope": str (the operation's protection clause)}
"set" replaces the active slots, "patch" adds or replaces the listed ones, "remove" restores the listed slots to
the current source, "reset" restores all of them.
"""
from __future__ import annotations

import re

NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
MAX_TARGET_SCOPE_CHARS = 4000


def validate_payload(payload, *, max_target_scope_chars=MAX_TARGET_SCOPE_CHARS):
    if not isinstance(payload, dict) or set(payload) != {"type", "operation", "references", "remove", "scope"}:
        raise ValueError("Reference input has unexpected fields")
    if payload["type"] != "reference_control" or payload["operation"] not in {"set", "patch", "remove", "reset"}:
        raise ValueError("Unknown reference operation")
    if not isinstance(payload["scope"], str) or not 0 < len(payload["scope"]) <= 4000:
        raise ValueError("A bounded current operation scope is required")
    references, removed = payload["references"], payload["remove"]
    if not isinstance(references, list) or not isinstance(removed, list):
        raise ValueError("Reference and removal fields must be arrays")
    seen = set()
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {"asset_id", "target_slot", "scope"}:
            raise ValueError("A current reference requires only asset, target slot and scope")
        if (not isinstance(reference["asset_id"], str) or not NAME.fullmatch(reference["asset_id"])
                or not reference["asset_id"].startswith("sa.clean_reference_image.")
                or not isinstance(reference["target_slot"], str) or not NAME.fullmatch(reference["target_slot"])
                or reference["target_slot"] in seen or not isinstance(reference["scope"], str)
                or not 0 < len(reference["scope"]) <= max_target_scope_chars):
            raise ValueError("Invalid or repeated clean reference target")
        seen.add(reference["target_slot"])
    if any(not isinstance(slot, str) or not NAME.fullmatch(slot) for slot in removed) or len(removed) != len(set(removed)):
        raise ValueError("Invalid or repeated removed target")
    op = payload["operation"]
    if ((op in {"set", "patch"} and (not references or removed))
            or (op == "remove" and (references or not removed))
            or (op == "reset" and (references or removed))):
        raise ValueError("Reference operation has contradictory reference/removal fields")
