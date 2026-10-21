"""Shared evaluation policy and input conditions, independent of model identity."""
from __future__ import annotations

import json

from . import REPO
from .util import digest, sha256_file

POLICY_PATH = REPO / "evaluation/protocol/evaluation_v2.json"
SCORING_PATH = REPO / "evaluation/protocol/scoring_v2.json"


def policy() -> dict:
    return json.loads(POLICY_PATH.read_text())


def policy_binding() -> dict:
    return {"version": policy()["version"], "sha256": sha256_file(POLICY_PATH)}


def input_contract(entry: dict, track: str) -> dict:
    """Declare the semantics presented to a model. Transport encoding does not change the task.

    A derived route must declare its transforms in system.json. No provider or model name is inspected.
    Renderer compositions additionally bind the shared dialogue configuration used to produce their input.
    """
    declarations = entry.get("input_adapters", {})
    if not isinstance(declarations, dict):
        raise ValueError("input_adapters must be an object keyed by track")
    unknown = set(declarations) - set(entry["tracks"])
    if unknown:
        raise ValueError(f"input_adapters names unregistered tracks: {sorted(unknown)}")
    route = declarations.get(track, {"mode": "native", "transforms": []})
    if not isinstance(route, dict):
        raise ValueError("each input adapter must be an object")
    if set(route) - {"mode", "transforms"}:
        raise ValueError("input adapter accepts mode and transforms only")
    mode, transforms = route.get("mode"), route.get("transforms")
    if mode not in ("native", "normalized", "derived") or not isinstance(transforms, list):
        raise ValueError("input adapter requires native/normalized/derived mode and a transforms list")
    if any(not isinstance(t, dict) or not isinstance(t.get("operation"), str) or not t["operation"].strip()
           or set(t) - {"operation", "parameters"} or not isinstance(t.get("parameters", {}), dict)
           for t in transforms):
        raise ValueError("each input transform requires an operation and optional parameters object")
    if (mode == "native") != (not transforms):
        raise ValueError("native input has no transforms; normalized and derived inputs must declare them")
    result = {"mode": mode, "transforms": transforms}
    if entry["system_kind"] == "composition":
        from .runtime.dialogue import STYLE, dialogue_models
        from .config import env
        comp = entry.get("composition") or {}
        result = {"mode": "derived", "transforms": [*transforms, {
            "operation": "shared_dialogue", "parameters": {
                **dialogue_models(entry), "stack": env("STREAMEVAL_DIALOGUE_STACK", "dashscope"),
                "reply_style": STYLE, "temperature": 0.3, "max_tokens": 200,
                "input": comp.get("input", "audio"),
                "audio_out": comp.get("audio_out", "client")}}]}
    # JSON round-trip makes the returned contract independent of mutable registry dictionaries.
    return json.loads(json.dumps(result, allow_nan=False))


def input_condition(contract: dict) -> str:
    """Stable comparison condition: native, or the declared normalization/conversion."""
    if contract == {"mode": "native", "transforms": []}:
        return "native"
    return contract["mode"] + ":" + digest(contract)[:16]
