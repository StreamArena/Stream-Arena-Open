"""Reported model identity is distinct from a task or composition registration.

Only the six explicitly reviewed renderer/dialogue aliases are collapsed. In
particular, distinct Scope pipelines and Krea generation/editing systems stay
distinct; this is not a count of independent weight sets or model architectures.
"""

ALIASES = {
    "aptavatar_comp": "aptavatar",
    "avatar_forever_comp": "avatar_forever",
    "avatarforcing_comp": "avatarforcing",
    "livetalk_comp": "livetalk",
    "quark_liveavatar_comp": "quark_liveavatar",
    "soulx_liveact_comp": "soulx_liveact",
}


def reported_model_id(entry):
    model_id = entry["modelId"]
    return ALIASES.get(model_id, model_id)


def roster_summary(entries):
    core = [e for e in entries if e.get("population") == "core"]
    keys = [(e["systemId"], e["population"]) for e in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate system/population result")
    tracks = sorted({e["group"] for e in core})
    full = [e for e in entries if e.get("population") == "full"]
    return {
        "fullConfigurations": len(full),
        "fullReportedModels": len({reported_model_id(e) for e in full}),
        "fullConfigurationsByTask": {t: sum(e["group"] == t for e in full) for t in sorted({e["group"] for e in full})},
        "coreConfigurations": len(core),
        "coreReportedModels": len({reported_model_id(e) for e in core}),
        "coreModelIdentifiers": len({e["modelId"] for e in core}),
        "coreConfigurationsByTask": {t: sum(e["group"] == t for e in core) for t in tracks},
        "exportEntries": len(entries),
        "systemIdentifiers": len({e["systemId"] for e in entries}),
        "modelIdentifiers": len({e["modelId"] for e in entries}),
    }
