"""Session contract of the v1 benchmark (one session = one condition's timeline).

A session: identity (case_id sa.v1.NNN.<group>, base_id, source_family_id, split, tier, duration_ms), group and
input_profile, initialization (portrait / opening prompt / source video / preloaded reference), asset_ids, and
timed events {event_id, release: {at_ms}, payload}. The scored rosters of benchmark/v1 use split private_test (tier
core | extension), a historical label kept so released sessions match the frozen release byte for byte; the
development roster (benchmark/v1/development) uses split public_dev (tier development). All are public.

The shared structure is checked by the frozen session schema (session.schema.json) and the input-profile rules
(protocols.py), which predate v1 and only admit the pre-v1 identities: a v1 session is validated by projecting it
onto that envelope and running both unchanged. v1-only differences are checked here explicitly:

* identity: ``sa.v1.NNN.<group>`` (or ``.conversation_text.digital_human``), ``dataset_version: streameval_v1.0``,
  ``split: private_test`` with ``tier: core|extension``, or ``split: public_dev`` with ``tier: development``;
* reference editing uses PROFILE_REFERENCE: reference A is preloaded in initialization (not a timed opportunity)
  and timed events carry the reference-control payload (set / patch / remove / reset, see reference.py).
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re

from . import protocols as input_contract
from .reference import validate_payload

DATASET_VERSION = "streameval_v1.0"
PROFILE_REFERENCE = "edit_reference_v1_preloaded_a"
SPLIT_TIERS = {"private_test": {"core", "extension"}, "public_dev": {"development"}}
# ranked conditions passed the v1 automated gate; development conditions projected from v0.2 never ran through it
READINESS = {"private_test": {"automated_gate_passed"}, "public_dev": {"automated_gate_passed", "design_only"}}
GROUPS = {"digital_human", "avatar_performance", "general_interaction", "instruction_edit", "reference_edit"}
CASE_ID = re.compile(r"sa\.v1\.([0-9]{3})\.(conversation_text\.digital_human|digital_human|avatar_performance"
                     r"|general_interaction|instruction_edit|reference_edit)\Z")
V1_KEYS = {"schema_version", "dataset_version", "base_id", "case_id", "source_family_id", "split", "tier",
           "title", "difficulty", "tags", "duration_ms", "readiness", "group", "task_family", "scenario",
           "input_profile", "initialization", "asset_ids", "events"}
SCHEMA = Path(__file__).with_name("session.schema.json")


@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft202012Validator
    return Draft202012Validator(json.loads(SCHEMA.read_text()))


def is_v1(session) -> bool:
    return isinstance(session, dict) and session.get("dataset_version") == DATASET_VERSION


def _identity(session):
    if not isinstance(session, dict) or set(session) != V1_KEYS:
        raise ValueError("v1 session has missing or unexpected fields")
    match = CASE_ID.fullmatch(session.get("case_id", ""))
    if (session["dataset_version"] != DATASET_VERSION or session["schema_version"] != "1" or not match
            or match.group(1) != session["base_id"] or session["split"] not in SPLIT_TIERS
            or session["tier"] not in SPLIT_TIERS[session["split"]] or session["group"] not in GROUPS
            or session["readiness"] not in READINESS[session["split"]]):
        raise ValueError("v1 session identity, split, tier or readiness is invalid")
    group_suffix = match.group(2).split(".")[-1]
    if group_suffix != session["group"]:
        raise ValueError("v1 case_id suffix disagrees with group")


def _legacy_envelope(session):
    """Pre-v1-shaped copy so the frozen schema checks the shared structure."""
    env = deepcopy(session)
    del env["tier"]
    env["readiness"] = "design_only"
    env["split"] = "public_dev"
    if session["group"] == "avatar_performance":
        env["dataset_version"] = "streameval_avatar_v0.1"
        env["base_id"] = "061"
        env["case_id"] = "sa.ap01.061.avatar_performance"
    else:
        env["dataset_version"] = "streameval_v0.2"
        env["case_id"] = session["case_id"].replace("sa.v1.", "sa.v02.", 1)
    return env


def _reference_session(session, env):
    init = session["initialization"]
    if set(init) != {"source_video_asset_id", "source_delivery", "reference"} or init["source_delivery"] != "realtime_frames":
        raise ValueError("v1 reference editing needs source video plus a preloaded reference")
    validate_payload(init["reference"])
    if init["reference"]["operation"] != "set":
        raise ValueError("The preloaded reference must be an explicit set")
    active = {r["target_slot"] for r in init["reference"]["references"]}
    used = {init["source_video_asset_id"]} | {r["asset_id"] for r in init["reference"]["references"]}
    previous = 0
    for event, shell in zip(session["events"], env["events"]):
        payload, release = event["payload"], event["release"]
        validate_payload(payload)
        if set(release) != {"at_ms"} or not previous < release["at_ms"] < session["duration_ms"]:
            raise ValueError("v1 reference controls need increasing fixed release times after t0")
        previous = release["at_ms"]
        slots = {r["target_slot"] for r in payload["references"]}
        used |= {r["asset_id"] for r in payload["references"]}
        op = payload["operation"]
        if op == "set":
            active = slots
        elif op == "patch":
            active |= slots
        elif op == "remove":
            if set(payload["remove"]) - active:
                raise ValueError("Removal references a slot that is not active")
            active -= set(payload["remove"])
        else:
            active = set()
        shell["payload"] = {"type": "reference_control", "asset_id": init["source_video_asset_id"],
                            "scope": payload["scope"]}
    if set(session["asset_ids"]) != used or len(session["asset_ids"]) != len(used):
        raise ValueError("v1 reference asset inventory must match source and released references")
    env["input_profile"] = "edit_reference"
    env["initialization"] = {"source_video_asset_id": init["source_video_asset_id"], "source_delivery": "realtime_frames"}


def validate_session(session):
    """Raise ValueError on any defect; return None on success."""
    _identity(session)
    env = _legacy_envelope(session)
    if session["input_profile"] == PROFILE_REFERENCE:
        if session["group"] != "reference_edit":
            raise ValueError("Preloaded-reference profile belongs to reference_edit")
        _reference_session(session, env)
    elif session["group"] == "reference_edit":
        raise ValueError("v1 reference editing must use the preloaded-reference profile")
    defects = list(_validator().iter_errors(env))
    if defects:
        where = ".".join(map(str, defects[0].absolute_path)) or "root"
        raise ValueError(f"v1 session envelope rejected at {where}: {defects[0].message[:160]}")
    errors = input_contract.validate_profile_session(env)
    if errors:
        raise ValueError("; ".join(errors))
