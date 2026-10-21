"""Pure input-semantic contracts; no vendor credentials, I/O or model calls.

Input profiles describe what an input means, independently of HTTP/WS/WebRTC: which event types a session of
the profile may release, what it is initialized with and where its latency is measured. The validators return
lists of defects for a payload, a session or the speech timing of an avatar-performance session.
"""
from __future__ import annotations

import re


def _profile(group, events, outputs, purpose, initialization, required_initialization=(),
             *, measurement_boundary):
    return {"group": group, "task_family": "streaming_editing" if group in {
        "reference_edit", "instruction_edit"} else "streaming_interaction",
        "event_types": list(events), "output_modalities": list(outputs), "purpose": purpose,
        "initialization_fields": list(initialization),
        "required_initialization": list(required_initialization),
        "measurement_boundary": {"clock": "evaluator_monotonic_with_media_pts_alignment",
                                 "include_all_system_components": True, **measurement_boundary},
        "continuous_input_required": True,
        "transport": "orthogonal_to_input_semantics"}


PROTOCOL_PROFILES = {
    "conversation_audio": _profile("digital_human", ["user_audio"], ["audio", "video"],
        "understand_user_and_generate_response", ["portrait_asset_id", "persona", "known_facts"],
        ["portrait_asset_id", "persona", "known_facts"], measurement_boundary={
            "origin": "end_of_required_user_turn_at_evaluator_release",
            "endpoint": "first_presented_substantive_response",
            "separate_modality_timestamps": ["audio", "video"],
            "excluded_origins": ["asr_completion", "dialogue_completion", "tts_completion"]}),
    "conversation_text": _profile("digital_human", ["user_prompt"], ["audio", "video"],
        "understand_user_and_generate_response", ["portrait_asset_id", "persona", "known_facts"],
        ["portrait_asset_id", "persona", "known_facts"], measurement_boundary={
            "origin": "user_prompt_release_at_evaluator",
            "endpoint": "first_presented_substantive_response",
            "separate_modality_timestamps": ["audio", "video"]}),
    "avatar_speech_audio": _profile("avatar_diagnostic", ["speech_audio"], ["video"],
        "render_supplied_speech_without_answering", ["portrait_asset_id"], ["portrait_asset_id"],
        measurement_boundary={"origin": "first_renderable_speech_audio_chunk_release",
            "endpoint": "first_presented_avatar_frame_corresponding_to_supplied_audio_pts",
            "secondary_measurements": ["render_lag_per_audio_pts", "av_sync_when_audio_is_presented"],
            "excluded_origins": ["complete_speech_audio_end"]}),
    "avatar_speech_text": _profile("avatar_diagnostic", ["speech_text"], ["audio", "video"],
        "speak_supplied_text_without_answering", ["portrait_asset_id"], ["portrait_asset_id"],
        measurement_boundary={"origin": "speech_text_release_at_evaluator",
            "endpoint": "first_presented_supplied_speech_and_corresponding_avatar_frame",
            "separate_modality_timestamps": ["audio", "video"], "includes_tts_when_used": True}),
    "general_text_to_video": _profile("general_interaction", ["user_prompt", "action", "camera_control"],
        ["video"], "respond_to_live_visual_generation_controls", ["prompt", "reference_asset_ids"], ["prompt"],
        measurement_boundary={"origin": "current_control_release_at_evaluator",
            "endpoint": "first_presented_effective_visual_change",
            "secondary_measurements": ["stable_effect_latency"], "source_age": "not_applicable"}),
    "general_image_text_to_video": _profile("general_interaction",
        ["user_prompt", "action", "camera_control", "reference_update"], ["video"],
        "respond_to_live_visual_generation_controls", ["prompt", "reference_asset_ids"], ["prompt"],
        measurement_boundary={"origin": "current_control_or_reference_release_at_evaluator",
            "endpoint": "first_presented_effective_visual_change",
            "secondary_measurements": ["stable_effect_latency"], "includes_reference_upload": True,
            "source_age": "not_applicable"}),
    "edit_reference": _profile("reference_edit", ["reference_control"], ["video"],
        "edit_continuous_source_from_released_reference", ["source_video_asset_id", "source_delivery"],
        ["source_video_asset_id", "source_delivery"], measurement_boundary={
            "origin": "current_reference_control_release_at_evaluator",
            "endpoint": "first_presented_effective_edit_on_corresponding_source_content",
            "secondary_measurements": ["stable_effect_latency", "source_age_from_source_pts"],
            "includes_reference_upload": True}),
    "edit_instruction": _profile("instruction_edit", ["text_control"], ["video"],
        "edit_continuous_source_from_released_instruction", ["source_video_asset_id", "source_delivery"],
        ["source_video_asset_id", "source_delivery"], measurement_boundary={
            "origin": "current_text_control_release_at_evaluator",
            "endpoint": "first_presented_effective_edit_on_corresponding_source_content",
            "secondary_measurements": ["stable_effect_latency", "source_age_from_source_pts"]}),
}

# Profiles added after PROTOCOL_PROFILES was frozen; kept in their own table so the base set stays unchanged.
EXTENSION_PROFILES = {
    "avatar_performance_audio": _profile("avatar_performance", ["speech_audio", "user_prompt"],
        ["video"], "render_supplied_speech_and_follow_online_visual_performance_controls",
        ["portrait_asset_id", "rendering_context"], ["portrait_asset_id", "rendering_context"],
        measurement_boundary={
            "origin": "per_event_semantics_do_not_pool_audio_and_visual_control_latency",
            "endpoint": "presented_avatar_behavior_with_verified_audio_pts_alignment",
            "event_boundaries": {
                "speech_audio": {"origin": "first_renderable_speech_audio_chunk_release",
                    "endpoint": "first_presented_avatar_frame_corresponding_to_supplied_audio_pts",
                    "response_speed_denominator": False,
                    "measurements": ["render_lag_per_audio_pts", "av_sync_when_audio_is_presented"],
                    "excluded_origins": ["complete_speech_audio_end", "end_of_user_turn"]},
                "user_prompt": {"origin": "current_visual_control_release_at_evaluator",
                    "endpoint": "checkpoint_defined_action_completion_or_stable_performance_effect",
                    "response_speed_denominator": True,
                    "semantic_role": "visual_performance_control_not_dialogue"}},
            "audio_output": "input_pcm_is_evaluator_presented_reference_not_required_generated_speech"}),
}


def get_profile(profile_id):
    """Resolve a legacy or explicitly versioned extension profile."""
    if not isinstance(profile_id, str):
        return None
    return PROTOCOL_PROFILES.get(profile_id) or EXTENSION_PROFILES.get(profile_id)

# This allowlist also prevents judge answers, transcripts and future plans riding
# alongside a legitimate payload.
PAYLOAD_FIELDS = {
    "user_audio": {"type", "asset_id"}, "speech_audio": {"type", "asset_id"},
    "user_prompt": {"type", "text"}, "speech_text": {"type", "text"},
    "text_control": {"type", "text"},
    "action": {"type", "action", "phase", "duration_ms"},
    "camera_control": {"type", "text"},
    "reference_control": {"type", "asset_id", "scope"},
    "reference_update": {"type", "asset_id", "instruction"},
}
ACTION_NAMES = {"move_forward", "move_backward", "turn_left", "turn_right",
                "strafe_left", "strafe_right", "jump", "interact"}
ACTION_PHASES = {"start", "stop", "pulse"}
MIN_ACTION_DURATION_MS = 1
MAX_ACTION_DURATION_MS = 10000


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _milliseconds(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_performance_audio_timing(session, audio_budgets_ms, measured_audio_durations_ns=None):
    """Check private authored budgets, optionally against measured audio lengths.

    Maps use event IDs. Omitting measured lengths is design validation only;
    supplying authored budgets as if measured is not a valid runtime probe.
    """
    if not isinstance(session, dict) or session.get("input_profile") != "avatar_performance_audio":
        return ["performance audio timing requires avatar_performance_audio"]
    if not _milliseconds(session.get("duration_ms")) or session["duration_ms"] == 0:
        return ["performance audio timing requires a positive session duration"]
    events = session.get("events")
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        return ["performance audio timing requires events"]
    speech = [e for e in events if isinstance(e.get("payload"), dict) and e["payload"].get("type") == "speech_audio"]
    ids = [e.get("event_id") for e in speech]
    if not ids or any(not _nonempty(i) for i in ids) or len(set(ids)) != len(ids):
        return ["performance requires distinct speech event IDs"]
    if (not isinstance(audio_budgets_ms, dict) or set(audio_budgets_ms) != set(ids) or
            any(not _milliseconds(v) or v == 0 for v in audio_budgets_ms.values())):
        return ["positive authored audio budgets are required for every speech event only"]
    measured = measured_audio_durations_ns
    if measured is not None and (not isinstance(measured, dict) or set(measured) != set(ids) or
            any(type(v) is not int or v <= 0 for v in measured.values())):
        return ["positive measured audio nanoseconds are required for every speech event only"]
    errors, intervals = [], []
    for event in speech:
        at = event.get("release", {}).get("at_ms") if isinstance(event.get("release"), dict) else None
        eid = event["event_id"]
        if not _milliseconds(at) or set(event["release"]) != {"at_ms"}:
            errors.append(f"{eid}: speech requires a fixed release time")
            continue
        end = at + audio_budgets_ms[eid]
        if end > session["duration_ms"]:
            errors.append(f"{eid}: audio budget exceeds session end")
        if measured is not None and measured[eid] > audio_budgets_ms[eid] * 1_000_000:
            errors.append(f"{eid}: measured audio exceeds authored budget")
        intervals.append((at, end, eid))
    intervals.sort()
    for prior, current in zip(intervals, intervals[1:]):
        if prior[1] > current[0]:
            errors.append(f"{current[2]}: speech budgets overlap on the single PCM timeline")
    return errors


def validate_payload(payload):
    if (not isinstance(payload, dict) or not _nonempty(payload.get("type"))
            or payload["type"] not in PAYLOAD_FIELDS):
        return ["unknown payload semantic type"]
    fields = PAYLOAD_FIELDS[payload["type"]]
    if set(payload) != fields:
        return ["payload fields must match semantic allowlist; no answers, transcripts or extra fields"]
    if payload["type"] == "action":
        if (not _nonempty(payload.get("action")) or payload["action"] not in ACTION_NAMES
                or not _nonempty(payload.get("phase")) or payload["phase"] not in ACTION_PHASES):
            return ["unknown structured action or phase"]
        duration = payload.get("duration_ms")
        if not _milliseconds(duration) or (duration != 0 if payload["phase"] == "stop"
                                          else not MIN_ACTION_DURATION_MS <= duration <= MAX_ACTION_DURATION_MS):
            return ["stop requires duration 0; start/pulse require a bounded positive duration"]
        return []
    return [] if all(_nonempty(payload[k]) for k in fields) else ["payload values must be nonempty strings"]


def validate_profile_session(session):
    """Return defects, rather than guessing a profile from an API field name."""
    if not isinstance(session, dict):
        return ["session must be an object"]
    profile_id = session.get("input_profile")
    profile = get_profile(profile_id)
    if profile is None:
        return ["unknown or missing input_profile"]
    errors = []
    if session.get("group") != profile["group"]:
        errors.append("input_profile cannot be pooled into this group")
    if session.get("task_family") != profile["task_family"]:
        errors.append("input_profile does not match task_family")
    initial = session.get("initialization", {})
    if not isinstance(initial, dict):
        return errors + ["initialization must be an object"]
    if set(initial) - set(profile["initialization_fields"]):
        errors.append("initialization contains non-profile fields")
    if set(profile["required_initialization"]) - set(initial):
        errors.append("initialization lacks required profile inputs")
    for key, value in initial.items():
        if key == "known_facts":
            if not isinstance(value, dict):
                errors.append("known_facts must be an object")
        elif key == "reference_asset_ids":
            if not isinstance(value, list) or any(not _nonempty(x) for x in value) or len(value) != len(set(value)):
                errors.append("initial reference_asset_ids must be distinct strings")
        elif not _nonempty(value):
            errors.append("initialization strings must be nonempty")
    if "source_delivery" in initial and initial["source_delivery"] != "realtime_frames":
        errors.append("source video must be released as realtime frames")
    initial_ids = _initial_asset_ids(initial)
    if session["input_profile"] == "general_text_to_video" and initial_ids:
        errors.append("text-only general profile cannot receive initial images")
    asset_ids = session.get("asset_ids", [])
    if not isinstance(asset_ids, list) or any(not _nonempty(x) for x in asset_ids) or len(set(asset_ids)) != len(asset_ids):
        return errors + ["asset_ids must be distinct strings"]
    if set(initial_ids) - set(asset_ids):
        errors.append("initial assets are absent from session asset_ids")
    events = session.get("events")
    if not isinstance(events, list) or not events:
        return errors + ["events must be a nonempty list"]
    seen, active_actions, preceding_payloads = set(), {}, {}
    for event in events:
        if not isinstance(event, dict) or not _nonempty(event.get("event_id")):
            errors.append("event requires a nonempty event_id")
            continue
        eid = event["event_id"]
        if eid in seen:
            errors.append("duplicate event_id")
        seen.add(eid)
        payload = event.get("payload")
        defects = validate_payload(payload)
        errors.extend(f"{eid}: {defect}" for defect in defects)
        if not defects:
            if payload["type"] not in profile["event_types"]:
                errors.append(f"{eid}: payload meaning is incompatible with input_profile")
            if "asset_id" in payload and payload["asset_id"] not in asset_ids:
                errors.append(f"{eid}: payload asset is absent from session asset_ids")
        release = event.get("release", {})
        if not isinstance(release, dict) or (("at_ms" in release) == ("trigger" in release)):
            errors.append(f"{eid}: release must select one fixed time or trigger")
        elif "at_ms" in release and not _milliseconds(release["at_ms"]):
            errors.append(f"{eid}: invalid release time")
        elif "trigger" in release and (not isinstance(release["trigger"], dict)
                                        or release["trigger"].get("on_timeout") != "fail_opportunity"):
            errors.append(f"{eid}: unfired trigger must retain its failure opportunity")
        if (profile_id in {"conversation_audio", "conversation_text"} and isinstance(release, dict)
                and isinstance(release.get("trigger"), dict)):
            trigger = release["trigger"]
            expected_type = "user_prompt" if profile_id == "conversation_text" else "user_audio"
            expected_origin = "user_prompt_release" if profile_id == "conversation_text" else "end_of_referenced_user_audio"
            if (set(trigger) != {"type", "after_turn", "offset_ms", "timeout_ms", "on_timeout", "timeout_origin"}
                    or trigger.get("type") != "response_started"
                    or trigger.get("timeout_origin") != expected_origin
                    or not _milliseconds(trigger.get("offset_ms"))
                    or not _milliseconds(trigger.get("timeout_ms")) or trigger["timeout_ms"] == 0):
                errors.append(f"{eid}: conversation trigger requires complete response_started timing with {expected_origin}")
            parent = trigger.get("after_turn")
            if not _nonempty(parent) or preceding_payloads.get(parent) != expected_type:
                errors.append(f"{eid}: conversation condition must reference an earlier {expected_type} event")
        preceding_payloads[eid] = payload.get("type") if isinstance(payload, dict) else None
        if not defects and payload["type"] == "action":
            at = release.get("at_ms") if isinstance(release, dict) else None
            if not _milliseconds(at):
                errors.append(f"{eid}: structured actions require fixed release times")
                continue
            action, phase = payload["action"], payload["phase"]
            if phase == "start":
                if action in active_actions:
                    errors.append(f"{eid}: action is already held")
                active_actions[action] = (at, at + payload["duration_ms"])
            elif phase == "stop":
                if action not in active_actions:
                    errors.append(f"{eid}: cannot release an action that is not held")
                else:
                    start, limit = active_actions.pop(action)
                    if not start < at <= limit:
                        errors.append(f"{eid}: action release exceeds its bounded hold")
            elif action in active_actions:
                errors.append(f"{eid}: pulse conflicts with an active hold")
            if _milliseconds(session.get("duration_ms")) and at + payload["duration_ms"] > session["duration_ms"]:
                errors.append(f"{eid}: action extends beyond session end")
    if active_actions:
        errors.append("held actions require explicit stop events before session end")
    if profile_id == "avatar_performance_audio":
        if (session.get("dataset_version") != "streameval_avatar_v0.1" or
                not isinstance(session.get("base_id"), str) or not re.fullmatch(r"06[1-9]|07[0-9]|080", session["base_id"]) or
                session.get("case_id") != f"sa.ap01.{session.get('base_id')}.avatar_performance" or
                session.get("scenario") != "avatar_performance"):
            errors.append("performance extension requires its own dataset, IDs 061–080, and scenario")
        times = [e.get("release", {}).get("at_ms") if isinstance(e, dict) and isinstance(e.get("release"), dict) else None for e in events]
        duration = session.get("duration_ms")
        if (not _milliseconds(duration) or not 15000 <= duration <= 300000 or
                any(not _milliseconds(t) or t >= duration for t in times) or
                any(set(e.get("release", {})) != {"at_ms"} for e in events if isinstance(e, dict))):
            errors.append("performance events require fixed release times before the session deadline")
        elif times != sorted(times):
            errors.append("performance events must be ordered by fixed release time")
        kinds = {e["payload"]["type"] for e in events if isinstance(e, dict) and isinstance(e.get("payload"), dict)
                 and isinstance(e["payload"].get("type"), str)}
        if kinds != {"speech_audio", "user_prompt"}:
            errors.append("performance sessions require supplied speech and online visual controls")
    return errors


def _initial_asset_ids(initial):
    ids = [initial[key] for key in ("portrait_asset_id", "source_video_asset_id")
           if _nonempty(initial.get(key))]
    if isinstance(initial.get("reference_asset_ids"), list):
        ids.extend(x for x in initial["reference_asset_ids"] if _nonempty(x))
    return list(dict.fromkeys(ids))
