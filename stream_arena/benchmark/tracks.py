"""Tracks and the operations their events need.

A *group* is one of the five benchmark tasks; a *track* is a group run under one input profile (the leaderboard
ranks within a track). A model declares the tracks it runs and, per track, the operations it supports; every
event whose operation is not supported is still released on time, recorded as ``unsupported`` without reaching
the model, and scored as a failure.

    track                      group                input_profile                    populations
    digital_human_audio        digital_human        conversation_audio               core, extension
    digital_human_text         digital_human        conversation_text                text, (text_extension)
    avatar_performance         avatar_performance   avatar_performance_audio         core, extension
    general_interaction        general_interaction  general_text_to_video            core
    general_interaction_image  general_interaction  general_image_text_to_video      extension
    instruction_edit           instruction_edit     edit_instruction                 core, extension
    reference_edit             reference_edit       edit_reference_v1_preloaded_a    core, extension

Operations (``event_op``): ``user_audio`` / ``speech_audio`` arrive as the 16 kHz audio input stream;
``user_prompt`` / ``text_control`` / ``camera_control`` as a text event; ``reference_update`` as a reference set on
slot ``subject``; ``reference:<set|patch|remove|reset>`` as a reference-control event; ``action`` (structured game
control: start / stop / pulse of a named action) as a control event. ``action`` is opt-in (OPTIONAL_OPS): a model
receives it only if its registry ``ops`` list it.
"""
from __future__ import annotations

# kind: avatar (portrait or provider avatar, speech), generation (text/image to video), editing (live source video)
TRACKS = {
    "digital_human_audio": {"group": "digital_human", "input_profile": "conversation_audio",
                            "populations": ("core", "extension"), "suffix": "audio", "kind": "avatar"},
    "digital_human_text": {"group": "digital_human", "input_profile": "conversation_text",
                           "populations": ("text",), "suffix": "text", "kind": "avatar"},
    "avatar_performance": {"group": "avatar_performance", "input_profile": "avatar_performance_audio",
                           "populations": ("core", "extension"), "suffix": "performance", "kind": "avatar"},
    "general_interaction": {"group": "general_interaction", "input_profile": "general_text_to_video",
                            "populations": ("core",), "suffix": "t2v", "kind": "generation"},
    "general_interaction_image": {"group": "general_interaction", "input_profile": "general_image_text_to_video",
                                  "populations": ("extension",), "suffix": "i2v", "kind": "generation"},
    "instruction_edit": {"group": "instruction_edit", "input_profile": "edit_instruction",
                         "populations": ("core", "extension"), "suffix": "instruction", "kind": "editing"},
    "reference_edit": {"group": "reference_edit", "input_profile": "edit_reference_v1_preloaded_a",
                       "populations": ("core", "extension"), "suffix": "reference", "kind": "editing"},
}
TRACK_OPS = {
    "digital_human_audio": ["user_audio"],
    "digital_human_text": ["user_prompt"],
    "avatar_performance": ["user_prompt", "speech_audio"],
    "general_interaction": ["user_prompt", "camera_control"],
    "general_interaction_image": ["user_prompt", "camera_control", "reference_update"],
    "instruction_edit": ["text_control"],
    "reference_edit": ["reference:set", "reference:patch", "reference:remove", "reference:reset"],
}
OPTIONAL_OPS = {"general_interaction": ["action"], "general_interaction_image": ["action"]}
# output frames kept for judging: the first received frame of every bin of this many ms of the case clock
SAMPLE_MS = {"digital_human": 500, "avatar_performance": 250, "general_interaction": 250,
             "instruction_edit": 250, "reference_edit": 250}


def event_op(payload: dict) -> str:
    kind = payload["type"]
    return "reference:" + payload["operation"] if kind == "reference_control" else kind
