"""Synthetic media for installation checks and tests: one valid v1 case per track kind, generated media, no real content.

    build(root) -> Benchmark with cases 901 (instruction_edit), 902 (digital_human audio), 903 (general t2v),
                   904 (reference_edit), 905 (avatar_performance); media under root/data.
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

from streameval.benchmark.dataset import Benchmark
from streameval.runtime.media import ffmpeg
from streameval.util import sha256_file, write_json, write_jsonl

CASE = {"instruction_edit": "sa.v1.901.instruction_edit", "digital_human_audio": "sa.v1.902.digital_human",
        "general_interaction": "sa.v1.903.general_interaction", "reference_edit": "sa.v1.904.reference_edit",
        "avatar_performance": "sa.v1.905.avatar_performance"}


def _case(n, group, profile, init, assets, events, family="streaming_interaction", scenario="synthetic"):
    return {"schema_version": "1", "dataset_version": "streameval_v1.0", "base_id": f"{n:03d}",
            "case_id": f"sa.v1.{n:03d}.{group}", "source_family_id": f"synthetic.{n:03d}", "split": "public_dev",
            "tier": "development", "title": "synthetic test case", "difficulty": "atomic", "tags": ["synthetic"],
            "duration_ms": 15000, "readiness": "automated_gate_passed", "group": group, "task_family": family,
            "scenario": scenario, "input_profile": profile, "initialization": init, "asset_ids": assets,
            "events": events}


def _ev(eid, at, **payload):
    return {"event_id": eid, "payload": payload, "release": {"at_ms": at}}


def cases() -> list[dict]:
    ref = lambda aid, scope: {"asset_id": aid, "scope": scope, "target_slot": "clothing"}
    return [
        _case(901, "instruction_edit", "edit_instruction",
              {"source_delivery": "realtime_frames", "source_video_asset_id": "sa.source_video.901"},
              ["sa.source_video.901"],
              [_ev("e000", 0, type="text_control", text="Make the square red."),
               _ev("e001", 5000, type="text_control", text="Reset the square.")],
              family="streaming_editing", scenario="continuous_video_editing"),
        _case(902, "digital_human", "conversation_audio",
              {"persona": "You are a museum guide.", "known_facts": {"closing_time": "17:00"},
               "portrait_asset_id": "sa.portrait.902"},
              ["sa.portrait.902", "sa.user_audio.902.u1"],
              [_ev("u1", 0, type="user_audio", asset_id="sa.user_audio.902.u1")], scenario="conversational_avatar"),
        _case(903, "general_interaction", "general_text_to_video",
              {"prompt": "A red robot on a grid.", "reference_asset_ids": []}, [],
              [_ev("e000", 0, type="user_prompt", text="Walk forward."),
               _ev("e001", 5000, type="camera_control", text="Pan left.")], scenario="game_control"),
        _case(904, "reference_edit", "edit_reference_v1_preloaded_a",
              {"source_delivery": "realtime_frames", "source_video_asset_id": "sa.source_video.901",
               "reference": {"type": "reference_control", "operation": "set", "remove": [], "scope": "Set A.",
                             "references": [ref("sa.clean_reference_image.904.a", "The square")]}},
              ["sa.source_video.901", "sa.clean_reference_image.904.a", "sa.clean_reference_image.904.b"],
              [_ev("e000", 5000, type="reference_control", operation="set", remove=[], scope="Set B.",
                   references=[ref("sa.clean_reference_image.904.b", "The square")]),
               _ev("e001", 10000, type="reference_control", operation="reset", references=[], remove=[],
                   scope="Reset all.")],
              family="streaming_editing", scenario="continuous_video_editing"),
        _case(905, "avatar_performance", "avatar_performance_audio",
              {"portrait_asset_id": "sa.performance_portrait.905", "rendering_context": "Fixed camera."},
              ["sa.performance_portrait.905", "sa.speech_audio.905.s01"],
              [_ev("e000", 0, type="user_prompt", text="Raise the right hand."),
               _ev("e001", 1000, type="speech_audio", asset_id="sa.speech_audio.905.s01")],
              scenario="avatar_performance"),
    ]


def oracles(sessions: list[dict]) -> list[dict]:
    out = []
    for s in sessions:
        cps = []
        for e in s["events"]:
            at, kind = e["release"]["at_ms"], e["payload"]["type"]
            if kind == "speech_audio":
                cps.append({"event_id": e["event_id"], "response_policy": "speech_render", "latency_opportunity": False,
                            "effect_kind": "supplied_speech_alignment", "audio_budget_ms": 1500,
                            "audio_asset_id": e["payload"]["asset_id"], "planned_speaking_interval_ms": [at, at + 1500],
                            "check_window_ms": [at, at + 1500]})
            elif s["group"] == "avatar_performance":
                cps.append({"event_id": e["event_id"], "response_policy": "control_effect", "latency_opportunity": True,
                            "effect_kind": "control_effect", "valid_until_ms": 15000, "effect_deadline_ms": 3000,
                            "stable_window_ms": 2000, "check_window_ms": [at + 3000, 15000]})
            elif s["group"] == "general_interaction":
                end = next((x["release"]["at_ms"] for x in s["events"] if x["release"]["at_ms"] > at), 15000)
                cps.append({"event_id": e["event_id"], "response_policy": "control_effect", "latency_opportunity": True,
                            "effect_kind": "control_effect", "valid_until_ms": end, "effect_deadline_ms": 2000,
                            "stable_window_ms": 1000, "check_window_ms": [at + 2000, end],
                            "expected": ["The requested change is visible."], "must_hold": []})
            elif s["group"] == "digital_human":
                cps.append({"event_id": e["event_id"], "response_policy": "substantive_response",
                            "expected": "Says the gallery closes at 17:00."})
            else:
                cps.append({"event_id": e["event_id"], "expected": "The requested change is visible.",
                            **({"active_state": e["payload"]["operation"] != "reset"}
                               if s["group"] == "reference_edit" else {})})
        out.append({"case_id": s["case_id"], "source_family_id": s["source_family_id"],
                    "failed_opportunity": "keep_in_denominator", "checkpoints": cps})
    return out


def _wav(path: Path, seconds: float, freq: float = 220.0):
    t = np.arange(int(16000 * seconds)) / 16000
    pcm = (0.4 * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(pcm.tobytes())


def _image(path: Path, color, fmt: str):
    from PIL import Image
    Image.new("RGB", (256, 256), color).save(path, fmt)


def media(data: Path) -> dict:
    data.mkdir(parents=True, exist_ok=True)
    files = {
        "sa.source_video.901": ("source_video", "video/source_901.mp4"),
        "sa.portrait.902": ("image", "image/portrait_902.jpg"),
        "sa.user_audio.902.u1": ("audio", "audio/user_902_u1.wav"),
        "sa.clean_reference_image.904.a": ("image", "image/reference_904_a.png"),
        "sa.clean_reference_image.904.b": ("image", "image/reference_904_b.png"),
        "sa.performance_portrait.905": ("image", "image/portrait_905.jpg"),
        "sa.speech_audio.905.s01": ("audio", "audio/speech_905_s01.wav"),
    }
    for _, rel in files.values():
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25", "-t", "16",
                    "-c:v", "mpeg4", "-q:v", "5", str(data / files["sa.source_video.901"][1])], check=True)
    _image(data / files["sa.portrait.902"][1], (180, 140, 120), "JPEG")
    _image(data / files["sa.performance_portrait.905"][1], (120, 140, 180), "JPEG")
    _image(data / files["sa.clean_reference_image.904.a"][1], (200, 30, 30), "PNG")
    _image(data / files["sa.clean_reference_image.904.b"][1], (30, 30, 200), "PNG")
    _wav(data / files["sa.user_audio.902.u1"][1], 1.5)
    _wav(data / files["sa.speech_audio.905.s01"][1], 1.2, 330.0)
    return {aid: {"path": rel, "kind": kind, "sha256": sha256_file(data / rel), "bytes": (data / rel).stat().st_size}
            for aid, (kind, rel) in files.items()}


def build(root: Path) -> Benchmark:
    root = Path(root)
    sessions = cases()
    (root / "set").mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "set/cases.jsonl", sessions)
    write_jsonl(root / "set/oracles.jsonl", oracles(sessions))
    write_json(root / "set/manifest.json", {"dataset_version": "streameval_v1.0", "split": "public_dev",
                                            "assets": media(root / "data")})
    return Benchmark(root / "set", root / "data")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be absent or empty; generated examples never overwrite existing runs')
    benchmark = build(args.out)
    print(f'Created {len(benchmark.cases)} synthetic software-test cases at {benchmark.root}.')
    print('These examples are not paper conditions or model-quality evaluations.')


if __name__ == '__main__':
    main()
