"""The run-artifact contract: what an attempt folder must hold for the judge to read it.

Any way of running a model (the harness, or a backend with its own media stack such as a browser WebRTC SDK) is
accepted if its attempt folders pass ``validate_run``; the judge reads nothing else. docs/protocol.md documents
every field. Errors make the attempt unreadable; warnings are reported but do not block judging.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..benchmark.tracks import TRACKS
from ..util import digest, read_json, sha256_file

BINDING_KEYS = ("job_id", "case_id", "repeat", "system_id", "system_config", "session_sha256")
RESULT_KEYS = ("job_id", "case_id", "status", "scheduler", "dispatch", "unsupported_events", "clock_offset_ns")
SCHEDULER_KEYS = ("started_at_ns", "deadline_ns", "finished_at_ns", "opportunities")
DISPATCH_STATUS = ("sent", "unsupported", "failed")


def _rows(path: Path, errors: list) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            errors.append(f"{path.name}:{n} is not JSON")
    return rows


def validate_run(attempt: Path, case: dict | None = None) -> dict:
    """{"attempt", "started", "ok", "errors", "warnings", "counts"} for one attempt folder."""
    attempt = Path(attempt)
    errors, warnings, counts = [], [], {}
    report = {"attempt": str(attempt), "started": None, "ok": False, "errors": errors, "warnings": warnings,
              "counts": counts}
    for name in ("binding.json", "result.json"):
        if not (attempt / name).is_file():
            errors.append(f"{name} missing")
    if errors:
        return report
    binding, result = read_json(attempt / "binding.json"), read_json(attempt / "result.json")
    errors += [f"binding.json lacks {k}" for k in BINDING_KEYS if k not in binding]
    errors += [f"result.json lacks {k}" for k in RESULT_KEYS if k not in result]
    if errors:
        return report
    config = binding["system_config"]
    for key in ("group", "input_profile", "portrait_policy", "sampling"):
        if key not in config:
            errors.append(f"binding.system_config lacks {key}")
    if binding["job_id"] != result["job_id"] or binding["case_id"] != result["case_id"]:
        errors.append("binding.json and result.json name different jobs")
    if not binding["job_id"].startswith(binding["case_id"] + ".r"):
        errors.append("job_id must be <case_id>.rNN")
    if case is not None:
        if digest(case) != binding["session_sha256"]:
            errors.append("session_sha256 differs from the benchmark case (the case was modified)")
        if (case["group"], case["input_profile"]) != (config.get("group"), config.get("input_profile")):
            errors.append("the system's track does not run this case")
    if not (attempt / "runner.json").is_file():
        warnings.append("runner.json missing (a killed runner cannot be told apart from a model failure)")

    scheduler = result["scheduler"]
    missing = [k for k in SCHEDULER_KEYS if k not in scheduler]
    if missing:
        errors.append(f"result.scheduler lacks {missing}")
        return report
    started = scheduler["started_at_ns"] is not None
    report["started"] = started
    if result["status"] not in ("completed", "failed"):
        errors.append("result.status must be completed or failed")
    rows = {r["event_id"]: r for r in scheduler["opportunities"]}
    if case is not None and set(rows) != {e["event_id"] for e in case["events"]}:
        errors.append("scheduler opportunities differ from the case events")
    for eid, record in result["dispatch"].items():
        if eid not in rows:
            errors.append(f"dispatch {eid} is not an event of the case")
        elif rows[eid]["status"] != "released":
            errors.append(f"dispatch {eid} was never released")
        if record.get("status") not in DISPATCH_STATUS:
            errors.append(f"dispatch {eid} status must be one of {DISPATCH_STATUS}")
    for eid, row in rows.items():
        if row["status"] == "released" and eid not in result["dispatch"]:
            errors.append(f"released event {eid} has no dispatch record")
    counts["released"] = sum(r["status"] == "released" for r in rows.values())
    counts["unsupported"] = sum(d.get("status") == "unsupported" for d in result["dispatch"].values())
    if not started:
        report["ok"] = not errors
        return report

    browser = attempt / "browser"
    duration_ms = (scheduler["deadline_ns"] - scheduler["started_at_ns"]) / 1e6
    sample_ms = config["sampling"].get("frame_sample_ms")
    samples_path = browser / "frame_samples.jsonl"
    if not samples_path.is_file():
        errors.append("browser/frame_samples.jsonl missing")
    else:
        samples = _rows(samples_path, errors)
        counts["frame_samples"] = len(samples)
        last_bin = -1
        for row in samples:
            if any(k not in row for k in ("case_ms", "path", "sha256")):
                errors.append(f"frame sample lacks case_ms/path/sha256: {row}")
                continue
            path = browser / row["path"]
            if not path.is_file():
                errors.append(f"frame {row['path']} missing")
            elif sha256_file(path) != row["sha256"]:
                errors.append(f"frame {row['path']} sha256 mismatch")
            if not 0 <= row["case_ms"] <= duration_ms:
                errors.append(f"frame {row['path']} case_ms {row['case_ms']:.1f} outside the case")
            b = row.get("sample_bin")
            if b is not None:
                if b <= last_bin:
                    errors.append(f"frame sample bins must increase ({row['path']})")
                if sample_ms and not b * sample_ms <= row["case_ms"] < (b + 1) * sample_ms:
                    errors.append(f"frame {row['path']} is outside its sample bin")
                last_bin = b
        if not samples:
            warnings.append("no output frame was captured during the case")
    reference = (result.get("browser_start") or {}).get("reference_frame")
    if reference:
        path = browser / reference["path"]
        if not path.is_file() or sha256_file(path) != reference["sha256"]:
            errors.append("browser/reference_frame.jpg missing or differs from result.browser_start")
    blocks_path = browser / "remote_audio_blocks.jsonl"
    if blocks_path.is_file():
        blocks = _rows(blocks_path, errors)
        total = sum(b.get("samples", 0) for b in blocks)
        size = (browser / "remote_audio.s16le").stat().st_size if (browser / "remote_audio.s16le").is_file() else 0
        if size != 2 * total:
            errors.append(f"remote_audio.s16le holds {size // 2} samples, the blocks list {total}")
        pages = [b["page_ms"] for b in blocks if "page_ms" in b]
        if len(pages) != len(blocks):
            errors.append("every remote audio block needs page_ms")
        elif any(b < a for a, b in zip(pages, pages[1:])):
            errors.append("remote audio block page_ms must not decrease")
        counts["audio_seconds"] = round(total / 48000, 3)
    elif TRACKS.get(config.get("track"), {}).get("kind") == "avatar":
        warnings.append("avatar track without captured output audio")
    if config.get("group") in ("instruction_edit", "reference_edit"):
        source = browser / "source_frames.jsonl"
        if not source.is_file():
            errors.append("browser/source_frames.jsonl missing (editing cases pair output with source frames)")
        else:
            # frame rows carry presentation_ms and media_time; a row with neither is an annotation (e.g. a browser
            # backend's held_first_frame_repeats_before_epoch)
            src = [r for r in _rows(source, errors) if "presentation_ms" in r or "media_time" in r]
            counts["source_frames"] = len(src)
            if any("presentation_ms" not in r or "media_time" not in r for r in src):
                errors.append("source frame rows need presentation_ms and media_time")
            elif not src:
                errors.append("browser/source_frames.jsonl lists no pushed source frame")
    report["ok"] = not errors
    return report
