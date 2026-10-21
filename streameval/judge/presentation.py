"""Client-presentation timing from recorded sink receipts.

The client presentation time of a received media sample is the host monotonic time at which the evaluator's client
delivered the decoded frame containing it to its sink, plus the sample's offset inside that frame, in the same
process clock the scheduler uses for input release (docs/evaluation.md). Display vsync and audio-device buffering after
the sink are not included and are reported as the measurement resolution (one frame).

This module is pure: it turns already-recorded receipts into onset times and presentation receipts shaped for
scoring.accounting.account_session. It never decides whether a response was *correct*; that verdict comes from the
judge.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

NS = 1_000_000_000


@dataclass(frozen=True)
class AudioFrame:
    host_ns: int      # sink delivery time of the frame
    first_sample: int  # index of the frame's first sample in the recorded track
    samples: int


def audio_clock(arrivals: list[tuple[int, int]]) -> list[AudioFrame]:
    """[(host_ns, samples_in_frame)] in arrival order -> cumulative sample clock."""
    frames, offset = [], 0
    for host_ns, samples in arrivals:
        if type(host_ns) is not int or type(samples) is not int or samples <= 0:
            raise ValueError("audio receipts need integer host ns and positive sample counts")
        frames.append(AudioFrame(host_ns, offset, samples))
        offset += samples
    if any(b.host_ns < a.host_ns for a, b in zip(frames, frames[1:])):
        raise ValueError("audio receipts must be in non-decreasing host time")
    return frames


def sample_host_ns(frames: list[AudioFrame], index: int, rate: int) -> int:
    """Presentation time of one sample: its frame's delivery + in-frame offset."""
    lo, hi = 0, len(frames) - 1
    if not frames or index < 0 or index >= frames[-1].first_sample + frames[-1].samples:
        raise ValueError("sample index outside the recorded track")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if frames[mid].first_sample <= index:
            lo = mid
        else:
            hi = mid - 1
    f = frames[lo]
    return f.host_ns + round((index - f.first_sample) * NS / rate)


def voiced_segments(pcm: np.ndarray, rate: int, *, window_ms: int = 20, floor_dbfs: float = -50.0,
                    relative_db: float = 18.0, min_voiced_ms: int = 300,
                    max_gap_ms: int = 250) -> list[tuple[int, int]]:
    """Speech-like segments as (start_sample, end_sample).

    Window RMS above max(floor_dbfs, 10th-percentile level + relative_db) is
    voiced; voiced windows separated by <= max_gap_ms merge; a segment needs
    >= min_voiced_ms of voiced windows. Parameters are recorded with results.
    """
    win = max(1, rate * window_ms // 1000)
    n = len(pcm) // win
    if n == 0:
        return []
    frames = pcm[: n * win].astype(np.float64).reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1)) / 32768.0
    level = 20 * np.log10(np.maximum(rms, 1e-9))
    threshold = max(floor_dbfs, float(np.percentile(level, 10)) + relative_db)
    voiced = level > threshold
    segments, start, last, count = [], None, None, 0
    gap = max(1, max_gap_ms // window_ms)
    for i, v in enumerate(voiced):
        if not v:
            continue
        if start is None or i - last > gap:
            if start is not None and count * window_ms >= min_voiced_ms:
                segments.append((start * win, (last + 1) * win))
            start, count = i, 0
        last, count = i, count + 1
    if start is not None and count * window_ms >= min_voiced_ms:
        segments.append((start * win, (last + 1) * win))
    return segments


TURN_GAP_NS = 600_000_000    # silence that separates a new turn from continuing speech
MIN_RESPONSE_NS = 200_000_000  # no genuine end-of-turn response starts faster than this


def first_onset_after(segments_ns: list[tuple[int, int]], after_ns: int, before_ns: int | None):
    """First NEW avatar turn starting in [after_ns, before_ns).

    Returns ("onset", (start, end)) or, when timing cannot be attributed,
    ("overlap" | "continuation" | "talk_over" | "none", None):
    * overlap: speech is already running at after_ns;
    * continuation: the candidate follows earlier speech by < TURN_GAP_NS, so it
      may be the tail of a previous utterance (e.g. a greeting over the user);
    * talk_over: the candidate starts < MIN_RESPONSE_NS after after_ns.
    These stay undetermined; they are never reported as fast responses."""
    previous_end = None
    for start, end in segments_ns:
        if start < after_ns < end:
            return "overlap", None
        if start >= after_ns and (before_ns is None or start < before_ns):
            if previous_end is not None and start - previous_end < TURN_GAP_NS:
                return "continuation", None
            if start - after_ns < MIN_RESPONSE_NS:
                return "talk_over", None
            return "onset", (start, end)
        if before_ns is not None and start >= before_ns:
            break
        previous_end = end
    return "none", None


def first_sustained(samples: list[tuple[int, bool]], stable_ns: int):
    """Visual effects: first judged-true sample that stays true for stable_ns.

    samples are (presentation_host_ns, judged_true) in time order. Returns
    (first_valid_ns, verified_ns) or None. Resolution = sample spacing."""
    run_start = None
    for host_ns, ok in samples:
        if ok:
            run_start = host_ns if run_start is None else run_start
            if host_ns - run_start >= stable_ns:
                return run_start, host_ns
        else:
            run_start = None
    return None


def nearest_rank(values, q):
    values = sorted(values)
    return values[math.ceil(q * len(values)) - 1] if values else None


def dialogue_receipt(*, clock_domain_id: str, origin_kind: str, origin_ns: int, onset_ns: int, end_ns: int,
                     evidence: dict, input_end: dict | None) -> dict:
    """Presentation receipt in scoring.accounting.account_session's exact schema."""
    return {"basis": "client_presentation", "clock_domain_id": clock_domain_id, "origin_kind": origin_kind,
            "origin_at_ns": origin_ns, "target": "first_substantive_response", "effect_started_at_ns": None,
            "first_valid_at_ns": onset_ns, "verified_at_ns": end_ns, "evidence": evidence, "input_end": input_end}
