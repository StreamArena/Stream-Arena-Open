"""Trusted, clock-driven release of a v1 session's events.

A dedicated thread owns time and a private queue; potentially slow dispatch runs elsewhere. Only ``take()`` crosses
the model boundary, returning one ReleasedInput whose release instant has passed. Nothing here talks to a model,
detects media, judges or scores.

Release rules:

* fixed events (``release.at_ms``) become due at epoch + at_ms, where the epoch is the instant the model was
  declared ready; an event handed out after the session deadline is never delivered;
* conditional events (``release.trigger``) wait for a trusted response-start signal after the referenced turn,
  then release at observed + offset_ms; without one by input_end + timeout_ms they fail (``trigger_not_fired``).
  input_end is the release of the referenced ``user_prompt`` (conversation_text) or release + measured duration
  of the referenced ``user_audio``;
* avatar-performance speech must fit its window: release + measured duration may not pass the next speech
  event's release (or the session end).
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
import threading
import time
import uuid

from ..benchmark import protocols, session as v1_session
from ..benchmark.reference import validate_payload as validate_reference_payload
from ..util import digest

MS = 1_000_000


class UnsupportedInput(ValueError):
    """Explicitly accounted unsupported operation, never an implicit conversion."""


@dataclass(frozen=True)
class ReleasedInput:
    event_id: str
    profile: str
    payload: dict
    released_at_ns: int

    @classmethod
    def at_release(cls, *, event_id, profile, payload, now_ns, due_ns):
        if type(now_ns) is not int or type(due_ns) is not int or min(now_ns, due_ns) < 0 or now_ns < due_ns:
            raise ValueError("A future event cannot enter the model")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id is required")
        if profile == v1_session.PROFILE_REFERENCE:
            validate_reference_payload(payload)
            return cls(event_id, profile, deepcopy(payload), now_ns)
        definition = protocols.get_profile(profile)
        if definition is None:
            raise UnsupportedInput("Unknown input profile")
        kind = payload.get("type") if isinstance(payload, dict) else None
        if kind not in definition["event_types"]:
            raise UnsupportedInput("Input semantic does not belong to this profile")
        defects = protocols.validate_payload(payload)
        if defects:
            raise ValueError("; ".join(defects))
        return cls(event_id, profile, deepcopy(payload), now_ns)


def _timestamp(value):
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class TrustedResponseStart:
    """Evaluator assertion of the first presented substantive reply to a turn.

    Supply only from the harness's own observer (online VAD on received audio, or the first received frame),
    never a model packet, keyword match or oracle. All timestamps share the scheduler's clock.
    """
    after_turn: str
    observed_at_ns: int
    evidence_ref: str


class TrustedScheduler:
    """Single-use release engine; call start only after the model is ready.

    ``audio_durations_ns`` maps event IDs to measured source-audio durations; every condition's referenced user
    audio requires one. ``conversation_text`` sessions take neither durations nor budgets. Avatar-performance
    speech requires all measured lengths plus ``audio_budgets_ns`` (authored integer milliseconds in ns).
    """

    def __init__(self, session, *, audio_durations_ns=None, audio_budgets_ns=None, clock=time.monotonic_ns):
        if not v1_session.is_v1(session):
            raise ValueError("the harness runs v1 sessions only")
        v1_session.validate_session(session)
        session = deepcopy(session)
        self._session_sha256 = digest(session)
        self._scheduler_id = uuid.uuid4().hex
        self._events = session["events"]
        self._profile = session["input_profile"]
        self._case_id = session["case_id"]
        self._duration_ns = session["duration_ms"] * MS
        self._performance = self._profile == "avatar_performance_audio"
        self._conversation_text = self._profile == "conversation_text"
        if self._conversation_text and (audio_durations_ns is not None or audio_budgets_ns is not None):
            raise ValueError("conversation_text must not receive audio durations or budgets")
        self._durations = dict(audio_durations_ns or {})
        audio_ids = {e["event_id"] for e in self._events if e["payload"]["type"] in {"user_audio", "speech_audio"}}
        if (set(self._durations) - audio_ids or
                any(not _timestamp(v) or v == 0 for v in self._durations.values())):
            raise ValueError("audio durations require known audio event IDs and positive measured nanoseconds")
        self._audio_budgets = dict(audio_budgets_ns or {})
        self._speech_windows = {}
        if self._performance:
            if any(not _timestamp(v) or v == 0 or v % MS for v in self._audio_budgets.values()):
                raise ValueError("speech budgets must be positive authored integer milliseconds expressed in nanoseconds")
            errors = protocols.validate_performance_audio_timing(
                session, {key: value // MS for key, value in self._audio_budgets.items()}, self._durations)
            if errors:
                raise ValueError("; ".join(errors))
            speech = [e for e in self._events if e["payload"]["type"] == "speech_audio"]
            for index, event in enumerate(speech):
                self._speech_windows[event["event_id"]] = (
                    speech[index + 1]["release"]["at_ms"] * MS if index + 1 < len(speech) else self._duration_ns)
        elif self._audio_budgets:
            raise ValueError("authored speech budgets belong only to avatar performance")
        seen = {}
        for event in self._events:
            release = event["release"]
            if "at_ms" in release:
                if release["at_ms"] * MS >= self._duration_ns:
                    raise ValueError("fixed events must precede the session deadline")
            else:
                trigger = release["trigger"]
                parent = trigger["after_turn"]
                parent_type = "user_prompt" if self._conversation_text else "user_audio"
                origin = "user_prompt_release" if self._conversation_text else "end_of_referenced_user_audio"
                if trigger["timeout_origin"] != origin:
                    raise ValueError("condition timeout origin must match the input profile")
                if parent not in seen or seen[parent]["payload"]["type"] != parent_type:
                    raise ValueError(f"conditions require an earlier {parent_type} event, without cycles or future dependencies")
                if not self._conversation_text and parent not in self._durations:
                    raise ValueError("condition timeout requires a measured duration for its referenced user audio")
            seen[event["event_id"]] = event
        self._clock = clock
        self._cv = threading.Condition()
        self._queue = deque()
        self._rows = {e["event_id"]: {
            "event_id": e["event_id"], "denominator": 1, "status": "waiting",
            "due_at_ns": None, "trigger_deadline_ns": None,
            "input_end_at_ns": None, "released_at_ns": None,
            "handoff_at_ns": None, "delivery_state": "not_released",
            "failure_reason": None, "opportunity_failed": None,
        } for e in self._events}
        for event_id in self._speech_windows:
            self._rows[event_id].update(speech_audio_duration_ns=self._durations[event_id],
                speech_audio_budget_ns=self._audio_budgets[event_id], speech_audio_window_end_at_ns=None,
                scheduled_speech_audio_end_at_ns=None)
        self._signals = {}
        self._state = "prepared"
        self._epoch = self._deadline = self._last_now = None
        self._finished_at = None
        self._thread = None

    def _now(self):
        now = self._clock()
        if not _timestamp(now) or (self._last_now is not None and now < self._last_now):
            if self._state == "running":
                self._finish("error", "invalid_monotonic_clock")
            raise ValueError("scheduler clock must be nonnegative integer monotonic nanoseconds")
        self._last_now = now
        return now

    def start(self, *, background=True, epoch_ns=None):
        """Start once, after readiness. background=False is for deterministic tests."""
        with self._cv:
            if self._state != "prepared":
                raise RuntimeError("scheduler is single-use")
            now = self._now()
            if epoch_ns is not None and (not _timestamp(epoch_ns) or epoch_ns > now):
                raise ValueError("Explicit activity epoch must be a released host-clock instant")
            self._epoch = now if epoch_ns is None else epoch_ns
            self._deadline = self._epoch + self._duration_ns
            self._state = "running"
            for event in self._events:
                if "at_ms" in event["release"]:
                    self._rows[event["event_id"]]["due_at_ns"] = self._epoch + event["release"]["at_ms"] * MS
                if event["event_id"] in self._speech_windows:
                    self._rows[event["event_id"]]["speech_audio_window_end_at_ns"] = self._epoch + self._speech_windows[event["event_id"]]
            self._advance(now)
            if background:
                self._thread = threading.Thread(target=self._run, name="arena-input-release", daemon=True)
                self._thread.start()
        return self

    @staticmethod
    def _fail(row, status, reason):
        row.update(status=status, failure_reason=reason, opportunity_failed=True)

    def _finish(self, state, reason):
        self._state = state
        self._finished_at = self._last_now
        for event in self._events:
            row = self._rows[event["event_id"]]
            if row["status"] in {"waiting", "armed"}:
                conditional = "trigger" in event["release"]
                status = "trigger_not_fired" if conditional and row["status"] == "waiting" else "not_released"
                self._fail(row, status, reason)
            elif row["delivery_state"] == "queued":
                row["delivery_state"] = "expired"
                self._fail(row, "release_expired", reason)
        self._queue.clear()
        self._cv.notify_all()

    def _update_conditions(self, now):
        # References point backwards, so failures propagate in one ordered pass.
        for event in self._events:
            row = self._rows[event["event_id"]]
            if row["status"] != "waiting" or "trigger" not in event["release"]:
                continue
            trigger = event["release"]["trigger"]
            parent = self._rows[trigger["after_turn"]]
            if parent["opportunity_failed"]:
                self._fail(row, "trigger_not_fired", "referenced_input_unavailable")
            elif parent["input_end_at_ns"] is not None:
                row["trigger_deadline_ns"] = parent["input_end_at_ns"] + trigger["timeout_ms"] * MS
                if now >= row["trigger_deadline_ns"]:
                    self._fail(row, "trigger_not_fired", "response_start_timeout")

    def _advance(self, now):
        if self._state != "running":
            return
        if now >= self._deadline:
            self._finish("ended", "session_deadline")
            return
        # Resolve only already released dependencies; no predicted future input.
        self._update_conditions(now)
        ready = [e for e in self._events if self._rows[e["event_id"]]["status"] in {"waiting", "armed"}
                 and self._rows[e["event_id"]]["due_at_ns"] is not None
                 and self._rows[e["event_id"]]["due_at_ns"] <= now]
        ready.sort(key=lambda e: self._rows[e["event_id"]]["due_at_ns"])
        for event in ready:
            row = self._rows[event["event_id"]]
            if event["event_id"] in self._speech_windows and now + self._durations[event["event_id"]] > row["speech_audio_window_end_at_ns"]:
                self._fail(row, "not_released", "speech_audio_window_expired")
                continue
            released = ReleasedInput.at_release(event_id=event["event_id"], profile=self._profile,
                payload=event["payload"], now_ns=now, due_ns=row["due_at_ns"])
            row.update(status="released", released_at_ns=now, delivery_state="queued")
            if event["payload"]["type"] == "user_audio" and event["event_id"] in self._durations:
                row["input_end_at_ns"] = now + self._durations[event["event_id"]]
            elif self._conversation_text:
                row["input_end_at_ns"] = now
            if event["event_id"] in self._speech_windows:
                row["scheduled_speech_audio_end_at_ns"] = now + self._durations[event["event_id"]]
            self._queue.append(released)
        self._update_conditions(now)
        self._cv.notify_all()

    def _next_wake(self):
        times = [self._deadline]
        for row in self._rows.values():
            if row["status"] in {"waiting", "armed"}:
                if row["due_at_ns"] is not None:
                    times.append(row["due_at_ns"])
                elif row["trigger_deadline_ns"] is not None:
                    times.append(row["trigger_deadline_ns"])
        return min(times)

    def _run(self):
        with self._cv:
            try:
                while self._state == "running":
                    now = self._now()
                    self._advance(now)
                    if self._state == "running":
                        self._cv.wait(max(0, (self._next_wake() - now) / 1e9))
            except Exception:
                # Clock/internal failure must not silently lose remaining rows.
                self._finish("error", "scheduler_error")

    def response_started(self, signal: TrustedResponseStart):
        """Accept trusted evidence online, before the condition timeout.

        Late observations cannot reopen failed opportunities. A valid signal arms release at observed_at_ns +
        offset; detector delay is not erased. The condition wait is half-open [input_end, input_end + timeout).
        """
        if (not isinstance(signal, TrustedResponseStart) or not _timestamp(signal.observed_at_ns)
                or not isinstance(signal.after_turn, str) or not signal.after_turn
                or not isinstance(signal.evidence_ref, str) or not signal.evidence_ref.strip()):
            raise ValueError("a complete trusted response-start signal is required")
        with self._cv:
            now = self._now()
            self._advance(now)
            if self._state != "running":
                raise ValueError("signals require a running scheduler")
            parent = self._rows.get(signal.after_turn)
            if (parent is None or parent["handoff_at_ns"] is None
                    or parent["input_end_at_ns"] is None
                    or not parent["input_end_at_ns"] <= signal.observed_at_ns <= now):
                raise ValueError("signal cannot precede the released input end or claim a future response")
            if signal.after_turn in self._signals:
                raise ValueError("first response-start signal is immutable; duplicates are rejected")
            children = [e for e in self._events if e["release"].get("trigger", {}).get("after_turn") == signal.after_turn]
            eligible = [e for e in children if self._rows[e["event_id"]]["status"] == "waiting"]
            if not eligible:
                raise ValueError("no waiting condition accepts this signal")
            self._signals[signal.after_turn] = {"observed_at_ns": signal.observed_at_ns,
                "received_at_ns": now, "evidence_ref": signal.evidence_ref}
            for event in eligible:
                row = self._rows[event["event_id"]]
                row.update(status="armed", due_at_ns=signal.observed_at_ns + event["release"]["trigger"]["offset_ms"] * MS)
            self._advance(now)
            self._cv.notify_all()

    def take(self, timeout=None):
        """Return one currently released input, or None on timeout/closure.

        Slow model calls happen outside this method. No queued input is handed out at/after the deadline or
        cancellation.
        """
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite nonnegative seconds")
        wait_until = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            if self._state == "prepared":
                raise RuntimeError("scheduler is not ready")
            while True:
                now = self._now()
                self._advance(now)
                if self._state != "running":
                    return None
                if self._queue:
                    event = self._queue.popleft()
                    if (event.event_id in self._speech_windows and now + self._durations[event.event_id] >
                            self._rows[event.event_id]["speech_audio_window_end_at_ns"]):
                        row = self._rows[event.event_id]
                        row["delivery_state"] = "expired"
                        self._fail(row, "release_expired", "speech_audio_handoff_window_expired")
                        continue
                    self._rows[event.event_id].update(handoff_at_ns=now, delivery_state="handed_off")
                    return event
                remaining = None if wait_until is None else wait_until - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                next_due = max(0, (self._next_wake() - now) / 1e9)
                self._cv.wait(next_due if remaining is None else min(remaining, next_due))

    def cancel(self):
        with self._cv:
            if self._state in {"prepared", "running"}:
                self._now()
                self._finish("cancelled", "scheduler_cancelled")

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def snapshot(self):
        """Trusted diagnostic ledger, never a model input or a score."""
        with self._cv:
            return deepcopy({"case_id": self._case_id, "state": self._state,
                "runner_session_sha256": self._session_sha256,
                "scheduler_id": self._scheduler_id,
                "clock_domain_id": "scheduler:" + self._scheduler_id,
                "finished_at_ns": self._finished_at,
                "started_at_ns": self._epoch, "deadline_ns": self._deadline,
                "denominator": len(self._rows), "official_evaluation": False,
                "opportunities": list(self._rows.values()), "response_signals": self._signals,
                **({"speech_audio_timing_basis": "caller_supplied_measured_lengths_not_actual_pcm_presentation",
                    "pcm_pacing_implemented": False} if self._performance else {})})
