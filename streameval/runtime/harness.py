"""The session harness: runs one case against one StreamingModel and writes the run artifacts the judge reads.

    open(init) [+ initial reference] -> ready + first frame (avatars: greeting/idle rule) -> activity epoch
    -> TrustedScheduler releases the case's events on the case clock -> dispatch to the model -> deadline -> close

Everything the model outputs is timed on the host clock as it reaches the sink (runtime.capture). Inputs follow
the case timeline exactly: user / supplied speech audio as a continuous 10 ms stream (silence between clips), the
source video of editing cases pushed in real time from the epoch, text and reference events at their release.
Conditional turns are released on the harness's own observation of a new reply (online VAD on the received audio),
never on a model message. Events outside the system's ``ops`` are recorded ``unsupported`` without reaching the
model. Artifacts (docs/protocol.md, checked by ``streameval validate-run``):

    <attempt>/runner.json   binding.json   result.json
    <attempt>/browser/      frame_samples.jsonl  frames/  reference_frame.jpg  remote_video_frames.jsonl
                            remote_audio.s16le  remote_audio_blocks.jsonl  vad_online.jsonl  source_frames.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import socket
import time

import numpy as np

from ..benchmark.tracks import event_op
from ..models.base import (ActionEvent, OutputSink, ReferenceEvent, ReferenceSlot, SessionInit, StreamingModel,
                           TextEvent)
from ..models.registry import DEFAULT_SEED, READY_TIMEOUT_S, composition, system_config
from ..util import digest, private_json, scrub, sha256_file
from . import media
from .capture import RATE as AUDIO_OUT_RATE, MediaCapture
from .scheduler import MS, TrustedResponseStart, TrustedScheduler

# Readiness of avatar systems: output flowing, the first frame at least
# GREETING_MIN_S old and IDLE_NS without speech; stop waiting for silence after IDLE_MAX_S (recorded idle=False).
IDLE_NS, IDLE_MAX_S, GREETING_MIN_S = 1500 * MS, 25, 4.0
TURN_GAP_NS, MIN_RESPONSE_NS = 600 * MS, 200 * MS
RUNNER = "runner.json"


class SessionEnded(RuntimeError):
    """The model ended the session or reported a fatal error during the case."""


def release_conditional(case: dict, scheduler, start: int, previous: int | None, evidence: str,
                        signals: list, note) -> None:
    """Release a conditional turn on a NEW avatar turn after its parent (host-clock ns): the reply starts at least
    MIN_RESPONSE_NS after the parent input ended and TURN_GAP_NS after the previous speech."""
    snapshot = scheduler.snapshot()
    rows = {r["event_id"]: r for r in snapshot["opportunities"]}
    for event in case["events"]:
        trigger = event["release"].get("trigger")
        if not trigger or rows[event["event_id"]]["status"] != "waiting" or trigger["after_turn"] in snapshot["response_signals"]:
            continue
        parent = rows[trigger["after_turn"]]
        if parent["handoff_at_ns"] is None or parent["input_end_at_ns"] is None:
            continue
        fresh = previous is None or start - previous >= TURN_GAP_NS
        if start >= parent["input_end_at_ns"] + MIN_RESPONSE_NS and fresh:
            try:
                scheduler.response_started(TrustedResponseStart(after_turn=trigger["after_turn"],
                                                                observed_at_ns=start, evidence_ref=evidence))
                signals.append({"after_turn": trigger["after_turn"], "observed_at_ns": start,
                                "previous_speech_end_ns": previous, "evidence_ref": evidence})
            except ValueError as exc:
                note("signal_rejected", after_turn=trigger["after_turn"], error=str(exc))


class _Sink(OutputSink):
    def __init__(self, harness: "Harness"):
        self.h = harness

    def ready(self, **meta):
        now = time.monotonic_ns()
        self.h.ready_msg = {"type": "ready", **meta, "host_ns": now}
        self.h.capture.events.put({"event": "server_ready", "host_ns": now})

    def video_frame(self, jpeg, pts_ns=None):
        self.h.capture.video_jpeg(jpeg, pts_ns)

    def video_rgb(self, width, height, rgb, pts_ns=None):
        self.h.capture.video(width, height, rgb, "RGB", pts_ns)

    def audio(self, pcm48k, pts_ns=None):
        if self.h.take_audio and pcm48k:
            self.h.capture.attach_audio()
            self.h.capture.audio(pcm48k[:len(pcm48k) // 2 * 2])

    def ack(self, event_id, status="applied", detail=None):
        self.h.acks.setdefault(str(event_id), {"type": "ack", "event_id": str(event_id), "status": status,
                                               **({"detail": detail} if detail else {}),
                                               "host_ns": time.monotonic_ns()})

    def fatal(self, message):
        self.h.capture.events.put({"event": "fatal", "error": str(message)[:500], "host_ns": time.monotonic_ns()})


class Harness:
    """One bounded session of `case` (a v1 session) for system `system_id` (registry spec) on `model`.

    assets: asset_id -> {path, sha256} for the case (benchmark.dataset.Benchmark.case_assets); verify=True checks
    every byte read. oracle: needed for avatar_performance (authored speech budgets), never shown to the model."""

    def __init__(self, model: StreamingModel, system_id: str, spec: dict, case: dict, assets: dict, out: Path, *,
                 repeat: int = 1, oracle: dict | None = None, smoke: bool = False, admission: dict | None = None,
                 verify: bool = True):
        if (case["group"], case["input_profile"]) != (spec["group"], spec["input_profile"]):
            raise ValueError(f"case {case['case_id']} is outside the track of {system_id}")
        self.model, self.system_id, self.spec, self.case = model, system_id, spec, case
        self.entry, self.track, self.kind, self.group = spec["entry"], spec["track"], spec["kind"], spec["group"]
        self.case_id, self.repeat = case["case_id"], int(repeat)
        self.job_id = f"{self.case_id}.r{self.repeat:02d}"
        self.assets, self.out, self.oracle, self.smoke, self.verify = assets, Path(out), oracle, smoke, verify
        self.admission = admission or {"case_id": self.case_id, "admitted": True, "smoke": True}
        self.config = system_config(system_id, spec)
        self.ops = set(self.config["ops"])
        self.comp = composition(self.entry)
        self.take_audio = not (self.comp and self.comp["audio_out"] == "client")
        self.seed = int(self.entry.get("seed", DEFAULT_SEED)) + self.repeat
        self.log, self.receipts, self.signals, self.speech, self.unsupported = [], {}, [], [], []
        self.acks, self.ready_msg, self.speaking = {}, None, False
        self.capture = self.pacer = self.pusher = self.dialogue = None

    def note(self, kind: str, **details):
        self.log.append({"kind": kind, "host_ns": time.monotonic_ns(), **details})

    @staticmethod
    def page_to_host(page_ms):
        return None if page_ms is None else round(page_ms * 1e6)

    # ---- preparation (reads and verifies media; no model call) ------------------------------------
    def asset(self, asset_id: str) -> Path:
        item = self.assets[asset_id]
        path = Path(item["path"])
        if self.verify and sha256_file(path) != item["sha256"]:
            raise ValueError(f"asset {asset_id} differs from the benchmark manifest: {path.name}")
        return path

    def prepare(self, read_media: bool = True):
        case, init = self.case, self.case["initialization"]
        self.durations, self.budgets, self.pcm, self.images, self.asset_sha = {}, {}, {}, {}, {}
        for event in case["events"]:
            payload = event["payload"]
            ids = [r["asset_id"] for r in payload.get("references") or []] + (
                [payload["asset_id"]] if payload.get("asset_id") else [])
            for asset_id in ids:
                self.asset_sha[asset_id] = self.assets[asset_id]["sha256"]
            if not read_media:
                continue
            if payload["type"] in ("user_audio", "speech_audio"):
                path = self.asset(payload["asset_id"])
                self.pcm[event["event_id"]] = media.load_pcm(path, media.AUDIO_IN_RATE)
                self.durations[event["event_id"]] = media.wav_ns(path)
            elif payload["type"] in ("reference_control", "reference_update"):
                for asset_id in ids:
                    self.images[asset_id] = self.asset(asset_id).read_bytes()
        if self.group == "avatar_performance":
            if self.oracle is None:
                raise ValueError("avatar_performance needs the case oracle (authored speech budgets)")
            self.budgets = {c["event_id"]: c["audio_budget_ms"] * MS for c in self.oracle["checkpoints"]
                            if c.get("audio_budget_ms") is not None}
        output = self.config["output"]
        fields = dict(track=self.track, case_id=self.case_id, width=output["width"], height=output["height"],
                      fps=output["fps"], seed=self.seed, output=output, duration_ms=case["duration_ms"])
        self.push = self.source = self.initial_reference = None
        image_id = None
        if self.kind == "avatar":
            if self.config["portrait_policy"] == "benchmark_portrait":
                image_id = init["portrait_asset_id"]
            if self.group == "digital_human":
                fields["persona"] = (init["persona"] + "\nKnown facts: "
                                     + json.dumps(init["known_facts"], ensure_ascii=False, sort_keys=True))
            else:
                fields["prompt"] = init.get("rendering_context", "")
        elif self.kind == "generation":
            fields["prompt"] = init.get("prompt", "")
            image_id = (init.get("reference_asset_ids") or [None])[0]
        else:
            source_id = init["source_video_asset_id"]
            self.asset_sha[source_id] = self.assets[source_id]["sha256"]
            if read_media:
                self.source = self.asset(source_id)
                self.push = media.push_spec(media.probe_video(self.source), self.config["source_push"])
                fields.update(width=self.push["width"], height=self.push["height"], fps=self.push["fps"])
            if self.group == "reference_edit":
                ref = init["reference"]
                for r in ref["references"]:
                    self.asset_sha[r["asset_id"]] = self.assets[r["asset_id"]]["sha256"]
                    if read_media:
                        self.images[r["asset_id"]] = self.asset(r["asset_id"]).read_bytes()
                self.initial_reference = ref
        if image_id:
            self.asset_sha[image_id] = self.assets[image_id]["sha256"]
            if read_media:
                fields["image"] = self.asset(image_id).read_bytes()
        self.init = SessionInit(**fields)
        self.init_msg = self.init.message()

    def reference_event(self, event_id: str, payload: dict) -> ReferenceEvent:
        slots = [ReferenceSlot(r["target_slot"], self.images[r["asset_id"]], r["scope"], r["asset_id"])
                 for r in payload.get("references") or []]
        slots += [ReferenceSlot(s) for s in payload.get("remove") or []]
        return ReferenceEvent(event_id, payload["operation"], tuple(slots), payload.get("scope"))

    def binding(self) -> dict:
        from .dialogue import stack_kind
        described = self.model.describe()
        public = {k: v for k, v in self.init_msg.items() if k != "image_b64"}
        return {"smoke_test_not_a_formal_sample": self.smoke, "job_id": self.job_id, "case_id": self.case_id,
                "repeat": self.repeat, "system_id": self.system_id, "system_config": self.config,
                "system_config_sha256": digest(self.config), "session_sha256": digest(self.case),
                "admission": self.admission, "assets": dict(self.asset_sha), "audio_durations_ns": self.durations,
                "portrait_used_as_avatar_input": "image_b64" in self.init_msg and self.kind == "avatar",
                "init": public, "init_sha256": digest(self.init_msg), "source_push": self.push,
                "server_url": described.get("url"), "model": described, "seed": self.seed,
                "dialogue_stack": stack_kind() if self.comp else None}

    # ---- dispatch -----------------------------------------------------------------------------------
    def send(self, event) -> int:
        self.model.send_event(event)
        return time.monotonic_ns()

    def dispatch(self, event):
        payload, eid = event.payload, event.event_id
        op = event_op(payload)
        record = {"event_id": eid, "operation": op, "released_at_ns": event.released_at_ns,
                  "dispatch_start_ns": time.monotonic_ns()}
        if op not in self.ops:
            record.update(status="unsupported", reason="not_in_registered_ops: " + op, provider_call=False)
            self.unsupported.append({"event_id": eid, "reason": record["reason"]})
        else:
            try:
                kind = payload["type"]
                if kind == "user_audio" and self.comp:
                    now = time.monotonic_ns()
                    end = now + self.durations[eid]
                    record.update(transport="composition_asr", provider_call=False, play={
                        "event_id": eid, "duration_s": self.durations[eid] / 1e9, "page_start_ms": now / 1e6,
                        "page_end_ms": end / 1e6, "basis": "user clip presented to the composition ASR in real time; "
                                                           "transcribed only after its end"})
                    self.dialogue.submit_audio(eid, self.pcm[eid], end)
                elif kind in ("user_audio", "speech_audio"):
                    record.update(transport="audio_in", provider_call=True, play=self.pacer.play(eid, self.pcm[eid]))
                elif kind == "user_prompt" and self.comp:
                    record.update(transport="composition_llm", provider_call=False)
                    self.dialogue.submit_text(eid, payload["text"], time.monotonic_ns())
                elif kind in ("user_prompt", "text_control", "camera_control"):
                    record.update(transport="text", provider_call=True, expects_ack=True,
                                  host_send_ns=self.send(TextEvent(eid, kind, payload["text"])))
                elif kind == "reference_update":
                    data = self.images[payload["asset_id"]]
                    slot = ReferenceSlot("subject", data, payload.get("instruction"), payload["asset_id"])
                    record.update(transport="reference", provider_call=True, expects_ack=True, host_send_ns=self.send(
                        ReferenceEvent(eid, "set", (slot,), text=payload.get("instruction"))))
                elif kind == "action":
                    record.update(transport="control", provider_call=True, expects_ack=True, host_send_ns=self.send(
                        ActionEvent(eid, payload["action"], payload["phase"], payload["duration_ms"])))
                elif kind == "reference_control":
                    record.update(transport="reference", provider_call=True, expects_ack=True,
                                  host_send_ns=self.send(self.reference_event(eid, payload)))
                else:
                    raise ValueError("unsupported payload " + kind)
                record["status"] = "sent"
            except Exception as exc:
                record.update(status="failed", error=type(exc).__name__ + ": " + scrub(str(exc))[:300])
        record["dispatch_end_ns"] = time.monotonic_ns()
        self.receipts[eid] = record

    def composition_out(self, kind: str, eid: str, payload):
        if kind == "audio":
            return self.pacer.play(eid + ".reply", payload)
        return {"event_id": eid + ".reply",
                "host_send_ns": self.send(TextEvent(eid + ".reply", "dialogue_reply", payload))}

    def pump(self, scheduler=None):
        while True:
            try:
                msg = self.capture.events.get_nowait()
            except queue.Empty:
                return
            kind = msg.get("event")
            if kind == "fatal":
                raise SessionEnded(kind + ": " + str(msg.get("error")))
            if kind == "speech_onset":
                self.speaking = True
                self.speech.append(msg)
                if scheduler is not None:
                    release_conditional(self.case, scheduler, self.page_to_host(msg["start_page_ms"]),
                                        self.page_to_host(msg.get("previous_end_page_ms")),
                                        f"browser/vad_online.jsonl#onset_frame={msg['start_frame']}",
                                        self.signals, self.note)
            elif kind == "speech_offset":
                self.speaking = False
                self.speech.append(msg)

    def wait_ready(self, clock=time.monotonic, sleep=time.sleep) -> dict:
        """`ready` + first frame; avatars additionally wait out any greeting."""
        start = clock()
        limit = float(READY_TIMEOUT_S)
        first_at = ready_at = quiet_since = None
        while clock() - start < limit:
            while True:
                try:
                    msg = self.capture.events.get_nowait()
                except queue.Empty:
                    break
                kind = msg.get("event")
                if kind == "fatal":
                    raise SessionEnded(kind + " before readiness: " + str(msg.get("error")))
                if kind == "server_ready" and ready_at is None:
                    ready_at = clock()
                    if self.kind == "avatar":
                        self.capture.attach_audio()   # the output audio track exists from `ready` (silence until audio)
                elif kind == "first_remote_frame" and first_at is None:
                    first_at = clock()
                elif kind == "speech_onset":
                    self.speaking, quiet_since = True, None
                    self.speech.append({**msg, "phase": "before_epoch"})
                elif kind == "speech_offset":
                    self.speaking = False
                    self.speech.append({**msg, "phase": "before_epoch"})
            ready = ready_at is not None and first_at is not None
            if ready and self.kind != "avatar":
                return {"waited_s": round(clock() - start, 3), "rule": "ready + first frame"}
            if ready and not self.speaking:
                quiet_since = quiet_since or clock()
                if clock() - quiet_since >= IDLE_NS / 1e9 and clock() - first_at >= GREETING_MIN_S:
                    return {"waited_s": round(clock() - start, 3), "idle": True,
                            "greeting_heard": any(e.get("event") == "speech_onset" for e in self.speech)}
            if ready and clock() - first_at >= IDLE_MAX_S:
                return {"waited_s": round(clock() - start, 3), "idle": False}
            sleep(0.01)
        raise TimeoutError(f"not ready after {limit:g} s (ready={ready_at is not None}, first_frame={first_at is not None})")

    # ---- the session ---------------------------------------------------------------------------------
    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=False, mode=0o700)
        # Local runner identity for diagnosing interrupted attempts; never implies permission to retry
        private_json(self.out / RUNNER, {"host": socket.gethostname(), "pid": os.getpid(),
                                         "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.prepare()
        private_json(self.out / "binding.json", self.binding())
        scheduler = TrustedScheduler(self.case, audio_durations_ns=self.durations or None,
                                     audio_budgets_ns=self.budgets or None)
        self.capture = MediaCapture(self.out / "browser", self.config["sampling"]["frame_sample_ms"])
        failure = cleanup_failure = None
        opened = end_sent = termination = False
        result = {"job_id": self.job_id, "case_id": self.case_id, "repeat": self.repeat,
                  "system_id": self.system_id, "system_config_sha256": digest(self.config)}
        try:
            opened = True
            self.model.open(self.init, _Sink(self))
            self.note("opened", **{k: v for k, v in self.model.describe().items() if k in ("transport", "url")})
            if self.initial_reference is not None:
                self.send(self.reference_event("initial", self.initial_reference))
            needs_audio = (self.group == "avatar_performance" or (self.track == "digital_human_audio" and not self.comp)
                           or (self.comp is not None and self.comp["input"] == "audio"))
            if needs_audio:
                on_chunk = None
                if self.comp and self.comp["audio_out"] == "client":
                    self.capture.attach_audio()
                    on_chunk = lambda data: self.capture.audio(media.resample(
                        np.frombuffer(data, dtype=np.int16), media.AUDIO_IN_RATE, AUDIO_OUT_RATE).tobytes())
                self.pacer = media.AudioInPacer(self.model.push_audio, self.capture.events, on_chunk)
                self.pacer.start()
            if self.kind == "editing":
                self.pusher = media.SourcePusher(self.source, self.push, self.model.push_source_frame,
                                                 self.capture.dir / "source_frames.jsonl", self.capture.events)
                self.pusher.start()
            if self.comp:
                from .dialogue import Composition, dialogue_models, make_stack
                stack = make_stack(dialogue_models(self.entry), self.seed)
                self.dialogue = Composition(stack, self.case, self.composition_out, self.comp["input"])
            result["readiness"] = self.wait_ready()
            result["server_ready"] = self.ready_msg
            result["clock_start"] = {"offset_ns": 0, "uncertainty_ns": 0,
                                     "method": "capture page clock is time.monotonic_ns (same host as the model client)"}
            epoch = time.monotonic_ns()
            scheduler.start(epoch_ns=epoch)
            result["browser_start"] = self.capture.start(epoch, self.case["duration_ms"])
            if self.pusher is not None:
                self.pusher.go_live(epoch)
            deadline = epoch + self.case["duration_ms"] * MS
            while time.monotonic_ns() < deadline:
                event = scheduler.take(timeout=0.005)
                if event is not None:
                    self.dispatch(event)
                self.pump(scheduler)
            scheduler.join(timeout=2)
        except BaseException as exc:
            failure = {"type": type(exc).__name__, "message": scrub(str(exc))[:2000]}
        finally:
            scheduler.cancel()
            scheduler.join(timeout=2)
            if self.dialogue is not None:
                self.dialogue.stop()
            if self.pacer is not None:
                self.pacer.stop()
                if self.pacer.error and failure is None:
                    failure = {"type": "AudioPacingError", "message": self.pacer.error}
            if self.pusher is not None:
                result["source_push_stats"] = self.pusher.stop()
            if opened:
                try:
                    termination = bool(self.model.close())
                    end_sent = True
                except Exception as exc:
                    cleanup_failure = {"type": type(exc).__name__, "message": scrub(str(exc))[:500]}
            result["browser_finish"] = self.capture.finish()
        try:
            receipt = self.model.receipt() if opened else {}
        except Exception as exc:
            receipt = {"error": type(exc).__name__ + ": " + scrub(str(exc))[:300]}
        snapshot = scheduler.snapshot()
        missing_acks = []
        for eid, record in self.receipts.items():
            if not record.pop("expects_ack", False):
                continue
            ack = self.acks.get(eid)
            record["ack"] = ack
            if ack is None:
                missing_acks.append(eid)
            elif ack.get("status") == "unsupported":
                record.update(status="unsupported", reason="server_ack_unsupported: " + str(ack.get("detail") or ""))
                self.unsupported.append({"event_id": eid, "reason": record["reason"]})
            elif ack.get("status") == "error":
                record.update(status="failed", error="server_ack_error: " + str(ack.get("detail") or ""))
        released = {r["event_id"] for r in snapshot["opportunities"] if r["status"] == "released"}
        termination = end_sent and termination
        result.update({
            "status": "completed" if failure is None else "failed", "failure": failure,
            "cleanup_failure": cleanup_failure, "termination_confirmed": termination,
            "provider": {"server_ready": self.ready_msg, "initial_reference_ack": self.acks.get("initial"),
                         "end_sent": end_sent, "server_closed": termination, "missing_acks": missing_acks,
                         **({"receipt": receipt} if receipt else {})},
            "provider_session_ended_early": failure is not None and failure["type"] == "SessionEnded",
            "scheduler": snapshot, "dispatch": self.receipts, "unsupported_events": self.unsupported,
            "response_signals": self.signals, "speech_events": self.speech, "notes": self.log,
            "composition_turns": self.dialogue.turns if self.dialogue is not None else None,
            "clock_offset_ns": 0,
            "activity_execution_complete": (failure is None and cleanup_failure is None and termination
                                            and released == {e["event_id"] for e in self.case["events"]}
                                            and released == set(self.receipts) and not missing_acks),
            "official_evaluation": False, "automatic_retries": 0,
        })
        private_json(self.out / "result.json", result)
        return result
