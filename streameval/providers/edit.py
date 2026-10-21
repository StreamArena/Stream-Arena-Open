"""Editing systems driven through a vendor's browser SDK (instruction_edit, reference_edit).

One session = one headless Chromium page (providers/browser.py, providers/js/edit_page_<vendor>.mjs) that pushes the
case's source video as a camera track, receives the edited stream and samples it; Python keeps the credentials,
the TrustedScheduler, the full-state planning (edit_state.py) and cleanup. Same artifacts as the harness
(docs/protocol.md), so the judge reads them unchanged.

Rules fixed for every system here (those of the v1 board):
* instruction events are sent verbatim as the full prompt (native semantics, no client-side accumulation);
* reference events become one English prompt from REFERENCE_TEMPLATE over the authored scope texts, plus the
  reference image; reference A of a reference_edit case is applied before the activity epoch;
* an event the client cannot express (two independent reference slots, an empty state) is released, recorded as
  ``unsupported`` without a provider call and scored as a failure;
* the source is pushed at the provider's highest supported input resolution, never upscaled;
* SDK reconnects inside a started session are recorded, never repeated by the harness.

A vendor subclass (models/providers/<vendor>/backend.py) sets ``page`` and ``push`` and implements ``mint``,
``connect_args``, ``set_args`` and ``close_receipt``; only those, ``preflight`` and ``close_receipt`` touch the
network.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import queue
import socket
import time

from ..benchmark.tracks import SAMPLE_MS
from ..runtime.harness import RUNNER
from ..runtime.media import probe_video
from ..runtime.scheduler import MS, ReleasedInput, TrustedScheduler, UnsupportedInput
from ..util import digest, private_json, scrub, sha256_file
from .base import SessionBackend
from .browser import BROWSER_FLAGS, Browser, BrowserError

from streameval.models.registry import READY_TIMEOUT_S

READY_MAX_S = READY_TIMEOUT_S
SESSION_CAP_MS = 300_000   # longest v1 case; no vendor here documents a shorter session cap
REFERENCE_TEMPLATE = ("Edit only this target so that it takes the look of the reference image "
                      "(its colors, pattern and material): {reference_scope}\n{operation_scope}")
REFERENCE_TEMPLATE_ID = "streameval.v1.reference-prompt.en/1"
MIMES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
         ".mp4": "video/mp4", ".webm": "video/webm"}


def template_sha256() -> str:
    return hashlib.sha256(REFERENCE_TEMPLATE.encode()).hexdigest()


def reference_dispositions(case: dict) -> list[dict]:
    """Which reference states a single-slot client can express, the preloaded reference A first.

    The authored state after each event (set replaces, patch merges, remove drops slots, reset empties) must hold
    exactly one edited slot: two independent slots cannot be one prompt + one image, and an empty state (restore
    the live source) has no verified native input. An unsupported event leaves the client state unchanged."""
    init = case["initialization"]["reference"]
    active = {r["target_slot"] for r in init["references"]}
    rows = [{"event_id": "initial", "operation": init["operation"], "supported": len(active) == 1,
             "reason": None if len(active) == 1 else "multiple_independent_reference_slots_not_supported"}]
    for event in case["events"]:
        payload, op = event["payload"], event["payload"]["operation"]
        requested = {r["target_slot"] for r in payload["references"]}
        active = (requested if op == "set" else active | requested if op == "patch"
                  else active - set(payload["remove"]) if op == "remove" else set())
        reason = (None if len(active) == 1 else "multiple_independent_reference_slots_not_supported"
                  if active else "empty_state_restore_has_no_expressible_input")
        rows.append({"event_id": event["event_id"], "operation": "reference:" + op, "supported": reason is None,
                     "reason": reason, "target_slots": sorted(active)})
    return rows


# ---- source push policy ------------------------------------------------------------------------------

def even(value: float) -> int:
    n = max(2, int(value))
    return n - n % 2


def push_size(width: int, height: int, policy: dict) -> tuple[int, int]:
    """The largest size the provider accepts, never above the source.

    policy {"kind": "max_pixels", "max_pixels", "align"}: below max_pixels as is, else the largest proportional
    align-multiple size below it; {"kind": "box", "box": [long, short]}: fit the box in either orientation."""
    if policy["kind"] == "max_pixels":
        limit, align = policy["max_pixels"], policy["align"]
        if width * height < limit:
            return even(width), even(height)
        scale = math.sqrt(limit / (width * height))
        w, h = (max(align, int(width * scale) // align * align), max(align, int(height * scale) // align * align))
        while w * h >= limit:
            w, h = (w - align, h) if w >= h else (w, h - align)
        return w, h
    long_edge, short_edge = policy["box"]
    landscape = width >= height
    scale = min(1.0, long_edge / (width if landscape else height), short_edge / (height if landscape else width))
    return even(width * scale), even(height * scale)


def push_spec(info: dict, policy: dict) -> dict:
    width, height = push_size(info["width"], info["height"], policy)
    spec = {"width": width, "height": height, "fps": min(policy["max_fps"], round(info["fps"])), "source": info,
            "scaled": (width, height) != (info["width"], info["height"]), "upscaled": False}
    if policy.get("bitrate_kbps"):
        high = width >= 1600 or height >= 1600
        spec["max_kbps"] = policy["bitrate_kbps"]["high" if high else "low"]
    return spec


# ---- the backend ------------------------------------------------------------------------------------

class EditBackend(SessionBackend):
    """Base class of the browser-SDK editing systems. A vendor subclass sets ``page`` and ``push`` and implements
    the vendor hooks below; run() executes one session as an EditJob."""

    page = ""          # provider page of providers/js/edit_browser.mjs (launch.json "provider")
    push: dict = {}    # source push policy (push_size)
    cost = {"unit": None, "per_active_second": None, "overhead_s": 30}   # overhead: connect + readiness

    # ---- declarations (no network) --------------------------------------------------------------
    def event_dispositions(self, system_id: str, spec: dict, case: dict) -> list[dict]:
        rows = super().event_dispositions(system_id, spec, case)
        if spec["track"] != "reference_edit":
            return rows
        client = {r["event_id"]: r for r in reference_dispositions(case)[1:]}
        return [r if not r["supported"] else {**r, **{k: client[r["event_id"]][k] for k in ("supported", "reason")}}
                for r in rows]

    def system_config(self, system_id: str, spec: dict) -> dict:
        reference = spec["track"] == "reference_edit"
        return {
            "session_cap_ms": SESSION_CAP_MS, "seed": None, "output": None,
            "instruction_events": None if reference else "text_control payload.text sent verbatim as the full "
                                                         "prompt (native semantics; no client-side accumulation)",
            "reference_events": {"prompt_template_id": REFERENCE_TEMPLATE_ID, "prompt_template": REFERENCE_TEMPLATE,
                                 "prompt_template_sha256": template_sha256(),
                                 "state": "single-slot full state (streameval.providers.edit_state); initial "
                                          "reference A planned and applied before the epoch"} if reference else None,
            "client": {"capture": "first presented remote frame per activity bin (requestVideoFrameCallback), page "
                                  "clock mapped to the host clock by a min-RTT round trip",
                       "browser_flags": BROWSER_FLAGS,
                       "reconnects": "SDK reconnects inside a started session are recorded, never repeated"},
            "readiness": ("first presented output frame" if self.expects_output_before_epoch(spec["track"])
                          else "connected; no output before the first instruction"),
            "unsupported": "events the single-slot client cannot express are released, recorded as dispatch status "
                           "'unsupported' without a provider call, client state unchanged; scored as failures",
            "sampling": {"frame_sample_ms": SAMPLE_MS[spec["group"]],
                         "frame_selection": "first presented frame per activity bin"},
        }

    def expects_output_before_epoch(self, track: str) -> bool:
        return True

    # ---- vendor hooks (network) ---------------------------------------------------------------------
    def preflight(self, job: "EditJob") -> None:
        missing = self.missing_credentials()
        if missing:
            raise RuntimeError("credentials missing: " + ", ".join(missing))

    def mint(self, job: "EditJob") -> dict:
        """A short-lived browser credential ({"key"} or {"token"}); never logged."""
        raise NotImplementedError

    def connect_args(self, job: "EditJob", initial) -> dict:
        raise NotImplementedError

    def set_args(self, plan) -> dict:
        raise NotImplementedError

    def browser_launch(self, job: "EditJob") -> dict:
        """Extra launch.json fields (e.g. an esbuild module alias)."""
        return {}

    def close_receipt(self, job: "EditJob", connected: dict | None, finished: dict | None) -> dict:
        """Termination receipt from the connect reply and the page's finish stats."""
        return {"termination_confirmed": False}

    def run(self, system_id, spec, case, assets, out, *, repeat=1, oracle=None, smoke=False, admission=None,
            verify=True) -> dict:
        return EditJob(self, system_id, spec, case, assets, Path(out), repeat=repeat, smoke=smoke,
                       admission=admission, verify=verify).run()


class EditJob:
    """One editing session of one case: browser page, scheduler, state planner and the attempt folder `out`.
    Created by EditBackend.run; the vendor hooks receive it as `job`."""

    def __init__(self, backend: EditBackend, system_id: str, spec: dict, case: dict, assets: dict, out: Path, *,
                 repeat: int, smoke: bool, admission: dict | None, verify: bool):
        from ..models.registry import system_config
        if (case["group"], case["input_profile"]) != (spec["group"], spec["input_profile"]):
            raise ValueError(f"case {case['case_id']} is outside the track of {system_id}")
        self.backend, self.system_id, self.spec, self.case, self.out = backend, system_id, spec, case, out
        self.track, self.case_id, self.repeat = spec["track"], case["case_id"], int(repeat)
        self.job_id = f"{self.case_id}.r{self.repeat:02d}"
        self.assets, self.smoke, self.verify = assets, smoke, verify
        self.admission = admission or {"case_id": self.case_id, "admitted": True, "smoke": True}
        self.config = system_config(system_id, spec)
        self.dispositions = {d["event_id"]: d for d in backend.event_dispositions(system_id, spec, case)}
        self.log, self.receipts, self.unsupported = [], {}, []
        self.offset_ns = None

    def note(self, kind: str, **details):
        self.log.append({"kind": kind, "host_ns": time.monotonic_ns(), **details})

    def page_to_host(self, page_ms):
        return None if page_ms is None or self.offset_ns is None else round(page_ms * 1e6 + self.offset_ns)

    def path(self, asset_id: str) -> Path:
        item = self.assets[asset_id]
        path = Path(item["path"])
        if self.verify and sha256_file(path) != item["sha256"]:
            raise ValueError(f"asset {asset_id} differs from the benchmark manifest: {path.name}")
        return path

    def resolve(self, asset_id: str) -> bytes:
        return self.path(asset_id).read_bytes()

    def prepare(self):
        source_id = self.case["initialization"]["source_video_asset_id"]
        source = self.path(source_id)
        self.push = push_spec(probe_video(source), self.backend.push)
        served = {"source": {"path": str(source), "sha256": self.assets[source_id]["sha256"],
                             "mime": MIMES.get(source.suffix.lower(), "application/octet-stream")}}
        for asset_id in self.case["asset_ids"]:
            path = Path(self.assets[asset_id]["path"])
            if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                served[asset_id] = {"path": str(path), "sha256": self.assets[asset_id]["sha256"],
                                    "mime": MIMES[path.suffix.lower()]}
        self.served = served

    def planner(self, epoch_ns: int):
        from .edit_state import EditStatePlanner
        return EditStatePlanner(asset_bytes_resolver=self.resolve, epoch_ns=epoch_ns, enhance=False,
                                reference_template=REFERENCE_TEMPLATE)

    def initial_plan(self, planner, now_ns: int):
        """Reference A (preloaded, not a timed opportunity) planned before the epoch."""
        if self.track != "reference_edit":
            return None
        return planner.plan(ReleasedInput("initial", self.case["input_profile"],
                                          self.case["initialization"]["reference"], now_ns))

    def dispatch(self, browser: Browser, planner, event):
        record = {"event_id": event.event_id, "released_at_ns": event.released_at_ns,
                  "dispatch_start_ns": time.monotonic_ns()}
        disposition = self.dispositions[event.event_id]
        try:
            if not disposition["supported"]:
                raise UnsupportedInput(disposition["reason"])
            plan = planner.plan(event)
        except UnsupportedInput as exc:
            record.update(status="unsupported", reason=str(exc), provider_call=False)
            self.unsupported.append({"event_id": event.event_id, "reason": str(exc)})
        else:
            meta = plan.metadata()
            record.update(plan={k: meta[k] for k in ("operation", "before", "after", "provider_prompt_sha256",
                                                     "enhance")}, provider_call=True)
            try:
                result, sent, replied = browser.call("set", {"event_id": event.event_id,
                                                             **self.backend.set_args(plan)}, timeout=60)
                record.update(status="sent", host_send_ns=sent, host_reply_ns=replied, page=result,
                              page_start_ns=self.page_to_host(result.get("page_start_ms")),
                              page_resolved_ns=self.page_to_host(result.get("page_resolved_ms")))
            except BrowserError as exc:
                record.update(status="failed", error=scrub(str(exc))[:1000])
        record["dispatch_end_ns"] = time.monotonic_ns()
        self.receipts[event.event_id] = record

    def pump(self, browser: Browser):
        while True:
            try:
                msg = browser.events.get_nowait()
            except queue.Empty:
                return
            kind = msg.get("event") or ""
            if kind in {"fatal", "process_exited", "page_error"}:
                raise BrowserError(kind + ": " + str(msg.get("error")))
            if kind.startswith("provider_"):
                self.note(kind, **{k: v for k, v in msg.items() if k != "event"})

    def wait_ready(self, browser: Browser) -> dict:
        if not self.backend.expects_output_before_epoch(self.track):
            return {"waited_s": 0.0, "first_remote_frame": None, "rule": "no output before the first instruction"}
        start = time.monotonic()
        while time.monotonic() - start < READY_MAX_S:
            try:
                msg = browser.events.get(timeout=0.05)
            except queue.Empty:
                continue
            kind = msg.get("event") or ""
            if kind in {"fatal", "process_exited", "page_error", "provider_disconnect", "provider_session_ended"}:
                raise BrowserError(kind + ": " + str(msg.get("error") or msg.get("reason")))
            if kind == "first_remote_frame":
                return {"waited_s": round(time.monotonic() - start, 3), "first_remote_frame": msg,
                        "rule": "first presented remote frame"}
            if kind.startswith("provider_"):
                self.note(kind, **{k: v for k, v in msg.items() if k != "event"})
        raise BrowserError("output not ready")

    def binding(self) -> dict:
        return {"smoke_test_not_a_formal_sample": self.smoke, "job_id": self.job_id, "case_id": self.case_id,
                "repeat": self.repeat, "system_id": self.system_id, "system_config": self.config,
                "system_config_sha256": digest(self.config), "session_sha256": digest(self.case),
                "admission": self.admission, "assets": {k: v["sha256"] for k, v in self.served.items()},
                "source_push": self.push}

    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=False, mode=0o700)
        private_json(self.out / RUNNER, {"host": socket.gethostname(), "pid": os.getpid(),
                                         "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.prepare()
        private_json(self.out / "binding.json", self.binding())
        scheduler = TrustedScheduler(self.case)
        prep_ns = time.monotonic_ns()
        planner = self.planner(prep_ns)
        browser = failure = cleanup_failure = connected = None
        termination = {"termination_confirmed": False}
        result = {"job_id": self.job_id, "case_id": self.case_id, "repeat": self.repeat,
                  "system_id": self.system_id, "system_config_sha256": digest(self.config)}
        try:
            initial = self.initial_plan(planner, prep_ns)
            if initial is not None:
                meta = initial.metadata()
                result["initial_plan"] = {k: meta[k] for k in ("operation", "after", "provider_prompt_sha256",
                                                               "enhance")}
            self.backend.preflight(self)
            browser = Browser(self.out / "browser", {"provider": self.backend.page, "assets": self.served,
                                                     **self.backend.browser_launch(self)})
            browser.wait_event("browser_ready", 90)
            result["browser_init"], _, _ = browser.call("init", {
                "sample_ms": self.config["sampling"]["frame_sample_ms"],
                "source": {"url": "/asset/source", **self.push["source"]},
                "push": {k: self.push[k] for k in ("width", "height", "fps")}}, timeout=180)
            credential = self.backend.mint(self)
            connected, _, _ = browser.call("connect", {**self.backend.connect_args(self, initial), **credential},
                                           timeout=READY_MAX_S)
            result["browser_connect"] = connected
            self.note("control_ready")
            result["readiness"] = self.wait_ready(browser)
            sync = browser.clock_sync()
            self.offset_ns = sync["offset_ns"]
            result["clock_start"] = sync
            epoch = time.monotonic_ns()
            scheduler.start(epoch_ns=epoch)
            result["browser_start"], _, _ = browser.call("start", {"zero_page_ms": (epoch - self.offset_ns) / 1e6,
                                                                   "duration_ms": self.case["duration_ms"]}, timeout=10)
            deadline = epoch + self.case["duration_ms"] * MS
            while time.monotonic_ns() < deadline:
                event = scheduler.take(timeout=0.01)
                if event is not None:
                    self.dispatch(browser, planner, event)
                self.pump(browser)
            scheduler.join(timeout=2)
        except BaseException as exc:
            failure = {"type": type(exc).__name__, "message": scrub(str(exc))[:2000]}
        finally:
            scheduler.cancel()
            scheduler.join(timeout=2)
            if browser is not None:
                try:
                    result["browser_finish"], _, _ = browser.call("finish", timeout=120)
                    if self.offset_ns is not None:
                        result["clock_end"] = browser.clock_sync()
                except Exception as exc:
                    cleanup_failure = {"type": type(exc).__name__, "message": scrub(str(exc))[:1000]}
            try:
                termination = self.backend.close_receipt(self, connected, result.get("browser_finish"))
            except BaseException as exc:
                cleanup_failure = {"type": type(exc).__name__, "message": scrub(str(exc))[:1000]}
            if browser is not None:
                browser.close()
        snapshot = scheduler.snapshot()
        released = {r["event_id"] for r in snapshot["opportunities"] if r["status"] == "released"}
        confirmed = termination.get("termination_confirmed", False)
        result.update({
            "status": "completed" if failure is None else "failed", "failure": failure,
            "cleanup_failure": cleanup_failure, "termination_confirmed": confirmed,
            "provider": termination, "scheduler": snapshot, "dispatch": self.receipts,
            "unsupported_events": self.unsupported, "notes": self.log, "clock_offset_ns": self.offset_ns,
            "activity_execution_complete": (failure is None and cleanup_failure is None and confirmed
                                            and released == {e["event_id"] for e in self.case["events"]}
                                            and released == set(self.receipts)),
            "official_evaluation": False, "automatic_retries": 0,
        })
        private_json(self.out / "result.json", result)
        return result
