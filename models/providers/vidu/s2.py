"""Vidu S2 wire adapter (realtime Avatar and Editing), pinned to the official parameters checked 2026-09-20.

    HTTP creates and queries a live; a WebSocket carries the controls; AliRTC (in the browser) carries the media.

One create attempt per session, never repeated; an input whose send outcome is unknown is never replayed; close()
hangs up (reconnecting the control socket if needed) and confirms termination with GET /live/v1/lives/{id}. The
account key stays in this process: the browser only receives the RTC join token of the created live.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from urllib.parse import urlencode
import uuid

from streameval.runtime.scheduler import UnsupportedInput
from streameval.util import digest, private_json, scrub

ORIGINS = {"https://api.vidu.cn": "cn", "https://api.vidu.com": "global"}
ROUTES = {"avatar": "/live/s_avatar/realtime", "editing": "/live/s_editing/realtime"}
EDITING_TYPES = {"style_transfer", "virtual_tryon", "subject_replacement", "background_replacement"}
MAX_WIRE_BYTES = 16 * 1024 * 1024
MAX_INLINE_BYTES = 4 * 1024 * 1024


class ProviderError(RuntimeError):
    def __init__(self, message, *, uncertain=False):
        super().__init__(scrub(message))
        self.uncertain = uncertain


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def image_uri(path: Path) -> str:
    from PIL import Image
    path = Path(path)
    data = path.read_bytes()
    if not 0 < len(data) <= MAX_INLINE_BYTES:
        raise ValueError("Inline reference image must be at most 4 MiB")
    with Image.open(path) as image:
        format_ = image.format
        image.verify()
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(format_)
    if not mime:
        raise ValueError("Unsupported reference image format")
    return "data:" + mime + ";base64," + base64.b64encode(data).decode()


def avatar_request(*, portrait: Path, persona: str, voice="Tina", active_seconds=30, recording=False) -> dict:
    if not isinstance(persona, str) or not 0 < len(persona) <= 50000:
        raise ValueError("A bounded initial persona is required")
    if type(active_seconds) is not int or not 1 <= active_seconds <= 300:
        raise ValueError("Activity duration must be 1-300 seconds")
    return {
        "model": "vidu-s2", "call_mode": "video",
        "avatar": {"image_uri": image_uri(portrait), "persona": persona, "voice": voice,
                   "persona_enhance": False, "farewell_enabled": False},
        "enable_recording": bool(recording), "audio": {"enable_transcription": True},
        "voice_model": {"provider": "qwen_omni"},
        "vad": {"type": "server", "silence_duration_ms": 400, "idle_timeout_ms": 0},
        "idle_timeout_seconds": min(7200, active_seconds + 120),
        "memory_retrieval": {"enabled": False}, "knowledge_retrieval": {"enabled": False},
    }


def editing_request(*, reference: Path, editing_type: str) -> dict:
    if editing_type not in EDITING_TYPES:
        raise UnsupportedInput("This reference scope has no documented Editing mapping")
    return {"image_url": image_uri(reference), "editing_type": editing_type}


class S2HTTP:
    def __init__(self, *, base: str, key: str, http=None, ws_factory=None):
        self.base = (base or "").rstrip("/")
        if self.base not in ORIGINS:
            raise ValueError("VIDU_S2_API_BASE must be an official Vidu regional origin: " + ", ".join(ORIGINS))
        if not (key or "").startswith("vda_"):
            raise ValueError("VIDU_S2_API_KEY is missing or malformed")
        self._key = key
        if http is None:
            import requests
            http = requests.Session()
        if ws_factory is None:
            import websocket
            ws_factory = websocket.create_connection
        self.http, self.ws_factory = http, ws_factory
        self.last_exchange = None

    def authorization(self, kind):
        # The Editing pages document the bare key; Avatar and the shared management endpoints use "Token <key>".
        return self._key if kind == "editing" else "Token " + self._key

    def request(self, method, path, *, kind="avatar", body=None, timeout=(10, 45)):
        import requests
        if not path.startswith("/live/") or "?" in path or ".." in path or "//" in path:
            raise ValueError("Invalid provider API path")
        request_id = str(uuid.uuid4())
        self.last_exchange = {"request_id": request_id, "method": method, "path": path, "started_at": utc_now()}
        try:
            response = self.http.request(method, self.base + path, json=body, timeout=timeout,
                                         allow_redirects=False,
                                         headers={"Authorization": self.authorization(kind),
                                                  "Content-Type": "application/json", "X-Request-ID": request_id})
        except requests.RequestException as exc:
            self.last_exchange["transport_error"] = type(exc).__name__
            raise ProviderError("Transport failed; creation outcome may be unknown",
                                uncertain=method == "POST") from None
        self.last_exchange["http_status"] = response.status_code
        try:
            data = response.json()
        except ValueError:
            raise ProviderError(f"Non-JSON HTTP response ({response.status_code})",
                                uncertain=method == "POST") from None
        if not isinstance(data, dict):
            raise ProviderError("Unexpected provider JSON", uncertain=method == "POST")
        if not 200 <= response.status_code < 300:
            detail = scrub({k: data[k] for k in ("code", "message", "reason", "error", "trace_id") if k in data})
            raise ProviderError(f"HTTP {response.status_code}: {json.dumps(detail, ensure_ascii=False)}",
                                uncertain=method == "POST" and response.status_code >= 500)
        return data

    def query(self, live_id, *, timeout=(10, 45)) -> dict:
        self._id(live_id)
        result = self.request("GET", "/live/v1/lives/" + live_id, kind="avatar", timeout=timeout)
        if not isinstance(result.get("live"), dict):
            raise ProviderError("Malformed live status response")
        return result

    @staticmethod
    def _id(live_id):
        if not isinstance(live_id, str) or not live_id.isdigit():
            raise ValueError("Provider live ID must remain a numeric string")

    def connect(self, *, live_id, conn_id, kind):
        self._id(live_id)
        url = (self.base.replace("https:", "wss:") + "/live/ws/live/connect?"
               + urlencode({"live_id": live_id, "conn_id": conn_id}))
        try:
            return self.ws_factory(url, header={"Authorization": self.authorization(kind)}, timeout=10,
                                   redirect_limit=0, enable_multithread=True)
        except Exception as exc:
            raise ProviderError("Control connection failed: " + type(exc).__name__) from None


class ViduS2Session:
    """One create attempt, ordered WS controls, bounded lifetime, confirmed end. The record (provider/record.json)
    keeps the control exchange without credentials or media."""

    def __init__(self, kind, *, client: S2HTTP, directory: Path, active_seconds=15, clock=time.monotonic):
        if kind not in ROUTES or type(active_seconds) is not int or not 1 <= active_seconds <= 300:
            raise ValueError("Invalid session kind or duration")
        self.kind, self.client, self.directory = kind, client, Path(directory)
        self.clock, self.active_seconds = clock, active_seconds
        self.live_id, self.ws = None, None
        self.conn_id, self.seq = str(uuid.uuid4()), 0
        self.started_at = self.ready_at = self.active_at = None
        self.created = self._closing = False
        self.create_timeout = 120
        self.record = {
            "provider": "vidu_s2", "kind": kind, "region": ORIGINS[client.base], "created_at": utc_now(),
            "status": "new", "active_seconds_requested": active_seconds, "control_ready": False,
            "conn_id": self.conn_id, "termination_confirmed": False, "metrics": {}, "events": [],
        }

    def persist(self):
        private_json(self.directory / "record.json", scrub(self.record))

    def event(self, kind, *, strict=True, **details):
        self.record["events"].append({
            "kind": kind, "monotonic_ns": time.monotonic_ns(),
            "elapsed_ms": None if self.started_at is None else round((self.clock() - self.started_at) * 1000, 3),
            **scrub(details)})
        try:
            self.persist()
        except OSError as exc:
            self.record["persistence_error"] = type(exc).__name__
            if strict:
                raise

    def create(self, body: dict) -> dict:
        """POST the create request once; returns {rtc, render_uid, kind} for the browser (never persisted)."""
        if self.created or (self.directory / "record.json").exists():
            raise ValueError("A creation attempt cannot be repeated; inspect or end the existing run")
        if self.kind == "avatar" and body.get("model") != "vidu-s2":
            raise ValueError("S2 must be selected explicitly; never fall back to the S1 default")
        self.created = True
        self.started_at = self.clock()
        self.record.update(status="creating", request_sha256=digest(body))
        self.persist()
        try:
            response = self.client.request("POST", ROUTES[self.kind], kind=self.kind, body=body,
                                           timeout=(10, self.create_timeout))
            live = response.get("live", {})
            self.client._id(live.get("id"))
            self.live_id = live["id"]
            # keep the remote id before any other fallible step, so a created live can always be hung up
            private_json(self.directory / "session.private.json",
                         {"base": self.client.base, "kind": self.kind, "live_id": self.live_id,
                          "live": {k: live.get(k) for k in ("id", "status", "model", "live_duration")}})
            self.record.update(live_id=self.live_id, status="created", requested_model=body.get("model"),
                               provider_model=live.get("model"), provider_duration_seconds=live.get("live_duration"))
            self.record["metrics"]["http_creation_ack_ms"] = round((self.clock() - self.started_at) * 1000, 3)
            self.event("created", initial_provider_status=live.get("status"))
            cap = live.get("live_duration")
            if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap < self.active_seconds:
                raise UnsupportedInput("Returned session cap does not cover the requested duration")
            rtc = response.get("rtc", {})
            if not isinstance(rtc, dict) or not all(rtc.get(k) for k in ("app_id", "channel_id", "user_id", "token")):
                raise ProviderError("RTC credentials are incomplete")
            try:
                expiry = float(rtc["token_expire_at"])
                if not math.isfinite(expiry) or expiry < time.time() + self.active_seconds:
                    raise ValueError()
            except (KeyError, ValueError, TypeError):
                raise ProviderError("RTC credential expiry does not cover the requested activity") from None
            if self.kind == "editing" and not response.get("render_uid"):
                raise ProviderError("Editing output identity render_uid is absent")
            return {"rtc": rtc, "render_uid": response.get("render_uid"), "kind": self.kind}
        except Exception as exc:
            unknown = getattr(exc, "uncertain", False) or (self.live_id is None and not isinstance(exc, ProviderError))
            self.record["status"] = "submission_unknown" if unknown else "failed"
            self.event("create_error", strict=False, http_exchange=self.client.last_exchange,
                       message=str(exc) if isinstance(exc, (ProviderError, UnsupportedInput)) else type(exc).__name__)
            raise

    def guard_input(self, event_id):
        if not self.record["control_ready"] or self.ws is None:
            raise ValueError("Runtime input cannot precede control readiness or follow cleanup")
        if self.ready_at is None or self.clock() >= self.activity_deadline():
            raise ValueError("Activity deadline has elapsed")
        if any(e.get("event_id") == event_id and e["kind"] == "input_attempt" for e in self.record["events"]):
            raise ValueError("Input was already attempted; never replay an uncertain input")

    def activity_deadline(self):
        start = self.active_at if self.active_at is not None else self.ready_at
        if start is None:
            raise ValueError("Control is not ready")
        deadline = start + self.active_seconds
        cap = self.record.get("provider_duration_seconds")
        if self.started_at is not None and isinstance(cap, (int, float)):
            deadline = min(deadline, self.started_at + cap)
        return deadline

    def start_activity(self, *, epoch_ns: int):
        if not self.record["control_ready"] or self.ws is None or self.active_at is not None:
            raise ValueError("Activity requires a ready control connection and cannot restart")
        cap = self.record.get("provider_duration_seconds")
        if cap is not None and self.clock() + self.active_seconds > self.started_at + cap:
            raise UnsupportedInput("Remaining provider lifetime is shorter than the activity")
        if type(epoch_ns) is not int or not self.ready_at <= epoch_ns / 1e9 <= self.clock():
            raise ValueError("The activity epoch must follow control readiness in the host clock")
        self.active_at = epoch_ns / 1e9
        self.event("activity_started", duration_seconds=self.active_seconds, activity_epoch_host_ns=epoch_ns)

    def send(self, type_, payload, *, event_id=None):
        if not self.ws:
            raise ValueError("Control socket is absent")
        if type_ not in {1, 5} and not self.record["control_ready"]:
            raise ValueError("Runtime input cannot precede control readiness")
        if type_ not in {1, 5} and self.ready_at is not None and self.clock() >= self.activity_deadline():
            raise ValueError("Activity deadline has elapsed")
        if event_id:
            self.guard_input(event_id)
        self.seq += 1
        envelope = {"type": type_, "live_id": self.live_id, "conn_id": self.conn_id, "seq_id": self.seq,
                    "payload": payload}
        wire = json.dumps(envelope, ensure_ascii=False, allow_nan=False)
        if len(wire.encode()) > MAX_WIRE_BYTES:
            raise ValueError("Control exceeds provider message size")
        strict = type_ != 5 and not self._closing
        self.event("input_attempt" if event_id else "control_attempt", strict=strict, type=type_, seq=self.seq,
                   event_id=event_id, wire_sha256=digest(envelope), bytes=len(wire.encode()))
        try:
            # receive() polls with a short timeout; a reference-image frame can be several MiB
            self.ws.settimeout(10)
            self.ws.send(wire)
        except Exception as exc:
            self.event("send_unknown", strict=strict, type=type_, seq=self.seq, event_id=event_id,
                       error_type=type(exc).__name__)
            # a partially written frame cannot carry the hangup: drop this socket, cleanup reconnects with the same
            # conn_id; the input itself is never replayed
            try:
                self.ws.close(timeout=1)
            except Exception:
                pass
            self.ws = None
            raise ProviderError("Control send outcome is unknown; do not retry this input") from None
        self.event("sent", strict=strict, type=type_, seq=self.seq, event_id=event_id)
        return self.seq

    def receive(self, timeout=.25):
        import websocket
        self.ws.settimeout(timeout)
        try:
            raw = self.ws.recv()
        except websocket.WebSocketTimeoutException:
            return None
        except Exception as exc:
            raise ProviderError("Control receive failed: " + type(exc).__name__) from None
        if not raw:
            raise ProviderError("Control connection closed")
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            raise ProviderError("Invalid control response") from None
        if not isinstance(data, dict) or not isinstance(data.get("payload", {}), dict):
            raise ProviderError("Malformed control envelope")
        self.event("received", type=data.get("type"), seq=data.get("seq_id"), payload=data.get("payload"))
        if data.get("type") == 6:
            self.record["provider_hangup"] = True
            raise ProviderError("Provider ended the active session")
        return data

    def initialize(self):
        if not self.live_id:
            raise ValueError("Create a session first")
        self.ws = self.client.connect(live_id=self.live_id, conn_id=self.conn_id, kind=self.kind)
        self.record["status"] = "initializing"
        self.send(1, {"conn_init": {"version": 1}})
        deadline = self.clock() + (60 if self.kind == "avatar" else 120)
        retry_at = None
        while self.clock() < deadline:
            if retry_at is not None and self.clock() >= retry_at:
                self.send(1, {"conn_init": {"version": 1}})
                retry_at = None
            data = self.receive()
            if not data or data.get("type") != 2:
                continue
            ack = data.get("payload", {}).get("conn_init_ack", {})
            if ack.get("success") is True:
                self.ready_at = self.clock()
                self.record.update(status="control_ready", control_ready=True)
                self.record["metrics"]["control_ready_ms"] = round((self.ready_at - self.started_at) * 1000, 3)
                self.event("control_ready")
                return
            if ack.get("error_code") == "NOT_READY":
                # documented for both flows: retry the handshake on the same connection, never a new live
                if retry_at is None:
                    retry_at = self.clock() + 2.5
            else:
                raise ProviderError("Control initialization rejected: " + str(ack.get("error_code")))
        raise ProviderError("Control initialization timed out")

    def close(self):
        self._closing = True
        if not self.live_id:
            self.record["completed_at"] = utc_now()
            self.event("no_known_session_to_close", strict=False)
            return
        try:
            if not self.record["termination_confirmed"]:
                if self.ws is None:
                    # every new socket starts with conn_init; the GET below decides whether termination succeeded
                    self.ws = self.client.connect(live_id=self.live_id, conn_id=self.conn_id, kind=self.kind)
                    self.send(1, {"conn_init": {"version": 1}})
                self.send(5, {"hangup": {"hangup_reason": "user_end" if self.kind == "avatar" else "user_hangup"}})
        except Exception as exc:
            self.event("hangup_unconfirmed", strict=False, error_type=type(exc).__name__)
        finally:
            if self.ws:
                try:
                    self.ws.close(timeout=2)
                except Exception:
                    pass
                self.ws = None
        deadline = self.clock() + 12
        for index in range(5):
            try:
                live = self.client.query(self.live_id, timeout=(3, 3))["live"]
                self.record["provider_final_status"] = live.get("status")
                for field in ("billed_seconds", "credits_cost"):
                    self.record["metrics"]["provider_" + field] = live.get(field)
                if live.get("status") == "ended":
                    self.record["termination_confirmed"] = True
                    self.event("termination_confirmed", strict=False)
                    break
            except Exception as exc:
                self.event("termination_query_failed", strict=False, error_type=type(exc).__name__)
            if self.clock() >= deadline:
                break
            if index < 4:
                time.sleep(.6)
        if not self.record["termination_confirmed"]:
            self.record.setdefault("status_before_cleanup", self.record["status"])
            self.record["status"] = "cleanup_pending"
        self.record["completed_at"] = utc_now()
        self.event("cleanup_finished", strict=False)
