"""Runway Characters (gwm1_avatars): REST session + LiveKit room, run by the ordinary harness (providers/rtc.py).

Per session: an account avatar for the exact benchmark portrait (POST /v1/avatars referenceImage, created once per
portrait sha256 and cached locally) -> POST /v1/realtime_sessions -> READY -> consume -> LiveKit room: avatar
audio/video captured in-process, user audio published as the room's microphone track. close(): DELETE the session,
leave the room, then GET its final status (COMPLETED | CANCELLED | FAILED confirms termination).

Runway has no runtime user-text input, so only digital_human_audio is registered. maxDuration is capped at 300 s by
the vendor: a case that cannot finish inside it is kept and fails in the denominator.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import ssl
import threading
import time
import urllib.error
import urllib.request

from streameval.config import env
from streameval.providers.livekit import LiveKitRoom
from streameval.providers.rtc import RtcAvatar
from streameval.util import private_json, scrub

API = "https://api.dev.runwayml.com"
VERSION = "2024-11-06"
VOICE = {"type": "runway-live-preset", "presetId": "clara"}
SESSION_CAP_S = 300          # documented maximum of one realtime session
SETUP_MARGIN_S = 45          # readiness wait + tail inside maxDuration
TLS_RETRIES = 4              # handshake-time certificate mismatches only (see Client.request)
CREATE_429_RETRIES, CREATE_429_WAIT_S = 6, 10   # session creation only, before any session exists


def avatar_cache() -> Path:
    return Path(env("STREAMEVAL_RUNWAY_AVATARS", "~/.cache/streameval/providers/runway_avatars")).expanduser()


class Client:
    def __init__(self, key: str):
        self.key = key

    def request(self, method, path, body=None, token=None):
        headers = {"Authorization": "Bearer " + (token or self.key), "X-Runway-Version": VERSION,
                   "Content-Type": "application/json", "User-Agent": "streameval/1.0"}
        req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers, method=method)
        for attempt in range(TLS_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=40) as r:
                    code, raw = r.status, r.read()
                break
            except urllib.error.HTTPError as e:
                code, raw = e.code, e.read()
                break
            except urllib.error.URLError as e:
                # Some resolved IPs of the API host have intermittently presented a certificate for another host.
                # Verification fails during the handshake, before any request byte is sent, so a retry
                # (verification kept on) cannot duplicate a POST.
                if not isinstance(e.reason, ssl.SSLCertVerificationError) or attempt == TLS_RETRIES:
                    raise
                time.sleep(1 + attempt)
        try:
            data = json.loads(raw) if raw else {}
        except ValueError:
            data = {"message": raw.decode(errors="replace")[:1000]}
        if code >= 400:
            raise RuntimeError(f"runway_http_{code}: " + scrub(json.dumps(data))[:500].replace(self.key, "<redacted>"))
        return data


_avatar_lock = threading.Lock()


def ensure_avatar(client: Client, portrait: bytes, cache: Path, sleep=time.sleep) -> str:
    """One account avatar per exact portrait; creation is never retried, a cached non-READY avatar halts."""
    sha = hashlib.sha256(portrait).hexdigest()
    with _avatar_lock:
        target = cache / (sha + ".json")
        if target.exists():
            cached = json.loads(target.read_text())
            if cached.get("status") != "READY":
                raise RuntimeError(f"cached avatar {target} is not READY; resolve its registration first")
            return cached["id"]
        mime = "image/png" if portrait[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        body = {"name": "streameval-v1-" + sha[:12],
                "referenceImage": f"data:{mime};base64," + base64.b64encode(portrait).decode(),
                "personality": "StreamEval benchmark avatar; the session personality replaces this.",
                "voice": VOICE, "imageProcessing": "none"}
        created = client.request("POST", "/v1/avatars", body)
        record = {"id": created["id"], "status": created.get("status"), "portrait_sha256": sha}
        private_json(target, record)
        for _ in range(90):
            detail = client.request("GET", "/v1/avatars/" + created["id"])
            record["status"] = detail.get("status")
            private_json(target, record)
            if record["status"] == "READY":
                return created["id"]
            if record["status"] == "FAILED":
                raise RuntimeError("avatar registration FAILED")
            sleep(2)
        raise RuntimeError(f"avatar registration timeout; its id is kept in {target} for recovery")


class RunwayCharacters(RtcAvatar):
    credentials_env = {"api_key": "RUNWAYML_API_SECRET"}
    cost = {"unit": "USD", "basis": "gwm1_avatars: 2 credits at session start + 2 credits per started 6 s, 1 credit "
                                    "= 0.01 USD (vendor pricing as of 2026-09); billed span estimated as case "
                                    f"duration + {SETUP_MARGIN_S} s, capped at {SESSION_CAP_S} s"}
    session_cap_s = SESSION_CAP_S
    client_transport = "LiveKit Python SDK (livekit.rtc), in-process"
    audio_transport = "LiveKit microphone track (48 kHz mono, 10 ms frames)"
    client = sid = join_with = None

    def estimate(self, case: dict) -> float:
        billed = min(SESSION_CAP_S, case["duration_ms"] / 1000 + SETUP_MARGIN_S)
        return (2 + 2 * math.ceil(billed / 6)) * 0.01

    def system_config(self, system_id: str, spec: dict) -> dict:
        return {**super().system_config(system_id, spec),
                "model": "gwm1_avatars", "product": "POST /v1/realtime_sessions (gwm1_avatars)",
                "api_origin": API, "api_version": VERSION,
                "avatar": {"creation": "POST /v1/avatars referenceImage = exact benchmark portrait (data URI), "
                                       "imageProcessing none, cached per portrait sha256",
                           "voice": VOICE, "llm": "vendor default conversational model (not selectable in the API)",
                           "persona": "session personality = persona + '\\nKnown facts: ' + known facts as sorted JSON",
                           "greeting": "no startScript sent; any vendor greeting is waited out and recorded "
                                       "(readiness.greeting_heard)",
                           "max_duration_s": f"min({SESSION_CAP_S}, case duration + {SETUP_MARGIN_S})",
                           "session_cap": "sessions the vendor ends before the case deadline are failures"}}

    def create(self, init) -> dict:
        if init.image is None:
            raise ValueError("Runway Characters needs the benchmark portrait (portrait_policy benchmark_portrait)")
        client = self.client = Client(self.credentials()["api_key"])
        started = time.monotonic()
        avatar = ensure_avatar(client, init.image, avatar_cache())
        self.record["avatar_id"] = avatar
        seconds = init.duration_ms // 1000
        request = {"model": "gwm1_avatars", "avatar": {"type": "custom", "avatarId": avatar},
                   "personality": init.persona, "maxDuration": int(min(SESSION_CAP_S, seconds + SETUP_MARGIN_S))}
        self.record["status"] = "submission_unknown"
        for attempt in range(CREATE_429_RETRIES + 1):
            try:
                created = client.request("POST", "/v1/realtime_sessions", request)
                break
            except RuntimeError as exc:
                # The previous session can still count against the account's concurrency limit for a few seconds
                # after DELETE; a 429 means no session was created, so waiting cannot duplicate one.
                if not str(exc).startswith("runway_http_429") or attempt == CREATE_429_RETRIES:
                    raise
                self.note("create_429_wait", attempt=attempt + 1)
                time.sleep(CREATE_429_WAIT_S)
        self.sid = created["id"]
        self.record.update(status="created", session_id=self.sid)
        for _ in range(90):
            state = client.request("GET", "/v1/realtime_sessions/" + self.sid)
            self.note("provider_readiness", status=state.get("status"))
            if state.get("status") == "READY":
                break
            if state.get("status") in ("FAILED", "CANCELLED", "COMPLETED"):
                raise RuntimeError("session " + str(state.get("status")))
            time.sleep(1)
        else:
            raise RuntimeError("session readiness timeout")
        self.join_with = client.request("POST", "/v1/realtime_sessions/" + self.sid + "/consume",
                                        token=state["sessionKey"])
        self.record["http_ready_ms"] = round((time.monotonic() - started) * 1000, 3)
        return {"session_id": self.sid, "avatar_id": avatar, "max_duration_s": request["maxDuration"]}

    def join(self):
        self.room = LiveKitRoom(self.sink, note=self.note, ended=self.ended)
        url, token = self.join_with["url"], self.join_with["token"]
        self.join_with = None
        self.room.connect(url, token)

    def end(self) -> bool:
        sid, errors = self.sid, []
        if sid:
            try:
                self.client.request("DELETE", "/v1/realtime_sessions/" + sid)
                self.note("session_deleted")
            except Exception as exc:
                errors.append(exc)
        if self.room is not None:
            self.room.close()
        if not sid:
            return self.record["status"] == "prepared"
        terminated = False
        try:
            final = self.client.request("GET", "/v1/realtime_sessions/" + sid)
            self.record.update(provider_final_status=final.get("status"), provider_duration=final.get("duration"))
            terminated = final.get("status") in ("COMPLETED", "CANCELLED", "FAILED")
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise errors[0]
        return terminated
