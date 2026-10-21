"""HeyGen LiveAvatar FULL: REST session + LiveKit room, run by the ordinary harness (providers/rtc.py).

Per session: a context (persona + known facts, empty opening text so FULL does not greet) -> POST /v1/sessions/token
(mode FULL) -> /v1/sessions/start -> LiveKit room: avatar audio/video captured in-process; user audio published as
the microphone track (digital_human_audio) or user text sent on the agent-control data topic as
avatar.speak_response (digital_human_text). close(): POST /v1/sessions/stop, then leave the room.

The public API has no single-image avatar creation, so the avatar is the pinned public stock avatar
(portrait_policy provider_stock; identity is not applicable). The plan's maximum session length bounds
max_session_duration; a session the vendor ends before the case deadline is a failure in the denominator.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
import uuid

from streameval.providers.livekit import LiveKitRoom
from streameval.providers.rtc import RtcAvatar

API = "https://api.liveavatar.com"
STOCK_AVATAR = "dd73ea75-1218-4ef3-92ce-606d5f7fbc0a"   # public stock avatar "Wayne"
STOCK_VOICE = "c2527536-6d1f-4412-a643-53a3497dada9"    # its English voice
AGENT_TOPIC, RESPONSE_TOPIC = "agent-control", "agent-response"
SETUP_MARGIN_S = 60          # join + readiness + stop inside max_session_duration
PLAN_MAX_S = 1200            # Essential plan maximum session length; confirm it for your account
CONTEXT_NAME_MAX = 64        # POST /v1/contexts rejects longer names (422)
VIDEO = {"quality": "high", "encoding": "H264"}


class Client:
    def __init__(self, key: str):
        self.key = key

    def request(self, method, path, body=None, *, bearer=None):
        headers = {"Accept": "application/json", "Content-Type": "application/json",
                   "User-Agent": "StreamEval/1.0 (benchmark client)"}
        headers["Authorization" if bearer else "X-API-KEY"] = ("Bearer " + bearer) if bearer else self.key
        encoded = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
        req = urllib.request.Request(API + path, data=encoded, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read(2048)
        try:
            value = json.loads(raw)
        except ValueError:
            value = {}
        if status >= 400 or value.get("code") not in (None, 1000):
            # error bodies can contain auth and signed URLs; report only fixed fields
            raise RuntimeError(f'liveavatar_http_{status}_code_{value.get("code", "unknown")}')
        return value.get("data")


def context_name(case_id: str) -> str:
    suffix = " " + uuid.uuid4().hex[:8]
    name = "StreamEval " + case_id + suffix
    return name if len(name) <= CONTEXT_NAME_MAX else ("SA " + case_id)[:CONTEXT_NAME_MAX - len(suffix)] + suffix


def is_avatar(identity: str) -> bool:
    return not identity.startswith("liveavatar-agent-")


class LiveAvatarFull(RtcAvatar):
    credentials_env = {"api_key": "LIVEAVATAR_API_KEY"}
    cost = {"unit": "credits", "basis": "FULL mode: 2 credits per started session minute (vendor docs as of 2026-09); "
                                        f"billed span estimated as case duration + {SETUP_MARGIN_S} s, rounded up to "
                                        "whole minutes; pay-as-you-go 1 credit = 0.10 USD"}
    session_cap_s = PLAN_MAX_S
    client_transport = "LiveKit Python SDK (livekit.rtc), in-process"
    audio_transport = "LiveKit microphone track (48 kHz mono, 10 ms frames; FULL server VAD/STT)"
    text_transport = "LiveKit data topic agent-control, event avatar.speak_response {text}"
    client = sid = join_with = None

    def estimate(self, case: dict) -> float:
        return 2 * math.ceil((case["duration_ms"] / 1000 + SETUP_MARGIN_S) / 60)

    def system_config(self, system_id: str, spec: dict) -> dict:
        return {**super().system_config(system_id, spec),
                "model": "liveavatar-full", "product": "POST /v1/sessions/token mode FULL", "api_origin": API,
                "avatar": {"avatar_id": STOCK_AVATAR, "voice_id": STOCK_VOICE, "language": "en",
                           "creation": "pinned public stock avatar; the benchmark portrait is not an avatar input",
                           "llm": "vendor default FULL-mode conversational model (context prompt only)",
                           "persona": "per-case context prompt = persona + '\\nKnown facts: ' + known facts as sorted "
                                      "JSON",
                           "greeting": "context opening_text empty (greeting disabled); any greeting is waited out "
                                       "and recorded (readiness.greeting_heard)",
                           "max_session_duration_s": f"min({PLAN_MAX_S}, ceil_minutes(case duration + "
                                                     f"{SETUP_MARGIN_S}))",
                           "session_cap": "sessions the vendor ends before the case deadline are failures",
                           "video_settings": VIDEO, "participants": "liveavatar-agent-* identities are not captured"}}

    def create(self, init) -> dict:
        client = self.client = Client(self.credentials()["api_key"])
        detail = client.request("GET", "/v1/avatars/" + STOCK_AVATAR)
        if not detail or detail.get("id") != STOCK_AVATAR or detail.get("status") != "ACTIVE":
            raise RuntimeError("pinned stock avatar is missing or not ACTIVE")
        context = client.request("POST", "/v1/contexts", {"name": context_name(init.case_id),
                                                          "prompt": init.persona, "opening_text": ""})
        if not context or not context.get("id"):
            raise RuntimeError("context_creation_missing_id")
        self.record["context_id"] = context["id"]
        seconds = min(PLAN_MAX_S, math.ceil((init.duration_ms / 1000 + SETUP_MARGIN_S) / 60) * 60)
        request = {"mode": "FULL", "is_sandbox": False, "avatar_id": STOCK_AVATAR,
                   "avatar_persona": {"voice_id": STOCK_VOICE, "context_id": context["id"], "language": "en"},
                   "video_settings": VIDEO, "max_session_duration": seconds}
        self.record["status"] = "submission_unknown"
        token = client.request("POST", "/v1/sessions/token", request)
        self.sid = token["session_id"]
        self.record["session_id"] = self.sid
        self.join_with = client.request("POST", "/v1/sessions/start", bearer=token["session_token"])
        self.record["status"] = "created"
        return {"session_id": self.sid, "max_session_duration": seconds}

    def join(self):
        self.room = LiveKitRoom(self.sink, note=self.note, ended=self.ended, accept=is_avatar,
                                data_topics=(RESPONSE_TOPIC,), publish_audio=self.publish_audio)
        url, token = self.join_with["livekit_url"], self.join_with["livekit_client_token"]
        self.join_with = None
        self.room.connect(url, token)
        for _ in range(300):   # avatar and agent participants both present
            if len(self.room.remote_identities()) >= 2:
                return
            time.sleep(0.1)
        raise RuntimeError("avatar_or_agent_not_ready")

    def send_text(self, event_id: str, text: str) -> dict:
        command_id = str(uuid.uuid4())
        self.room.publish_data({"event_id": command_id, "event_type": "avatar.speak_response",
                                "session_id": self.sid, "source_event_id": None, "text": text}, AGENT_TOPIC)
        return {"command_event_id": command_id}

    def end(self) -> bool:
        error, terminated = None, self.record["status"] == "prepared"
        if self.sid:
            try:
                self.client.request("POST", "/v1/sessions/stop", {"session_id": self.sid, "reason": "USER_CLOSED"})
                terminated = True
                self.note("session_stopped")
            except Exception as exc:
                error = exc
        if self.room is not None:
            self.note("agent_response_events", count=len(self.room.data))
            self.room.close()
        if error:
            raise error
        return terminated
