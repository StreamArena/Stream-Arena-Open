"""Tavus CVI Full: REST conversation + Daily call, run by the ordinary harness (providers/rtc.py).

Per session: POST /v2/conversations (the frozen Full PAL, the pinned stock face, per-case conversational_context,
dynamic greeting off) -> Daily call joined by the Python SDK: avatar audio/video captured in-process; user audio
written to a virtual microphone (digital_human_audio) or user text sent as the app message conversation.respond
(digital_human_text). close(): leave the call, then POST /v2/conversations/{id}/end.

Faces are trained from an HTTPS image URL, which the benchmark portraits do not have, so the face is the pinned stock
replica of pal_v1.json (portrait_policy provider_stock; identity not applicable). The PAL (persona) lives on your
account: create it once from pal_v1.json `request` and set TAVUS_PAL_ID. Before every session the live persona is
compared with the frozen one on pal_v1.json `pinned_fields`; a difference refuses the session unless
TAVUS_ALLOW_PAL_DRIFT=1 (then it is recorded in the receipt and the run is no longer the frozen configuration).
max_call_duration above the plan limit is silently truncated by Tavus; a session that ends before the case
deadline is a failure in the denominator.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import urllib.error
import urllib.request

from streameval.config import env
from streameval.providers.daily import DailyCall
from streameval.providers.rtc import RtcAvatar
from streameval.util import digest, scrub, sha256_file

API = "https://tavusapi.com/v2/conversations"
PERSONAS = "https://tavusapi.com/v2/personas/"
PAL_CONFIG = Path(__file__).with_name("pal_v1.json")
SETUP_MARGIN_S = 45


def frozen_pal() -> dict:
    return json.loads(PAL_CONFIG.read_text())


def pal_drift(live: dict, frozen: dict) -> list[str]:
    """Pinned persona fields whose live value differs from the frozen PAL."""
    return [k for k in frozen["pinned_fields"] if live.get(k) != frozen["persona"].get(k)]


class Client:
    def __init__(self, key: str):
        self.key = key

    def request(self, method, url, body=None):
        payload = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, payload, method=method,
                                     headers={"x-api-key": self.key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                raw = response.read()
                return json.loads(raw) if raw else {"http_status": response.status}
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"tavus_http_{exc.code}: " + scrub(exc.read(1000).decode(errors="replace"))) from None


class TavusCviFull(RtcAvatar):
    credentials_env = {"api_key": "TAVUS_API_KEY", "pal_id": "TAVUS_PAL_ID"}
    required_credentials = ("api_key", "pal_id")
    cost = {"unit": "USD", "basis": "CVI: connection-to-disconnection minutes billed in 6 s increments with a 30 s "
                                    "minimum, 0.37 USD/min Starter overage (vendor pricing as of 2026-08); billed span "
                                    f"estimated as case duration + {SETUP_MARGIN_S} s; included plan minutes are not "
                                    "subtracted"}
    client_transport = "Daily Python SDK (daily-python), in-process"
    audio_transport = "Daily virtual microphone (48 kHz mono, 10 ms frames; Full STT/turn-taking)"
    text_transport = "Daily app message conversation.respond {text}"
    client = cid = join_with = call = None

    def estimate(self, case: dict) -> float:
        billed = max(30, math.ceil((case["duration_ms"] / 1000 + SETUP_MARGIN_S) / 6) * 6)
        return billed / 60 * 0.37

    def system_config(self, system_id: str, spec: dict) -> dict:
        frozen = frozen_pal()
        return {**super().system_config(system_id, spec),
                "model": "tavus-cvi-full", "product": "POST /v2/conversations (Full PAL)", "api_origin": API,
                "avatar": {"face_id": frozen["face_id"], "pipeline_mode": "full",
                           "creation": "pinned stock replica; the benchmark portrait is not an avatar input",
                           "llm": "the Full PAL's default conversational LLM",
                           "persona": "conversational_context = persona + '\\nKnown facts: ' + known facts as sorted "
                                      "JSON",
                           "greeting": "dynamic_greeting false; any greeting is waited out and recorded "
                                       "(readiness.greeting_heard)",
                           "max_call_duration_s": f"case duration + {SETUP_MARGIN_S} (the plan limit truncates "
                                                  "silently)",
                           "session_cap": "sessions the vendor ends before the case deadline are failures"},
                "pal": {"config_file": "models/providers/tavus/" + PAL_CONFIG.name,
                        "config_sha256": sha256_file(PAL_CONFIG), "persona_sha256": digest(frozen["persona"]),
                        "pinned_fields": frozen["pinned_fields"], "replica_selection": frozen["replica_selection"],
                        "pal_id": "account-specific (TAVUS_PAL_ID); checked against the frozen persona per session"}}

    def create(self, init) -> dict:
        values = self.credentials()
        client = self.client = Client(values["api_key"])
        frozen = frozen_pal()
        drift = pal_drift(client.request("GET", PERSONAS + values["pal_id"]), frozen)
        if drift:
            if env("TAVUS_ALLOW_PAL_DRIFT") != "1":
                raise RuntimeError("the live PAL differs from the frozen pal_v1.json in: " + ", ".join(drift))
            self.record["pal_drift"] = drift
        body = {"pal_id": values["pal_id"], "face_id": frozen["face_id"],
                "conversation_name": "streameval-v1-" + init.case_id, "conversational_context": init.persona,
                "dynamic_greeting": False,
                "properties": {"max_call_duration": init.duration_ms // 1000 + SETUP_MARGIN_S,
                               "participant_absent_timeout": 60, "participant_left_timeout": 0}}
        self.record["status"] = "submission_unknown"
        created = client.request("POST", API, body)
        self.cid, self.join_with = created["conversation_id"], created["conversation_url"]
        self.record.update(status="created", conversation_id=self.cid)
        return {"conversation_id": self.cid, "max_call_duration": body["properties"]["max_call_duration"]}

    def join(self):
        self.room = self.call = DailyCall(self.sink, note=self.note, ended=self.ended,
                                          publish_audio=self.publish_audio)
        url, self.join_with = self.join_with, None
        self.call.join(url)

    def send_text(self, event_id: str, text: str) -> dict:
        self.call.send({"message_type": "conversation", "event_type": "conversation.respond",
                        "conversation_id": self.cid, "properties": {"text": text}})
        return {}

    def end(self) -> bool:
        error, terminated = None, self.record["status"] == "prepared"
        if self.call is not None:
            try:
                self.note("app_messages", count=len(self.call.messages))
                self.call.close()
            except Exception as exc:
                error = exc
        if self.cid:
            try:
                self.client.request("POST", API + "/" + self.cid + "/end")
                terminated = True
                self.note("conversation_ended")
            except Exception as exc:
                error = error or exc
        if error:
            raise error
        return terminated
