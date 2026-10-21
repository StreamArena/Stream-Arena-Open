"""Vidu S2 (realtime Avatar and Editing): HTTP create + WebSocket control (s2.py) + AliRTC media in headless Chromium.

One session = one create attempt -> WS control ready -> the browser (streameval/providers/js/vidu_browser.mjs)
joins the RTC channel -> readiness -> activity epoch -> TrustedScheduler releases inputs -> hangup confirmed. Python
owns creation, control, scheduling and cleanup; the browser owns only the RTC media (fake devices, synthetic user
tracks, recording). Inputs are never replayed. The account key stays in this process: the page only receives the
RTC join token of the created live.

Routes (the v1 systems):
* digital_human_audio: user WAVs published as the RTC microphone track;
* digital_human_text: each user turn as a WS type 99 text message;
* avatar_performance (derived text route, "text-driven"): S2 cannot take a supplied speech track, so each visual
  instruction goes out as "ACTION: <instruction>" and each supplied speech clip as "LINE: <its script>" (the
  manifest's ``script`` field), with action matching on and idle motion off;
* reference_edit: every condition through mapping.py (one reference + one editing_type at a time);
* instruction_edit (composition): each instruction -> one qwen-image reference (DashScope) -> the same mapping;
  instructions outside the six v1 templates are released as unsupported without a provider call.

Credentials (environment only): VIDU_S2_API_KEY (vda_...), VIDU_S2_API_BASE (https://api.vidu.cn or
https://api.vidu.com), DASHSCOPE_API_KEY (instruction_edit only). The pinned AliRTC SDK file is read from
STREAMEVAL_VIDU_SDK (default ~/.cache/streameval/providers/vidu/aliyun-rtc-sdk-7.1.9.js) and checked against
its sha256 before every session.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import socket
import sys
import time
import wave

from streameval.benchmark.tracks import SAMPLE_MS
from streameval.config import env
from streameval.providers.base import SessionBackend
from streameval.providers.browser import BROWSER_FLAGS, Browser, BrowserError
from streameval.runtime.harness import GREETING_MIN_S, IDLE_MAX_S, IDLE_NS, RUNNER, release_conditional
from streameval.runtime.media import probe_video, wav_ns
from streameval.runtime.scheduler import MS, TrustedScheduler
from streameval.util import digest, private_json, scrub, sha256_file

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """mapping.py / s2.py of this folder (system.json backends are loaded by file path, not as a package)."""
    module_name = "streameval_vidu_" + name
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, HERE / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


mapping = _sibling("mapping")
s2 = _sibling("s2")

SDK = {"version": "7.1.9", "sha256": "8100338351ce622a1d25e26997247e55b09135f458b90e748995b357ce26e376",
       "url": "https://g.alicdn.com/apsara-media-box/imp-web-rtc/7.1.9/aliyun-rtc-sdk.js"}
DEFAULT_BASE = "https://api.vidu.cn"
VOICE = "Tina"
# S2 always opens with a greeting (default: Chinese). The benchmark is English, so every session uses the same short
# in-character English greeting; the activity epoch starts only after it has ended (wait_ready).
GREETING = "Greet the visitor with one short sentence in English, in character."
PERFORMER = ("You are a performer in a live English-language video. Speak English only. Messages come in two kinds. "
             "'LINE: <text>': say exactly those words, verbatim, then stop; do not reply, comment, translate or add "
             "anything. 'ACTION: <text>': do not speak at all; stay silent and only move your face and body as "
             "instructed, then hold the final pose. Never read an ACTION aloud.")
PERFORMANCE_GREETING = "Do not greet. Stay silent and still until you receive a LINE or an ACTION."
PERFORMANCE_ROUTE = ("derived: LINE/ACTION text messages to the realtime avatar; action matching on, idle motion off; "
                     "not the supplied-audio condition")
EDITING_ALL_ROUTE = ("every condition; models/providers/vidu/mapping.py (worn garment -> virtual_tryon, else "
                     "subject_replacement; newest target wins; empty state -> style_transfer with the current "
                     "source frame)")
EDITING_T2I_ROUTE = ("composed: released instruction -> qwen-image reference (" + mapping.T2I_PROMPT + ") -> S2 "
                     "Editing via models/providers/vidu/mapping.py; before the first instruction S2 holds "
                     "style_transfer with the first source frame; instructions outside the six v1 templates are "
                     "unsupported")
T2I_COMPONENTS = [
    {"role": "reference_image_generator", "model": "qwen-image", "provider": "DashScope",
     "settings": {"size": "1024*1024", "prompt": "fixed product-photo template with the instructed target and look; "
                                                 "one image per instruction"}},
    {"role": "editor", "model": "vidu-s2", "provider": "Vidu",
     "settings": {"mapping": "newest active target wins; worn garment -> virtual_tryon, else subject_replacement; "
                             "no active target -> style_transfer with the current source frame"}}]
# a session is refused when the account holds fewer credits than this
CREDIT_FLOOR = int(env("VIDU_S2_CREDIT_FLOOR", "100000"))
# One benchmark portrait took S2 241 s to create (2026-09-29); a 120 s client timeout gave up too early.
CREATE_TIMEOUT_S = 300
from streameval.models.registry import READY_TIMEOUT_S

READY_MAX_S = READY_TIMEOUT_S
SNAPSHOT_TIMEOUT_S = 5
FAKE_AUDIO = "--use-file-for-fake-audio-capture="
# the recorded flags (the silence file's path is a run-local detail)
CONFIG_FLAGS = BROWSER_FLAGS[:2] + [FAKE_AUDIO + "<1 s digital silence>"] + BROWSER_FLAGS[2:]
# track -> (S2 product, page mode)
MODES = {"digital_human_audio": ("avatar", "audio"), "digital_human_text": ("avatar", "text"),
         "avatar_performance": ("avatar", "performance"), "reference_edit": ("editing", "editing"),
         "instruction_edit": ("editing", "editing")}
INPUTS = {"audio": "user WAV published as RTC microphone track", "text": "WS type 99 text_msg",
          "performance": "WS type 99 'LINE: <script>' / 'ACTION: <instruction>'"}


def sdk_path() -> Path:
    return Path(env("STREAMEVAL_VIDU_SDK", "~/.cache/streameval/providers/vidu/aliyun-rtc-sdk-7.1.9.js")
                ).expanduser()


def write_silence(path: Path, seconds: int = 1, rate: int = 48000) -> Path:
    """The fake microphone capture file: digital silence, so no physical device is ever opened."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * rate * seconds)
    return path


class ViduS2(SessionBackend):
    credentials_env = {"api_key": "VIDU_S2_API_KEY", "base": "VIDU_S2_API_BASE", "dashscope": "DASHSCOPE_API_KEY"}
    cost = {"unit": "credits", "per_active_second": None, "overhead_s": 0,
            "basis": "Vidu account credits per live second at the account's rate (no public fixed rate); each "
                     "session records provider_billed_seconds and provider_credits_cost"}

    def __init__(self, compose_t2i: bool = False):
        self.compose_t2i = compose_t2i
        self.required_credentials = ("api_key", "dashscope") if compose_t2i else ("api_key",)

    def base(self) -> str:
        base = (self.credentials().get("base") or DEFAULT_BASE).rstrip("/")
        if base not in s2.ORIGINS:
            raise ValueError("VIDU_S2_API_BASE must be one of " + ", ".join(s2.ORIGINS))
        return base

    # ---- declarations (no network) --------------------------------------------------------------
    def event_dispositions(self, system_id: str, spec: dict, case: dict) -> list[dict]:
        rows = super().event_dispositions(system_id, spec, case)
        if spec["track"] != "instruction_edit":
            return rows
        texts = {e["event_id"]: e["payload"].get("text") for e in case["events"]}
        out = []
        for row in rows:
            if row["supported"]:
                try:
                    mapping.parse_instruction(texts[row["event_id"]] or "")
                except ValueError:
                    row = {**row, "supported": False, "reason": "instruction_outside_v1_templates"}
            out.append(row)
        return out

    def system_config(self, system_id: str, spec: dict) -> dict:
        product, mode = MODES[spec["track"]]
        avatar, performance = product == "avatar", mode == "performance"
        reference = spec["track"] == "reference_edit"
        config = {
            "model": "vidu-s2",
            "product": "POST " + s2.ROUTES[product], "api_origin": self.base(),
            "avatar": {"voice": VOICE, "voice_model": "qwen_omni", "server_vad_silence_ms": 400, "recording": False,
                       "greeting_instruction": PERFORMANCE_GREETING if performance else GREETING,
                       **({"performer_persona": PERFORMER} if performance else {}),
                       "activity_start": f"after the first remote frame + {GREETING_MIN_S:g} s and "
                                         f"{IDLE_NS // MS} ms without speech",
                       "persona": ("rendering_context + '\\n' + performer_persona" if performance else
                                   "session persona + '\\nKnown facts: ' + known_facts as sorted JSON"),
                       "input": INPUTS[mode]} if avatar else None,
            "editing": {"editing_type": "models/providers/vidu/mapping.py (worn garment -> virtual_tryon, else "
                                        "subject_replacement; no active target -> style_transfer)",
                        "initial_reference": ("reference A inline in create (profile edit_reference_v1_preloaded_a)"
                                              if reference else "the first source frame inline in create, "
                                                                "style_transfer"),
                        "updates": "WS type 13 switch_prompt with the mapped reference image inline",
                        "source": "canvas track redrawn per decoded source frame; source resolution and rounded "
                                  "frame rate; 5000 kbps if width >= 1600 else 3000 kbps; first frame held before "
                                  "epoch"} if not avatar else None,
            "client": {"transport": "AliRTC Web SDK " + SDK["version"] + " in headless Chromium (Playwright)",
                       "sdk": dict(SDK), "browser_flags": CONFIG_FLAGS,
                       "remote_audio": "AudioWorklet capture at 48 kHz, context clock", "output_volume": 0,
                       "capture": "first presented remote frame per activity bin (requestVideoFrameCallback), page "
                                  "clock mapped to the host clock by a min-RTT round trip"},
            "readiness": ("first remote frame and remote audio, then the first frame >= 4 s old and 1500 ms without "
                          "speech (at most 25 s)" if avatar else "first presented remote frame"),
            "session_cap_ms": None, "seed": None, "output": None, "source_push": None,
            "sampling": {"frame_sample_ms": SAMPLE_MS[spec["group"]],
                         "frame_selection": "first presented frame per activity bin"},
        }
        if performance:
            config["route"] = PERFORMANCE_ROUTE
        elif reference:
            config["route"] = EDITING_ALL_ROUTE
        elif spec["track"] == "instruction_edit":
            config.update(route=EDITING_T2I_ROUTE, system_kind="composition", components=T2I_COMPONENTS,
                          unsupported="instructions outside the six v1 templates (mapping.INSTRUCTIONS) are "
                                      "released, recorded as dispatch status 'unsupported' without a provider call; "
                                      "scored as failures")
        return config

    def run(self, system_id, spec, case, assets, out, *, repeat=1, oracle=None, smoke=False, admission=None,
            verify=True) -> dict:
        return ViduJob(self, system_id, spec, case, assets, Path(out), repeat=repeat, oracle=oracle, smoke=smoke,
                       admission=admission, verify=verify).run()


class ViduJob:
    def __init__(self, backend: ViduS2, system_id: str, spec: dict, case: dict, assets: dict, out: Path, *,
                 repeat: int, oracle: dict | None, smoke: bool, admission: dict | None, verify: bool):
        from streameval.models.registry import system_config
        if (case["group"], case["input_profile"]) != (spec["group"], spec["input_profile"]):
            raise ValueError(f"case {case['case_id']} is outside the track of {system_id}")
        self.backend, self.system_id, self.spec, self.case, self.out = backend, system_id, spec, case, out
        self.track, self.case_id, self.repeat = spec["track"], case["case_id"], int(repeat)
        self.kind, self.mode = MODES[self.track]
        self.job_id = f"{self.case_id}.r{self.repeat:02d}"
        self.assets, self.oracle, self.smoke, self.verify = assets, oracle, smoke, verify
        self.admission = admission or {"case_id": self.case_id, "admitted": True, "smoke": True}
        self.config = system_config(system_id, spec)
        self.dispositions = {d["event_id"]: d for d in backend.event_dispositions(system_id, spec, case)}
        self.log, self.receipts, self.unsupported = [], {}, []
        self.signals, self.speech, self.snapshots = [], [], []
        self.speaking = False
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

    # ---- preparation (no network) ---------------------------------------------------------------
    def speech_budgets(self) -> dict:
        """Authored speech budgets (ns) for the TrustedScheduler: the oracle's audio_budget_ms. A smoke run without
        an oracle gives each clip its whole window (to the next speech release or the session end)."""
        if self.oracle is not None:
            return {c["event_id"]: c["audio_budget_ms"] * MS for c in self.oracle["checkpoints"]
                    if c.get("audio_budget_ms") is not None}
        if not self.smoke:
            raise ValueError("avatar_performance needs the case oracle (authored speech budgets)")
        speech = [e for e in self.case["events"] if e["payload"]["type"] == "speech_audio"]
        return {e["event_id"]: ((speech[i + 1]["release"]["at_ms"] if i + 1 < len(speech)
                                 else self.case["duration_ms"]) - e["release"]["at_ms"]) * MS
                for i, e in enumerate(speech)}

    def prepare(self):
        case, init = self.case, self.case["initialization"]
        seconds = case["duration_ms"] // 1000
        served, durations, source, encoder = {}, {}, None, None
        self.lines, self.budgets, self.state = {}, {}, None
        if self.mode == "performance":
            # the benchmark portrait, the rendering context as persona, no user media in the browser; each supplied
            # speech clip becomes a LINE with its script, its measured length and authored budget still bound the
            # scheduler's speech window exactly as for the supplied-audio condition
            body = s2.avatar_request(portrait=self.path(init["portrait_asset_id"]),
                                     persona=init["rendering_context"] + "\n" + PERFORMER, voice=VOICE,
                                     active_seconds=seconds, recording=False)
            body["avatar"]["greeting_instruction"] = PERFORMANCE_GREETING
            body["action"], body["idle_motion"] = {"enabled": True}, {"enabled": False}
            for event in case["events"]:
                if event["payload"]["type"] == "speech_audio":
                    asset_id = event["payload"]["asset_id"]
                    if not self.assets[asset_id].get("script"):
                        raise ValueError("supplied speech has no script in the manifest: " + asset_id)
                    durations[event["event_id"]] = wav_ns(self.path(asset_id))
                    self.lines[event["event_id"]] = self.assets[asset_id]["script"]
            self.budgets = self.speech_budgets()
        elif self.kind == "avatar":
            persona = init["persona"] + "\nKnown facts: " + json.dumps(init["known_facts"], ensure_ascii=False,
                                                                        sort_keys=True)
            body = s2.avatar_request(portrait=self.path(init["portrait_asset_id"]), persona=persona, voice=VOICE,
                                     active_seconds=seconds, recording=False)
            body["avatar"]["greeting_instruction"] = GREETING
            for event in case["events"]:
                if event["payload"]["type"] == "user_audio":
                    asset_id = event["payload"]["asset_id"]
                    path = self.path(asset_id)
                    served[event["event_id"]] = {"path": str(path), "sha256": self.assets[asset_id]["sha256"],
                                                 "mime": "audio/wav"}
                    durations[event["event_id"]] = wav_ns(path)
        else:
            self.state = mapping.EditState()
            if self.track == "reference_edit":
                mapping.apply_reference(self.state, init["reference"])
                _, asset_id, worn = self.state.newest()
                body = s2.editing_request(reference=self.path(asset_id), editing_type=mapping.editing_type(worn))
            else:
                # the create request carries the first source frame (style_transfer), snapshotted by the browser
                # in run() before the session is created; nothing is edited until an instruction
                body = None
            source_id = init["source_video_asset_id"]
            path = self.path(source_id)
            info = probe_video(path)
            served["source"] = {"path": str(path), "sha256": self.assets[source_id]["sha256"], "mime": "video/mp4"}
            source = {"url": "/asset/source", **info}
            encoder = {"width": info["width"], "height": info["height"], "frameRate": round(info["fps"]),
                       "bitrate": 5000 if info["width"] >= 1600 else 3000}
        self.body, self.seconds = body, seconds
        self.served, self.durations = served, durations
        self.source, self.encoder = source, encoder

    def binding(self) -> dict:
        return {"smoke_test_not_a_formal_sample": self.smoke, "job_id": self.job_id, "case_id": self.case_id,
                "repeat": self.repeat, "system_id": self.system_id, "system_config": self.config,
                "system_config_sha256": digest(self.config), "session_sha256": digest(self.case),
                "admission": self.admission, "assets": {k: v["sha256"] for k, v in self.served.items()},
                "audio_durations_ns": self.durations, "source": self.source, "encoder": self.encoder,
                "request_sha256": None if self.body is None else digest(self.body)}

    # ---- network ----------------------------------------------------------------------------------
    def preflight(self) -> tuple[Path, "s2.S2HTTP"]:
        missing = self.backend.missing_credentials()
        if missing:
            raise RuntimeError("credentials missing: " + ", ".join(missing))
        sdk = sdk_path()
        if not sdk.is_file() or sha256_file(sdk) != SDK["sha256"]:
            raise RuntimeError(f"AliRTC Web SDK {SDK['version']} with the pinned sha256 is not at {sdk}: download "
                               f"{SDK['url']} there (or set STREAMEVAL_VIDU_SDK)")
        credentials = self.backend.credentials()
        client = s2.S2HTTP(base=self.backend.base(), key=credentials["api_key"])
        credits = self.credits(client.base, credentials["api_key"])
        if credits is not None and credits < CREDIT_FLOOR:
            raise RuntimeError(f"Vidu credits ({credits}) below the floor of {CREDIT_FLOOR} (VIDU_S2_CREDIT_FLOOR)")
        return sdk, client

    @staticmethod
    def credits(base: str, key: str) -> int | None:
        import requests
        try:
            reply = requests.get(base + "/ent/v2/credits", timeout=20, headers={"Authorization": "Token " + key})
            return int(reply.json()["remains"][0]["credit_remain"])
        except Exception:
            return None

    def snapshot(self, browser: Browser) -> str:
        """The source frame the browser is showing now, as an inline JPEG; its receipt goes to source_snapshots."""
        shot, _, _ = browser.call("snapshot_source", timeout=SNAPSHOT_TIMEOUT_S)
        data = shot.pop("jpeg_base64")
        if not 0 < len(data) * 3 // 4 <= s2.MAX_INLINE_BYTES:
            raise ValueError("source snapshot is empty or above the 4 MiB inline limit")
        self.snapshots.append(shot)
        return "data:image/jpeg;base64," + data

    def switch(self, session, browser: Browser, eid: str, record: dict) -> None:
        """Send S2 what the mapping holds now: the newest active target's reference, or the current source frame."""
        newest = self.state.newest()
        if newest is None:
            content, editing_type = self.snapshot(browser), "style_transfer"
        else:
            content, editing_type = newest[1], mapping.editing_type(newest[2])
        if isinstance(content, bytes):
            content = "data:image/png;base64," + base64.b64encode(content).decode()
        elif not content.startswith("data:"):
            content = s2.image_uri(self.path(content))
        session.send(13, {"switch_prompt": {"prompts": [{"type": "image", "content": content}],
                                            "editing_type": editing_type}}, event_id=eid)
        record.update(transport="ws_type13", editing_type=editing_type, s2_basis=self.state.basis())

    def send_text(self, session, eid: str, text: str) -> None:
        session.send(99, {"text_msg": {"msg_id": eid, "content": text, "timestamp": int(time.time() * 1000)}},
                     event_id=eid)

    def dispatch(self, session, browser: Browser, event):
        payload, eid = event.payload, event.event_id
        record = {"event_id": eid, "released_at_ns": event.released_at_ns, "dispatch_start_ns": time.monotonic_ns()}
        self.receipts[eid] = record
        disposition = self.dispositions[eid]
        if not disposition["supported"]:
            record.update(status="unsupported", reason=disposition["reason"], provider_call=False,
                          dispatch_end_ns=time.monotonic_ns())
            self.unsupported.append({"event_id": eid, "reason": disposition["reason"]})
            return
        try:
            if self.mode == "performance":
                text = ("ACTION: " + payload["text"] if payload["type"] == "user_prompt"
                        else "LINE: " + self.lines[eid])
                self.send_text(session, eid, text)
                record.update(transport="ws_type99", derived_text=text)
            elif payload["type"] == "user_prompt":
                self.send_text(session, eid, payload["text"])
                record["transport"] = "ws_type99"
            elif payload["type"] == "user_audio":
                session.guard_input(eid)
                session.event("input_attempt", event_id=eid, semantic="user_audio", transport="browser_rtc")
                result, sent, replied = browser.call("play", {"event_id": eid}, timeout=5)
                session.event("sent", event_id=eid, transport="browser_rtc")
                record.update(transport="browser_rtc_webaudio", play=result, host_send_ns=sent, host_reply_ns=replied)
            elif payload["type"] == "reference_control":
                record["operation"] = mapping.apply_reference(self.state, payload)
                self.switch(session, browser, eid, record)
            elif payload["type"] == "text_control":
                session.guard_input(eid)
                parsed = mapping.parse_instruction(payload["text"])
                record["parsed"] = parsed
                if parsed["op"] == "restore":
                    mapping.apply_instruction(self.state, parsed)
                else:
                    category, prompt = mapping.reference_prompt(parsed["target"], parsed["look"])
                    begun = time.monotonic_ns()
                    image = mapping.generate_reference(prompt, self.backend.credentials()["dashscope"])
                    done = time.monotonic_ns()
                    record["generation"] = {"model": mapping.T2I_MODEL, "prompt": prompt, "category": category,
                                            "image_sha256": hashlib.sha256(image).hexdigest(), "bytes": len(image),
                                            "started_ns": begun, "finished_ns": done,
                                            "generation_ms": (done - begun) // MS}
                    mapping.apply_instruction(self.state, {**parsed, "look": image})
                self.switch(session, browser, eid, record)
            else:
                raise ValueError("unsupported payload " + payload["type"])
            record.update(status="sent", provider_call=True)
        except Exception as exc:
            record.update(status="failed", error=scrub(str(exc))[:1000], dispatch_end_ns=time.monotonic_ns())
            if str(exc) == "Activity deadline has elapsed":
                # the input was not ready before the activity window closed: nothing was sent, the session goes on
                record.update(not_sent=str(exc), provider_call=False)
                return
            raise   # an input whose outcome is unknown ends the attempt; it is never replayed
        record["dispatch_end_ns"] = time.monotonic_ns()

    # ---- browser events ---------------------------------------------------------------------------
    def pump(self, browser: Browser, scheduler=None):
        while True:
            try:
                msg = browser.events.get_nowait()
            except queue.Empty:
                return
            kind = msg.get("event")
            if kind in {"fatal", "process_exited", "page_error"}:
                raise BrowserError(kind + ": " + str(msg.get("error")))
            if kind in {"rtc_bye", "rtc_remote_offline"}:
                self.note(kind, **{k: v for k, v in msg.items() if k != "event"})
            if kind == "speech_onset":
                self.speaking = True
                self.speech.append(msg)
                if scheduler is not None and self.offset_ns is not None:
                    release_conditional(self.case, scheduler, self.page_to_host(msg["start_page_ms"]),
                                        self.page_to_host(msg.get("previous_end_page_ms")),
                                        f"browser/vad_online.jsonl#onset_frame={msg['start_frame']}", self.signals,
                                        self.note)
            elif kind == "speech_offset":
                self.speaking = False
                self.speech.append(msg)

    def wait_ready(self, session, browser: Browser) -> dict:
        """Output flowing, and for avatars no speech for IDLE_NS (any greeting is over)."""
        start = time.monotonic()
        first_frame = audio = False
        first_frame_at = quiet_since = None
        while time.monotonic() - start < READY_MAX_S:
            session.receive(timeout=0.02)
            while True:
                try:
                    msg = browser.events.get_nowait()
                except queue.Empty:
                    break
                kind = msg.get("event")
                if kind in {"fatal", "process_exited", "page_error"}:
                    raise BrowserError(kind + ": " + str(msg.get("error")))
                if kind == "first_remote_frame" and not first_frame:
                    first_frame, first_frame_at = True, time.monotonic()
                audio |= kind == "remote_audio_attached"
                if kind == "speech_onset":
                    self.speaking, quiet_since = True, None
                    self.speech.append({**msg, "phase": "before_epoch"})
                elif kind == "speech_offset":
                    self.speaking = False
                    self.speech.append({**msg, "phase": "before_epoch"})
            if first_frame and self.kind == "editing":
                return {"waited_s": round(time.monotonic() - start, 3), "idle": None}
            ready = first_frame and audio
            if ready and not self.speaking:
                quiet_since = quiet_since or time.monotonic_ns()
                if time.monotonic_ns() - quiet_since >= IDLE_NS and time.monotonic() - first_frame_at >= GREETING_MIN_S:
                    return {"waited_s": round(time.monotonic() - start, 3), "idle": True,
                            "greeting_heard": any(e.get("event") == "speech_onset" for e in self.speech)}
            if ready and time.monotonic() - first_frame_at >= IDLE_MAX_S:
                return {"waited_s": round(time.monotonic() - start, 3), "idle": False}
        raise BrowserError(f"output not ready (first_frame={first_frame}, audio={audio})")

    # ---- one session ------------------------------------------------------------------------------
    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=False, mode=0o700)
        private_json(self.out / RUNNER, {"host": socket.gethostname(), "pid": os.getpid(),
                                         "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.prepare()
        private_json(self.out / "binding.json", self.binding())
        scheduler = TrustedScheduler(self.case, audio_durations_ns=self.durations or None,
                                     audio_budgets_ns=self.budgets or None)
        session = browser = failure = cleanup_failure = None
        result = {"job_id": self.job_id, "case_id": self.case_id, "repeat": self.repeat,
                  "system_id": self.system_id, "system_config_sha256": digest(self.config)}
        try:
            sdk, client = self.preflight()
            session = s2.ViduS2Session(self.kind, client=client, directory=self.out / "provider",
                                       active_seconds=self.seconds)
            session.create_timeout = CREATE_TIMEOUT_S
            silence = write_silence(self.out / "fake_microphone_silence.wav")
            browser = Browser(self.out / "browser", {"sdk": {"path": str(sdk), "sha256": SDK["sha256"]},
                                                     "assets": self.served,
                                                     "flags": BROWSER_FLAGS[:2] + [FAKE_AUDIO + str(silence)]
                                                              + BROWSER_FLAGS[2:]}, script="vidu_browser.mjs")
            browser.wait_event("browser_ready", 60)
            # supplied speech on the performance route is sent as LINE text, never played by the browser
            user_audio = {eid: "/asset/" + eid for eid in self.durations} if self.mode == "audio" else {}
            result["browser_init"], _, _ = browser.call("init", {
                "mode": self.mode, "sample_ms": self.config["sampling"]["frame_sample_ms"],
                "user_audio": user_audio, "source": self.source}, timeout=180)
            if self.track == "instruction_edit":
                self.body = {"image_url": self.snapshot(browser), "editing_type": "style_transfer"}
            credentials = session.create(self.body)
            session.initialize()
            self.note("control_ready")
            result["browser_join"], _, _ = browser.call("join", {
                "rtc": credentials["rtc"], "render_uid": credentials.get("render_uid"), "kind": self.kind,
                "publish_audio": self.mode == "audio", "encoder": self.encoder}, timeout=READY_MAX_S)
            del credentials
            result["readiness"] = self.wait_ready(session, browser)
            sync = browser.clock_sync()
            self.offset_ns = sync["offset_ns"]
            result["clock_start"] = sync
            epoch = time.monotonic_ns()
            session.start_activity(epoch_ns=epoch)
            scheduler.start(epoch_ns=epoch)
            result["browser_start"], _, _ = browser.call("start", {"zero_page_ms": (epoch - self.offset_ns) / 1e6,
                                                                   "duration_ms": self.case["duration_ms"]}, timeout=10)
            deadline = epoch + self.case["duration_ms"] * MS
            while time.monotonic_ns() < deadline:
                event = scheduler.take(timeout=0.005)
                if event is not None:
                    self.dispatch(session, browser, event)
                self.pump(browser, scheduler)
                session.receive(timeout=0.005)
            scheduler.join(timeout=2)
            session.record["status"] = "activity_complete"
        except BaseException as exc:
            failure = {"type": type(exc).__name__, "message": scrub(str(exc))[:2000]}
            if session is not None and session.record["status"] != "submission_unknown":
                session.record["status"] = "run_failed"
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
            if session is not None:
                try:
                    session.close()
                except BaseException as exc:
                    cleanup_failure = {"type": type(exc).__name__, "message": scrub(str(exc))[:1000]}
            if browser is not None:
                browser.close()
        snapshot = scheduler.snapshot()
        released = {r["event_id"] for r in snapshot["opportunities"] if r["status"] == "released"}
        confirmed = session is not None and session.record["termination_confirmed"]
        metrics = session.record["metrics"] if session is not None else {}
        result.update({
            "status": "completed" if failure is None else "failed", "failure": failure,
            "cleanup_failure": cleanup_failure, "termination_confirmed": confirmed,
            "provider": {k: metrics.get(k) for k in ("provider_billed_seconds", "provider_credits_cost",
                                                     "control_ready_ms", "http_creation_ack_ms")},
            "scheduler": snapshot, "dispatch": self.receipts, "unsupported_events": self.unsupported,
            "response_signals": self.signals, "speech_events": self.speech, "notes": self.log,
            "clock_offset_ns": self.offset_ns,
            **({"source_snapshots": self.snapshots} if self.kind == "editing" else {}),
            "activity_execution_complete": (failure is None and cleanup_failure is None and confirmed
                                            and released == {e["event_id"] for e in self.case["events"]}
                                            and released == set(self.receipts)
                                            and not any("not_sent" in r for r in self.receipts.values())),
            "official_evaluation": False, "automatic_retries": 0,
        })
        private_json(self.out / "result.json", result)
        return result
