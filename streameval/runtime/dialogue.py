"""Shared dialogue stack for composition systems (renderer-only avatars that cannot hold a conversation).

ASR (qwen3-asr-flash, OpenAI-compatible mode) -> chat model (qwen3.8-max-0902, thinking off, temperature 0.3) ->
TTS (qwen3-tts-flash, voice Cherry, SSE streaming), as configured for the evaluation. The Composition worker turns each
released user turn into speech for the renderer in real time: a user clip is transcribed only after its presented
end (never early); the reply is synthesized and pushed to the model as audio input while it streams in (or sent as
a ``dialogue_reply`` text event for text-input renderers).

Environment: DASHSCOPE_API_KEY (ASR, TTS and by default the chat model), STREAMEVAL_DIALOGUE_BASE_URL,
STREAMEVAL_DIALOGUE_API_KEY, STREAMEVAL_DIALOGUE_LLM, STREAMEVAL_DIALOGUE_ASR, STREAMEVAL_DIALOGUE_TTS,
STREAMEVAL_DIALOGUE_VOICE. STREAMEVAL_DIALOGUE_STACK=offline selects a network-free test double.
"""
from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
import wave

import numpy as np

from ..config import env
from ..util import scrub
from .media import resample

TTS_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
TTS_RATE = 24000
DEFAULTS = {"asr": "qwen3-asr-flash", "llm": "qwen3.8-max-0902", "tts": "qwen3-tts-flash", "tts_voice": "Cherry"}
STYLE = ("Reply in one to three short spoken English sentences, in character, using only the known facts. "
         "No lists, markdown or stage directions.")


def dialogue_models(entry: dict) -> dict:
    """ASR / LLM / TTS model names (registry composition > environment > default)."""
    comp = entry.get("composition") or {}
    names = {"asr": "STREAMEVAL_DIALOGUE_ASR", "llm": "STREAMEVAL_DIALOGUE_LLM",
             "tts": "STREAMEVAL_DIALOGUE_TTS", "tts_voice": "STREAMEVAL_DIALOGUE_VOICE"}
    return {k: comp.get(k) or env(var) or DEFAULTS[k] for k, var in names.items()}


def components(entry: dict) -> list[dict]:
    """The components list of the system configuration ([] for a native system)."""
    if entry["system_kind"] != "composition":
        return []
    comp = entry.get("composition") or {}
    models = dialogue_models(entry)
    return [{"role": "asr", "model": models["asr"], "provider": "DashScope (compatible mode)"},
            {"role": "llm", "model": models["llm"], "provider": "DashScope (compatible mode)",
             "settings": {"enable_thinking": False, "temperature": 0.3, "max_tokens": 200,
                          "system": "persona + known facts; one to three short spoken English sentences"}},
            {"role": "tts", "model": models["tts"], "voice": models["tts_voice"], "provider": "DashScope",
             "streaming": True},
            {"role": "renderer", "model": entry["model_id"], "input": comp.get("input", "audio"),
             "audio_out": comp.get("audio_out", "client")}]


def system_prompt(case: dict) -> str:
    init = case["initialization"]
    return (init["persona"] + "\nKnown facts: " + json.dumps(init["known_facts"], ensure_ascii=False, sort_keys=True)
            + "\n" + STYLE)


class DashScopeStack:
    kind = "dashscope"

    def __init__(self, models: dict, seed: int | None = None):
        import requests
        self.models, self.seed = models, seed
        self.base_url = env("STREAMEVAL_DIALOGUE_BASE_URL",
                            "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.key = env("DASHSCOPE_API_KEY")
        self.llm_key = env("STREAMEVAL_DIALOGUE_API_KEY") or self.key
        if not self.key:
            raise RuntimeError("composition systems need DASHSCOPE_API_KEY for the shared dialogue stack")
        self.http = requests.Session()

    def _post(self, url, body, key, **kw):
        for attempt in range(3):
            r = self.http.post(url, json=body, headers={"Authorization": "Bearer " + key}, timeout=(10, 60), **kw)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(1 + attempt)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"http_{r.status_code}: " + scrub(r.text)[:300])
            return r
        raise RuntimeError("unreachable")

    def asr(self, pcm16k: np.ndarray) -> str:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(pcm16k.tobytes())
        body = {"model": self.models["asr"], "messages": [{"role": "user", "content": [{"type": "input_audio",
                "input_audio": {"data": "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()}}]}],
                "asr_options": {"enable_itn": False}}
        r = self._post(self.base_url + "/chat/completions", body, self.key)
        return r.json()["choices"][0]["message"]["content"].strip()

    def chat(self, messages: list[dict]) -> str:
        body = {"model": self.models["llm"], "messages": messages, "temperature": 0.3, "max_tokens": 200,
                "enable_thinking": False}
        if self.seed is not None:
            body["seed"] = self.seed
        r = self._post(self.base_url + "/chat/completions", body, self.llm_key)
        return r.json()["choices"][0]["message"]["content"].strip()

    def tts(self, text: str):
        """Yield 16 kHz int16 chunks while the SSE stream arrives; fall back to the final audio URL."""
        body = {"model": self.models["tts"], "input": {"text": text, "voice": self.models["tts_voice"],
                                                       "language_type": "English"}}
        r = self.http.post(TTS_ENDPOINT, json=body, stream=True, timeout=(10, 60),
                           headers={"Authorization": "Bearer " + self.key, "X-DashScope-SSE": "enable"})
        if r.status_code != 200:
            raise RuntimeError(f"tts http_{r.status_code}: " + scrub(r.text)[:300])
        streamed, url = False, None
        for raw in r.iter_lines():
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:])
            audio = (data.get("output") or {}).get("audio") or {}
            if audio.get("data"):
                pcm = np.frombuffer(base64.b64decode(audio["data"]), dtype=np.int16)
                if len(pcm):
                    streamed = True
                    yield resample(pcm, TTS_RATE, 16000)
            url = audio.get("url") or url
        if not streamed and url:
            import subprocess
            from .media import ffmpeg
            raw = self.http.get(url, timeout=(10, 60)).content
            out = subprocess.run([ffmpeg(), "-v", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "s16le",
                                  "pipe:1"], input=raw, capture_output=True, check=True).stdout
            yield np.frombuffer(out, dtype=np.int16)


class OfflineStack:
    """Network-free test double; never a formal configuration."""
    kind = "offline_stub"

    def __init__(self, models: dict | None = None, seed: int | None = None):
        self.models = models or {}

    def asr(self, pcm16k):
        return "(offline transcript)"

    def chat(self, messages):
        return "Hello, I am an offline test double."

    def tts(self, text):
        t = np.arange(int(16000 * 1.2)) / 16000
        tone = (0.3 * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        for i in range(0, len(tone), 3200):
            yield tone[i:i + 3200]


def stack_kind() -> str:
    return "offline" if env("STREAMEVAL_DIALOGUE_STACK") == "offline" else "dashscope"


def make_stack(models: dict, seed: int | None = None):
    return OfflineStack(models, seed) if stack_kind() == "offline" else DashScopeStack(models, seed)


class Composition:
    """Serial dialogue turns: user audio (after its presented end) or user text -> reply speech / text.

    out(kind, eid, payload): kind "audio" -> payload int16 16 kHz chunk to push (returns the pacer schedule);
    kind "text" -> payload reply text to send. Every turn's timings (host ns) are recorded in .turns."""

    def __init__(self, stack, case: dict, out, input_kind: str, clock=time.monotonic_ns):
        self.stack, self.out, self.input_kind, self.clock = stack, out, input_kind, clock
        self.messages = [{"role": "system", "content": system_prompt(case)}]
        self.jobs: queue.Queue = queue.Queue()
        self.turns = []
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name="arena-composition", daemon=True)
        self.thread.start()

    def submit_audio(self, event_id: str, pcm16k: np.ndarray, end_ns: int):
        self.jobs.put(("audio", event_id, pcm16k, end_ns))

    def submit_text(self, event_id: str, text: str, at_ns: int):
        self.jobs.put(("text", event_id, text, at_ns))

    def _run(self):
        while not self.stopped.is_set():
            try:
                kind, eid, data, at = self.jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            turn = {"event_id": eid, "input": kind, "input_end_ns": at}
            try:
                while self.clock() < at and not self.stopped.is_set():
                    time.sleep(min(0.01, (at - self.clock()) / 1e9))
                if kind == "audio":
                    turn["asr_start_ns"] = self.clock()
                    turn["transcript"] = self.stack.asr(data)
                    turn["asr_end_ns"] = self.clock()
                    user = turn["transcript"]
                else:
                    user = data
                self.messages.append({"role": "user", "content": user})
                turn["llm_start_ns"] = self.clock()
                reply = self.stack.chat(self.messages)
                turn["llm_end_ns"] = self.clock()
                turn["reply"] = reply
                self.messages.append({"role": "assistant", "content": reply})
                if self.stopped.is_set():
                    break
                if self.input_kind == "text":
                    turn["sent"] = self.out("text", eid, reply)
                else:
                    turn["tts_start_ns"] = self.clock()
                    schedule = []
                    for chunk in self.stack.tts(reply):
                        if self.stopped.is_set():
                            break
                        schedule.append(self.out("audio", eid, chunk))
                    turn["tts_end_ns"] = self.clock()
                    turn["speech"] = {"chunks": len(schedule),
                                      "first_page_start_ms": schedule[0]["page_start_ms"] if schedule else None,
                                      "last_page_end_ms": schedule[-1]["page_end_ms"] if schedule else None,
                                      "duration_s": round(sum(s["duration_s"] for s in schedule), 3)}
            except Exception as exc:
                turn["error"] = type(exc).__name__ + ": " + scrub(str(exc))[:500]
            self.turns.append(turn)

    def stop(self):
        self.stopped.set()
        self.thread.join(5)
