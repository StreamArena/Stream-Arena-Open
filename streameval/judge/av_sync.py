"""A-V Sync for the avatar tracks (digital human, avatar performance).

The frozen rule is evaluation/protocol/av_sync_rule_v2.json (its sha256 is recorded in every record); docs/evaluation.md
explains the rule and validation requirements. Nothing asks a model
for milliseconds:

* audio (deterministic): the captured OUTPUT audio (the avatar's voice as received by the evaluation client) on its
  presentation clock -> 10 ms levels -> speech activity and a speech envelope;
* video (judge): a vision model rates the lip opening (0 closed .. 3 wide open, or "n" not visible) of face crops cut
  at native resolution from presentation-timed output frames (browser/frame_samples.jsonl). The frames are chosen
  from the unit window only (never from the audio), shuffled, and the judge never hears the audio;
* alignment (deterministic): one A/V offset per system - the lag maximising the correlation of lip opening with the
  speech envelope over every judged frame of every judged session of the system, with a bootstrap CI over sessions
  (one frame per 250-500 ms bin cannot time a single event). tau > 0: the lips move after the sound.

A unit (required speech event) passes when its own lips follow its speech at the system offset and the offset's CI lies
within the tolerance; see the rule file for the exact order, undetermined cases and known failures.

Container video timestamps do not define client observation time. Timing uses the separately captured
presentation samples and received audio clock.

Two steps (the judge step is the only one that calls a model; decide/splice read JSON and the captured audio):
`streameval judge` runs the judge step for every judged avatar job (EVAL/<system>/<job>/units/av_sync.json) and
decides at the offset of the jobs judged so far; `streameval assemble` re-decides every system over its whole
judged population (EVAL/<system>/av_sync_offset.json, per-job units/av_sync_decision.json, and the av_sync decisions
spliced into observations.json / score.json; every other metric must come out unchanged) before results are written.
The official v1 judge is gpt-6-astra on the OpenAI Responses API (STREAMEVAL_AV_JUDGE_*); another model is allowed
but its results are not comparable with the published board.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import threading
import time

import numpy as np
from PIL import Image
import requests

from .. import REPO
from ..config import Endpoint, av_judge_endpoint
from ..util import digest
from ..scoring.report import score_prepared_run
from .prompts import sha
from .units import Run, artifact, attempts, jsonl, read_json

RULE_PATH = REPO / "evaluation/protocol/av_sync_rule_v2.json"
RULE_BYTES = RULE_PATH.read_bytes()
RULE = json.loads(RULE_BYTES)
RULE_SHA256 = hashlib.sha256(RULE_BYTES).hexdigest()
AV_VERSION = RULE["judge_version"]
JUDGE_MODEL, JUDGE_ENDPOINT = RULE["judge"]["model"], RULE["judge"]["endpoint"]
MS = 1_000_000
RATE = 48000
HOP = 480  # 10 ms at 48 kHz
GROUPS = ("digital_human", "avatar_performance")

GROUND = """Each image is a frame of a video of one animated or filmed character. For every image give the bounding box of
the character's whole face (forehead to chin, ear to ear) in that image's own pixel coordinates, or null if no face is
visible. Return only JSON: {{{schema}}}"""

MOUTH = """Each image is a crop around the face of the same character, taken from different moments of one video. The
images are shuffled: they are NOT in time order. For every image, rate how open the lips are:
0 = lips closed (touching or pressed together), 1 = slightly parted, 2 = clearly open, 3 = wide open;
"n" = the mouth cannot be seen (occluded, cut off by the crop, or the face turned away).
Judge each image on its own. Return only JSON with one entry per image id: {{{schema}}}"""

PROMPTS_SHA256 = hashlib.sha256("\n".join((GROUND, MOUTH)).encode()).hexdigest()
OPENING = (0, 1, 2, 3)


class GatewayRefused(RuntimeError):
    """Quota, arrears or authentication refusal (or no key): stop the batch (never retried)."""


# ------------------------------------------------------------------ helpers

def jpeg_bytes(image: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


# ------------------------------------------------------------------ audio: speech activity and envelope

def levels_db(pcm: np.ndarray) -> np.ndarray:
    n = len(pcm) // HOP
    frames = pcm[: n * HOP].astype(np.float64).reshape(n, HOP) / 32768.0
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def speech_hops(level: np.ndarray, rule=RULE["audio"]):
    """(speech intervals [a, b) in hops, threshold dBFS, reference dBFS); no active audio -> ([], None, None)."""
    active = level[level > rule["active_floor_dbfs"]]
    if not len(active):
        return [], None, None
    reference = float(np.percentile(active, 95))
    threshold = max(rule["floor_dbfs"], reference - rule["relative_db"])
    voiced = np.concatenate([[False], level > threshold, [False]])
    edges = np.flatnonzero(voiced[1:] != voiced[:-1])
    merged = []
    bridge = rule["bridge_pause_ms"] // rule["hop_ms"]
    for a, b in zip(edges[::2], edges[1::2]):
        if merged and a - merged[-1][1] < bridge:
            merged[-1][1] = int(b)
        else:
            merged.append([int(a), int(b)])
    keep = rule["min_speech_ms"] // rule["hop_ms"]
    return [(a, b) for a, b in merged if b - a >= keep], threshold, reference


class OutputAudio:
    """The captured output audio on its presentation clock: speech intervals and envelope in host ns."""

    def __init__(self, pcm: np.ndarray, block_hosts, block_firsts, rule=RULE["audio"]):
        self.pcm = pcm
        n = len(pcm) // HOP
        self.level = levels_db(pcm) if n else np.zeros(0)
        starts = np.arange(n, dtype=np.int64) * HOP
        firsts, hosts = np.asarray(block_firsts, dtype=np.int64), np.asarray(block_hosts, dtype=np.int64)
        k = np.clip(np.searchsorted(firsts, starts, side="right") - 1, 0, max(0, len(firsts) - 1))
        self.hop_ns = (hosts[k] + (starts - firsts[k]) * 1_000_000_000 // RATE) if n else np.zeros(0, np.int64)
        hops, self.threshold, self.reference = speech_hops(self.level, rule)
        self.speech = [(int(self.hop_ns[a]), int(self.hop_ns[b - 1]) + 10 * MS) for a, b in hops]
        if self.threshold is None:
            self.env = np.zeros(n)
        else:
            raw = np.clip((self.level - self.threshold) / max(1e-6, self.reference - self.threshold), 0, 1)
            self.env = np.convolve(raw, np.ones(5) / 5, mode="same")

    @classmethod
    def from_run(cls, run) -> "OutputAudio | None":
        pcm_path, blocks_path = run.browser / "remote_audio.s16le", run.browser / "remote_audio_blocks.jsonl"
        if not pcm_path.exists() or not blocks_path.exists():
            return None
        blocks = jsonl(blocks_path)
        if not blocks:
            return None
        pcm = np.fromfile(pcm_path, dtype=np.int16)[: sum(b["samples"] for b in blocks)]
        firsts = np.cumsum([0] + [b["samples"] for b in blocks[:-1]])
        hosts = [run.host(b["page_ms"]) for b in blocks]
        if any(b < a for a, b in zip(hosts, hosts[1:])):
            raise ValueError("audio receipts must be in non-decreasing host time")
        return cls(pcm, hosts, firsts)

    def envelope(self, times: np.ndarray) -> np.ndarray:
        if not len(self.hop_ns):
            return np.zeros(len(times))
        return np.interp(times, self.hop_ns + 5 * MS, self.env, left=0.0, right=0.0)


class AudioView:
    """What the rule sees of the audio: segments [(lo, hi, OutputAudio | None, shift_ns)]. Production: one segment,
    no shift. Known-answer probes shift it, silence it or splice in another session's speech."""

    def __init__(self, segments):
        self.segments = segments

    @classmethod
    def of(cls, audio: OutputAudio | None) -> "AudioView":
        return cls([(-2 ** 62, 2 ** 62, audio, 0)])

    def speech(self, lo: int = -2 ** 62, hi: int = 2 ** 62) -> list[tuple[int, int]]:
        out = []
        for a, b, audio, d in self.segments:
            if audio is None:
                continue
            s0, s1 = max(a, lo), min(b, hi)
            out += [(max(x + d, s0), min(y + d, s1)) for x, y in audio.speech if y + d > s0 and x + d < s1]
        return sorted(out)

    def envelope(self, times: np.ndarray) -> np.ndarray:
        out = np.zeros(len(times))
        for a, b, audio, d in self.segments:
            m = (times >= a) & (times < b)
            if audio is not None and m.any():
                out[m] = audio.envelope(times[m] - d)
        return out

    def active(self, times: np.ndarray) -> np.ndarray:
        speech = self.speech()
        if not speech:
            return np.zeros(len(times))
        starts = np.array([a for a, _ in speech], dtype=np.int64)
        ends = np.array([b for _, b in speech], dtype=np.int64)
        i = np.searchsorted(starts, times, side="right") - 1
        ok = i >= 0
        out = np.zeros(len(times))
        out[ok] = (times[ok] < ends[i[ok]]).astype(float)
        return out


def run_audio(run) -> AudioView | None:
    raw = OutputAudio.from_run(run)
    return AudioView.of(raw) if raw is not None else None


# ------------------------------------------------------------------ video: presentation-timed samples

class Frames:
    """Output frames on the host clock (frame_samples.jsonl). load(indices, boxes) -> {index: PIL image}, cropped to
    boxes[index] (native pixels) when given."""

    def __init__(self, run):
        self.run, self.samples = run, run.samples
        self.times = np.array([run.host(s["presentation_ms"]) for s in self.samples], dtype=np.int64)
        self.meta = {"source": "browser/frame_samples.jsonl", "frames": len(self.samples),
                     "presentation_basis": (self.samples[0].get("presentation_basis", "requestVideoFrameCallback")
                                            if self.samples else None),
                     "bin_ms": (self.samples[1]["scheduled_bin_start_ms"] - self.samples[0]["scheduled_bin_start_ms"]
                                if len(self.samples) > 1 else None)}

    def load(self, indices, boxes: dict | None = None) -> dict[int, Image.Image]:
        out = {}
        for i in sorted(set(int(i) for i in indices)):
            with Image.open(io.BytesIO(self.run.image(self.samples[i]))) as im:
                im = im.convert("RGB")
                box = (boxes or {}).get(i)
                out[i] = im.crop(tuple(int(v) for v in box)) if box else im
        return out

    def size(self, i: int):
        s = self.samples[i]
        return s.get("width"), s.get("height")


# ------------------------------------------------------------------ units and frame selection

def unit_windows(run, rel, plan) -> list[dict]:
    """[{unit_id, event_id, window: (lo, hi) host ns | None, status}] for every applicable av_sync unit."""
    units = [u for u in plan["metrics"].get("av_sync", {}).get("units", []) if u["applicable"]]
    session = rel.sessions[run.case_id]
    snap = run.result["scheduler"]
    rows = {r["event_id"]: r for r in snap["opportunities"]}
    out = []
    if plan["group"] == "digital_human":
        releases = sorted((r["released_at_ns"], r["event_id"]) for r in snap["opportunities"] if r["released_at_ns"])
        audio_mode = session["input_profile"] == "conversation_audio"
        for u in units:
            eid = u["event_id"]
            row = rows.get(eid, {})
            released = row.get("released_at_ns")
            if released is None or row.get("status") != "released":
                out.append({"unit_id": u["unit_id"], "event_id": eid, "window": None,
                            "status": "not_released: " + str(row.get("status"))})
                continue
            if audio_mode:
                play = run.result["dispatch"].get(eid, {}).get("play")
                origin = run.host(play["page_end_ms"]) if play else None
            else:
                origin = released
            nxt = next((t for t, _ in releases if t > released), None)
            end = min(x for x in (nxt, run.deadline, run.finished) if x)
            ok = origin is not None and end > origin
            out.append({"unit_id": u["unit_id"], "event_id": eid, "window": (origin, end) if ok else None,
                        "status": "ok" if ok else "no_input_end_receipt"})
    else:
        for u in units:
            eid = u["event_id"]
            row = rows.get(eid, {})
            a, b = u["check_window_ms"]
            lo, hi = run.epoch + a * MS, min(run.epoch + b * MS, run.finished or run.deadline)
            ok = row.get("status") == "released" and hi > lo
            out.append({"unit_id": u["unit_id"], "event_id": eid, "window": (lo, hi) if ok else None,
                        "status": "ok" if ok else "not_released: " + str(row.get("status"))})
    return out


def _thin(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(k * step)] for k in range(n)]


def select_frames(times: np.ndarray, window, rule=RULE["frames"]) -> dict:
    """Audio-independent: every frame in the first dense_s of the window, then the first frame of each sparse cell."""
    lo, hi = window
    inside = [int(i) for i in np.flatnonzero((times >= lo) & (times < hi))]
    dense_end = lo + rule["dense_s"] * 1000 * MS
    dense = [i for i in inside if times[i] < dense_end]
    sparse, cells = [], set()
    for i in inside:
        if times[i] >= dense_end:
            cell = (int(times[i]) - dense_end) // (rule["sparse_step_ms"] * MS)
            if cell not in cells:
                cells.add(cell)
                sparse.append(i)
    cap = rule["max_frames_per_unit"]
    sparse = _thin(sparse, max(0, cap - len(dense)))
    dense = _thin(dense, cap - len(sparse))
    kinds = {**{i: "dense" for i in dense}, **{i: "sparse" for i in sparse}}
    return {"indices": sorted(kinds), "kinds": {str(k): v for k, v in sorted(kinds.items())}}


# ------------------------------------------------------------------ judge (OpenAI Responses API)

class MouthJudge:
    """One request per batch; retries transport errors, rate limits and one malformed reply. Gateway refusals
    (quota, arrears, authentication) and a missing endpoint or key raise GatewayRefused."""

    def __init__(self, endpoint: Endpoint | None = None, model: str | None = None,
                 batch: int = RULE["judge"]["batch"]):
        endpoint = endpoint or av_judge_endpoint()
        if not endpoint.base_url or not endpoint.api_key:
            raise GatewayRefused("A-V Sync judge: set STREAMEVAL_AV_JUDGE_BASE_URL and STREAMEVAL_AV_JUDGE_API_KEY")
        base = endpoint.base_url.rstrip("/")
        self.model = model or endpoint.model
        self.endpoint = base if base.endswith("/responses") else base + "/responses"
        self.key, self.batch = endpoint.api_key, batch
        self.local = threading.local()
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        self.lock = threading.Lock()

    def _request(self, content: list, max_tokens: int) -> tuple[dict, dict]:
        body = {"model": self.model, "input": [{"role": "user", "content": content}], "max_output_tokens": max_tokens}
        record = {"model": self.model, "endpoint": self.endpoint, "judge_version": AV_VERSION,
                  "prompts_sha256": PROMPTS_SHA256, "request_sha256": sha(json.dumps(body, sort_keys=True).encode()),
                  "attempts": 0}
        return body, record

    def _post(self, content: list, validate, max_tokens: int) -> dict:
        body, record = self._request(content, max_tokens)
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
        malformed = 0
        for attempt in range(8):
            record["attempts"] = attempt + 1
            try:
                r = self.local.s.post(self.endpoint, json=body, timeout=(10, 300),
                                      headers={"Authorization": "Bearer " + self.key})
                text = r.text[:800]
                low = text.lower()
                if r.status_code in (401, 402, 403) or any(w in low for w in (
                        "insufficient_quota", "arrear", "quota exceeded", "exceeded your current quota", "billing",
                        "invalid api key", "invalid_api_key")):
                    raise GatewayRefused(f"http_{r.status_code}: {text[:300]}")
                if r.status_code in (408, 409, 429, 500, 502, 503, 504, 520, 522, 524):
                    record["last_error"] = f"http_{r.status_code}"
                    time.sleep(min(120, 5 * 2 ** attempt))
                    continue
                if r.status_code != 200:
                    record.update(status="failed", error=f"http_{r.status_code}: {text[:300]}")
                    return record
                envelope = r.json()
                reply_text = "".join(c.get("text", "") for o in envelope.get("output", [])
                                     for c in (o.get("content") or []) if c.get("type") == "output_text").strip()
                start, end = reply_text.find("{"), reply_text.rfind("}")
                reply = json.loads(reply_text[start:end + 1])
                if not validate(reply):
                    record["invalid_reply"] = reply_text[:1500]
                    raise ValueError("reply does not follow the schema")
                usage = envelope.get("usage") or {}
                with self.lock:
                    self.usage["requests"] += 1
                    self.usage["input_tokens"] += usage.get("input_tokens", 0)
                    self.usage["output_tokens"] += usage.get("output_tokens", 0)
                    self.usage["reasoning_tokens"] += (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
                record.update(status="completed", reply=reply, usage=usage, returned_model=envelope.get("model"))
                return record
            except GatewayRefused:
                raise
            except requests.RequestException as exc:
                record["last_error"] = type(exc).__name__
                time.sleep(min(120, 5 * 2 ** attempt))
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                malformed += 1
                record["last_error"] = str(exc)[:300]
                if malformed >= 2:
                    break
        record.setdefault("status", "failed")
        return record

    @staticmethod
    def _image(label: str, data: bytes) -> list:
        return [{"type": "input_text", "text": label},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}]

    def ground(self, images: list[tuple[str, bytes, tuple[int, int]]]) -> dict:
        content = []
        for fid, data, (w, h) in images:
            content += self._image(f"Image {fid} ({w}x{h} pixels):", data)
        content.append({"type": "input_text", "text": GROUND.format(
            schema=", ".join(f'"{fid}": [x0, y0, x1, y1] or null' for fid, _, _ in images))})
        ids = {fid for fid, _, _ in images}

        def ok(reply):
            return set(reply) == ids and all(
                v is None or (isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v))
                for v in reply.values())
        record = self._post(content, ok, 2000)
        record.update(kind="ground", images=[{"id": fid, "sha256": sha(data), "size": list(size)}
                                             for fid, data, size in images])
        return record

    def mouth(self, images: list[tuple[str, bytes]]) -> dict:
        content = []
        for fid, data in images:
            content += self._image(f"Image {fid}:", data)
        content.append({"type": "input_text", "text": MOUTH.format(
            schema=", ".join(f'"{fid}": <0|1|2|3|"n">' for fid, _ in images))})
        ids = {fid for fid, _ in images}

        def ok(reply):
            return set(reply) == ids and all(v in OPENING or v == "n" for v in reply.values())
        record = self._post(content, ok, 3000)
        record.update(kind="mouth", images=[{"id": fid, "sha256": sha(data)} for fid, data in images])
        return record


class ReplayMouthJudge(MouthJudge):
    """Answers each request with the stored record of the identical request (by request_sha256); a request the stored
    job never made is a failed record ("not_in_replay"). No network call."""

    def __init__(self, records: dict, model: str = JUDGE_MODEL, endpoint: str = JUDGE_ENDPOINT,
                 batch: int = RULE["judge"]["batch"]):
        self.records, self.model, self.endpoint, self.key, self.batch = records, model, endpoint, None, batch
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        self.lock = threading.Lock()
        self.misses: list[dict] = []

    def _post(self, content: list, validate, max_tokens: int) -> dict:
        _, record = self._request(content, max_tokens)
        stored = self.records.get(record["request_sha256"])
        if stored is None:
            with self.lock:
                self.misses.append({"request_sha256": record["request_sha256"]})
            return {**record, "status": "failed", "error": "not_in_replay"}
        return copy.deepcopy(stored)


def replay_mouth_judge(job_dir: Path) -> ReplayMouthJudge | None:
    """The A-V Sync judge answering from a judged job folder's units/av_sync.json (None when it has none)."""
    path = Path(job_dir) / "units" / "av_sync.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return ReplayMouthJudge({r["request_sha256"]: r for r in record.get("requests", []) if r.get("request_sha256")},
                            record.get("judge_model") or JUDGE_MODEL, record.get("judge_endpoint") or JUDGE_ENDPOINT)


def face_boxes(frames: Frames, picks: list[int], judge, rule=RULE["frames"]) -> tuple[dict, list]:
    """{frame index: face box in native pixels} for the grounding frames (one request per 8 frames)."""
    images = frames.load(picks)
    boxes, requests_ = {}, []
    order = [i for i in picks if i in images]
    for k in range(0, len(order), 8):
        payload, scale = [], {}
        for i in order[k:k + 8]:
            im = images[i].copy()
            full = im.size
            im.thumbnail((rule["ground_max_side"], rule["ground_max_side"]))
            scale[f"g{i}"] = (full[0] / im.size[0], full[1] / im.size[1], full)
            payload.append((f"g{i}", jpeg_bytes(im, 90), im.size))
        record = judge.ground(payload)
        requests_.append(record)
        if record["status"] != "completed":
            continue
        for fid, box in record["reply"].items():
            if box is None:
                continue
            sx, sy, (w, h) = scale[fid]
            x0, x1 = sorted((box[0] * sx, box[2] * sx))
            y0, y1 = sorted((box[1] * sy, box[3] * sy))
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                boxes[int(fid[1:])] = [max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)]
    return boxes, requests_


def crop_box(t: int, grounds: list[tuple[int, list]], size, margin: float) -> list[int]:
    """Union of the nearest grounding boxes before and after time t, expanded by margin on every side."""
    before = [b for g, b in grounds if g <= t]
    after = [b for g, b in grounds if g > t]
    near = ([before[-1]] if before else []) + ([after[0]] if after else [])
    x0, y0 = min(b[0] for b in near), min(b[1] for b in near)
    x1, y1 = max(b[2] for b in near), max(b[3] for b in near)
    w, h = x1 - x0, y1 - y0
    return [max(0, int(x0 - margin * w)), max(0, int(y0 - margin * h)),
            min(int(size[0]), int(math.ceil(x1 + margin * w))), min(int(size[1]), int(math.ceil(y1 + margin * h)))]


def label_crops(crops: dict, judge, seed: str) -> tuple[dict, list]:
    """{key: 0..3 | "n"} for {key: PIL crop}, from shuffled batches."""
    order = sorted(crops)
    random.Random(int(sha(seed.encode())[:16], 16)).shuffle(order)
    labels, requests_ = {}, []
    for k in range(0, len(order), judge.batch):
        part = order[k:k + judge.batch]
        payload = [(f"f{n:03d}", jpeg_bytes(crops[i])) for n, i in enumerate(part)]
        record = judge.mouth(payload)
        record["frames"] = {fid: i for (fid, _), i in zip(payload, part)}
        requests_.append(record)
        if record["status"] == "completed":
            for fid, i in record["frames"].items():
                labels[i] = record["reply"][fid]
    return labels, requests_


def ground_unit(frames: Frames, indices: list[int], judge, rule=RULE["frames"]):
    """(crop boxes {index: box} or None when no face was found, grounding {index: box}, requests)."""
    t = frames.times[indices]
    step = rule["ground_step_ms"] * MS
    picks = sorted({indices[0], indices[-1]} |
                   {indices[int(np.argmin(np.abs(t - g)))] for g in range(int(t[0]), int(t[-1]), step)})
    found, reqs = face_boxes(frames, picks, judge, rule)
    grounds = [(int(frames.times[i]), found[i]) for i in picks if i in found]
    grounding = {str(i): found.get(i) for i in picks}
    if not grounds:
        return None, grounding, reqs
    boxes = {i: crop_box(int(frames.times[i]), grounds, frames.size(i), rule["crop_margin"]) for i in indices}
    return boxes, grounding, reqs


# ------------------------------------------------------------------ step 1: the judged evidence of one job

def unit_evidence(run, rel, plan, audio: AudioView | None, frames: Frames | None) -> list[dict]:
    """Deterministic part of every unit: window, speech inside it, frame selection (no judge call)."""
    out = []
    for w in unit_windows(run, rel, plan):
        u = {"unit_id": w["unit_id"], "event_id": w["event_id"], "status": w["status"],
             "window_ns": list(w["window"]) if w["window"] else None}
        if w["window"] is None:
            u["missing"] = "required speech event " + w["status"]
        elif audio is None:
            u["missing"] = "no output audio captured"
        else:
            lo, hi = w["window"]
            speech = audio.speech(lo, hi)
            u["speech_ns"] = [list(s) for s in speech]
            u["voiced_ms"] = round(sum(b - a for a, b in speech) / MS)
            if u["voiced_ms"] < RULE["unit"]["min_required_speech_ms"]:
                u["missing"] = "no required speech in the unit window"
            elif frames is None or not len(frames.times):
                u["missing"] = "no output video captured"
            else:
                u["selection"] = select_frames(frames.times, w["window"])
                if not u["selection"]["indices"]:
                    u["missing"] = "no presented output frame in the unit window"
        out.append(u)
    return out


def judge_evidence(frames: Frames, units: list[dict], judge, seed: str) -> list:
    """Ground and label the selected frames of every unit with frames (in place); returns the requests."""
    requests_ = []
    for u in units:
        if "selection" not in u or u.get("missing"):
            continue
        idx = u["selection"]["indices"]
        boxes, u["grounding"], reqs = ground_unit(frames, idx, judge)
        requests_ += reqs
        if boxes is None:
            u["undetermined"] = "no face found in the unit's frames"
            continue
        labels, reqs = label_crops(frames.load(idx, boxes), judge, seed + u["unit_id"])
        requests_ += reqs
        if len(labels) < len(idx):
            u["undetermined"] = "judge request failed"
            continue
        u["frames"] = [{"index": i, "host_ns": int(frames.times[i]), "kind": u["selection"]["kinds"][str(i)],
                        "crop": boxes[i], "opening": labels[i]} for i in idx]
    return requests_


def job_record(run, rel, plan, judge) -> dict:
    raw = OutputAudio.from_run(run)
    audio = AudioView.of(raw) if raw is not None else None
    frames = Frames(run) if run.samples else None
    units = unit_evidence(run, rel, plan, audio, frames)
    requests_ = judge_evidence(frames, units, judge, run.job_id) if frames is not None else []
    return {"unit_id": "av_sync", "job_id": run.job_id, "case_id": run.case_id, "system_id": run.system_id,
            "attempt": run.dir.name, "judge_version": AV_VERSION, "rule_sha256": RULE_SHA256,
            "prompts_sha256": PROMPTS_SHA256, "judge_model": judge.model, "judge_endpoint": judge.endpoint,
            "final": True,
            "audio": None if raw is None else {"source": "browser/remote_audio.s16le + remote_audio_blocks.jsonl",
                                               "threshold_dbfs": raw.threshold, "reference_dbfs": raw.reference,
                                               "speech_ns": [list(s) for s in raw.speech]},
            "video": None if frames is None else frames.meta, "units": units, "requests": requests_}


def load_record(out: Path, attempt: str | None = None):
    """The job's complete record for the current rule (and attempt), else None (missing, stale or a failed request)."""
    path = out / "units" / "av_sync.json"
    record = read_json(path)
    context = read_json(out / "judge_context.json")
    if record and context:
        if (record.get("judge_version") == AV_VERSION and record.get("rule_sha256") == RULE_SHA256
                and record.get("final") and (attempt is None or record.get("attempt") == attempt)
                and record.get("context_sha256") == digest(context)
                and all(r.get("status") == "completed" for r in record.get("requests", []))):
            return record
    return None


# ------------------------------------------------------------------ step 2: system offset and unit decisions

def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def lag_grid(rule=RULE["offset"]) -> np.ndarray:
    lo, hi = rule["search_ms"]
    return np.arange(lo, hi + 1, rule["step_ms"])


def visible_frames(record: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per unit with judged frames: (times, openings) of its visible frames."""
    out = []
    for u in record["units"]:
        vis = [f for f in u.get("frames") or [] if f["opening"] != "n"]
        if vis:
            out.append((np.array([f["host_ns"] for f in vis], dtype=np.int64),
                        np.array([f["opening"] for f in vis], dtype=float)))
    return out


def session_moments(record: dict, audio: AudioView, grid: np.ndarray):
    """Session-centred sufficient statistics over the lag grid: (C[tau], Vo, Vx[tau]) or None without frames."""
    parts = visible_frames(record)
    if not parts:
        return None
    t = np.concatenate([p[0] for p in parts])
    o = np.concatenate([p[1] for p in parts])
    oc = o - o.mean()
    x = np.stack([audio.envelope(t - int(tau) * MS) for tau in grid])
    xc = x - x.mean(axis=1, keepdims=True)
    return xc @ oc, float(oc @ oc), (xc * xc).sum(axis=1)


def argmax_lag(curve: np.ndarray, grid: np.ndarray) -> float:
    top = curve.max()
    at = np.flatnonzero(curve >= top - 1e-12)
    blocks, start = [], at[0]
    for a, b in zip(at, at[1:]):
        if b != a + 1:
            blocks.append((start, a))
            start = b
    blocks.append((start, at[-1]))
    widest = max(b - a for a, b in blocks)
    chosen = min((blk for blk in blocks if blk[1] - blk[0] == widest), key=lambda blk: abs(grid[blk[0]] + grid[blk[1]]))
    return float((grid[chosen[0]] + grid[chosen[1]]) / 2)


def pooled_curve(moments: list, weights=None) -> np.ndarray:
    """Pooled session-centred correlation over the lag grid; weights = bootstrap multiplicities."""
    c = np.stack([m[0] for m in moments])
    vo = np.array([m[1] for m in moments])
    vx = np.stack([m[2] for m in moments])
    w = np.ones(len(moments)) if weights is None else np.asarray(weights, dtype=float)
    num, den = w @ c, np.sqrt((w @ vo) * (w @ vx))
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def system_offset(moments: dict, rule=RULE["offset"]) -> dict:
    """moments: {job_id: session moments}. Point estimate, bootstrap CI over sessions, offset condition."""
    grid = lag_grid(rule)
    jobs = sorted(j for j, m in moments.items() if m is not None and m[1] > 0)
    base = {"sessions": len(jobs), "min_sessions": rule["min_sessions"], "tolerance_ms": rule["tolerance_ms"]}
    if not jobs:
        return {**base, "tau_ms": None, "ci95_ms": None, "condition": "undetermined",
                "reason": "no judged session with changing lip opening"}
    ms = [moments[j] for j in jobs]
    curve = pooled_curve(ms)
    tau = argmax_lag(curve, grid)
    cis = None
    if len(jobs) >= 2:
        draws, seed = 2000, 20260930
        rng = np.random.default_rng(seed)
        c, vo, vx = np.stack([m[0] for m in ms]), np.array([m[1] for m in ms]), np.stack([m[2] for m in ms])
        taus = []
        for _ in range(draws):
            w = np.bincount(rng.integers(0, len(ms), len(ms)), minlength=len(ms)).astype(float)
            den = np.sqrt((w @ vo) * (w @ vx))
            taus.append(argmax_lag(np.where(den > 0, (w @ c) / np.where(den > 0, den, 1.0), 0.0), grid))
        taus.sort()
        cis = [taus[math.ceil(0.025 * draws) - 1], taus[math.ceil(0.975 * draws) - 1]]
    tol = rule["tolerance_ms"]
    if len(jobs) < rule["min_sessions"]:
        condition, reason = "undetermined", f"fewer than {rule['min_sessions']} judged sessions"
    elif -tol <= cis[0] and cis[1] <= tol:
        condition, reason = "pass", "95% CI within the tolerance"
    elif cis[1] < -tol or cis[0] > tol:
        condition, reason = "fail", "95% CI entirely outside the tolerance"
    else:
        condition, reason = "undetermined", "95% CI straddles the tolerance"
    return {**base, "tau_ms": tau, "ci95_ms": cis, "correlation": round(float(curve.max()), 4),
            "at_zero": round(float(curve[grid == 0][0]), 4), "condition": condition, "reason": reason,
            "curve": [round(float(c), 4) for c in curve], "grid_ms": [int(grid[0]), int(grid[-1]), int(grid[1] - grid[0])],
            "bootstrap": {"draws": 2000, "seed": 20260930, "unit": "session"}, "jobs": jobs}


def decide_units(record: dict, audio: AudioView | None, offset: dict, rule=RULE["decision"]) -> list[dict]:
    """Every unit's decision under the frozen order (rule file, decision.order)."""
    out = []
    tau = offset.get("tau_ms")
    for u in record["units"]:
        d = {"unit_id": u["unit_id"]}
        if u.get("missing"):
            out.append({**d, "decision": "output_missing", "reason": u["missing"]})
            continue
        if audio is None or u.get("voiced_ms", 0) < RULE["unit"]["min_required_speech_ms"]:
            out.append({**d, "decision": "output_missing", "reason": "no required speech in the unit window"})
            continue
        if u.get("undetermined"):
            out.append({**d, "decision": "undetermined", "reason": u["undetermined"]})
            continue
        frames = u.get("frames") or []
        vis = [f for f in frames if f["opening"] != "n"]
        t = np.array([f["host_ns"] for f in vis], dtype=np.int64)
        o = np.array([f["opening"] for f in vis], dtype=float)
        s = audio.active(t) if len(t) else np.zeros(0)
        d.update(frames=len(frames), visible=len(vis), speech_frames=int(s.sum()), silence_frames=int(len(s) - s.sum()))
        if len(vis) < (1 - rule["max_not_visible_fraction"]) * len(frames):
            out.append({**d, "decision": "undetermined", "reason": "mouth not visible in most judged frames"})
        elif d["speech_frames"] < rule["min_speech_frames"] or d["silence_frames"] < rule["min_silence_frames"]:
            out.append({**d, "decision": "undetermined", "reason": "too few judged frames in speech or in silence"})
        elif o.std() == 0:
            out.append({**d, "decision": "fail", "reason": "the lip opening never changes over the event"})
        elif tau is None or offset["sessions"] < RULE["offset"]["min_sessions"]:
            out.append({**d, "decision": "undetermined", "reason": "system offset not estimable: "
                        + offset.get("reason", "too few sessions")})
        else:
            r = correlation(o, audio.envelope(t - int(round(tau)) * MS))
            d["correlation_at_offset"] = round(r, 4)
            if r < rule["min_correlation"]:
                out.append({**d, "decision": "fail", "reason": "the lips do not follow this event's speech"})
            elif offset["condition"] == "fail":
                side = "after" if tau > 0 else "before"
                out.append({**d, "decision": "fail", "reason": f"system lips move {abs(tau):.0f} ms {side} the sound "
                                                               f"(95% CI {offset['ci95_ms']})"})
            elif offset["condition"] == "undetermined":
                out.append({**d, "decision": "undetermined", "reason": "system offset " + offset["reason"]})
            else:
                out.append({**d, "decision": "pass", "reason": "lips follow the speech; system offset within tolerance"})
    return out


DECISION_STATUS = {"pass": ("observed", 100), "fail": ("observed", 0), "output_missing": ("output_missing", None),
                   "undetermined": ("undetermined", None)}


def metric_decisions(units: list[dict], path: Path, root: Path) -> list[dict]:
    """av_sync metric decisions (bounded_score units: a passing event scores 100, a failing one 0)."""
    out = []
    for u in units:
        status, value = DECISION_STATUS[u["decision"]]
        d = {"unit_id": u["unit_id"], "status": status, "evidence": artifact(path, root)}
        if value is not None:
            d["value"] = value
        out.append(d)
    return out


def judge_of(record: dict | None) -> tuple[str, str]:
    """(model, endpoint) that produced a record; the frozen v1 judge when unknown."""
    record = record or {}
    return record.get("judge_model") or JUDGE_MODEL, record.get("judge_endpoint") or JUDGE_ENDPOINT


def splice(out: Path, decisions: list[dict], prep: Path, root: Path, judge: tuple[str, str]) -> dict:
    """Replace the job's av_sync decisions and re-score; every other metric must come out unchanged."""
    obs_path, score_path = out / "observations.json", out / "score.json"
    bundle = json.loads(obs_path.read_text())
    bundle["metric_decisions"]["av_sync"] = decisions
    old = json.loads(score_path.read_text())
    tmp = out / "observations.av_sync.json"
    tmp.write_text(json.dumps(bundle, indent=1) + "\n")
    try:
        new = score_prepared_run(prep, tmp, evidence_root=root)
        for name, value in old["metrics"].items():
            if name != "av_sync" and new["metrics"].get(name) != value:
                raise RuntimeError(f"{out.name}: re-scoring changed {name}")
        tmp.replace(obs_path)
    finally:
        tmp.unlink(missing_ok=True)
    new = score_prepared_run(prep, obs_path, evidence_root=root)
    merged = {**old, **{k: new[k] for k in ("metrics", "observations_sha256", "verified_evidence_artifacts")},
              "av_sync_version": AV_VERSION, "av_sync_rule_sha256": RULE_SHA256,
              "av_sync_judge_model": judge[0], "av_sync_judge_endpoint": judge[1]}
    write_json(score_path, merged)
    return merged["metrics"]["av_sync"]


# ------------------------------------------------------------------ drivers

def scored_attempts(runs: Path, out_root: Path, system: str | None, rel) -> list[tuple[Path, dict]]:
    """(attempt, score) for every judged avatar job whose score.json names that attempt and whose run started."""
    found = []
    for attempt in sorted(attempts(runs, system, None)):
        b = json.loads((attempt / "binding.json").read_text())
        if rel.plans.get(b["case_id"], {}).get("group") not in GROUPS:
            continue
        score_path = out_root / b["system_id"] / b["job_id"] / "score.json"
        if not score_path.exists():
            continue
        score = json.loads(score_path.read_text())
        if score.get("attempt") == attempt.name and score.get("run_started") and not score.get("unsupported"):
            found.append((attempt, score))
    return found


def job_moments(out: Path, record: dict, audio: AudioView | None):
    """The session's pooled-correlation moments (cached in units/av_sync_moments.json), or None."""
    path = out / "units" / "av_sync_moments.json"
    record_sha = sha((out / "units" / "av_sync.json").read_bytes())
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("rule_sha256") == RULE_SHA256 and cached.get("record_sha256") == record_sha:
            return None if cached["c"] is None else (np.array(cached["c"]), cached["vo"], np.array(cached["vx"]))
    m = session_moments(record, audio, lag_grid()) if audio is not None else None
    write_json(path, {"job_id": record["job_id"], "rule_sha256": RULE_SHA256, "record_sha256": record_sha,
                      "c": None if m is None else m[0].tolist(), "vo": None if m is None else m[1],
                      "vx": None if m is None else m[2].tolist()})
    return m


def offset_file(out_root: Path, system: str, moments: dict, judge: tuple[str, str]) -> tuple[dict, Path]:
    offset = {"system_id": system, "judge_version": AV_VERSION, "rule_sha256": RULE_SHA256,
              "judge_model": judge[0], "judge_endpoint": judge[1], **system_offset(moments)}
    path = out_root / system / "av_sync_offset.json"
    write_json(path, offset)
    return offset, path


def pooled_moments(out_root: Path, system: str) -> dict:
    """Moments of every job of the system whose cached moments match its current complete record."""
    out = {}
    for path in sorted((out_root / system).glob("*/units/av_sync_moments.json")):
        cached = json.loads(path.read_text())
        rec = path.parent / "av_sync.json"
        if (cached.get("rule_sha256") == RULE_SHA256 and rec.exists() and cached.get("c") is not None
                and cached.get("record_sha256") == sha(rec.read_bytes())
                and load_record(path.parent.parent) is not None):
            out[cached["job_id"]] = (np.array(cached["c"]), cached["vo"], np.array(cached["vx"]))
    return out


def decision_file(out: Path, job_id: str, units: list[dict], offset: dict, offset_path: Path, root: Path) -> Path:
    path = out / "units" / "av_sync_decision.json"
    write_json(path, {"unit_id": "av_sync_decision", "job_id": job_id, "judge_version": AV_VERSION,
                      "rule_sha256": RULE_SHA256, "record": artifact(out / "units" / "av_sync.json", root),
                      "system_offset": {**artifact(offset_path, root), **{k: offset[k] for k in
                                                                          ("tau_ms", "ci95_ms", "condition", "sessions")}},
                      "units": units})
    return path


def job_decisions(run, out: Path, record: dict, out_root: Path) -> list[dict]:
    """av_sync metric decisions of one freshly judged job, at the system offset of every job judged so far
    (`streameval assemble` re-decides the whole system once its population is complete)."""
    audio = run_audio(run)
    job_moments(out, record, audio)
    offset, offset_path = offset_file(out_root, run.system_id, pooled_moments(out_root, run.system_id),
                                      judge_of(record))
    units = decide_units(record, audio, offset)
    return metric_decisions(units, decision_file(out, run.job_id, units, offset, offset_path, run.root), run.root)


def needs_av_sync(score: dict, group: str) -> bool:
    """A judged, started avatar job whose score does not carry the current A-V Sync version yet."""
    return (group in GROUPS and bool(score.get("run_started")) and not score.get("unsupported")
            and score.get("av_sync_version") != AV_VERSION)


def av_sync_job(attempt: Path, out_root: Path, rel, judge, root: Path) -> dict:
    """`streameval judge` hook for one judged job: judge (or reuse) its record, decide at the current system offset
    and splice the decisions into observations.json / score.json."""
    run = Run(attempt, root)
    out = out_root / run.system_id / run.job_id
    record = load_record(out, attempt.name)
    if record is None:
        if judge is None:
            return {"job_id": run.job_id, "av_sync": "no record"}
        record = job_record(run, rel, rel.plans[run.case_id], judge)
        record["context_sha256"] = digest(json.loads((out / "judge_context.json").read_text()))
        write_json(out / "units" / "av_sync.json", record)
        if not all(r.get("status") == "completed" for r in record["requests"]):
            return {"job_id": run.job_id, "av_sync": "a judge request failed; retried on the next pass"}
    metric = splice(out, job_decisions(run, out, record, out_root), rel.preparation(run.case_id), root,
                    judge_of(record))
    return {"job_id": run.job_id, "av_sync": (metric["status"], None if metric["score"] is None
                                             else round(metric["score"], 1))}


def decide_system(runs: Path, out_root: Path, system: str, rel, root: Path, *, splice_scores: bool = True) -> dict:
    """System offset from every complete record of the judged jobs, then decisions (+ splice) for each of them."""
    if not Path(out_root).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"{out_root} resolves outside the evidence root {root}: A-V Sync evidence refs must stay "
                         "inside it (scoring rejects references that leave it)")
    jobs = []
    for attempt, _score in scored_attempts(runs, out_root, system, rel):
        run = Run(attempt, root)
        out = out_root / run.system_id / run.job_id
        record = load_record(out, attempt.name)
        audio = run_audio(run)
        m = job_moments(out, record, audio) if record is not None else None
        jobs.append((run, out, record, audio, m))
    moments = {run.job_id: m for run, _, rec, _, m in jobs if m is not None}
    judge = judge_of(next((rec for _, _, rec, _, _ in jobs if rec is not None), None))
    offset, offset_path = offset_file(out_root, system, moments, judge)
    summary = {"system_id": system, "tau_ms": offset["tau_ms"], "ci95_ms": offset["ci95_ms"],
               "condition": offset["condition"], "sessions": offset["sessions"], "jobs": 0, "pending": 0, "units": {}}
    for run, out, record, audio, _ in jobs:
        if record is None:
            summary["pending"] += 1
            continue
        units = decide_units(record, audio, offset)
        path = decision_file(out, run.job_id, units, offset, offset_path, root)
        summary["jobs"] += 1
        for u in units:
            summary["units"][u["decision"]] = summary["units"].get(u["decision"], 0) + 1
        if splice_scores:
            try:
                splice(out, metric_decisions(units, path, root), rel.preparation(run.case_id), root, judge_of(record))
            except (ValueError, RuntimeError, OSError) as exc:  # e.g. evidence outside the evidence root
                summary.setdefault("splice_errors", []).append(f"{run.job_id}: {exc}"[:200])
    return summary
