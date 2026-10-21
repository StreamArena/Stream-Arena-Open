"""The v1 judge: one attempt folder in, per-unit judge records, observations, score and latency out (docs/evaluation.md).

Per job, under OUT/<system_id>/<job_id>/:

  units/*.json        one record per judged clip, edit/control checkpoint, dense latency window and dialogue session
  observations.json   metric decisions with cited evidence (streameval.scoring.report input)
  score.json          the job's scores (streameval.scoring.report output) + judge versions and prompt hashes
  latency.json        opportunity accounting (streameval.scoring.accounting.account_session)
  source_alignment.json   editing only: the measured processing lag used to pair output and source frames

The vision judge inspects frames at full resolution against a reference and reports named defects with a severity;
scores follow the fixed rules in streameval.judge.prompts, so every lost point is a named, cited defect:

* avatar clip (5 s): the initial portrait + 5 output frames 1 s apart -> quality, temporal and identity classes
  (identity is the lower of the primary and second judge ratings);
* editing clip (5 s): 5 pairs of (lag-matched source frame, output frame);
* edit checkpoint / edit state: the reference(s) + lag-matched pairs at observable seconds -> per-target state,
  coverage, reference match, leakage, preservation; latency from one request over the 0.25 s frames after release;
* general interaction: 5 s clip -> quality / stability / continuity; each control -> expected outcome per frame;
* avatar performance: avatar clips; each visual control -> following;
* dialogue: the avatar's received audio is transcribed per turn and judged by the text judge.

Events a model declared unsupported, and conditions a system cannot run (score_unsupported_job), are failures kept in
every denominator. A clip with no presented frame in a started run is a known missing output (0); a run that never
started has no decisions (unknown). A-V Sync is a separate stage (streameval.judge.av_sync).

Artifact references in observations/score/latency are relative to the evidence root, the directory that holds both
the runs and the judged outputs.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import subprocess
import threading

import numpy as np
from PIL import Image

from ..runtime.media import ffmpeg
from ..scoring.accounting import account_session
from ..scoring.report import score_prepared_run
from ..util import digest
from . import presentation as pr
from ..evaluation import policy_binding
from ..config import av_judge_endpoint
from .prompts import (AVATAR_CLIP, EDIT_CHECK, EDIT_CLIP, EDIT_CLIP_MULTI, EDIT_DENSE, EDIT_STATE, EDIT_STATE_DENSE,
                      EXT_PROMPTS_SHA256, EXT_VERSION, GEN_CLIP, GEN_CONTINUITY, GEN_DENSE, GEN_FOLLOW, GEN_QUALITY,
                      GEN_REFERENCE_KEYS, GEN_TEMPORAL, JUDGE_VERSION, MS, PROMPTS_SHA256, STABILITY_RULE, STYLES, VAD,
                      defect_rating, edit_state_verdicts, following_verdict, identity_rating, jpeg, match_rating,
                      picture_stable, preservation_pass, quality_of, reference_rating, sha, stable)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def artifact(path: Path, root: Path) -> dict:
    return {"artifact_ref": str(Path(path).relative_to(root)), "sha256": sha(Path(path).read_bytes())}


class Run:
    """One attempt folder (docs/protocol.md) as the judge reads it. Artifact references in the judged outputs are
    relative to `root`, the evidence root that holds both the runs and the judged outputs."""

    def __init__(self, attempt: Path, root: Path):
        self.dir, self.root = attempt, root
        self.binding = json.loads((attempt / "binding.json").read_text())
        self.result = json.loads((attempt / "result.json").read_text())
        self.job_id, self.case_id = self.binding["job_id"], self.binding["case_id"]
        self.system_id = self.binding["system_id"]
        self.browser = attempt / "browser"
        snap = self.result["scheduler"]
        self.epoch, self.deadline, self.finished = snap["started_at_ns"], snap["deadline_ns"], snap["finished_at_ns"]
        self.started = self.epoch is not None
        self.offset = self.result.get("clock_offset_ns")
        policy = self.binding["system_config"].get("portrait_policy")
        self.benchmark_portrait = policy in (None, "benchmark_portrait")
        path = self.browser / "frame_samples.jsonl"
        self.samples = sorted(jsonl(path), key=lambda s: s["case_ms"]) if path.exists() else []

    def reference_frame(self) -> bytes:
        receipt = self.result.get("browser_start", {}).get("reference_frame")
        data = (self.browser / "reference_frame.jpg").read_bytes()
        if not receipt or sha(data) != receipt["sha256"]:
            raise ValueError("reference frame missing or changed")
        return data

    def artifact(self, path: Path) -> dict:
        return artifact(path, self.root)

    def host(self, page_ms):
        return None if page_ms is None else round(page_ms * MS + self.offset)

    def near(self, t_ms, lo, hi, tolerance_ms=400):
        inside = [s for s in self.samples if lo <= s["case_ms"] < hi]
        best = min(inside, key=lambda s: abs(s["case_ms"] - t_ms), default=None)
        return best if best is not None and abs(best["case_ms"] - t_ms) <= tolerance_ms else None

    def image(self, sample) -> bytes:
        data = (self.browser / sample["path"]).read_bytes()
        if sha(data) != sample["sha256"]:
            raise ValueError("sample bytes changed: " + sample["path"])
        return data

    def audio(self):
        """(pcm int16, rate, voiced segments in host ns, sample clock) of the received avatar audio."""
        pcm_path, blocks_path = self.browser / "remote_audio.s16le", self.browser / "remote_audio_blocks.jsonl"
        if not pcm_path.exists() or not blocks_path.exists():
            return None, 48000, [], []
        blocks = jsonl(blocks_path)
        pcm = np.fromfile(pcm_path, dtype=np.int16)[: sum(b["samples"] for b in blocks)]
        clock = pr.audio_clock([(self.host(b["page_ms"]), b["samples"]) for b in blocks])
        segs = [(pr.sample_host_ns(clock, a, 48000), pr.sample_host_ns(clock, b - 1, 48000))
                for a, b in pr.voiced_segments(pcm, 48000, **VAD)]
        return pcm, 48000, segs, clock


def sample_index(clock, host_ns, rate=48000):
    for f in reversed(clock):
        if f.host_ns <= host_ns:
            return f.first_sample + int((host_ns - f.host_ns) * rate / 1e9)
    return 0


class JudgeSet:
    """What the judge reads besides the run: sessions, oracles, asset files, and per case the frozen metric plan and the
    preparation it belongs to (streameval.scoring.plan)."""

    def __init__(self, sessions: dict, oracles: dict, assets: dict, preparations):
        """assets: asset_id -> {"path": absolute Path, "sha256", "text" (user_audio transcript, dialogue only)};
        preparations: preparation folders (each with manifest.json and metric_plans.jsonl)."""
        self.sessions, self.oracles, self.assets = sessions, oracles, assets
        self.plans, self.population, self._prep = {}, {}, {}
        for folder in preparations:
            folder = Path(folder)
            pop = json.loads((folder / "manifest.json").read_text())["population"]
            for plan in jsonl(folder / "metric_plans.jsonl"):
                self.plans[plan["case_id"]] = plan
                self.population[plan["case_id"]] = pop
                self._prep[plan["case_id"]] = folder
        self._cache, self._lock = {}, threading.Lock()

    @classmethod
    def from_benchmark(cls, bench, preparations) -> "JudgeSet":
        assets = {aid: {"path": bench.path(aid), "sha256": a["sha256"], "text": a.get("script") or a.get("text")}
                  for aid, a in bench.assets.items()}
        return cls(bench.cases, bench.oracles, assets, preparations)

    def preparation(self, case_id: str) -> Path:
        return self._prep[case_id]

    def asset_jpeg(self, asset_id, max_side=1024):
        with self._lock:
            key = (asset_id, max_side)
            if key not in self._cache:
                asset = self.assets[asset_id]
                raw = Path(asset["path"]).read_bytes()
                if sha(raw) != asset["sha256"]:
                    raise ValueError("asset changed: " + asset_id)
                self._cache[key] = jpeg(Image.open(io.BytesIO(raw)), max_side)
            return self._cache[key]


class Source:
    """Source frames matched to outputs by the measured processing lag (matched source PTS)."""

    def __init__(self, run: Run, rel: JudgeSet):
        session = rel.sessions[run.case_id]
        self.path = rel.assets[session["initialization"]["source_video_asset_id"]]["path"]
        zero = (run.epoch - run.offset) / MS
        self.rows = [(r["presentation_ms"] - zero, r["media_time"]) for r in jsonl(run.browser / "source_frames.jsonl")
                     if r.get("presentation_ms") is not None and "media_time" in r]
        self.lag_ms, self.cache, self.lock = None, {}, threading.Lock()

    def pts_at(self, case_ms):
        best = min(self.rows, key=lambda r: abs(r[0] - case_ms), default=None)
        return None if best is None or abs(best[0] - case_ms) > 500 else best[1]

    def estimate_lag(self, run: Run, max_ms: int = 4000, step_ms: int = 40) -> dict:
        """Lag minimising the median correlation distance of 32x18 grey thumbnails (output vs source at t - lag)."""
        raw = subprocess.run([ffmpeg(), "-v", "error", "-threads", "2", "-i", str(self.path), "-vf",
                              "fps=20,scale=32:18,format=gray", "-f", "rawvideo", "-"],
                             capture_output=True, check=True).stdout
        norm = lambda a: (a - a.mean()) / (a.std() + 1e-6)
        thumbs = np.array([norm(t) for t in np.frombuffer(raw, np.uint8).reshape(-1, 18 * 32).astype(np.float32)])
        outs = []
        for smp in run.samples[:: max(1, len(run.samples) // 120)]:
            with Image.open(run.browser / smp["path"]) as im:
                outs.append((smp["case_ms"], norm(np.asarray(im.convert("L").resize((32, 18)), np.float32).ravel())))
        curve = []
        for d in range(0, max_ms + 1, step_ms):
            dist = [1 - float(np.dot(thumbs[min(len(thumbs) - 1, int(round(p * 20)))], th)) / len(th)
                    for t, th in outs if (p := self.pts_at(t - d)) is not None]
            if len(dist) >= 8:
                curve.append((d, float(np.median(dist)), len(dist)))
        if not curve:
            return {"lag_ms": None, "reason": "too few aligned samples"}
        best = min(curve, key=lambda c: c[1])
        self.lag_ms = best[0]
        return {"lag_ms": best[0], "median_distance": round(best[1], 4), "at_zero": round(curve[0][1], 4),
                "samples": best[2], "search_ms": [0, max_ms, step_ms]}

    def frame(self, case_ms, width):
        pts = self.pts_at(case_ms - (self.lag_ms or 0))
        if pts is None:
            return None
        key = (round(pts, 3), width)
        with self.lock:
            if key not in self.cache:
                self.cache[key] = subprocess.run(
                    [ffmpeg(), "-v", "error", "-threads", "2", "-ss", f"{pts:.3f}", "-i", str(self.path), "-frames:v", "1",
                     "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "-"],
                    capture_output=True, check=True).stdout
            raw = self.cache[key]
        return raw


def target_text(scope: str) -> str:
    return "the " + scope.split(". Apply the reference look")[0].strip().rstrip(".")


# ------------------------------------------------------------------ units

def save_unit(out: Path, name: str, record: dict) -> Path:
    path = out / "units" / (name + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    context = read_json(out / "judge_context.json")
    record = {**record, "context_sha256": digest(context) if context else None}
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    return path


def load_unit(out: Path, name: str, version: str = JUDGE_VERSION):
    """A cached record, unless it is from another judge version or a judge request failed (retry those)."""
    path = out / "units" / (name + ".json")
    record = read_json(path)
    if record:
        if (record.get("judge_version") == version and record.get("final")
                and record.get("context_sha256") == digest(read_json(out / "judge_context.json"))
                and not record.get("judge_error")
                and all(j.get("status") == "completed" for j in record.get("judges", []))):
            return record
    return None


def has_failed_units(out: Path) -> bool:
    """Any unit with a judge that did not complete. A unit caught mid-write is skipped, not counted: it would inflate
    the failed count and, in the pass loop, would not be a reason to re-judge anyway (the next pass reads it whole)."""
    for path in (out / "units").glob("*.json"):
        record = read_json(path)
        if record and record.get("context_sha256") == digest(read_json(out / "judge_context.json")) and (
                record.get("judge_error") or any(
                j.get("status") != "completed" for j in record.get("judges", []))):
            return True
    return False


def clip_windows(plan):
    units = plan["metrics"].get("visual_quality", {}).get("units", [])
    return [(u["unit_id"].split(".", 1)[1], u["window_ms"]) for u in units if u["applicable"]]


def judge_avatar_clip(run, rel, clip, window, style, portrait_id, out, judges):
    if load_unit(out, clip):
        return
    lo, hi = window
    base = {"unit_id": clip, "window_ms": window, "judge_version": JUDGE_VERSION, "final": True}
    picks = list({s["path"]: s for t in range(lo + 500, hi, 1000) if (s := run.near(t, lo, hi))}.values())
    if not picks:
        save_unit(out, clip, {**base, "status": "output_missing"})
        return
    reference = (rel.asset_jpeg(portrait_id) if run.benchmark_portrait
                 else jpeg(Image.open(io.BytesIO(run.reference_frame()))))
    images = [("portrait", reference)] + [(f"out_{i}", run.image(s)) for i, s in enumerate(picks)]
    text = AVATAR_CLIP.format(style=style.replace("_", " "), last=len(picks) - 1)
    first = judges["primary"].ask(text, images, "avatar")
    second = judges["second"].ask(text, images, "avatar")
    rec = {**base, "frames": [{"id": f"out_{i}", "case_ms": s["case_ms"], "path": s["path"]} for i, s in enumerate(picks)],
           "judges": [first, second]}
    if first["status"] == "completed":
        rec["visual_quality"] = {"rating": defect_rating(quality_of(first["reply"], "avatar"))}
        rec["stability"] = {"stable": stable(first["reply"]["temporal"].values())}
    ratings = [identity_rating(j["reply"]["identity"]) if j["status"] == "completed" else None for j in (first, second)]
    rec["identity"] = {"ratings": ratings, "rating": min(ratings) if all(ratings) else None,
                       "consensus": "lower of primary and second judge",
                       "reference": "initialization portrait" if run.benchmark_portrait else "browser/reference_frame.jpg"}
    save_unit(out, clip, rec)


def judge_edit_clip(run, clip, window, target, source, out, judges):
    if load_unit(out, clip):
        return
    lo, hi = window
    base = {"unit_id": clip, "window_ms": window, "judge_version": JUDGE_VERSION, "final": True,
            "source_lag_ms": source.lag_ms}
    picks = list({s["path"]: s for t in range(lo + 500, hi, 1000) if (s := run.near(t, lo, hi, 300))}.values())
    if not picks:
        save_unit(out, clip, {**base, "status": "output_missing"})
        return
    images = []
    for i, s in enumerate(picks):
        src = source.frame(s["case_ms"], s["width"])
        if src is not None:
            images.append((f"src_{i}", src))
        images.append((f"out_{i}", run.image(s)))
    reply = judges["primary"].ask(EDIT_CLIP.format(target=target), images, "edit_clip")
    rec = {**base, "frames": [{"id": f"out_{i}", "case_ms": s["case_ms"], "path": s["path"]} for i, s in enumerate(picks)],
           "judges": [reply]}
    if reply["status"] == "completed":
        rec["visual_quality"] = {"rating": defect_rating(quality_of(reply["reply"], "edit_clip"))}
        rec["stability"] = {"stable": stable(reply["reply"]["temporal"].values())}
    save_unit(out, clip, rec)


def clip_decisions(plan, clip_records, root: Path):
    decisions = {}
    for metric in ("visual_quality", "stability", "identity", "continuity"):
        for unit in plan["metrics"].get(metric, {}).get("units", []):
            if not unit["applicable"]:
                continue
            rec, path = clip_records[unit["unit_id"].split(".", 1)[1]]
            d = {"unit_id": unit["unit_id"], "evidence": artifact(path, root)}
            if rec.get("status") == "output_missing":
                d["status"] = "output_missing"
            elif metric == "stability":
                v = picture_stable(rec)
                d["status"] = "undetermined" if v is None else "pass" if v else "fail"
            elif metric == "continuity":
                v = rec.get(metric)
                d["status"] = "undetermined" if v is None else "pass" if v["continuous"] else "fail"
            else:
                rating = (rec.get(metric) or {}).get("rating")
                d["status"] = "observed" if rating else "undetermined"
                if rating:
                    d["value"] = 25 * (rating - 1)
            decisions.setdefault(metric, []).append(d)
    return decisions


# ------------------------------------------------------------------ dialogue (text judge unchanged)

def dialogue(run: Run, rel: JudgeSet, plan: dict, out: Path, asr, text_judge):
    """Per-turn reply text from the received audio; one text-judge call per session; latency decisions."""
    session, oracle = rel.sessions[run.case_id], rel.oracles[run.case_id]
    snap = run.result["scheduler"]
    rows = {r["event_id"]: r for r in snap["opportunities"]}
    releases = sorted((r["released_at_ns"], r["event_id"]) for r in snap["opportunities"] if r["released_at_ns"])
    cps = {c["event_id"]: c for c in oracle["checkpoints"]}
    audio_mode = session["input_profile"] == "conversation_audio"
    pcm, rate, segs, clock = run.audio()
    turns, windows, lates = [], {}, {}
    for event in session["events"]:
        eid, row, cp = event["event_id"], rows[event["event_id"]], cps[event["event_id"]]
        released = row["released_at_ns"]
        if audio_mode and released is not None:
            play = run.result["dispatch"].get(eid, {}).get("play")
            origin = run.host(play["page_end_ms"]) if play else None
        else:
            origin = released
        nxt = next((t for t, e in releases if released is not None and t > released), None)
        # Content: every avatar utterance that starts after this input ends and before the next user
        # turn (or session end). Timing: the reply must START within the response deadline.
        end = min(x for x in (nxt, run.deadline, run.finished) if x) if origin is not None else None
        windows[eid] = (origin, end, nxt)
        reply, late = [], None
        if origin is not None and pcm is not None:
            parts = [(a, b) for a, b in segs if origin <= a < end]
            if parts:
                i, j = sample_index(clock, parts[0][0]), sample_index(clock, parts[-1][1]) + int(0.2 * rate)
                text = asr.transcribe(pcm[i:j], rate, key=eid) if j > i else ""
                deadline = cp.get("response_deadline_ms")
                if cp["response_policy"] == "substantive_response" and deadline and parts[0][0] > origin + deadline * MS:
                    late = {"onset_after_input_end_ms": round((parts[0][0] - origin) / MS), "text": text}
                elif text:
                    reply = [text]
        lates[eid] = late
        prompt = event["payload"].get("text") or rel.assets[event["payload"]["asset_id"]]["text"]
        turns.append({"event_id": eid, "user_prompt": prompt, "expected": cp["expected"], "forbidden": cp["forbidden"],
                      "response_policy": cp["response_policy"], "service_reply_text": reply,
                      "released": released is not None and origin is not None, "semantics_assessable": True,
                      "reply_source": "client_received_audio_asr_" + asr.model,
                      "window_ms": None if origin is None else [round((origin - run.epoch) / MS), round((end - run.epoch) / MS)]})
    judged = text_judge.judge({"job_id": run.job_id, "turns": turns})
    verdicts = {v["event_id"]: v for v in judged["verdicts"]}
    path = save_unit(out, "dialogue_turns", {"unit_id": "dialogue_turns", "judge_version": JUDGE_VERSION, "final": True,
                                             "turns": turns, "late_replies": lates,
                                             "window_rule": "content: utterances starting after the input ends and before "
                                                            "the next user turn; timing: the first utterance must start "
                                                            "within response_deadline_ms",
                                             "verdicts": judged["verdicts"], "judge_model": judged["judge_model"],
                                             "judge_error": judged.get("judge_error"),
                                             "judge_raw_reply": judged.get("judge_raw_reply"), "vad": VAD})
    decisions = []
    for unit in plan["metrics"]["dialogue"]["units"]:
        eid = unit["event_id"]
        v = verdicts.get(eid, {}).get("verdict", "undetermined")
        if rows[eid]["status"] in ("trigger_not_fired", "not_released", "release_expired"):
            v = "fail"  # oracle failure_policy: the opportunity fails even if its trigger never fires
        decisions.append({"unit_id": unit["unit_id"], "status": v if v in ("pass", "fail") else "undetermined",
                          "evidence": run.artifact(path)})
    evidence_audio = (run.artifact(run.browser / "remote_audio_blocks.jsonl") if (run.browser / "remote_audio_blocks.jsonl").exists()
                      else run.artifact(run.dir / "result.json"))
    adjudications = []
    for event in session["events"]:
        eid = event["event_id"]
        row, cp = rows[eid], cps[eid]
        origin, end, nxt = windows[eid]
        base = {"event_id": eid, "checkpoint_sha256": digest(cp), "evidence": run.artifact(path)}
        if row["status"] != "released":
            adjudications.append({**base, "verdict": "fail", "reason": row["status"], "presentation": None})
            continue
        verdict = verdicts.get(eid, {"verdict": "undetermined"})
        if cp["response_policy"] != "substantive_response":
            status = verdict["verdict"]
            adjudications.append({**base, "verdict": status if status in ("pass", "fail") else "undetermined",
                                  "reason": {"pass": "criterion_satisfied", "fail": "semantic_failure"}.get(status, "not_adjudicated"),
                                  "presentation": None})
            continue
        if verdict["verdict"] == "fail":
            reason = "timeout" if verdict.get("judge") == "timing_rule" else "semantic_failure"
            adjudications.append({**base, "verdict": "fail", "reason": reason, "presentation": None})
            continue
        if verdict["verdict"] != "pass":
            adjudications.append({**base, "verdict": "undetermined", "reason": "not_adjudicated", "presentation": None})
            continue
        window_end = min(x for x in (nxt, run.deadline, run.finished) if x is not None)
        kind, seg = pr.first_onset_after(segs, max(origin, row["handoff_at_ns"]), window_end)
        if kind != "onset":
            adjudications.append({**base, "verdict": "undetermined",
                                  "reason": "insufficient_evidence" if kind == "none" else "unobservable",
                                  "presentation": None, "_timing": kind})
            continue
        onset, seg_end = seg
        if onset > origin + cp["response_deadline_ms"] * MS:
            adjudications.append({**base, "verdict": "fail", "reason": "timeout", "presentation": None})
            continue
        input_end = ({"event_id": eid, "clock_domain_id": snap["clock_domain_id"], "presented_end_at_ns": origin,
                      "evidence": run.artifact(run.dir / "result.json")} if audio_mode else None)
        receipt = pr.dialogue_receipt(clock_domain_id=snap["clock_domain_id"],
                                      origin_kind="end_of_presented_user_audio" if audio_mode else "user_prompt_release",
                                      origin_ns=origin, onset_ns=onset,
                                      end_ns=min(seg_end, onset + VAD["min_voiced_ms"] * MS, window_end),
                                      evidence=evidence_audio, input_end=input_end)
        adjudications.append({**base, "verdict": "pass", "reason": "criterion_satisfied", "presentation": receipt})
    return decisions, adjudications


# ------------------------------------------------------------------ edit checkpoints

def edit_checkpoints(run: Run, rel: JudgeSet, plan: dict, out: Path, judges: dict, source: Source):
    session, oracle = rel.sessions[run.case_id], rel.oracles[run.case_id]
    cps = {c["event_id"]: c for c in oracle["checkpoints"]}
    snap = run.result["scheduler"]
    rows = {r["event_id"]: r for r in snap["opportunities"]}
    results, adjudications = {}, []
    for event in session["events"]:
        eid, cp = event["event_id"], cps[event["event_id"]]
        ref = event["payload"]["references"][0]
        target = target_text(ref["scope"])
        reference = ("reference", rel.asset_jpeg(ref["asset_id"]))
        seconds = cp["observable_s_in_check_window"].get(ref["target_slot"]) or []
        rec = load_unit(out, "edit_" + eid)
        if rec is None:
            rec = {"unit_id": "edit_" + eid, "event_id": eid, "target": target, "observable_seconds": seconds,
                   "judge_version": JUDGE_VERSION, "final": True, "source_lag_ms": source.lag_ms}
            picks = [seconds[i] for i in sorted({0, len(seconds) // 3, 2 * len(seconds) // 3, len(seconds) - 1})] if seconds else []
            samples = [s for t in picks if (s := run.near(t * 1000 + 500, t * 1000, t * 1000 + 1000, 500))]
            if not seconds:
                rec.update(edit="undetermined", preservation="undetermined", reference_rating=None,
                           reason="target not observable in the check window")
            elif not samples:
                rec.update(edit="output_missing", preservation="output_missing", reference_rating=None,
                           reason="no presented output frame at the observable seconds")
            else:
                images, ids = [reference], []
                for i, s in enumerate(samples):
                    src = source.frame(s["case_ms"], s["width"])
                    if src is not None:
                        images.append((f"src_{i}", src))
                    images.append((f"out_{i}", run.image(s)))
                    ids.append(f"out_{i}")
                text = EDIT_CHECK.format(target=target, frames=", ".join(f'"{i}": <value>' for i in ids))
                reply = judges["primary"].ask(text, images, "edit_check", ids)
                rec["judges"] = [reply]
                rec["frames"] = [{"id": f"out_{i}", "case_ms": s["case_ms"], "path": s["path"]} for i, s in enumerate(samples)]
                if reply["status"] == "completed":
                    check = reply["reply"]
                    visible = [v for v in check["target_per_frame"].values() if v != "not_visible"]
                    rec["edit"] = ("undetermined" if not visible else
                                   "pass" if all(v == "matches" for v in visible) and check["target_coverage"] == "full"
                                   else "fail")
                    rec["preservation"] = "pass" if preservation_pass(check) else "fail"
                    rec["reference_rating"] = reference_rating(check)
                else:
                    rec.update(edit="undetermined", preservation="undetermined", reference_rating=None)
            save_unit(out, "edit_" + eid, rec)
        path = out / "units" / f"edit_{eid}.json"
        results[eid] = (rec, path)
        row = rows[eid]
        base = {"event_id": eid, "checkpoint_sha256": digest(cp), "evidence": run.artifact(path)}
        if row["status"] != "released":
            adjudications.append({**base, "verdict": "fail", "reason": row["status"], "presentation": None})
            continue
        if rec["edit"] in ("fail", "output_missing") or rec["preservation"] == "fail":
            adjudications.append({**base, "verdict": "fail",
                                  "reason": "no_effect" if rec["edit"] in ("fail", "output_missing") else "semantic_failure",
                                  "presentation": None})
            continue
        if rec["edit"] != "pass" or rec["preservation"] != "pass":
            adjudications.append({**base, "verdict": "undetermined", "reason": "insufficient_evidence", "presentation": None})
            continue
        # latency: one request over the 0.25 s frames from release to deadline + stable window
        at = event["release"]["at_ms"]
        dense = [s for s in run.samples if at <= s["case_ms"] < at + cp["effect_deadline_ms"] + cp["stable_window_ms"] + 250]
        drec = load_unit(out, f"edit_{eid}_dense")
        if drec is None:
            ids = [f"out_{i}" for i in range(len(dense))]
            text = EDIT_DENSE.format(target=target, last=len(dense) - 1, frames=", ".join(f'"{i}": <value>' for i in ids))
            reply = judges["primary"].ask(text, [reference] + [(ids[i], run.image(s)) for i, s in enumerate(dense)],
                                          "edit_dense", ids)
            marks = ([{"case_ms": s["case_ms"], "host_ns": run.host(s["presentation_ms"]),
                       "value": reply["reply"]["frames"][ids[i]]} for i, s in enumerate(dense)]
                     if reply["status"] == "completed" else [])
            drec = {"unit_id": f"edit_{eid}_dense", "judge_version": JUDGE_VERSION, "final": True,
                    "judges": [reply], "marks": marks}
            save_unit(out, f"edit_{eid}_dense", drec)
        dense_path = out / "units" / f"edit_{eid}_dense.json"
        found = pr.first_sustained([(m["host_ns"], m["value"] == "matches") for m in drec["marks"]],
                                   cp["stable_window_ms"] * MS)
        release = row["released_at_ns"]
        first = max(found[0], row["handoff_at_ns"]) if found else None
        if (found and first <= release + cp["effect_deadline_ms"] * MS and found[1] - first >= cp["stable_window_ms"] * MS
                and found[1] <= run.epoch + cp["valid_until_ms"] * MS):
            receipt = {"basis": "client_presentation", "clock_domain_id": snap["clock_domain_id"],
                       "origin_kind": "input_release", "origin_at_ns": release, "target": "stable_correct_effect",
                       "effect_started_at_ns": None, "first_valid_at_ns": first, "verified_at_ns": found[1],
                       "evidence": run.artifact(dense_path), "input_end": None}
            adjudications.append({**base, "verdict": "pass", "reason": "criterion_satisfied", "presentation": receipt})
        else:
            adjudications.append({**base, "verdict": "pass", "reason": "criterion_satisfied", "presentation": None})
    decisions = {"edit_success": [], "preservation": [], "reference": []}
    for metric in decisions:
        for unit in plan["metrics"].get(metric, {}).get("units", []):
            if not unit["applicable"]:
                continue
            rec, path = results[unit["event_id"]]
            d = {"unit_id": unit["unit_id"], "evidence": run.artifact(path)}
            if metric == "reference":
                d["status"] = ("output_missing" if rec["edit"] == "output_missing" else
                               "observed" if rec["reference_rating"] else "undetermined")
                if d["status"] == "observed":
                    d["value"] = 25 * (rec["reference_rating"] - 1)
            else:
                value = rec["edit"] if metric == "edit_success" else rec["preservation"]
                d["status"] = value if value in ("pass", "fail", "output_missing") else "undetermined"
            decisions[metric].append(d)
    return decisions, adjudications


# ------------------------------------------------------------------ extension: unsupported events

def unsupported_events(run: Run) -> dict:
    """event_id -> reason for events the sampler explicitly declared the system cannot express natively.

    Read from result.json: a dispatch receipt {"status": "unsupported", "reason": ...} (the harness), the list or
    mapping result["unsupported_events"], or a receipt {"unsupported": reason}. Never inferred from outputs.
    Each such event is a failure kept in every denominator: edit success, preservation and following fail, reference is a known missing output (0), and the latency opportunity is adjudicated fail
    (unsupported_during_run). Any job with such an event is judged on the extension path (one implementation)."""
    declared = run.result.get("unsupported_events") or {}
    if isinstance(declared, list):
        declared = {d["event_id"]: d.get("reason") for d in declared}
    found = {str(k): str(v) for k, v in declared.items()}
    for eid, receipt in (run.result.get("dispatch") or {}).items():
        if isinstance(receipt, dict) and receipt.get("status") == "unsupported":
            found.setdefault(eid, str(receipt.get("reason")))
        elif isinstance(receipt, dict) and receipt.get("unsupported"):
            found.setdefault(eid, str(receipt["unsupported"]))
    return found


def unsupported_record(unit_id: str, eid: str, reason: str) -> dict:
    return {"unit_id": unit_id, "event_id": eid, "judge_version": EXT_VERSION, "final": True, "status": "unsupported",
            "unsupported": True, "reason": "unsupported_during_run: " + reason, "edit": "fail", "preservation": "fail",
            "following": "fail", "reference_rating": None,
            "rule": "an event the system cannot express natively is a failure, kept in the denominator"}


def event_adjudication(row, base, rec):
    """Shared release / unsupported / failure / undetermined adjudication; None when the event passed its checks."""
    if row["status"] != "released":
        return {**base, "verdict": "fail", "reason": row["status"], "presentation": None}
    if rec.get("status") == "unsupported":
        return {**base, "verdict": "fail", "reason": "unsupported_during_run", "presentation": None}
    return None


def sustained_receipt(run, marks, cp, row, snap, evidence_path, target):
    """The v4 latency rule over dense marks [{host_ns, ok}]: first sustained correct frame within deadline and validity."""
    found = pr.first_sustained([(m["host_ns"], m["ok"]) for m in marks], stable_ms(cp) * MS)
    release = row["released_at_ns"]
    first = max(found[0], row["handoff_at_ns"]) if found else None
    if (found and first <= release + cp["effect_deadline_ms"] * MS and found[1] - first >= stable_ms(cp) * MS
            and found[1] <= run.epoch + cp["valid_until_ms"] * MS):
        # an action checkpoint times its onset separately; the first sustained correct frame bounds it from above
        onset = first if cp.get("effect_deadline_target") == "first_observable_correct_action_response" else None
        return {"basis": "client_presentation", "clock_domain_id": snap["clock_domain_id"],
                "origin_kind": "input_release", "origin_at_ns": release, "target": target,
                "effect_started_at_ns": onset, "first_valid_at_ns": first, "verified_at_ns": found[1],
                "evidence": run.artifact(evidence_path), "input_end": None}
    return None


def dense_window(run, event, cp):
    at = event["release"]["at_ms"]
    return [s for s in run.samples if at <= s["case_ms"] < at + cp["effect_deadline_ms"] + stable_ms(cp) + 250]


def frames_schema(ids, value="<value>"):
    return ", ".join(f'"{i}": {value}' for i in ids)


# ------------------------------------------------------------------ extension: edit states (instruction, patch, remove/reset)

INSTRUCTION = tuple(re.compile(p) for p in (r"^Change only (.+?) to: ", r"^Update only (.+?) to: ",
                                             r"^(?:Additionally|Also) change (.+?) to: ",
                                             r"^Cancel the edit: restore (.+?) to its "))


def is_legacy_edit(session, oracle) -> bool:
    """The judge-v4 edit path: reference_edit whose every checkpoint `set`s exactly the one slot its event references."""
    if session["group"] != "reference_edit":
        return False
    cps = {c["event_id"]: c for c in oracle["checkpoints"]}
    for event in session["events"]:
        cp, refs = cps[event["event_id"]], event["payload"].get("references") or []
        if (cp.get("operation") != "set" or len(refs) != 1 or cp.get("restored_source_slots")
                or list(cp.get("active_state") or {}) != [refs[0]["target_slot"]]):
            return False
    return True


def edit_targets(session, oracle) -> dict:
    """event_id -> {"request", "targets": [{slot, description, state: active|restored, look, reference_asset_id}]}.

    Targets are the checkpoint's whole declared state: every active_state slot (so a patch also checks that the earlier
    slot is retained) plus every restored_source_slots slot (remove/reset restore to the current source). Descriptions
    come from reference scopes or from the instruction text that introduced the slot."""
    cps = {c["event_id"]: c for c in oracle["checkpoints"]}
    reference = session["group"] == "reference_edit"
    desc, assets = {}, {}

    def take(payload):
        for r in payload.get("references") or []:
            desc[r["target_slot"]] = target_text(r["scope"])
            assets[r["target_slot"]] = r["asset_id"]

    take(session["initialization"].get("reference") or {})
    previous, result = {}, {}
    for event in session["events"]:
        eid, payload, cp = event["event_id"], event["payload"], cps[event["event_id"]]
        take(payload)
        active, restored = cp.get("active_state") or {}, cp.get("restored_source_slots") or []
        text = payload.get("text")
        if text:
            changed = [s for s in active if previous.get(s) != active[s]] + list(restored)
            match = next((m for p in INSTRUCTION if (m := p.match(text))), None)
            if match and len(changed) == 1:
                desc.setdefault(changed[0], "the " + match.group(1).strip())
        previous = active
        name = lambda s: desc.get(s, "the " + s.replace("_", " "))
        if text:
            request = text
        else:
            parts = [f'apply the look of reference image "ref_{r["target_slot"]}" to {name(r["target_slot"])}'
                     for r in payload.get("references") or []]
            parts += [f"cancel the edit of {name(s)} so it follows the live source" for s in payload.get("remove") or []]
            request = f'{payload.get("operation", "set")}: ' + "; ".join(parts)
        targets = [{"slot": s, "description": name(s), "state": "active", "look": None if reference else look,
                    "reference_asset_id": assets.get(s) if reference else None} for s, look in active.items()]
        targets += [{"slot": s, "description": name(s), "state": "restored", "look": None, "reference_asset_id": None}
                    for s in restored if s not in active]
        result[eid] = {"request": request, "targets": targets}
    return result


def target_lines(targets) -> str:
    lines = []
    for t in targets:
        if t["state"] == "restored":
            need = "the edit is cancelled; it must look as in the live source (its paired src frame)"
        elif t["reference_asset_id"]:
            need = f'must show the look of image "ref_{t["slot"]}" (its colors, pattern and material)'
        else:
            need = f"must show: {t['look']}"
        lines.append(f'- "{t["slot"]}" = {t["description"]}: {need}')
    return "\n".join(lines)


def edit_state_prompt(request, targets, protected, ids) -> tuple:
    """EDIT_STATE text and its schema spec. Images must be ref_<slot> for active reference targets, then src_i / out_i."""
    slots = [t["slot"] for t in targets]
    refs = [t["slot"] for t in targets if t["state"] == "active" and t["reference_asset_id"]]
    per_frame = "{" + frames_schema(ids) + "}"
    schema = ", ".join(f'"{s}": {{"per_frame": {per_frame}, "coverage": <"full" | "partial" | "none">}}' for s in slots)
    match = '<"match" | "partial" | "no">'
    ref_schema = (' "reference": {' + ", ".join(
        f'"{s}": {{"reference_color": {match}, "reference_pattern": {match}, "reference_material": {match}}}'
        for s in refs) + "},\n") if refs else ""
    text = EDIT_STATE.format(request=request, targets=target_lines(targets),
                             protected=", ".join(protected) or "none declared",
                             refs=("reference images " + ", ".join(f'"ref_{s}"' for s in refs) + ", then ") if refs else "",
                             target_schema=schema, reference_schema=ref_schema)
    return text, {"frames": list(ids), "slots": slots, "reference_slots": refs}


def follow_prompt(*, domain, setup, kind, control, cp, named, ids, reference) -> str:
    """GEN_FOLLOW text. named: descriptions of the context images sent before out_0 .. (portrait / reference / before)."""
    match = '<"match" | "partial" | "no"'
    ref_schema = (f' "reference": {{"reference_subject": {match}: the same subject or object as image "reference">, '
                  f'"reference_appearance": {match}: its colors, pattern and material>, '
                  f'"reference_shape": {match}: its shape, structure and fine details>}},\n') if reference else ""
    return GEN_FOLLOW.format(
        domain=domain, setup=setup, kind=kind, control=control,
        expected="; ".join(cp.get("expected") or []), protected="; ".join(cp.get("protected") or []) or "nothing declared",
        failures="; ".join(cp.get("failure_examples") or []) or "none declared",
        images=(", ".join(named) + ", then ") if named else "", last=len(ids) - 1,
        frames=frames_schema(ids, '<"yes" | "partial" | "no" | "not_visible">'), reference_schema=ref_schema)


def gen_clip_prompt(setup, controls, n_frames, reference) -> str:
    return GEN_CLIP.format(setup=setup, controls=controls or "none yet", last=n_frames - 1,
                           reference='Image "reference" is the opening reference image the scene is based on.\n'
                           if reference else "")


def gen_clip_verdicts(reply) -> dict:
    return {"visual_quality": {"rating": defect_rating(reply["quality"][n] for n in GEN_QUALITY)},
            "stability": {"stable": stable(reply["temporal"][n] for n in GEN_TEMPORAL)},
            "continuity": {"continuous": stable(reply["continuity"][n] for n in GEN_CONTINUITY)}}


def pick_seconds(seconds):
    return [seconds[i] for i in sorted({0, len(seconds) // 3, 2 * len(seconds) // 3, len(seconds) - 1})] if seconds else []


def edit_state_unit(run, rel, cp, eid, info, out, judges, source) -> dict:
    name = "edit_" + eid
    rec = load_unit(out, name, EXT_VERSION)
    if rec is not None:
        return rec
    targets = info["targets"]
    slots = [t["slot"] for t in targets]
    observable = {s: cp["observable_s_in_check_window"].get(s) or [] for s in slots}
    rec = {"unit_id": name, "event_id": eid, "operation": cp.get("operation"), "request": info["request"],
           "targets": targets, "observable_seconds": observable, "judge_version": EXT_VERSION, "final": True,
           "source_lag_ms": source.lag_ms}
    seconds = sorted({x for s in slots for x in pick_seconds(observable[s])})
    samples = [(t, s) for t in seconds if (s := run.near(t * 1000 + 500, t * 1000, t * 1000 + 1000, 500))]
    empty = dict(edit="undetermined", preservation="undetermined", following="undetermined", reference_rating=None)
    if not seconds:
        rec.update(empty, reason="no declared target observable in the check window")
    elif not samples:
        rec.update(edit="output_missing", preservation="output_missing", following="output_missing",
                   reference_rating=None, reason="no presented output frame at the observable seconds")
    else:
        refs = [t["slot"] for t in targets if t["state"] == "active" and t["reference_asset_id"]]
        images = [(f"ref_{s}", rel.asset_jpeg(next(t["reference_asset_id"] for t in targets if t["slot"] == s)))
                  for s in refs]
        ids = []
        for i, (_, s) in enumerate(samples):
            src = source.frame(s["case_ms"], s["width"])
            if src is not None:
                images.append((f"src_{i}", src))
            images.append((f"out_{i}", run.image(s)))
            ids.append(f"out_{i}")
        text, spec = edit_state_prompt(info["request"], targets, cp.get("protected") or [], ids)
        refs = spec["reference_slots"]
        reply = judges["primary"].ask(text, images, "edit_state", spec)
        rec["judges"] = [reply]
        rec["frames"] = [{"id": f"out_{i}", "second": t, "case_ms": s["case_ms"], "path": s["path"]}
                         for i, (t, s) in enumerate(samples)]
        if reply["status"] == "completed":
            observed = {slot: [f"out_{i}" for i, (t, _) in enumerate(samples) if t in observable[slot]] for slot in slots}
            rec.update(edit_state_verdicts(reply["reply"], observed, refs))
        else:
            rec.update(empty)
    save_unit(out, name, rec)
    return rec


def edit_state_dense(run, rel, event, cp, info, out, judges, source) -> tuple:
    name = f"edit_{event['event_id']}_dense"
    drec = load_unit(out, name, EXT_VERSION)
    if drec is None:
        targets = info["targets"]
        slots = [t["slot"] for t in targets]
        refs = [t for t in targets if t["state"] == "active" and t["reference_asset_id"]]
        restored = any(t["state"] == "restored" for t in targets)
        dense = dense_window(run, event, cp)
        ids = [f"out_{i}" for i in range(len(dense))]
        judged = []
        if dense:
            images = [(f"ref_{t['slot']}", rel.asset_jpeg(t["reference_asset_id"])) for t in refs]
            for i, s in enumerate(dense):
                src = source.frame(s["case_ms"], s["width"]) if restored else None
                if src is not None:
                    images.append((f"src_{i}", src))
                images.append((ids[i], run.image(s)))
            text = EDIT_STATE_DENSE.format(
                targets=target_lines(targets), last=len(dense) - 1,
                refs=("reference images " + ", ".join(f'"ref_{t["slot"]}"' for t in refs) + ", then ") if refs else "",
                src=", each preceded by its paired live source frame src_i" if restored else "",
                frames=frames_schema(ids, "{" + ", ".join(f'"{s}": <value>' for s in slots) + "}"))
            judged = [judges["primary"].ask(text, images, "edit_state_dense", {"frames": ids, "slots": slots})]
        marks = ([{"case_ms": s["case_ms"], "host_ns": run.host(s["presentation_ms"]),
                   "values": judged[0]["reply"]["frames"][ids[i]],
                   "ok": all(v == "matches" for v in judged[0]["reply"]["frames"][ids[i]].values())}
                  for i, s in enumerate(dense)] if judged and judged[0]["status"] == "completed" else [])
        drec = {"unit_id": name, "judge_version": EXT_VERSION, "final": True, "judges": judged, "marks": marks,
                "rule": "a frame counts when every declared target matches its required state"}
        save_unit(out, name, drec)
    return drec, out / "units" / f"{name}.json"


def edit_state_checkpoints(run: Run, rel: JudgeSet, plan: dict, out: Path, judges: dict, source: Source):
    session, oracle = rel.sessions[run.case_id], rel.oracles[run.case_id]
    cps = {c["event_id"]: c for c in oracle["checkpoints"]}
    snap = run.result["scheduler"]
    rows = {r["event_id"]: r for r in snap["opportunities"]}
    infos, unsupported = edit_targets(session, oracle), unsupported_events(run)
    results, adjudications = {}, []
    for event in session["events"]:
        eid, cp = event["event_id"], cps[event["event_id"]]
        if eid in unsupported:
            rec = unsupported_record("edit_" + eid, eid, unsupported[eid])
            save_unit(out, "edit_" + eid, rec)
        else:
            rec = edit_state_unit(run, rel, cp, eid, infos[eid], out, judges, source)
        path = out / "units" / f"edit_{eid}.json"
        results[eid] = (rec, path)
        row = rows[eid]
        base = {"event_id": eid, "checkpoint_sha256": digest(cp), "evidence": run.artifact(path)}
        early = event_adjudication(row, base, rec)
        if early:
            adjudications.append(early)
            continue
        if rec["edit"] in ("fail", "output_missing") or rec["preservation"] == "fail":
            adjudications.append({**base, "verdict": "fail",
                                  "reason": "no_effect" if rec["edit"] in ("fail", "output_missing") else "semantic_failure",
                                  "presentation": None})
            continue
        if rec["edit"] != "pass" or rec["preservation"] != "pass":
            adjudications.append({**base, "verdict": "undetermined", "reason": "insufficient_evidence", "presentation": None})
            continue
        drec, dense_path = edit_state_dense(run, rel, event, cp, infos[eid], out, judges, source)
        receipt = sustained_receipt(run, drec["marks"], cp, row, snap, dense_path, "stable_correct_effect")
        adjudications.append({**base, "verdict": "pass", "reason": "criterion_satisfied", "presentation": receipt})
    decisions = {"edit_success": [], "preservation": [], "reference": [], "following": []}
    for metric in decisions:
        for unit in plan["metrics"].get(metric, {}).get("units", []):
            if not unit["applicable"]:
                continue
            rec, path = results[unit["event_id"]]
            d = {"unit_id": unit["unit_id"], "evidence": run.artifact(path)}
            if metric == "reference":
                d["status"] = ("output_missing" if rec["edit"] == "output_missing" or rec.get("unsupported") else
                               "observed" if rec["reference_rating"] else "undetermined")
                if d["status"] == "observed":
                    d["value"] = 25 * (rec["reference_rating"] - 1)
            else:
                value = {"edit_success": rec["edit"], "preservation": rec["preservation"],
                         "following": rec["following"]}[metric]
                d["status"] = value if value in ("pass", "fail", "output_missing") else "undetermined"
            decisions[metric].append(d)
    return {k: v for k, v in decisions.items() if v}, adjudications


def edit_timeline(session, oracle) -> str:
    infos = edit_targets(session, oracle)
    return "\n".join(f'- at {e["release"]["at_ms"] / 1000:g} s: {infos[e["event_id"]]["request"]}' for e in session["events"])


def judge_edit_clip_multi(run, clip, window, timeline, source, out, judges):
    if load_unit(out, clip, EXT_VERSION):
        return
    lo, hi = window
    base = {"unit_id": clip, "window_ms": window, "judge_version": EXT_VERSION, "final": True,
            "source_lag_ms": source.lag_ms}
    picks = list({s["path"]: s for t in range(lo + 500, hi, 1000) if (s := run.near(t, lo, hi, 300))}.values())
    if not picks:
        save_unit(out, clip, {**base, "status": "output_missing"})
        return
    images = []
    for i, s in enumerate(picks):
        src = source.frame(s["case_ms"], s["width"])
        if src is not None:
            images.append((f"src_{i}", src))
        images.append((f"out_{i}", run.image(s)))
    reply = judges["primary"].ask(EDIT_CLIP_MULTI.format(plan=timeline), images, "edit_clip")
    rec = {**base, "frames": [{"id": f"out_{i}", "case_ms": s["case_ms"], "path": s["path"]} for i, s in enumerate(picks)],
           "judges": [reply]}
    if reply["status"] == "completed":
        rec["visual_quality"] = {"rating": defect_rating(quality_of(reply["reply"], "edit_clip"))}
        rec["stability"] = {"stable": stable(reply["reply"]["temporal"].values())}
    save_unit(out, clip, rec)


# ------------------------------------------------------------------ extension: general interaction and performance controls

CONTROL_KIND = {"user_prompt": "text control", "camera_control": "camera control",
                "reference_update": 'reference update (image "reference")', "action": "game action control"}
ACTION_PHASE = {"start": "start holding {a} (released by a later stop, at most {s:g} s)", "stop": "stop {a}",
                "pulse": "{a} once for {s:g} s"}


def control_text(payload) -> str:
    if payload["type"] == "action":
        return ACTION_PHASE[payload["phase"]].format(a=payload["action"].replace("_", " "),
                                                     s=payload["duration_ms"] / 1000)
    return payload.get("text") or payload.get("instruction") or ""


def stable_ms(cp) -> int:
    """The checkpoint's stability window: the completion-state window of an action, else the effect window."""
    return cp.get("completion_state_stable_window_ms") or cp["stable_window_ms"]


def active_reference(session, event):
    """General interaction: the latest reference released at or before this event, else the opening reference."""
    asset = (session["initialization"].get("reference_asset_ids") or [None])[0]
    for e in session["events"]:
        if e["release"]["at_ms"] > event["release"]["at_ms"]:
            break
        if e["payload"]["type"] == "reference_update":
            asset = e["payload"]["asset_id"]
    return asset


def judge_gen_clip(run, rel, session, clip, window, out, judges):
    if load_unit(out, clip, EXT_VERSION):
        return
    lo, hi = window
    base = {"unit_id": clip, "window_ms": window, "judge_version": EXT_VERSION, "final": True}
    picks = list({s["path"]: s for t in range(lo + 500, hi, 1000) if (s := run.near(t, lo, hi))}.values())
    if not picks:
        save_unit(out, clip, {**base, "status": "output_missing"})
        return
    ref = (session["initialization"].get("reference_asset_ids") or [None])[0]
    controls = "; ".join(f'at {e["release"]["at_ms"] / 1000:g} s "{control_text(e["payload"])}"'
                         for e in session["events"] if e["release"]["at_ms"] < hi)
    text = gen_clip_prompt(session["initialization"].get("prompt", ""), controls, len(picks), bool(ref))
    images = ([("reference", rel.asset_jpeg(ref))] if ref else []) + [(f"out_{i}", run.image(s)) for i, s in enumerate(picks)]
    reply = judges["primary"].ask(text, images, "gen_clip", {})
    rec = {**base, "frames": [{"id": f"out_{i}", "case_ms": s["case_ms"], "path": s["path"]} for i, s in enumerate(picks)],
           "judges": [reply]}
    if reply["status"] == "completed":
        rec.update(gen_clip_verdicts(reply["reply"]))
    save_unit(out, clip, rec)


def follow_unit(run, rel, session, event, cp, out, judges, performance) -> dict:
    """Judge a released control against its frozen checkpoint."""
    eid = event["event_id"]
    name = "follow_" + eid
    rec = load_unit(out, name, EXT_VERSION)
    if rec is not None:
        return rec
    payload, at = event["payload"], event["release"]["at_ms"]
    reference = None if performance else active_reference(session, event)
    rec = {"unit_id": name, "event_id": eid, "control": control_text(payload), "expected": cp.get("expected"),
           "reference_asset_id": reference, "judge_version": EXT_VERSION, "final": True}
    a, b = cp["check_window_ms"]
    times = sorted({a + (b - a) * (2 * k + 1) // 8 for k in range(4)})
    samples = list({s["path"]: s for t in times if (s := run.near(t, a, b, 500))}.values())
    if not samples:
        rec.update(following="output_missing", reference_rating=None,
                   reason="no presented output frame in the check window")
        save_unit(out, name, rec)
        return rec
    before = run.near(at - 250, max(0, at - 1500), at, 1250) if at > 0 else None
    images, named = [], []
    if performance:
        images.append(("portrait", rel.asset_jpeg(session["initialization"]["portrait_asset_id"]) if run.benchmark_portrait
                       else jpeg(Image.open(io.BytesIO(run.reference_frame())))))
        named.append('"portrait" (the avatar\'s initial appearance)')
    if reference:
        images.append(("reference", rel.asset_jpeg(reference)))
        named.append('"reference" (the active reference image)')
    if before:
        images.append(("before", run.image(before)))
        named.append('"before" (the last output frame before the control)')
    ids = [f"out_{i}" for i in range(len(samples))]
    images += [(ids[i], run.image(s)) for i, s in enumerate(samples)]
    text = follow_prompt(
        domain="avatar performance" if performance else "video-generation",
        setup=session["initialization"].get("rendering_context" if performance else "prompt", ""),
        kind=CONTROL_KIND.get(payload["type"], payload["type"]), control=rec["control"], cp=cp, named=named, ids=ids,
        reference=bool(reference))
    reply = judges["primary"].ask(text, images, "gen_follow", {"frames": ids, "reference": bool(reference)})
    rec["judges"] = [reply]
    rec["frames"] = [{"id": ids[i], "case_ms": s["case_ms"], "path": s["path"]} for i, s in enumerate(samples)]
    if before:
        rec["before"] = {"case_ms": before["case_ms"], "path": before["path"]}
    if reply["status"] == "completed":
        rec["following"] = following_verdict(reply["reply"])
        rec["reference_rating"] = match_rating(reply["reply"]["reference"], GEN_REFERENCE_KEYS) if reference else None
    else:
        rec.update(following="undetermined", reference_rating=None)
    save_unit(out, name, rec)
    return rec


def follow_dense(run, event, cp, out, judges, performance) -> tuple:
    name = f"follow_{event['event_id']}_dense"
    drec = load_unit(out, name, EXT_VERSION)
    if drec is None:
        dense = dense_window(run, event, cp)
        ids = [f"out_{i}" for i in range(len(dense))]
        judged = []
        if dense:
            text = GEN_DENSE.format(domain="avatar performance" if performance else "video-generation",
                                    kind=CONTROL_KIND.get(event["payload"]["type"], event["payload"]["type"]),
                                    control=control_text(event["payload"]), expected="; ".join(cp.get("expected") or []),
                                    last=len(dense) - 1, frames=frames_schema(ids))
            judged = [judges["primary"].ask(text, [(ids[i], run.image(s)) for i, s in enumerate(dense)], "gen_dense",
                                            {"frames": ids})]
        marks = ([{"case_ms": s["case_ms"], "host_ns": run.host(s["presentation_ms"]),
                   "value": judged[0]["reply"]["frames"][ids[i]], "ok": judged[0]["reply"]["frames"][ids[i]] == "yes"}
                  for i, s in enumerate(dense)] if judged and judged[0]["status"] == "completed" else [])
        drec = {"unit_id": name, "judge_version": EXT_VERSION, "final": True, "judges": judged, "marks": marks}
        save_unit(out, name, drec)
    return drec, out / "units" / f"{name}.json"


def control_checkpoints(run: Run, rel: JudgeSet, plan: dict, out: Path, judges: dict, performance: bool):
    """Following (+ reference for general interaction) per visual control event; speech_audio events are not controls."""
    session, oracle = rel.sessions[run.case_id], rel.oracles[run.case_id]
    cps = {c["event_id"]: c for c in oracle["checkpoints"]}
    snap = run.result["scheduler"]
    rows = {r["event_id"]: r for r in snap["opportunities"]}
    unsupported = unsupported_events(run)
    results, adjudications = {}, []
    for event in session["events"]:
        eid, cp = event["event_id"], cps[event["event_id"]]
        if event["payload"]["type"] == "speech_audio":
            continue  # supplied speech is not a control; its alignment needs a calibrated A/V model (undetermined)
        if eid in unsupported:
            rec = unsupported_record("follow_" + eid, eid, unsupported[eid])
            save_unit(out, "follow_" + eid, rec)
        else:
            rec = follow_unit(run, rel, session, event, cp, out, judges, performance)
        path = out / "units" / f"follow_{eid}.json"
        results[eid] = (rec, path)
        row = rows[eid]
        base = {"event_id": eid, "checkpoint_sha256": digest(cp), "evidence": run.artifact(path)}
        early = event_adjudication(row, base, rec)
        if early:
            adjudications.append(early)
            continue
        if rec["following"] in ("fail", "output_missing"):
            adjudications.append({**base, "verdict": "fail", "reason": "no_effect", "presentation": None})
            continue
        if rec["following"] != "pass":
            adjudications.append({**base, "verdict": "undetermined", "reason": "insufficient_evidence", "presentation": None})
            continue
        drec, dense_path = follow_dense(run, event, cp, out, judges, performance)
        target = cp.get("stable_effect_latency_target",
                        "stable_correct_performance_effect" if performance else "stable_correct_effect")
        receipt = sustained_receipt(run, drec["marks"], cp, row, snap, dense_path, target)
        adjudications.append({**base, "verdict": "pass", "reason": "criterion_satisfied", "presentation": receipt})
    decisions = {"following": [], "reference": []}
    for metric in decisions:
        for unit in plan["metrics"].get(metric, {}).get("units", []):
            if not unit["applicable"]:
                continue
            rec, path = results[unit["event_id"]]
            d = {"unit_id": unit["unit_id"], "evidence": run.artifact(path)}
            if metric == "reference":
                d["status"] = ("output_missing" if rec["following"] == "output_missing" or rec.get("unsupported") else
                               "observed" if rec.get("reference_rating") else "undetermined")
                if d["status"] == "observed":
                    d["value"] = 25 * (rec["reference_rating"] - 1)
            else:
                d["status"] = rec["following"] if rec["following"] in ("pass", "fail", "output_missing") else "undetermined"
            decisions[metric].append(d)
    return {k: v for k, v in decisions.items() if v}, adjudications


# ------------------------------------------------------------------ extension: a whole condition the system cannot run

def score_unsupported_job(case_id: str, repeat: int, system_id: str, system_config_sha256: str, reason: str,
                          out_root: Path, rel: JudgeSet, evidence_root: Path) -> dict:
    """A condition the system cannot express at all (not admitted) scored as a known failure in every denominator:
    binary units fail, rubric/bounded units are known missing output (0). No run exists, so latency has no measured
    success; latency.json records the eligible denominator with zero measurements."""
    plan = rel.plans[case_id]
    job_id = f"{case_id}.r{repeat:02d}"
    out = out_root / system_id / job_id
    out.mkdir(parents=True, exist_ok=True)
    path = save_unit(out, "unsupported", {"unit_id": "unsupported", "case_id": case_id, "judge_version": EXT_VERSION,
                                          "final": True, "status": "unsupported", "reason": reason,
                                          "rule": "unsupported condition = failure, kept in the denominator"})
    decisions = {}
    for name, metric in plan["metrics"].items():
        if name == "latency":
            continue
        decisions[name] = [{"unit_id": u["unit_id"], "evidence": artifact(path, evidence_root),
                            "status": "fail" if u["kind"] == "binary" else "output_missing"}
                           for u in metric["units"] if u["applicable"]]
    bundle = {"job_id": job_id, "system_id": system_id, "system_config_sha256": system_config_sha256,
              "metric_decisions": decisions}
    (out / "observations.json").write_text(json.dumps(bundle, indent=1) + "\n")
    score = score_prepared_run(rel.preparation(case_id), out / "observations.json", evidence_root=evidence_root)
    score.update(attempt=None, run_started=False, unsupported=reason, judge_version=JUDGE_VERSION,
                 prompts_sha256=PROMPTS_SHA256, extension_judge_version=EXT_VERSION)
    (out / "score.json").write_text(json.dumps(score, indent=1) + "\n")
    eligible = plan["metrics"].get("latency", {}).get("denominator", 0)
    (out / "latency.json").write_text(json.dumps({"job_id": job_id, "unsupported": reason, "latency": {
        "denominator": eligible, "semantically_successful": 0, "measured_successes": 0, "unit": "ns",
        "measured_success_latencies_ns": []}}, indent=1) + "\n")
    return {"job_id": job_id, "unsupported": reason}

def judge_context(attempt: Path, rel: JudgeSet, judges: dict, asr) -> dict:
    """Identity shared by job and unit caches; evidence and configuration changes invalidate both."""
    binding = json.loads((attempt / "binding.json").read_text())
    cid = binding["case_id"]
    evidence = {}
    browser = attempt / "browser"
    for name in ("frame_samples.jsonl", "remote_audio_blocks.jsonl", "remote_audio.s16le",
                 "source_frames.jsonl", "reference_frame.jpg"):
        path = browser / name
        if path.exists():
            evidence[name] = sha(path.read_bytes())
    samples = browser / "frame_samples.jsonl"
    if samples.exists():
        for frame in jsonl(samples):
            if sha((browser / frame["path"]).read_bytes()) != frame["sha256"]:
                raise ValueError("captured frame does not match its declared hash")
    def identity(judge):
        stored = getattr(judge, "frozen_identity", None)
        if stored is not None:
            return dict(stored)
        endpoint = (getattr(judge, "base_url", None) or "").rstrip("/")
        return {"model": judge.model, "endpoint_sha256": digest(endpoint)}

    configuration = {k: identity(j) for k, j in judges.items()}
    configuration["asr"] = identity(asr)
    configuration["rules"] = {"judge": JUDGE_VERSION, "extension": EXT_VERSION,
                               "prompts": PROMPTS_SHA256, "extension_prompts": EXT_PROMPTS_SHA256}
    if rel.plans[cid]["group"] in ("digital_human", "avatar_performance"):
        from .av_sync import AV_VERSION, RULE_SHA256
        endpoint = av_judge_endpoint()
        stored = getattr(judges["primary"], "replay_av_configuration", None)
        configuration["av_sync"] = {
            "model": stored["model"] if stored else endpoint.model,
            "endpoint_sha256": (stored.get("endpoint_sha256") if stored else
                                digest((endpoint.base_url or "").rstrip("/"))),
            "version": AV_VERSION, "rule_sha256": RULE_SHA256}
    return {"attempt": attempt.name, "binding": digest(binding),
            "result": sha((attempt / "result.json").read_bytes()), "evidence": evidence,
            "session": digest(rel.sessions[cid]), "oracle": digest(rel.oracles[cid]),
            "metric_plan": digest(rel.plans[cid]),
            "policy": policy_binding(), "versions": [JUDGE_VERSION, EXT_VERSION],
            "prompts": [PROMPTS_SHA256, EXT_PROMPTS_SHA256],
            "judges": configuration, "asr_model": asr.model}


def judge_job(attempt: Path, out_root: Path, rel: JudgeSet, judges: dict, asr, pool, evidence_root: Path) -> dict:
    """Judge one attempt into out_root/<system_id>/<job_id>/ (units, observations.json, score.json, latency.json).

    judges: {"primary", "second"} vision judges and "text" (dialogue), streameval.judge.client; asr transcribes the
    avatar's received speech; pool runs the clip units concurrently."""
    run = Run(attempt, evidence_root)
    plan = rel.plans[run.case_id]
    session = rel.sessions[run.case_id]
    if digest(session) != plan["session_sha256"] or digest(rel.oracles[run.case_id]) != plan["oracle_sha256"]:
        raise ValueError("judge inputs do not match the frozen metric plan")
    if run.binding.get("session_sha256") != plan["session_sha256"]:
        raise ValueError("attempt does not match the frozen session")
    if run.binding["system_config"].get("evaluation_policy") != policy_binding():
        raise ValueError("attempt does not bind the current evaluation policy")
    out = out_root / run.system_id / run.job_id
    out.mkdir(parents=True, exist_ok=True)
    context = judge_context(attempt, rel, judges, asr)
    (out / "judge_context.json").write_text(json.dumps(context, indent=1) + "\n")
    decisions, adjudications = {}, []
    group = plan["group"]
    # Use task-specific units for dialogue and single-slot reference edits.
    legacy = group == "digital_human" or (is_legacy_edit(session, rel.oracles[run.case_id]) and not unsupported_events(run))
    if run.started:
        windows = clip_windows(plan)
        if group in ("digital_human", "avatar_performance"):
            style = next((t for t in session.get("tags", []) if t in STYLES), "photorealistic")
            portrait = session["initialization"]["portrait_asset_id"]
            list(pool.map(lambda cw: judge_avatar_clip(run, rel, cw[0], cw[1], style, portrait, out, judges), windows))
        elif group == "general_interaction":
            list(pool.map(lambda cw: judge_gen_clip(run, rel, session, cw[0], cw[1], out, judges), windows))
        else:
            source = Source(run, rel)
            lag = source.estimate_lag(run)
            (out / "source_alignment.json").write_text(json.dumps(lag, indent=1) + "\n")
            if legacy:
                target = target_text(session["events"][0]["payload"]["references"][0]["scope"])
                list(pool.map(lambda cw: judge_edit_clip(run, cw[0], cw[1], target, source, out, judges), windows))
            else:
                timeline = edit_timeline(session, rel.oracles[run.case_id])
                list(pool.map(lambda cw: judge_edit_clip_multi(run, cw[0], cw[1], timeline, source, out, judges), windows))
        clips = {clip: (json.loads((out / "units" / f"{clip}.json").read_text()), out / "units" / f"{clip}.json")
                 for clip, _ in windows}
        decisions.update(clip_decisions(plan, clips, run.root))
        if group == "digital_human":
            decisions["dialogue"], adjudications = dialogue(run, rel, plan, out, asr, judges["text"])
        elif group in ("general_interaction", "avatar_performance"):
            controls, adjudications = control_checkpoints(run, rel, plan, out, judges, group == "avatar_performance")
            decisions.update(controls)
        elif legacy:
            edits, adjudications = edit_checkpoints(run, rel, plan, out, judges, source)
            decisions.update(edits)
        else:
            edits, adjudications = edit_state_checkpoints(run, rel, plan, out, judges, source)
            decisions.update(edits)
    bundle = {"job_id": run.job_id, "system_id": run.system_id,
              "system_config_sha256": run.binding["system_config_sha256"], "metric_decisions": decisions}
    (out / "observations.json").write_text(json.dumps(bundle, indent=1) + "\n")
    score = score_prepared_run(rel.preparation(run.case_id), out / "observations.json", evidence_root=run.root)
    score.update(attempt=attempt.name, run_started=run.started, judge_version=JUDGE_VERSION, prompts_sha256=PROMPTS_SHA256,
                 stability_rule=STABILITY_RULE, judge_context_sha256=digest(context),
                 judge_configuration=context["judges"], evaluation_policy=policy_binding())
    if not legacy:
        score.update(extension_judge_version=EXT_VERSION, extension_prompts_sha256=EXT_PROMPTS_SHA256,
                     unsupported_events=unsupported_events(run))
    (out / "score.json").write_text(json.dumps(score, indent=1) + "\n")
    if run.started:
        oracle = rel.oracles[run.case_id]
        snap = run.result["scheduler"]
        adjud = {"schema_version": ("streameval.avatar-performance-accounting.v1" if group == "avatar_performance"
                                   else "streameval.opportunity-accounting.v1"), "runner_session_sha256": digest(session),
                 "oracle_sha256": digest(oracle), "scheduler_sha256": digest(snap), "evidence_kind": "automated_judge_run",
                 "decisions": [{k: v for k, v in a.items() if not k.startswith("_")} for a in adjudications]}
        latency = account_session(session, oracle, snap, adjud)
        (out / "latency.json").write_text(json.dumps({"job_id": run.job_id, "latency": latency["latency"],
                                                     "behavior": latency["behavior"], "adjudications": adjud,
                                                     "timing_audit": [a.get("_timing") for a in adjudications]},
                                                    indent=1) + "\n")
    return {"job_id": run.job_id, "started": run.started,
            "metrics": {k: (v["status"], None if v.get("score") is None else round(v["score"], 1))
                        for k, v in score["metrics"].items()}}


def attempt_order(job_id: str, name: str):
    """Order stored attempts by their sequence suffix."""
    suffix = name[len(job_id):].lstrip(".")
    for rank, kind in ((1, "retry"), (2, "rerun")):
        if suffix.startswith(kind):
            return rank, int(suffix[len(kind):])
    return 0, 0


def read_json(path: Path) -> dict | None:
    """Read an artifact, returning None if absent or being written."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return None


def attempts(runs: Path, system: str | None, only: list[str] | None):
    """Select the first started attempt. An earlier unfinished attempt keeps the job unknown.

    Only a documented pre-epoch failure permits moving to another attempt. Client diagnostics and provider
    identity never select a replacement. Single-session and smoke runs are outside the batch roster.
    """
    tries = {}
    for binding in sorted(runs.glob("*/*/sa.*/binding.json")):
        attempt = binding.parent
        if attempt.parent.name == "single":
            continue
        b = read_json(binding)
        if b is None:
            continue
        if b.get("smoke_test_not_a_formal_sample") or (system and b["system_id"] != system) or (only and b["job_id"] not in only):
            continue
        tries.setdefault((b["system_id"], b["job_id"]), []).append(attempt)
    chosen = []
    for (_, job_id), found in tries.items():
        found.sort(key=lambda a: attempt_order(job_id, a.name))
        for attempt in found:
            result = read_json(attempt / "result.json")
            if result is None:
                break
            if result["scheduler"]["started_at_ns"] is not None:
                chosen.append(attempt)
                break
        else:
            chosen.append(found[0])
    return chosen
