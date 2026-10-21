"""Running a model: one attempt (``run_attempt``) or every job of a track on a benchmark set (``run_batch``).

A batch of one system on one population lives in one folder; everything that decides which samples count is
frozen there before the first output exists:

    <out>/<system_id>/<population>/
        admission.json      the cases of the system's track that are run, and the events it cannot handle
        batch_plan.json     seeded job order, the frozen system configuration, the rerun rule
        batch_log.jsonl     one line per finished attempt (exit code)
        <job_id>/           one attempt (runtime.harness; docs/protocol.md)
        <job_id>.retryK/    re-attempts of a job whose earlier attempts never started (rerun rule)

Each attempt runs in its own process (``streameval run``), so a crash or a hung model costs one attempt only.
A file ``<out>/STOP`` stops new attempts from starting. ``dry_run`` reports what a batch would run and cost
without writing, reading media or making a request (a vendor backend is never started by it).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

from ..benchmark.dataset import Benchmark
from ..benchmark.tracks import event_op
from ..models.registry import (READY_TIMEOUT_S, LaunchedServer, account, build_backend, build_model, health, ops,
                               public, system_config)
from ..evaluation import policy
from ..util import digest, private_json, read_json, read_jsonl
from .harness import Harness

MAX_RETRIES = policy()["execution"]["pre_epoch_retries"]


def rerun_rule(max_retries: int = MAX_RETRIES) -> str:
    return ("An attempt that failed before its activity epoch (scheduler never started, no input released) "
            f"is an infrastructure non-start and is re-attempted as <job_id>.retry1 .. .retry{max_retries} (at most "
            f"{max_retries} times); the first started attempt is the sample. Any attempt that started keeps its "
            "outcome and is never re-run, including a session the model ended before the case deadline: that "
            "attempt is a failure in the denominator.")


# ---- admission --------------------------------------------------------------------------------------

def unsupported_events(system_id: str, spec: dict, case: dict, backend=None) -> list[dict]:
    """The events of `case` the system cannot handle: outside its registered ops, or (a session backend) refused by
    the backend's own event dispositions. Decided from the case alone, before any output exists."""
    if backend is not None:
        return [{"event_id": d["event_id"], "operation": d["operation"], "reason": d["reason"]}
                for d in backend.event_dispositions(system_id, spec, case) if not d["supported"]]
    supported = set(ops(spec["entry"], spec["track"]))
    return [{"event_id": e["event_id"], "operation": event_op(e["payload"]), "reason": "not_in_registered_ops"}
            for e in case["events"] if event_op(e["payload"]) not in supported]


def admission(system_id: str, spec: dict, bench: Benchmark) -> dict:
    """Every case of the system's track is admitted; events outside its ops stay in, flagged (scored as failures)."""
    supported = set(ops(spec["entry"], spec["track"]))
    backend = build_backend(spec["entry"]) if "backend" in spec["entry"] else None
    rows = []
    for cid, case in sorted(bench.cases.items()):
        if (case["group"], case["input_profile"]) != (spec["group"], spec["input_profile"]):
            continue
        unsupported = unsupported_events(system_id, spec, case, backend)
        rows.append({"case_id": cid, "group": case["group"], "input_profile": case["input_profile"],
                     "duration_ms": case["duration_ms"], "admitted": True, "system_id": system_id,
                     "unsupported_events": unsupported,
                     "status": "admitted_with_unsupported_events" if unsupported else "admitted"})
    entry = public(spec["entry"])
    return {"schema_version": "streameval.system-admission/1", "system_id": system_id, "track": spec["track"],
            "decided_before_outputs": True, "dataset_version": bench.manifest.get("dataset_version"),
            "cases_sha256": digest(sorted(bench.cases)), "registry_entry_sha256": digest(entry),
            "ops": sorted(supported),
            "rule": "every condition of the registered track is admitted; events outside the registered ops (or "
                    "that the session backend cannot express) are admitted-with-unsupported-events and scored as "
                    "failures",
            "admitted": sum(r["admitted"] for r in rows),
            "unsupported_events": sum(len(r["unsupported_events"]) for r in rows), "cases": rows}


# ---- attempt states and the rerun rule ---------------------------------------------------------------

def started(folder: Path) -> bool | None:
    """True if the epoch started, False for a documented non-start, None without a result."""
    path = folder / "result.json"
    if not path.exists():
        return None
    return read_json(path)["scheduler"]["started_at_ns"] is not None


def pending_attempts(folder: Path, order: list[dict], retry_unstarted: bool, max_retries: int = MAX_RETRIES):
    """The attempts still to run: first attempts of jobs without one, or (retry_unstarted) the next .retryK of
    jobs whose every attempt so far never started."""
    if retry_unstarted:
        pending = []
        for j in order:
            tries = [folder / j["job_id"]] + [folder / f"{j['job_id']}.retry{k}" for k in range(1, max_retries + 1)]
            done = [t for t in tries if t.exists()]
            if done and all(started(t) is False for t in done) and len(done) < len(tries):
                pending.append({**j, "attempt": tries[len(done)].name})
        return pending
    return [{**j, "attempt": j["job_id"]} for j in order if not (folder / j["job_id"]).exists()]


def status(folder: Path, order: list[dict], max_retries: int = MAX_RETRIES) -> dict:
    """Per job: the attempt that is its sample and how it ended."""
    counts = {"completed": 0, "failed": 0, "not_started": 0, "no_result": 0, "pending": 0}
    for j in order:
        tries = [folder / j["job_id"]] + [folder / f"{j['job_id']}.retry{k}" for k in range(1, max_retries + 1)]
        done = [t for t in tries if t.exists()]
        if not done:
            counts["pending"] += 1
            continue
        last = done[-1]
        state = started(last)
        if state is None:
            counts["no_result"] += 1
        elif state is False:
            counts["not_started" if len(done) == len(tries) else "pending"] += 1
        else:
            counts[read_json(last / "result.json")["status"]] += 1
    return counts


# ---- one attempt ------------------------------------------------------------------------------------

def run_attempt(system_id: str, spec: dict, bench: Benchmark, case_id: str, out: Path, *, repeat: int = 1,
                url: str | None = None, launch: bool = False, smoke: bool = False, admission_row: dict | None = None,
                verify: bool = True) -> dict:
    """Run one session of `case_id` and write the attempt folder `out`; returns result.json."""
    if case_id not in bench.cases:
        raise SystemExit(f"case {case_id} is not in {bench.root}")
    entry, server = spec["entry"], None
    if "backend" in entry:
        return build_backend(entry).run(system_id, spec, bench.cases[case_id], bench.case_assets(case_id), Path(out),
                                        repeat=repeat, oracle=bench.oracles.get(case_id), smoke=smoke,
                                        admission=admission_row, verify=verify)
    try:
        if launch:
            if "server" not in entry:
                raise SystemExit(f"{system_id} has no server command to launch")
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            server = LaunchedServer(entry, log=Path(out).with_name(Path(out).name + ".server.log"))
            url = server.url
        model = build_model(entry, url)
        harness = Harness(model, system_id, spec, bench.cases[case_id], bench.case_assets(case_id), out,
                          repeat=repeat, oracle=bench.oracles.get(case_id), smoke=smoke, admission=admission_row,
                          verify=verify)
        return harness.run()
    finally:
        if server is not None:
            server.stop()


# ---- a dry run --------------------------------------------------------------------------------------

def dry_run(system_id: str, spec: dict, bench: Benchmark, *, population: str, repetitions: int = 3,
            preparation: Path | None = None) -> dict:
    """What `run_batch` would run: sessions, active seconds, unsupported events, estimated cost and the missing
    credentials. Nothing is written, no media file is read and no request is made."""
    from ..scoring.plan import prepare_files
    entry = spec["entry"]
    backend = build_backend(entry) if "backend" in entry else None
    vendor = account(entry)
    if preparation is not None and (Path(preparation) / "sampling_jobs.jsonl").exists():
        jobs = read_jsonl(Path(preparation) / "sampling_jobs.jsonl")
    else:
        if not (bench.root / "oracles.jsonl").exists():
            raise SystemExit(f"{bench.root} has no oracles; it cannot be prepared (and run) locally")
        jobs = prepare_files(bench.root / "cases.jsonl", bench.root / "oracles.jsonl", population=population,
                             repetitions=repetitions)[2]
    track_cases = {cid for cid, c in bench.cases.items()
                   if (c["group"], c["input_profile"]) == (spec["group"], spec["input_profile"])}
    jobs = [j for j in jobs if j["case_id"] in track_cases]
    cases = sorted({j["case_id"] for j in jobs})
    unsupported = {cid: unsupported_events(system_id, spec, bench.cases[cid], backend) for cid in cases}
    seconds = [bench.cases[j["case_id"]]["duration_ms"] / 1000 for j in jobs]
    by_duration: dict[str, int] = {}
    for s in seconds:
        by_duration[f"{s:g}s"] = by_duration.get(f"{s:g}s", 0) + 1
    estimates = [vendor.estimate(bench.cases[j["case_id"]]) for j in jobs] if vendor else []
    config = system_config(system_id, spec)
    return {"system_id": system_id, "track": spec["track"], "population": population, "dry_run": True,
            "sessions": len(jobs), "cases": len(cases), "repetitions": repetitions,
            "sessions_by_duration": dict(sorted(by_duration.items(), key=lambda kv: float(kv[0][:-1]))),
            "active_seconds": round(sum(seconds), 3),
            "event_dispositions_all_repeats": {
                "events": sum(len(bench.cases[j["case_id"]]["events"]) for j in jobs),
                "unsupported": sum(len(unsupported[j["case_id"]]) for j in jobs)},
            "cases_with_unsupported_events": {cid: [u["event_id"] + " " + u["operation"] + ": " + u["reason"]
                                                    for u in rows] for cid, rows in unsupported.items() if rows},
            "estimated_cost": (None if not estimates or None in estimates else round(sum(estimates), 2)),
            "cost_unit": vendor.cost.get("unit") if vendor else None,
            "cost_basis": (vendor.cost.get("basis") or ("(case seconds + %s s overhead) x %s per second" % (
                vendor.cost.get("overhead_s", 0), vendor.cost.get("per_active_second")))
                if estimates and None not in estimates else vendor.cost.get("basis")) if vendor else None,
            "session_cap_s": vendor.session_cap_s if vendor else None,
            "sessions_beyond_session_cap": sum(vendor.exceeds_session_cap(bench.cases[j["case_id"]], spec["kind"])
                                               for j in jobs) if vendor else 0,
            "missing_credentials": vendor.missing_credentials() if vendor else [],
            "system_config_sha256": digest(config),
            "network_requests": 0, "media_read": False}


def spent_estimate(folder: Path, bench: Benchmark, vendor) -> float:
    """Estimated cost of the attempts already in `folder` that reached the provider (every attempt with a binding)."""
    total = 0.0
    if folder.is_dir():
        for attempt in folder.iterdir():
            binding = attempt / "binding.json"
            if binding.exists():
                case = bench.cases.get((read_json(binding) or {}).get("case_id"))
                total += (vendor.estimate(case) or 0) if case else 0
    return total


# ---- a batch ----------------------------------------------------------------------------------------

def sampling_jobs(bench: Benchmark, preparation: Path, *, population: str, repetitions: int = 3) -> list[dict]:
    """Jobs of the frozen preparation (written from the benchmark set the first time it is needed)."""
    from ..scoring.plan import prepare_files, write_preparation
    if not (preparation / "sampling_jobs.jsonl").exists():
        if not (bench.root / "oracles.jsonl").exists():
            raise SystemExit(f"{bench.root} has no oracles; it cannot be prepared (and run) locally")
        write_preparation(preparation, *prepare_files(bench.root / "cases.jsonl", bench.root / "oracles.jsonl",
                                                      population=population, repetitions=repetitions))
    manifest = read_json(preparation / "manifest.json")
    if manifest["population"] != population:
        raise SystemExit(f"{preparation} was prepared for population {manifest['population']}, not {population}")
    from ..scoring.report import load_preparation
    from ..evaluation import policy_binding
    manifest, plans, jobs, _ = load_preparation(preparation)
    if manifest["repetitions"] != repetitions or manifest.get("evaluation_policy") != policy_binding():
        raise SystemExit("preparation uses different repetition or policy settings; use a separate output directory")
    for cid, plan in plans.items():
        if cid not in bench.cases or cid not in bench.oracles or any(
                digest(value) != plan[key] for value, key in (
                    (bench.cases[cid], "session_sha256"), (bench.oracles[cid], "oracle_sha256"))):
            raise SystemExit("benchmark content differs from the frozen preparation")
    return list(jobs.values())


def _frozen(path: Path, value: dict, keys: tuple[str, ...]) -> dict:
    """Write `value` once; later batches in the same folder must agree on `keys`."""
    if not path.exists():
        private_json(path, value)
        return value
    frozen = read_json(path)
    changed = [k for k in keys if frozen.get(k) != value.get(k)]
    if changed:
        raise SystemExit(f"{path} was frozen with a different {changed}; start a new --out for a changed system "
                         "or benchmark")
    return frozen


def run_batch(system_id: str, spec: dict, bench: Benchmark, out: Path, *, population: str,
              preparation: Path | None = None, repetitions: int = 3, concurrency: int = 1, seed: int = 0,
              limit: int | None = None, only: list[str] | None = None, auto_retry: bool = True,
              url: str | None = None, launch: bool = False, registry: tuple = (), stagger_s: float = 1.0,
              max_retries: int = MAX_RETRIES, budget: float | None = None, log=print) -> dict:
    """Run every pending attempt of `system_id` on `bench`; returns the batch status.

    budget (vendor systems with a cost estimate): no attempt starts once the estimated cost of the attempts in the folder
    plus the next one would exceed it."""
    if type(max_retries) is not int or max_retries != MAX_RETRIES:
        raise ValueError("retry allowance is fixed by the shared evaluation policy")
    out = Path(out)
    folder = out / system_id / population
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    config = system_config(system_id, spec)
    adm = _frozen(folder / "admission.json", admission(system_id, spec, bench),
                  ("registry_entry_sha256", "cases_sha256", "ops"))
    admitted = {r["case_id"]: r for r in adm["cases"] if r["admitted"]}
    preparation = Path(preparation or out / "preparation" / population)
    jobs = [j for j in sampling_jobs(bench, preparation, population=population, repetitions=repetitions)
            if j["case_id"] in admitted]
    order = sorted(jobs, key=lambda j: j["job_id"])
    random.Random(seed).shuffle(order)
    _frozen(folder / "batch_plan.json",
            {"system_id": system_id, "population": population, "seed": seed, "order": [j["job_id"] for j in order],
             "system_config": config, "system_config_sha256": digest(config), "rerun_rule": rerun_rule(max_retries),
             "benchmark": str(bench.root), "admission_sha256": digest(adm)},
            ("system_config_sha256", "order"))
    if only:
        order = [j for j in order if j["job_id"] in only]

    # model servers: one per concurrent attempt (a server runs one session at a time)
    entry, servers, urls = spec["entry"], [], queue.Queue()
    stop, lock, state = out / "STOP", threading.Lock(), {"halted": None}
    backend, vendor = "backend" in entry, account(entry)
    if budget is not None and (vendor is None or vendor.estimate(next(iter(bench.cases.values()))) is None):
        raise SystemExit(f"--budget needs a vendor system with a cost estimate; {system_id} has none")
    state["spent"] = spent_estimate(folder, bench, vendor) if budget is not None else 0.0
    try:
        if launch:
            if "server" not in entry:
                raise SystemExit(f"{system_id} has no server command to launch")
            for i in range(concurrency):
                servers.append(LaunchedServer(entry, entry["server"]["port"] + i,
                                              log=folder / f"server_{i}.log"))
        for u in [s.url for s in servers] or [None if backend else url or entry.get("url")] * concurrency:
            urls.put(u)
        common = ["--benchmark", str(bench.root), "--data", str(bench.data),
                  *[a for p in registry for a in ("--registry", str(p))]]
        timeout = lambda job: (bench.cases[job["case_id"]]["duration_ms"] // 1000
                               + READY_TIMEOUT_S + 600)

        def one(job):
            target = urls.get()
            try:
                with lock:
                    if stop.exists():
                        return {"job_id": job["job_id"], "skipped": "stop_file"}
                    if state["halted"]:
                        return {"job_id": job["job_id"], "skipped": state["halted"]}
                    if target and health(urlparse(target).port or 80, urlparse(target).hostname or "127.0.0.1") is None:
                        state["halted"] = "server_down: " + target
                        return {"job_id": job["job_id"], "skipped": state["halted"]}
                    if budget is not None:
                        cost = vendor.estimate(bench.cases[job["case_id"]])
                        if state["spent"] + cost > budget:
                            state["halted"] = (f"budget: {state['spent']:.2f} + {cost:.2f} {vendor.cost['unit']} "
                                               f"> {budget:g}")
                            return {"job_id": job["job_id"], "skipped": state["halted"]}
                        state["spent"] += cost
                    time.sleep(stagger_s)  # stagger session creation
                cmd = [sys.executable, "-m", "streameval.cli", "run", "--system", system_id,
                       "--case", job["case_id"], "--repeat", str(job["repeat"]), "--out", str(folder / job["attempt"]),
                       "--admission", str(folder / "admission.json"), *common, *(["--url", target] if target else [])]
                with (folder / (job["attempt"] + ".log")).open("w") as handle:
                    try:
                        code = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                              timeout=timeout(job)).returncode
                    except subprocess.TimeoutExpired:
                        code = "timeout"
                line = {"job_id": job["job_id"], "attempt": job["attempt"], "exit": code,
                        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                with lock, (folder / "batch_log.jsonl").open("a") as handle:
                    handle.write(json.dumps(line) + "\n")
                log(json.dumps(line))
                return line
            finally:
                urls.put(target)

        passes = [False] + [True] * (max_retries if auto_retry else 0)
        for retry in passes:
            if stop.exists() or state["halted"]:
                break
            pending = pending_attempts(folder, order, retry, max_retries)
            if limit is not None and not retry:
                pending = pending[:limit]
            if not pending:
                continue
            with ThreadPoolExecutor(concurrency) as pool:
                list(pool.map(one, pending))
    finally:
        for s in servers:
            s.stop()
    report = {"system_id": system_id, "population": population, "folder": str(folder), "jobs": len(order),
              **status(folder, order, max_retries)}
    if budget is not None:
        report["estimated_spent"] = round(state["spent"], 2)
    if state["halted"]:
        report["halted"] = state["halted"]
    return report
