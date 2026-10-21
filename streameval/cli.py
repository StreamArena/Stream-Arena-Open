"""streameval command line.

    streameval list                                         registered systems and the tracks
    streameval validate [--benchmark benchmark/v1/full]     every asset present with its manifest sha256
    streameval run --system ID --case CASE [--launch|--url] one session -> an attempt folder
    streameval run --model file.py:Class --track TRACK --case CASE      (an unregistered in-process model)
    streameval batch --system ID [--concurrency N]          every job of the system's track (3 repeats per case)
    streameval batch --system ID --dry-run                  sessions, unsupported events, cost; nothing is run
    streameval validate-run DIR...                          check attempt folders against the run contract
    streameval serve --model module:Class [--port 8765]     expose an in-process model over the WebSocket protocol
    streameval judge --runs runs --out runs/eval [--replay DIR]   judge attempts -> per-job units and score.json
    streameval assemble --runs runs --eval runs/eval --out runs/results  per-system results for the leaderboard
    streameval leaderboard --results runs/results --roster runs/preparation/full --out FILE   a board file
    streameval leaderboard --table --out FILE     a local board file as Markdown tables
    streameval evaluate --system ID [--launch]   batch + judge + assemble + leaderboard on the complete benchmark

Paths. The defaults (benchmark/v1/full, runs/, data/) are those of the repository checkout the package is
installed from (pip install -e .), whatever the current directory; relative paths given on the command line are
relative to the current directory. Judging and scoring: every evidence reference is a path relative to
--evidence-root (default: the run root, --runs), so the eval root must lie inside it (runs/eval).

Exit codes: 0 done; 1 the check failed (a failed session, a halted batch, invalid attempts, a replay miss); 2 usage; 3 a vendor credential is missing, nothing was sent;
4 assemble could not re-decide A-V Sync for a system (see docs/protocol.md#exit-codes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import REPO

RUNS = REPO / "runs"


def _bench(a):
    from .benchmark.dataset import Benchmark
    return Benchmark(a.benchmark, a.data)


def _system(a) -> tuple[str, dict]:
    from .models import registry
    if a.system:
        return a.system, registry.spec_of(a.system, a.registry)
    if not a.track or not (a.model or a.url):
        raise SystemExit("give --system ID (registered), or --track with --model module:Class or --url ws://...")
    return registry.adhoc(a.track, python=a.model, url=None if a.model else a.url)


def _refuse_without_credentials(spec: dict) -> int | None:
    """Exit code 3 (and the missing variable names) when a vendor system lacks a credential; nothing is sent."""
    from .models.registry import account
    vendor = account(spec["entry"])
    if vendor is None:
        return None
    missing = vendor.missing_credentials()
    if not missing:
        return None
    print(json.dumps({"refused": "credentials missing", "env": missing}))
    return 3


def _list(a) -> int:
    from .benchmark.tracks import TRACK_OPS, TRACKS
    from .models import registry
    print("tracks:")
    for name, t in TRACKS.items():
        print(f"  {name:27s} {t['group']:20s} {t['input_profile']:30s} ops: {', '.join(TRACK_OPS[name])}")
    print("systems:")
    for sid, spec in sorted(registry.systems(a.registry).items()):
        entry = spec["entry"]
        how = next(f"{k} {entry[k]}" if k != "server" else k for k in ("python", "backend", "server", "url") if k in entry)
        print(f"  {sid:40s} {spec['track']:27s} {entry['name']} ({how})")
    return 0


def _validate(a) -> int:
    bench = _bench(a)
    problems = bench.check(hashes=not a.quick)
    print(json.dumps({"benchmark": str(bench.root), "data": str(bench.data), "cases": len(bench.cases),
                      "assets": len(bench.assets), "problems": problems}, indent=1))
    return 1 if problems else 0


def _run(a) -> int:
    import time
    from .runtime.batch import admission, run_attempt
    from .util import read_json
    system_id, spec = _system(a)
    refused = _refuse_without_credentials(spec)
    if refused:
        return refused
    bench = _bench(a)
    if a.admission:
        rows = {r["case_id"]: r for r in read_json(a.admission)["cases"]}
        if a.case not in rows or not rows[a.case]["admitted"]:
            raise SystemExit(f"{a.case} is not admitted in {a.admission}")
        row = rows[a.case]
    else:
        rows = {r["case_id"]: r for r in admission(system_id, spec, bench)["cases"]}
        if a.case not in rows:
            raise SystemExit(f"{a.case} is not a case of track {spec['track']}")
        row = {**rows[a.case], "outside_batch": True}
    problems = bench.check([a.case], hashes=False)
    if problems:
        raise SystemExit(f"media of {a.case} not found under {bench.data}: {', '.join(sorted(problems))} "
                         "(set STREAMEVAL_DATA or --data; check with `streameval validate`)")
    out = a.out or RUNS / system_id / "single" / f"{a.case}.r{a.repeat:02d}.{time.strftime('%Y%m%dT%H%M%S')}"
    result = run_attempt(system_id, spec, bench, a.case, out, repeat=a.repeat, url=a.url if a.system else None,
                         launch=a.launch, smoke=a.smoke, admission_row=row, verify=not a.no_verify)
    failure = result.get("failure") or {}
    print(json.dumps({"attempt": str(out), "status": result["status"],
                      "started": result["scheduler"]["started_at_ns"] is not None,
                      "complete": result["activity_execution_complete"],
                      "frames_received": (result.get("browser_finish") or {}).get("frames"),
                      "unsupported": len(result["unsupported_events"]),
                      **({"failure": failure.get("type") + ": " + failure.get("message", "")[:200]} if failure else {})}))
    return 0 if result["status"] == "completed" else 1


def _batch(a) -> int:
    from .models import registry
    from .runtime.batch import dry_run, run_batch
    spec = registry.spec_of(a.system, a.registry)
    if a.dry_run:
        print(json.dumps(dry_run(a.system, spec, _bench(a), population=a.population, repetitions=a.repetitions,
                                 preparation=a.preparation), indent=1))
        return 0
    refused = _refuse_without_credentials(spec)
    if refused:
        return refused
    report = run_batch(a.system, spec, _bench(a), a.out, population=a.population, preparation=a.preparation,
                       repetitions=a.repetitions, concurrency=a.concurrency, seed=a.seed, limit=a.limit,
                       only=a.only, auto_retry=not a.no_auto_retry, url=a.url, launch=a.launch,
                       registry=tuple(a.registry), stagger_s=a.stagger, budget=a.budget,
                       log=lambda line: print(line, flush=True))
    print(json.dumps(report, indent=1))
    return 1 if report.get("halted") else 0     # server down or budget reached: jobs are left unrun


def _validate_run(a) -> int:
    from .runtime.contract import validate_run
    from .util import read_json
    cases = _bench(a).cases if (a.benchmark / "cases.jsonl").exists() else {}
    attempts = []
    for path in a.paths:
        attempts += [path] if (path / "binding.json").exists() or (path / "result.json").exists() else sorted(
            p.parent for p in path.rglob("binding.json"))
    bad = 0
    for attempt in attempts:
        case = cases.get(read_json(attempt / "binding.json")["case_id"]) if (attempt / "binding.json").exists() else None
        report = validate_run(attempt, case)
        bad += not report["ok"]
        if a.verbose or not report["ok"]:
            print(json.dumps(report, ensure_ascii=False))
    print(json.dumps({"attempts": len(attempts), "invalid": bad}))
    return 1 if bad or not attempts else 0


def _serve(a) -> int:
    from .models.server import main_serve
    main_serve(a.model, a.kwargs, a.host, a.port)
    return 0


def _leaderboard(a) -> int:
    from .benchmark.roster import Roster
    from .scoring import leaderboard
    if a.out is None:
        print("give --out FILE (a local board file to read or write)", file=sys.stderr)
        return 2
    if a.table:
        if not a.out.is_file():
            print(f"local board file not found: {a.out}", file=sys.stderr)
            return 2
        print(leaderboard.markdown(json.loads(a.out.read_text())))
        return 0
    board = leaderboard.build(a.results, Roster(a.roster))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(leaderboard.encode(board))
    print(f"wrote {a.out}: {len(board['entries'])} entries")
    return 0


def _judge_set(a):
    """The JudgeSet of --benchmark with the preparations under --runs (or --preparation)."""
    from .judge.units import JudgeSet
    preps = a.preparation or sorted(p.parent for p in Path(a.runs).glob("preparation/*/manifest.json"))
    if not preps:
        raise SystemExit(f"no preparation under {a.runs}/preparation; give --preparation DIR")
    return JudgeSet.from_benchmark(_bench(a), preps)


def _evidence_root(a) -> Path:
    return Path(a.evidence_root or a.runs).resolve()


def _roots(a) -> tuple[Path, Path, Path]:
    runs, out, root = Path(a.runs).resolve(), Path(a.out).resolve(), _evidence_root(a)
    for path in (runs, out):
        if not path.is_relative_to(root):
            raise SystemExit(f"{path} is outside the evidence root {root}: put --out inside --runs (e.g. "
                             f"{runs / 'eval'}), or give an --evidence-root that holds both")
    return runs, out, root


def _score_unsupported(a, rel, out: Path, root: Path) -> int:
    _score_unsupported_plan(a.score_unsupported, rel, out, root, a.admission)
    return 0


def _score_unsupported_plan(plan_path, rel, out: Path, root: Path, admission_path=None) -> int:
    """Account for known unsupported conditions using the frozen repeat count."""
    from .judge.units import score_unsupported_job
    from .util import read_json, digest
    plan_path = Path(plan_path)
    batch = read_json(plan_path)
    config = batch["system_config"]
    admission = read_json(admission_path or plan_path.parent / "admission.json")
    if digest(config) != batch["system_config_sha256"] or digest(admission) != batch["admission_sha256"]:
        raise ValueError("unsupported accounting does not match the frozen batch")
    scored = 0
    for case in admission["cases"]:
        if (case["admitted"] or case["group"] != config["group"] or case["input_profile"] != config["input_profile"]
                or case["case_id"] not in rel.plans
                or rel.population[case["case_id"]] != batch["population"]):
            continue
        reason = "not_admitted: " + "; ".join(case.get("reasons") or ["unsupported"])
        manifest = read_json(rel.preparation(case["case_id"]) / "manifest.json")
        for repeat in range(1, manifest["repetitions"] + 1):
            print(json.dumps(score_unsupported_job(case["case_id"], repeat, batch["system_id"],
                                                   batch["system_config_sha256"], reason, out, rel, root)))
            scored += 1
    return scored


def _judge(a) -> int:
    from concurrent.futures import ThreadPoolExecutor
    from .judge import av_sync
    from .judge.client import live_judges, replay_judges
    from .judge.prompts import JUDGE_VERSION
    from .judge.units import attempts, has_failed_units, judge_context, judge_job, read_json
    from .util import digest
    runs, out, root = _roots(a)
    rel = _judge_set(a)
    if a.score_unsupported:
        return _score_unsupported(a, rel, out, root)
    shard, shards = (int(x) for x in a.shard.split("/"))
    if a.replay and Path(a.replay).resolve() == out:
        raise SystemExit("--replay must name another eval root than --out (the stored jobs would be overwritten)")
    if shards < 1 or not 0 <= shard < shards:
        raise SystemExit("--shard must be i/n with n > 0 and 0 <= i < n")
    available = sorted(attempts(runs, a.system, a.only))
    todo = [x for i, x in enumerate(available) if i % shards == shard]
    unsupported = 0
    if shard == 0 and not a.only:
        for path in sorted(runs.glob("*/*/batch_plan.json")):
            if not a.system or read_json(path)["system_id"] == a.system:
                unsupported += _score_unsupported_plan(path, rel, out, root)
    if not todo:
        print(json.dumps({"judged": 0, "attempts": 0, "unsupported": unsupported}))
        return 0 if unsupported or available else 1
    live = None if a.replay else live_judges()
    done, misses, failures = 0, 0, 0
    mouth = []  # the live A-V Sync lip-opening judge, created on first use

    def with_av_sync(attempt, b, replay_mouth):
        """Judge (or reuse) the job's A-V Sync record and splice its decisions; never re-runs the other metrics."""
        nonlocal failures
        score = read_json(out / b["system_id"] / b["job_id"] / "score.json")
        if score is None or not av_sync.needs_av_sync(score, rel.plans[b["case_id"]]["group"]):
            return None
        try:
            have = av_sync.load_record(out / b["system_id"] / b["job_id"], attempt.name) is not None
            judge = None
            if not have:
                if a.replay:
                    judge = replay_mouth
                else:
                    if not mouth:
                        mouth.append(av_sync.MouthJudge())
                    judge = mouth[0]
            summary = av_sync.av_sync_job(attempt, out, rel, judge, root)
            if isinstance(summary.get("av_sync"), str):
                failures += 1
            return summary
        except av_sync.GatewayRefused as exc:
            failures += 1
            return {"job_id": b["job_id"], "av_sync_refused": str(exc)[:300]}
        except Exception as exc:  # noqa: BLE001 - the other metrics of the job stay valid
            failures += 1
            return {"job_id": b["job_id"], "av_sync_error": f"{type(exc).__name__}: {exc}"[:500]}

    with ThreadPoolExecutor(a.workers) as pool:
        for attempt in todo:
            b = read_json(attempt / "binding.json")
            if b is None:
                print(json.dumps({"attempt": attempt.name, "skipped": "binding.json went away (voided mid-pass)"}),
                      flush=True)
                continue
            replay_mouth = None
            if a.replay:
                stored = Path(a.replay) / b["system_id"] / b["job_id"]
                judges, asr = replay_judges(stored)
                replay_mouth = av_sync.replay_mouth_judge(stored)
            else:
                judges, asr = live
            score_path = out / b["system_id"] / b["job_id"] / "score.json"
            previous = read_json(score_path) if score_path.exists() and not a.only and not a.replay else None
            if previous is not None:
                if (previous.get("judge_version") == JUDGE_VERSION and previous.get("attempt") == attempt.name
                        and previous.get("judge_context_sha256") == digest(judge_context(attempt, rel, judges, asr))
                        and not has_failed_units(score_path.parent)):
                    av = with_av_sync(attempt, b, replay_mouth)
                    if av:
                        print(json.dumps(av), flush=True)
                    continue
            try:
                summary = judge_job(attempt, out, rel, judges, asr, pool, root)
                done += 1
                if has_failed_units(score_path.parent):
                    failures += 1
                print(json.dumps(summary), flush=True)
                av = with_av_sync(attempt, b, replay_mouth)
                if av:
                    print(json.dumps(av), flush=True)
            except Exception as exc:  # noqa: BLE001 - one job never stops the pass
                failures += 1
                print(json.dumps({"job_id": b["job_id"], "error": f"{type(exc).__name__}: {exc}"[:500]}), flush=True)
            if a.replay:
                missed = [m for j in (judges["primary"], judges["second"], replay_mouth) if j is not None
                          for m in j.misses]
                if missed:
                    misses += len(missed)
                    print(json.dumps({"job_id": b["job_id"], "not_in_replay": missed[:5], "count": len(missed)}),
                          flush=True)
    print(json.dumps({"judged": done, "attempts": len(todo), "failures": failures, **({"replay_misses": misses} if a.replay else {})}))
    return 1 if misses or failures else 0


def _assemble(a) -> int:
    from .benchmark.roster import Roster
    from .scoring.assemble import assemble
    rel = _judge_set(a)
    release = Roster(a.roster).release_manifest_sha256 if a.roster else None
    return assemble(a.runs, a.eval, a.out, rel, _evidence_root(a), release_manifest_sha256=release,
                    av_sync=not a.no_av_sync, log=lambda line: print(line, flush=True))


def _evaluate(a) -> int:
    """batch -> judge -> assemble -> leaderboard for one system on one population, all under --out."""
    out = a.out.resolve()
    common = ["--benchmark", str(a.benchmark)] + (["--data", str(a.data)] if a.data else [])
    registry = [x for r in a.registry for x in ("--registry", str(r))]
    model = registry + (["--url", a.url] if a.url else []) + (["--launch"] if a.launch else [])
    judged = ["--runs", str(out), "--evidence-root", str(out)] + common
    steps = [
        ["batch", "--system", a.system, "--out", str(out), "--population", a.population,
         "--repetitions", str(a.repetitions), "--concurrency", str(a.concurrency)]
        + (["--budget", str(a.budget)] if a.budget is not None else []) + common + model,
        ["judge", "--system", a.system, "--out", str(out / "eval"), "--workers", str(a.judge_workers)] + judged,
        ["assemble", "--eval", str(out / "eval"), "--out", str(out / "results")] + judged,
        ["leaderboard", "--results", str(out / "results"), "--roster", str(out / "preparation" / a.population),
         "--out", str(out / f"leaderboard_{a.population}.json")],
    ]
    for argv in steps:
        print(json.dumps({"evaluate": argv[0]}), flush=True)
        code = main(argv)
        if code:
            print(json.dumps({"evaluate": argv[0], "exit": code}), flush=True)
            return code
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="streameval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def bench_args(s):
        s.add_argument("--benchmark", type=Path, default=REPO / "benchmark/v1/full",
                       help="roster folder of benchmark/v1")
        s.add_argument("--data", type=Path, help="media root (default: STREAMEVAL_DATA, else data/ in the checkout)")

    def model_args(s):
        s.add_argument("--registry", action="append", default=[], type=Path,
                       help="extra system.json files or folders (repeatable; also STREAMEVAL_MODELS)")
        s.add_argument("--url", help="an already running model server (ws://host:port/session)")
        s.add_argument("--launch", action="store_true", help="start the registered server command for the run")

    s = sub.add_parser("list", help="registered systems and the tracks")
    s.add_argument("--registry", action="append", default=[], type=Path)
    s.set_defaults(func=_list)

    s = sub.add_parser("validate", help="check the benchmark media against the manifest")
    bench_args(s)
    s.add_argument("--quick", action="store_true", help="presence only, no sha256")
    s.set_defaults(func=_validate)

    s = sub.add_parser("run", help="run one session of one case")
    s.add_argument("--system", help="registered system id (streameval list)")
    s.add_argument("--model", help="unregistered in-process model: module:Class or file.py:Class (with --track)")
    s.add_argument("--track", help="track of an unregistered model")
    s.add_argument("--case", required=True)
    s.add_argument("--repeat", type=int, default=1)
    s.add_argument("--out", type=Path, help="attempt folder (default runs/<system>/single/<case>.rNN.<time>)")
    s.add_argument("--admission", type=Path, help=argparse.SUPPRESS)
    s.add_argument("--smoke", action="store_true", help="mark the attempt as a smoke test, not a sample")
    s.add_argument("--no-verify", action="store_true", help="skip the sha256 check of the media read")
    bench_args(s)
    model_args(s)
    s.set_defaults(func=_run)

    s = sub.add_parser("batch", help="run every job of a registered system on a benchmark set")
    s.add_argument("--system", required=True)
    s.add_argument("--out", type=Path, default=RUNS)
    s.add_argument("--population", default="full")
    s.add_argument("--preparation", type=Path, help="frozen preparation (default <out>/preparation/<population>, made if missing)")
    s.add_argument("--repetitions", type=int, default=3)
    s.add_argument("--concurrency", type=int, default=1)
    s.add_argument("--seed", type=int, default=0, help="seed of the job order")
    s.add_argument("--limit", type=int, help="at most this many first attempts")
    s.add_argument("--only", action="append", help="only these job ids (repeatable)")
    s.add_argument("--no-auto-retry", action="store_true", help="do not re-attempt non-starts")
    s.add_argument("--stagger", type=float, default=1.0, help="seconds between session starts")
    s.add_argument("--dry-run", action="store_true",
                   help="report sessions, active seconds, unsupported events, estimated cost and missing credentials; "
                        "no file is written, no media read, no request made")
    s.add_argument("--budget", type=float,
                   help="vendor systems: start no attempt that would bring the estimated cost of the folder's "
                        "attempts above this (in the backend's cost unit)")
    bench_args(s)
    model_args(s)
    s.set_defaults(func=_batch)

    s = sub.add_parser("validate-run", help="check attempt folders (or folders containing them) against the contract")
    s.add_argument("paths", nargs="+", type=Path)
    s.add_argument("--verbose", action="store_true")
    bench_args(s)
    s.set_defaults(func=_validate_run)

    s = sub.add_parser("serve", help="serve an in-process model over the WebSocket protocol")
    s.add_argument("--model", required=True, help="module:Class or file.py:Class")
    s.add_argument("--kwargs", help="JSON constructor arguments")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=_serve)

    s = sub.add_parser("leaderboard", help="aggregate per-job scores into the leaderboard file")
    s.add_argument("--results", type=Path, default=RUNS / "results", help="results root: <results>/<system>.<population>/")
    s.add_argument("--roster", type=Path, default=RUNS / "preparation/full",
                   help="a local frozen preparation folder or an explicitly supplied roster.json")
    s.add_argument("--out", type=Path, help="required local board file to write; with --table, the local file to read")
    s.add_argument("--table", action="store_true", help="print the board file --out as Markdown tables; no rebuild")
    s.set_defaults(func=_leaderboard)

    def judge_args(s, out_default=True):
        s.add_argument("--runs", type=Path, default=RUNS, help="run root: <runs>/<system>/<population>/<job>/")
        if out_default:
            s.add_argument("--out", type=Path, default=RUNS / "eval", help="eval root: <out>/<system>/<job>/")
        s.add_argument("--evidence-root", type=Path,
                       help="evidence references are relative to it; holds --runs and the eval root (default: --runs)")
        s.add_argument("--preparation", type=Path, action="append",
                       help="preparation folder(s) (default: every <runs>/preparation/*)")
        bench_args(s)

    s = sub.add_parser("judge", help="judge attempts: units, observations.json, score.json, latency.json per job")
    judge_args(s)
    s.add_argument("--workers", type=int, default=12, help="concurrent judge requests within a job")
    s.add_argument("--system")
    s.add_argument("--only", nargs="*", help="these job ids only (always re-judged)")
    s.add_argument("--shard", default="0/1", help="i/n: judge only every n-th attempt, starting at i")
    s.add_argument("--replay", type=Path, metavar="EVAL",
                   help="answer every judge request from the stored judged jobs EVAL/<system>/<job>/ (no API call); "
                        "exit 1 when a request is not found there")
    s.add_argument("--score-unsupported", type=Path, metavar="BATCH_PLAN",
                   help="score every repeat of each condition the batch's admission did not admit as an unsupported "
                        "failure (no model call), then exit")
    s.add_argument("--admission", type=Path, help="with --score-unsupported (default: admission.json beside the plan)")
    s.add_argument("--repetitions", type=int, default=3)
    s.set_defaults(func=_judge)

    s = sub.add_parser("assemble", help="collect judged jobs into <out>/<system>.<population>/ for the leaderboard")
    judge_args(s, out_default=False)
    s.add_argument("--eval", type=Path, default=RUNS / "eval")
    s.add_argument("--out", type=Path, required=True, help="results root")
    s.add_argument("--roster", type=Path, help="optional dataset roster")
    s.add_argument("--no-av-sync", action="store_true",
                   help="copy scores as they are, without re-deciding A-V Sync over the system population")
    s.set_defaults(func=_assemble)

    s = sub.add_parser("evaluate", help="batch, judge, assemble and leaderboard of one system in one go")
    s.add_argument("--system", required=True)
    s.add_argument("--out", type=Path, default=RUNS, help="run root; eval/, results/ and the leaderboard go inside")
    s.add_argument("--population", default="full")
    s.add_argument("--repetitions", type=int, default=3)
    s.add_argument("--concurrency", type=int, default=1)
    s.add_argument("--budget", type=float, help="vendor systems: as for batch")
    s.add_argument("--judge-workers", type=int, default=12)
    bench_args(s)
    model_args(s)
    s.set_defaults(func=_evaluate)

    a = p.parse_args(argv)
    if not (REPO / "evaluation/protocol").is_dir():
        raise SystemExit(f"{REPO} is not a StreamEval checkout (no evaluation/protocol): install the package from "
                         "a clone with `pip install -e .`, or set STREAMEVAL_ROOT to the clone")
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
