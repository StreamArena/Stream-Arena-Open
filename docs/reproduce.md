# Running evaluations

## Setup and model launch

Follow the homepage's [Quick start](../README.md#quick-start) to install the package, then [download and validate the benchmark](../README.md#download-and-validation). Evaluation uses `benchmark/v1/full`.

Before running a real model, follow [Model evaluation](../README.md#model-evaluation) to install its dependencies, configure `.env` and choose how to start or connect to its service. Use `--launch` for a registered self-hosted server, or `--url ws://HOST:PORT/session` for a running server. Hosted services follow their adapter instructions.

Settings for comparable runs, including repetitions, input settings and judge configuration, are in the [evaluation protocol](evaluation.md).

## Individual stages

The batch example below launches a registered self-hosted server; replace `--launch` with `--url ws://HOST:PORT/session` for a running server. For hosted services, follow the adapter instructions.

```bash
streameval batch --system ID --launch --out runs --repetitions 3 --benchmark benchmark/v1/full --population full
streameval judge --runs runs --out runs/eval --system ID --evidence-root runs --benchmark benchmark/v1/full
streameval assemble --runs runs --eval runs/eval --out runs/results --evidence-root runs --benchmark benchmark/v1/full
streameval leaderboard --results runs/results --roster runs/preparation/full --out runs/leaderboard_full.json
streameval leaderboard --table --out runs/leaderboard_full.json
```

`evaluate` runs the first four stages and stops at a nonzero exit code. A resumed batch keeps its frozen settings
and sample selection; only documented pre-epoch failures are retried. Use a new run directory to change the
dataset, judge or input contract.

| Path under the run root | Content |
|---|---|
| `preparation/<population>/` | frozen metric plans and repeated jobs |
| `<system>/<population>/<job>/` | input bindings, scheduler result, media and timestamps |
| `eval/<system>/<job>/` | judge context, structured units, scores and latency accounting |
| `results/<system>.<population>/` | system input contract and assembled job scores |
| `leaderboard_full.json` | local aggregate content results |

`streameval validate-run runs/<system> --benchmark benchmark/v1/full` checks attempts against the
[run contract](protocol.md#run-contract-the-attempt-folder). Assembly verifies the configuration, case, oracle and
metric-plan bindings and keeps every expected repeat, and every eligible latency opportunity, in the denominator.

## Replay

```bash
streameval judge --runs runs --out runs/replay --replay runs/eval --evidence-root runs --benchmark benchmark/v1/full
```

Stored request hashes select the recorded replies; a missing request exits with code 1. Both eval roots must lie
inside the evidence root and differ. Exact image bytes require the same ffmpeg and Pillow versions.

## Checks

`pytest -q` runs the test suite, including 15-second sessions and a scripted evaluation and replay chain (the
scripted judge exercises code paths only). `python scripts/verify_release.py` checks the release contents,
scoring policy and frozen source hashes; maintainers refresh the hashes with `--freeze` after review. A
non-editable installation needs `STREAMEVAL_ROOT` pointing at a full checkout, since the wheel contains neither
the model registry nor the benchmark files.

The code release contains evaluation tools and benchmark definitions. Keep generated scores, tables,
calibration records and session outputs under your local run root. Table rendering requires an explicit
`--out FILE`; no evaluation results are bundled.
