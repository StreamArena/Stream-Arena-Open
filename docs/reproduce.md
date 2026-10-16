# Running evaluations

## Benchmark data

| Roster | Folder | Conditions | Use |
|---|---|---:|---|
| Core | `benchmark/v1/core` | 149 | main results |
| Extension | `benchmark/v1/extension` | 50 | local evaluation |
| Text-input dialogue | `benchmark/v1/text`, `benchmark/v1/text_extension` | 30 + 10 | local evaluation |
| Development | `benchmark/v1/development` | 100 + 20 | command-line default, integration |

Each folder holds `cases.jsonl` (input timelines), `oracles.jsonl` (expected outcomes, read only by the judge) and
`manifest.json` (media paths and hashes). Media are not in git: download them from
[Hugging Face or ModelScope](../README.md#data) and place `v1/` under `data/` or `$STREAM_ARENA_DATA`.

```bash
pip install -e ".[runtime,judge]"
cp .env.example .env            # model and judge credentials; all run output stays local
stream-arena validate           # media present with their manifest hashes
stream-arena list
stream-arena evaluate --system ID --out runs                     # development roster
stream-arena evaluate --system ID --out runs_core --benchmark benchmark/v1/core --population core
```

Model installation is described in [models/](../models/README.md). Scores describe the roster they were run on;
comparisons with the paper require `benchmark/v1/core`, three repeats and the same judge configuration.

## Individual stages

```bash
stream-arena batch --system ID --out runs --repetitions 3
stream-arena judge --runs runs --out runs/eval --system ID --evidence-root runs
stream-arena assemble --runs runs --eval runs/eval --out runs/results --evidence-root runs
stream-arena leaderboard --results runs/results --roster runs/preparation/development --out runs/leaderboard_development.json
stream-arena leaderboard --table --out runs/leaderboard_development.json
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
| `leaderboard_development.json` | local aggregate content results |

`stream-arena validate-run runs/<system>` checks attempts against the
[run contract](protocol.md#run-contract-the-attempt-folder). Assembly verifies the configuration, case, oracle and
metric-plan bindings and keeps every expected repeat, and every eligible latency opportunity, in the denominator.

## Replay

```bash
stream-arena judge --runs runs --out runs/replay --replay runs/eval --evidence-root runs
```

Stored request hashes select the recorded replies; a missing request exits with code 1. Both eval roots must lie
inside the evidence root and differ. Exact image bytes require the same ffmpeg and Pillow versions.

## Checks

`pytest -q` runs the test suite, including 15-second sessions and a scripted evaluation and replay chain (the
scripted judge exercises code paths only). `python scripts/verify_release.py` checks every paper score, Common-4,
the display rules and the frozen source hashes; maintainers refresh the hashes with `--freeze` after review. A
non-editable installation needs `STREAM_ARENA_ROOT` pointing at a full checkout, since the wheel contains neither
the model registry nor the benchmark files.

The maintained publication includes only manuscript-disclosed numerical results. Keep other evaluation
outputs local; see [leaderboard/README.md](../leaderboard/README.md).
