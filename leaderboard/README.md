# Published paper results

Results of **Stream-Arena: Evaluating Interactive Streaming Video Generation Systems**, Section 3 (System Evaluation).

| File | Content |
|---|---|
| [v1_leaderboard.json](v1_leaderboard.json) | Only the paper's 49 core configurations, metric values and status |
| [v1_leaderboard.md](v1_leaderboard.md) | The five core result tables, rendered from the JSON |
| [paper_results.json](paper_results.json) | The manuscript values used as the publication allowlist |

The core population contains 49 configurations of 39 reported models. Extension and text-input data remain
available for evaluation; their historical results are not published here. Timing, coverage, confidence
intervals, per-run evidence and numerical diagnostics absent from the paper stay with the evaluator.

## Display

Content scores have two decimals. Unresolved measurements show their lower bound. N/A means not applicable;
\* marks a provisional metric. † shared ASR, dialogue and TTS stack; ‡ provider-supplied avatar (Identity N/A);
§ separate input route. Rows are alphabetical; no significance ranking is assigned.

## Publication boundary

Each entry contains model identity, task and input groups, and only the metrics printed for that row in the
paper. Metrics retain their status and provisional flag. Latency, coverage and auxiliary diagnostic fields
are not part of the publication schema. The maintained release must pass `scripts/verify_release.py`, which
rejects extra rows, extra fields and changed paper scores. `paper_board` projects internal aggregate records
onto the manuscript allowlist and rejects mismatched values; it never substitutes paper scores into new runs.

```bash
stream-arena leaderboard --table   # render the maintained paper tables
```

Local evaluations of new systems or other populations remain supported. Their outputs under the run root are
local evaluation results, not additions to this historical publication. Updating the manuscript allowlist
requires checking the corresponding revised paper first.

[Evaluation rules](../docs/evaluation.md) · [Benchmark](../benchmark/v1/README.md) ·
[Reproduction](../docs/reproduce.md)
