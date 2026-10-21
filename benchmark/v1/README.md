# StreamEval benchmark (v1)

The complete benchmark is available in [`full/`](full/). It covers five streaming-video tasks and the
15, 30, 60, 120 and 300 second duration strata. Evaluate every applicable condition with three independent
repetitions. Text-dialogue variants are also provided.

Each benchmark folder contains:

| File | Contents |
|---|---|
| `cases.jsonl` | Input timelines, initialization, duration and source-family identifiers |
| `oracles.jsonl` | Expected outcomes, timing windows and protected content; evaluator only |
| `manifest.json` | Asset paths under the data root, SHA-256 hashes, types and roles |

[`ATTRIBUTION.md`](ATTRIBUTION.md) credits the 55 third-party input videos and their licences.
The other 683 assets were generated for the benchmark.

## Figure statistics

The homepage statistics figure describes a benchmark subset. Its panels show task composition, planned input types, inputs per session and gaps between adjacent scheduled inputs. Event counts exclude initialization and continuous source frames; gap statistics exclude adjacent pairs containing a trigger. These figure statistics do not describe the complete evaluation set.

## Running it

Follow the homepage's [Quick start](../../README.md#quick-start) for installation, [download and validation](../../README.md#download-and-validation), and [model evaluation](../../README.md#model-evaluation). The model instructions cover dependencies, credentials and choosing between `--launch` and `--url`.

Oracles and future inputs are never passed to the participating model. Settings for comparable runs are in the [evaluation protocol](../../docs/evaluation.md).
