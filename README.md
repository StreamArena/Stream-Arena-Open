<div align="center">

# Stream-Arena

**Evaluating Interactive Streaming Video Generation Systems**

[![Leaderboard](https://img.shields.io/badge/Leaderboard-v1-4C6EF5)](leaderboard/v1_leaderboard.md)
[![Hugging Face](https://img.shields.io/badge/Data-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/StreamArena/Stream-Arena-Open)
[![ModelScope](https://img.shields.io/badge/Data-ModelScope-624AFF)](https://www.modelscope.ai/datasets/StreamArena/Stream-Arena-Open)
[![Code License](https://img.shields.io/badge/Code-Apache%202.0-4CAF50)](LICENSE)
[![Data License](https://img.shields.io/badge/Data-CC%20BY%204.0-4CAF50)](#license)

**English** | [简体中文](README.zh-CN.md)

</div>

Interactive streaming video generation requires systems to respond to evolving user inputs while preserving
relevant content and state throughout a session. **Stream-Arena** is a benchmark of five tasks across avatar
interaction, world generation and real-time video editing. Sessions span 15–300 s, and multimodal inputs arrive
incrementally — a user speaks, an instruction or a reference image is released, a source video keeps playing —
to test both the execution of new requests and the retention of prior context. The output is recorded as a viewer
receives it, and an automated judge scores every session and response.

## At a glance

| | |
|---|---|
| Tasks | digital-human dialogue, avatar performance, world generation, instruction editing, reference editing |
| Benchmark | 300 conditions and 60 text-dialogue derivatives; reported core roster of 149 effective conditions |
| Sessions | 15, 30, 60, 120 and 300 s duration strata; 3 repeats per condition |
| Results | 39 models, 49 core configurations, 4,389 runs |
| Pipeline | one session harness, one model interface (in-process or WebSocket), one automated judge |
| Release | code, the complete benchmark (timelines, oracles, media) and the published results |

## Tasks and metrics

| Task | The system receives | Scored on |
|---|---|---|
| Digital-human dialogue | a portrait and a persona; the user's speech (or text) turns | Quality, A–V Sync, Stability, Dialogue, Identity |
| Avatar performance | a portrait; speech to perform and live directions | Quality, A–V Sync, Stability, Following, Identity |
| World generation | an opening prompt (or image); live prompts and camera controls | Quality, Stability, Following, Continuity (+ Reference in the image-conditioned extension) |
| Instruction editing | a live source video; text edit instructions | Quality, Stability, Following, Edit Success, Preservation |
| Reference editing | a live source video; reference images to apply, patch, remove or reset | Quality, Stability, Reference, Edit Success, Preservation |

Metrics are scored 0–100. A task's Overall is the equal-weight mean of its applicable content metrics;
only the paper's content scores are published, and tasks are never merged into one total.
[Definitions](docs/evaluation.md).

## Benchmark

![Benchmark construction and verification](assets/construction.jpg)

<sub>**Benchmark construction and verification.** Model-assisted authoring and input synthesis precede
modality-specific checks. A Qwen-Image portrait with hand defects is regenerated and passes the same screening.</sub>

![Benchmark composition](assets/statistics.png)

<sub>**Benchmark composition.** (a) Task and function composition. (b) Planned input types. (c) Inputs per session.
(d) Gaps between adjacent scheduled inputs. The 200 core and extension conditions contain 562 online events, excluding initialization
and continuous source frames. Gap statistics exclude pairs containing a trigger.</sub>

The benchmark ([`benchmark/v1`](benchmark/v1/README.md)) has 300 conditions, 60 per task and 12 per task in each
duration stratum, from 240 source families, plus 60 text-dialogue derivatives. Results are reported on fixed
rosters of it:

| Roster | Folder | Conditions | Reported as |
|---|---|---:|---|
| Core | [`core`](benchmark/v1/core) | 149 | main results |
| Extension | [`extension`](benchmark/v1/extension) | 50 | additional conditions |
| Text dialogue | [`text`](benchmark/v1/text), [`text_extension`](benchmark/v1/text_extension) | 30 + 10 | text-input dialogue |
| Development | [`development`](benchmark/v1/development) | 100 + 20 | default of the command line, for integration |

Each roster folder has `cases.jsonl` (the timeline of inputs), `oracles.jsonl` (expected outcomes, read only by the
judge) and `manifest.json` (every media file with its sha256). The media are on the dataset hubs (see [Data](#data)).

## Evaluation

- **Harness:** every task uses the same harness and model interface. It releases inputs on a clock and timestamps
  the output, so each response is attributed to the input that triggered it.
- **Judge:** structured visual and text observations are converted to scores by fixed rules; recorded requests
  support offline replay.
- **Scoring:** equal aggregation over repeats, conditions and duration strata. Unresolved measurements remain unresolved and contribute zero
  only when computing the displayed lower bound.

## Results

The [results tables](leaderboard/v1_leaderboard.md) reproduce the paper's five core tables. The
[JSON file](leaderboard/v1_leaderboard.json) contains their 49 configurations and metric status. Other populations,
timing, coverage and detailed numerical diagnostics stay in the evaluator's local records. Rows are alphabetical; A–V Sync is provisional.

## Data

This code repository publishes the benchmark implementation, definitions, scoring rules and aggregate results.
Session logs, per-run judge records, sampled model outputs and output demonstrations stay with the evaluator.
Hugging Face and ModelScope contain benchmark data only; implementation and published result tables are here.

The benchmark definitions are in this repository; the media (≈3.1 GB) are on
[Hugging Face](https://huggingface.co/datasets/StreamArena/Stream-Arena-Open) and
[ModelScope](https://www.modelscope.ai/datasets/StreamArena/Stream-Arena-Open), which hold the same files:

```python
from huggingface_hub import snapshot_download        # or: from modelscope import dataset_snapshot_download
path = snapshot_download("StreamArena/Stream-Arena-Open", repo_type="dataset")
```

```bash
export STREAM_ARENA_DATA=<path>/data                  # contains v1/
stream-arena validate --benchmark benchmark/v1/core   # every file present with its manifest sha256
```

## Quick start

Python ≥ 3.10 on Linux or macOS. The first check uses generated CPU-only media; it needs no API key, GPU or download.

```bash
git clone https://github.com/StreamArena/Stream-Arena-Open.git stream-arena && cd stream-arena
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[runtime,judge,dev]"
python -m stream_arena.benchmark.synthetic --out .smoke
stream-arena run --model stream_arena.models.mock:MockModel --track instruction_edit \
  --case sa.v1.901.instruction_edit --benchmark .smoke/set --data .smoke/data --out .smoke/attempt
stream-arena validate-run .smoke/attempt --benchmark .smoke/set --data .smoke/data
pytest -q
```

To evaluate a registered system, download the media, install the model's dependencies and set the judge
credentials:

```bash
cp .env.example .env
stream-arena list
stream-arena evaluate --system ID --out runs                                         # development roster
stream-arena evaluate --system ID --out runs_core --benchmark benchmark/v1/core --population core
```

Results comparable with the paper use the core roster. [docs/reproduce.md](docs/reproduce.md) covers the individual stages
and judge replay; [models/README.md](models/README.md) lists the v1 systems.

## Add your model

Implement one class and register it in a `system.json`:

```python
from stream_arena.models import OutputSink, SessionInit, StreamingModel

class MyModel(StreamingModel):
    def open(self, init: SessionInit, sink: OutputSink): ...   # portrait / prompt / output size; call sink.ready(...)
    def push_audio(self, pts_ns, pcm16k): ...                  # avatar tasks: speech, 10 ms chunks
    def push_source_frame(self, pts_ns, jpeg): ...             # editing tasks: the live source video
    def send_event(self, event): ...                           # released text / reference events; call sink.ack(...)
    def close(self) -> bool: ...
# output: sink.video_frame(jpeg), sink.audio(pcm48k)
```

A model can also serve sessions over a WebSocket, in any language. Start from [`examples/my_model`](examples/my_model)
and [docs/add_a_model.md](docs/add_a_model.md).

## Documentation

| Document | Content |
|---|---|
| [docs/add_a_model.md](docs/add_a_model.md) | the model interface, `system.json`, servers, composed avatars, vendor adapters |
| [docs/protocol.md](docs/protocol.md) | the WebSocket protocol and what each session writes |
| [docs/evaluation.md](docs/evaluation.md) | metrics, aggregation and the judge |
| [docs/reproduce.md](docs/reproduce.md) | running a set, individual stages, judge replay, result files |
| [benchmark/v1/README.md](benchmark/v1/README.md) | the benchmark, its rosters and errata |
| [leaderboard/README.md](leaderboard/README.md) | the published results and their format |
| [models/README.md](models/README.md) | the v1 systems: upstream code, weights and licences |

<details>
<summary>Repository layout</summary>

```text
stream_arena/          Python package: benchmark, runtime (harness), models, providers, judge, scoring, cli
benchmark/v1/          the benchmark: core, extension, text, text_extension, development rosters (+ errata, attribution)
models/                v1 systems: self-hosted model servers and vendor adapters
examples/my_model/     template model, in-process and as a WebSocket server
evaluation/protocol/   evaluation policy, scoring specification, judge validation summaries
leaderboard/           published results (JSON and tables)
assets/                benchmark construction and composition figures
tests/                 CPU test suite; generates its own media; no keys or GPU needed
```

</details>

## Citation

```bibtex
@misc{streamarena2026,
  title        = {Stream-Arena: Evaluating Interactive Streaming Video Generation Systems},
  author       = {{Stream-Arena Team}},
  year         = {2026},
  howpublished = {\url{https://github.com/StreamArena/Stream-Arena-Open}}
}
```

## License

- **Code:** [Apache License 2.0](LICENSE). Wrapped third-party models keep their own licences, several of them
  non-commercial ([models/README.md](models/README.md)); vendor adapters are subject to each vendor's terms.
- **Benchmark data** (timelines, oracles, generated media, results): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  The editing source videos are excerpts of third-party footage and keep their own licences (CC BY, CC BY-SA,
  CC0 or public domain); credit them as listed in [benchmark/v1/ATTRIBUTION.md](benchmark/v1/ATTRIBUTION.md), including in
  edited outputs you publish. See [NOTICE](NOTICE).
