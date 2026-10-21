# Adding a model

From an empty folder to a score on the full roster. The wire protocol and the attempt folder are in
[protocol.md](protocol.md); a working template is [`examples/my_model`](../examples/my_model).

```bash
pip install -e ".[runtime,judge]"
streameval validate                                  # the full roster's media are in place
cp -r examples/my_model ~/my_lab_model                 # then edit model.py / server.py and system.json:
                                                       # every "system_id" must be new (here: my_lab, my_lab_ws)
export STREAMEVAL_MODELS=~/my_lab_model
streameval list | grep my_lab                        # the registry sees it, one system id per track
streameval batch --system my_lab_instruction --dry-run      # sessions, events it cannot take, cost
streameval run --system my_lab_instruction --case sa.v1.001.instruction_edit
streameval validate-run runs/my_lab_instruction
streameval evaluate --system my_lab_instruction      # 3 repeats of every case of the track, judged and scored
```

## 1. Choose a shape

| shape | `system.json` key | when |
|---|---|---|
| in-process Python class | `"python": "model.py:MyModel"` (or `package.module:Class`) | the model runs in the harness's Python environment |
| model server started by the harness | `"server": {"cmd": ..., "port": ...}` + `--launch` | the model needs its own environment (CUDA stack, pinned torch), or another language |
| model server already running | `"url": "ws://host:port/session"` (or `--url` on the command line) | the model runs on another machine, or you start it yourself |
| session backend | `"backend": "backend.py:MyVendor"` | a vendor whose output only exists inside its own media stack (a browser SDK); you write the whole attempt folder ([below](#vendor-systems)) |

The first three implement `StreamingModel` and run through the same harness, which owns the clock, the release of
events, readiness, capture and timing. An in-process class can become a server with no code change:

```bash
streameval serve --model ~/my_lab_model/model.py:MyModel --port 8765 [--kwargs '{"device": "cuda"}']
streameval run --system my_lab_instruction --case sa.v1.001.instruction_edit --url ws://127.0.0.1:8765/session
```

A server in another language implements the messages of [protocol.md](protocol.md);
`streameval/runtime/protocol.py` is standard library only and can be copied next to a Python server. Whatever the
shape, the judge reads only the attempt folder, so every model is scored by the same code.

## 2. The `StreamingModel` interface

```python
from streameval.models import OutputSink, ReferenceEvent, SessionInit, StreamingModel, TextEvent

class MyModel(StreamingModel):
    def open(self, init: SessionInit, sink: OutputSink) -> None: ...          # start; sink.ready(...) when ready
    def send_event(self, event: TextEvent | ReferenceEvent) -> None: ...      # answer with sink.ack(...)
    def push_audio(self, pts_ns: int, pcm16k: bytes) -> None: ...             # audio tracks: 10 ms, 160 samples
    def push_source_frame(self, pts_ns: int, jpeg: bytes) -> None: ...        # editing tracks: the live source
    def close(self) -> bool: ...                                              # True once the session has ended
    def describe(self) -> dict: return {}                                     # optional, into binding.json
    def receipt(self) -> dict: return {}                                      # optional, into result.json
```

### Session lifecycle

```
harness                                           model
open(SessionInit, sink)                    ->     start a session; sink.ready(...) once it can take inputs
send_event(ReferenceEvent "initial")       ->     (reference_edit only) the preloaded reference A
                                           <-     sink.video_frame(jpeg) ... continuously, from the start
                                           <-     sink.audio(pcm 48 kHz)  ... avatar tracks
-- readiness: ready + first frame (avatars: + first frame >= 4 s old and 1.5 s without speech, max 25 s) --
-- activity epoch: the case clock starts --
push_source_frame(pts_ns, jpeg)            ->     editing tracks: the source video, in real time
push_audio(pts_ns, pcm 16 kHz)             ->     audio tracks: a continuous 10 ms stream (silence between clips)
send_event(TextEvent | ReferenceEvent)     ->     a released event; reply with sink.ack(event_id, status)
close()                                    ->     at the case deadline (or on failure)
```

### Inputs

`SessionInit` (frozen dataclass):

| field | meaning |
|---|---|
| `track`, `case_id` | which track and case |
| `width`, `height`, `fps` | requested output (registry `output`); editing: the pushed source size and rate |
| `seed` | registry seed (default 20260929) |
| `output` | the registry `output` dict |
| `prompt` | opening prompt (generation), rendering context (avatar performance), else `""` |
| `persona` | digital human: persona + known facts |
| `image` | bytes: the portrait (avatars with `benchmark_portrait`), the opening reference image (image-to-video), else `None` |
| `duration_ms` | activity length after readiness (lets a vendor adapter size its session; not sent over the WebSocket) |

`TextEvent(event_id, kind, text)`, kind one of `user_prompt`, `text_control`, `camera_control`,
`dialogue_reply` (see [protocol.md](protocol.md#client--server)).

`ReferenceEvent(event_id, op, slots, scope, text)`: `op` is `set | patch | remove | reset`; `slots` is a tuple of
`ReferenceSlot(slot, image, scope, asset_id)`, where `image=None` means restore that slot to the live source.
`text` is set only for an image-to-video reference update (op `set`, slot `subject`).

### Output: `OutputSink`

| method | use |
|---|---|
| `ready(**meta)` | the session accepts inputs; `model`, `version`, `width`, `height`, `fps`, `config` … are recorded |
| `video_frame(jpeg, pts_ns=None)` | one output frame as JPEG |
| `video_rgb(width, height, rgb, pts_ns=None)` | one frame as packed RGB24, or a zero-argument callable returning it (converted only if the frame is sampled) |
| `audio(pcm48k, pts_ns=None)` | speech, s16le mono 48 kHz, any block size, gapless (send silence rather than nothing) |
| `ack(event_id, status="applied", detail=None)` | `applied`, `unsupported` or `error`; one per event, later ones ignored |
| `fatal(message)` | the session cannot continue; the attempt fails |

`pts_ns` is your own clock and only a diagnostic: output is timed when it reaches the sink.

### Threading and guarantees

- `open` must return promptly; report readiness from any thread.
- The harness calls `push_audio` and `push_source_frame` from dedicated pacer threads and `send_event` from its
  dispatch loop. Do not block them; hand the work to your own threads. Every sink method is thread-safe and
  non-blocking.
- A model only sees inputs whose release time has passed. It never sees the oracle or a future event.
- An event whose operation the model does not declare is never sent: it is recorded `unsupported` and scored as a
  failure. `ack(..., "unsupported")` has the same effect; `"error"` records a failed dispatch.
- Conditional turns (a user turn released after the avatar answers) are released on the harness's own voice
  activity detection of the received audio, never on a model message.

### Capabilities and operations

What a model can do is declared in its registry entry, not in code: `tracks`, and optionally `ops` /
`unsupported_ops`. Operations per track (`streameval/benchmark/tracks.py`):

| track | operations | arrive as |
|---|---|---|
| `digital_human_audio` | `user_audio` | `push_audio` |
| `digital_human_text` | `user_prompt` | `TextEvent` |
| `avatar_performance` | `user_prompt`, `speech_audio` | `TextEvent` (directions), `push_audio` (the supplied speech) |
| `general_interaction` | `user_prompt`, `camera_control`; opt-in `action` | `TextEvent`; `ActionEvent(action, phase, duration_ms)` |
| `general_interaction_image` | + `reference_update` | `ReferenceEvent(op="set", slot "subject", text=...)` |
| `instruction_edit` | `text_control` | `TextEvent` |
| `reference_edit` | `reference:set`, `reference:patch`, `reference:remove`, `reference:reset` | `ReferenceEvent` |

Implementations to read: `streameval/models/mock.py:MockModel` (a CPU stand-in for every track),
`streameval/models/websocket.py:WebSocketModel` (the protocol client used by every `server` / `url` entry),
`streameval/providers/rtc.py` (RTC avatars) and `examples/my_model/model.py`. An unregistered in-process model
runs directly:

```bash
streameval run --model examples/my_model/model.py:MyModel --track instruction_edit --case sa.v1.001.instruction_edit
```

## 3. Write `system.json`

One file holds one entry or a list. It is found in `models/selfhost/*/`, `models/providers/*/`, `examples/*/`,
under any folder or file in `STREAMEVAL_MODELS` (`:`-separated), or with `--registry PATH` on any command.

```json
{"system_id": "my_lab", "model_id": "my_lab_v2", "name": "My Lab Live Editor v2", "provider": "My Lab",
 "system_kind": "native", "tracks": ["instruction_edit", "reference_edit"], "portrait_policy": null,
 "python": "model.py:MyModel", "kwargs": {"device": "cuda"},
 "ops": ["text_control", "reference:set", "reference:remove"],
 "push": {"max_side": 832, "max_fps": 16},
 "model_version": "my_lab_v2 @ commit 1a2b3c", "weights": {"repo": "my-lab/v2", "sha256": {"model.safetensors": "..."}},
 "steps": 4, "gpu": {"count": 1, "type": "H100 80GB"}}
```

| key | required | meaning |
|---|---|---|
| `system_id` | yes | unique among all registered entries (a copy of an example must be renamed); prefix of the per-track ids: `<system_id>_<suffix>` (`my_lab_instruction`, `my_lab_reference`); `system_ids` {track: id} overrides |
| `model_id`, `name`, `provider` | yes | identity on the board (`provider` is the organisation) |
| `system_kind` | yes | `native`: the model does the whole task itself. `composition`: a renderer-only avatar driven by a shared dialogue stack (below) |
| `tracks` | yes | any of the registered tracks ([above](#capabilities-and-operations)) |
| `portrait_policy` | yes | avatars: `benchmark_portrait` (the case's portrait) or `provider_stock` (the vendor's own avatar; Identity is measured against its opening frame and included in Overall); `null` otherwise |
| `python` / `server` / `url` / `backend` | exactly one | the shape above |
| `kwargs` | | constructor arguments of the `python` or `backend` class |
| `ops`, `unsupported_ops` | | operations the model takes; default every operation of its tracks. Events outside them are never sent, recorded `unsupported` and scored as failures |
| `output` | | requested {width, height, fps} (default 512 × 512 at 25); editing tracks receive the source size instead |
| `push` | | editing: {max_side, max_fps} of the pushed source (default 1280, 30); scaled down, never up |
| `seed` | | base seed (default 20260929); a session gets base + repeat |
| `model_version`, `weights`, `steps`, `gpu`, `notes`, `license` | | recorded locally with every run; publish model identity and the comparison conditions |
| `composition` | | composition systems only (below) |
| `model` | | `server` shape: the folder name under `STREAMEVAL_MODEL_ROOT` (default `system_id`) |

Every key not starting with `_` is frozen into the system configuration (`system_config` in `binding.json`) and
hashed: change a setting and the runs are a different configuration. Use a new `system_id` for a new setting
rather than mixing runs in one folder.

### Input adapters

Declare required input conversions per registered track before evaluation. The declaration records a conversion
implemented by the adapter; it does not apply that conversion automatically.

```json
{
  "input_adapters": {
    "avatar_performance": {
      "mode": "derived",
      "transforms": [
        {"operation": "speech_to_script", "parameters": {"source": "asset.script"}}
      ]
    }
  }
}
```

Modes are `native` (empty transforms), `normalized` (representation changes) and `derived` (semantic conversion).
Each transform has an `operation` and optional `parameters` object. Name tracks exactly as in `tracks`.
The registry freezes the declaration as `input_contract` and hashes it into `input_condition` for comparison.
Composition renderers additionally bind the shared dialogue models and voice. Record resizing, resampling,
control mapping and instruction-started generation where used. Conversion may only read released inputs;
its latency remains part of the response. The policy's 180-second readiness limit applies to all systems.

### Server launch

```json
"server": {"cmd": "python {model_dir}/server.py --port {port} --weights {weights}", "port": 18790,
           "health_timeout_s": 1800, "cwd": "."}
```

| placeholder | value |
|---|---|
| `{port}` | `server.port` (or the port given to the launcher) |
| `{model_dir}` | the folder of the `system.json` |
| `{weights}`, `{env}` | `STREAMEVAL_MODEL_ROOT/<model>/weights` and `.../env` (default root `~/.cache/streameval/models`) |
| `{python}` | the interpreter: `STREAMEVAL_SERVER_PYTHON`, else the harness's own |

A leading `python` is replaced by the same interpreter, so point `STREAMEVAL_SERVER_PYTHON` at the model's venv
(for example `$STREAMEVAL_MODEL_ROOT/<model>/env/bin/python`). `--launch` starts the command, waits for
`GET /health` to answer (up to `health_timeout_s`, default 900 s, for weight loading), runs the session(s) and stops
the server. The self-hosted models in `models/selfhost/<model>/` follow one layout: `server.py`, `system.json`, a
`setup.sh` that downloads code and weights and builds the environment under `STREAMEVAL_MODEL_ROOT/<model>/`,
and a README ([models/README.md](../models/README.md)).

### Composition avatars

A renderer that only animates speech (no language model of its own) is registered as `"system_kind":
"composition"` and gets the shared v1 dialogue stack: ASR → LLM → streaming TTS on DashScope, run by the harness
(`DASHSCOPE_API_KEY`; model names in `STREAMEVAL_DIALOGUE_*` or the entry, defaults are the v1 models). The
stack is recorded as `components` of the configuration, and the board entry carries `systemKind: composition`.

| `composition` key | values |
|---|---|
| `input` | `audio` (default): the TTS speech arrives through `push_audio`; `text`: the reply text arrives as a `TextEvent` of kind `dialogue_reply` and the renderer speaks it |
| `audio_out` | `client` (default): the harness records the TTS speech it pushed as the avatar's audio; `server`: the renderer returns the speech with `sink.audio`, in sync with the frames it drove |
| `asr`, `llm`, `tts`, `tts_voice` | override the stack's model names (a different stack is a different configuration) |

## 4. What costs points

| rule | why |
|---|---|
| output frames from the start, before any event | readiness is `ready` + first frame; editing models receive the source (first frame held) before the activity clock starts |
| ack every event; ack `unsupported` for what you cannot do | an ignored instruction is scored the same as a failed one, but `unsupported` is visible in the evidence and in `batch --dry-run` |
| keep the identity of the portrait and the unedited regions | Identity and Preservation are judged against the portrait and the live source |
| avatars: speech audio continuous at 48 kHz, silence rather than gaps | A–V Sync and Dialogue read the received audio on the presentation clock |
| `open` returns at once; do heavy work on your own threads | push and dispatch threads must never block |

## 5. Check and score

```bash
streameval run --system my_lab_instruction --case sa.v1.001.instruction_edit [--launch]
streameval validate-run runs/my_lab_instruction             # every attempt against the run contract
streameval batch --system my_lab_instruction --dry-run       # 60 sessions on instruction_edit, unsupported events
streameval evaluate --system my_lab_instruction [--launch] [--concurrency 2]
```

`evaluate` runs batch → judge → assemble → leaderboard for one system and writes
`runs/leaderboard_full.json`; the steps one by one are in [reproduce.md](reproduce.md#individual-stages).
It needs the judge keys of [evaluation.md](evaluation.md#automated-judge) (and the dialogue stack's key for
composition systems). Development-roster sessions per track at 3 repeats:

| track | cases | sessions |
|---|---|---|
| `instruction_edit`, `reference_edit`, `digital_human_audio`, `digital_human_text`, `avatar_performance` | 20 each (four per duration stratum) | 60 each |
| `general_interaction_image` (image-to-video) | 10 | 30 |
| `general_interaction` (text-to-video) | 10 | 30 |

Seven text-to-video cases also release structured game actions (`action`); a system that does not list the `action`
op records them as unsupported, which counts as failed opportunities.

Active duration follows the case, with readiness before the session epoch. Judging processes five-second
clips and event windows. Configure judge credentials before starting an evaluation.

The preparation freezes the repetition count. Use three repeats for paper-protocol comparisons; other counts
are diagnostic. Use a separate output directory to change a frozen setting.

Several systems scored into the same `--out` share one board: run `evaluate` per system, then
`streameval leaderboard --results runs/results --roster runs/preparation/full --out board.json`. These
scores describe the full roster; for results comparable with the paper use
`--benchmark benchmark/v1/full --population full` (see [evaluation.md](evaluation.md#aggregation-and-display)).
Only aggregate values should be published.

Interrupted runs resume: `batch` skips finished jobs, re-attempts sessions that never started (up to 3 times,
unless `--no-auto-retry`), and `judge` skips judged jobs. `--concurrency N` runs N sessions at once, each in its own
process; with `--launch` it starts N servers on `port` … `port + N − 1`, and with `--url` all sessions share that
server, so use it only if the server takes several sessions at once.

## Vendor systems

The commercial systems of the v1 board are sampled through adapters in `models/providers/`. They exist so
that a vendor system can be sampled again, or a new one added, under the same rules as the open models.

| folder | systems | shape | client | extra |
|---|---|---|---|---|
| `vidu` | Vidu S2 avatar (audio, text, performance), S2 editing (reference, instruction via qwen-image) | `SessionBackend` | headless Chromium + AliRTC Web SDK 7.1.9 | `providers` + Node runtime |
| `xmax` | XMax X2.0 instruction and reference editing | `EditBackend` | headless Chromium + XMax JS SDK 1.1.13 | `providers` + Node runtime |
| `runway` | Runway Characters (audio) | `StreamingModel` + `Account` | in-process LiveKit | `rtc` |
| `liveavatar` | HeyGen LiveAvatar FULL (audio, text; stock avatar) | `StreamingModel` + `Account` | in-process LiveKit | `rtc` |
| `tavus` | Tavus CVI Full (audio, text) | `StreamingModel` + `Account` | in-process Daily | `rtc` |
| `vivix` | Vivix A1 (audio) | `StreamingModel` | in-process Agora | `vivix` |

Each folder's README gives the system ids, setup, the vendor calls of a session and what is specific to that
vendor (session cap, avatar creation, the frozen persona, how operations map onto the vendor API).

```bash
pip install -e ".[providers]"          # browser backends (Vidu, XMax)
pip install -e ".[rtc]"                # RTC avatars (Runway, LiveAvatar, Tavus): livekit==1.1.19, daily-python==0.32.0
cd streameval/providers/js && npm ci && npx playwright install chromium && cd -    # browser backends only
cp .env.example .env                   # then fill in the keys of the vendors you use
```

SDK versions are pinned (npm integrity in `package-lock.json`, exact pip
versions); the XMax and Vidu backends refuse a session with a different SDK build. `STREAMEVAL_NODE`,
`STREAMEVAL_BROWSER_RUNTIME` and `STREAMEVAL_CHROMIUM` point at another Node binary, `js` folder or Chromium.

### Accounts, cost and limits

Every vendor system is an `Account` (`streameval/providers/account.py`), whether it is a session backend or a
`StreamingModel`:

| | |
|---|---|
| credentials (`credentials_env`, `required_credentials`) | read from environment variables (or the git-ignored `.env`) only, never written to an artifact. Browser pages never see an account key: the backend mints a short-lived, scoped client token per session. A run without a required variable is refused with **exit code 3** before any request, listing the missing names |
| cost estimate (`cost`, `estimate(case)`) | `(case seconds + overhead) × rate` in the vendor's unit (USD, credits). `batch --dry-run` sums it over the jobs; `--budget X` starts no attempt that would bring the folder's estimated total above X. An adapter without a rate (`None`) cannot be run with `--budget` |
| session cap (`session_cap_s`) | the vendor's longest session. Cases beyond it are still run and fail in the denominator; `--dry-run` counts them (`sessions_beyond_session_cap`) |
| system configuration (`system_config(system_id, spec)`) | the adapter adds what it fixes (client, SDK version, readiness, region, session cap) to the frozen configuration, so a run records the route it took |
| receipt | `result.json` records the vendor session facts (ids, request bodies without keys, termination confirmation, per-event dispatch records) |

Always start with the dry run, which makes no request, reads no media and writes nothing:

```bash
streameval batch --system xmax_x2_reference --dry-run
#   sessions, active seconds, unsupported events per case, estimated cost, session-cap overruns, missing credentials
streameval batch --system xmax_x2_reference --budget 2000 --concurrency 1
```

A session the vendor ends early, a moderation refusal or a disconnect after the session started is a started
failure and stays in the denominator. Only a session that never started is re-attempted (up to 3 times). Create a
file `STOP` in the run root (`--out`) to stop starting new attempts.

### Recording

* **Browser backends** run one headless Chromium page per session, which joins the vendor's WebRTC session,
  publishes the benchmark's tracks (the source video as a camera, user audio as a microphone) and samples the
  received stream with `requestVideoFrameCallback`; Python keeps the account key, the trusted scheduler and the
  control channel. They write the [run contract](protocol.md#run-contract-the-attempt-folder) themselves,
  including `webrtc_stats.jsonl`.
* **RTC avatars** join the room in-process (LiveKit or Daily) and run under the ordinary harness, so readiness,
  capture and timing are those of every self-hosted avatar. The room's microphone takes 48 kHz, so the 16 kHz
  input is upsampled linearly (continuous across chunks, 62.5 µs per input sample).

Account-specific avatar and persona identifiers are configured locally. Use the provider's official API or SDK
with your own credentials. Adapters may upload benchmark inputs and record received output as documented in their
READMEs. Keep those recordings and account details in the run directory. Judging uses the same output
preprocessing for all providers.

### Adding a vendor

A vendor reachable with an in-process client (an RTC room, a WebSocket API) is best written as a `StreamingModel`
that also inherits `Account`, like `streameval/providers/rtc.py`: the harness then does all timing and capture.
Write a `SessionBackend` only when the output exists solely inside the vendor's own media stack:

```python
from streameval.providers import SessionBackend

class MyVendor(SessionBackend):
    credentials_env = {"api_key": "MYVENDOR_API_KEY"}
    cost = {"unit": "USD", "per_active_second": 0.01, "overhead_s": 30}
    session_cap_s = None

    def run(self, system_id, spec, case, assets, out, *, repeat=1, oracle=None, smoke=False, admission=None,
            verify=True) -> dict:
        """Run one session of `case` into the new folder `out`; write the run contract; return result.json."""

    def event_dispositions(self, system_id, spec, case) -> list[dict]:
        """Optional: per event {event_id, operation, supported, reason}, decided before any output exists
        (default: the registered ops)."""
```

`streameval/providers/edit.py:EditBackend` is a ready base for browser-SDK editing vendors (one headless Chromium
page pushes the source as a camera track and samples the received stream; Python keeps the trusted scheduler).
`oracle` is passed only for adapters that need public timing budgets from it (Vidu avatar performance); it must
never reach the model. Register either with a `system.json`, test with `--dry-run`, then one `run --smoke`, then
`validate-run`.
