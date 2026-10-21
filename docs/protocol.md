# Protocol and run folders

Two interfaces fix what a model sees and what the judge reads: the WebSocket protocol between the harness and a
model server, and the run contract, the attempt folder every session writes.

**MUST** and **MUST NOT** state requirements; **SHOULD** states a recommendation. Every implementation MUST follow
[the shared evaluation policy](evaluation.md#equal-treatment). API differences MUST be declared through an
[input adapter](add_a_model.md#input-adapters) before execution. Wire compatibility remains `streameval-selfhost/1`.

## WebSocket model protocol (`streameval-selfhost/1`)

A model written in any language joins the benchmark by implementing one **model server**. The harness
(`streameval.models.websocket.WebSocketModel`) is the client; it does the timing, capture and evidence keeping,
so the server only turns inputs into output frames (and speech) as fast as it can.

Reference implementation: `streameval/runtime/protocol.py` (standard library only: pack/unpack, an asyncio
server `serve(handler, host, port, health)` and a blocking client). A Python server may import it or copy the
file. `examples/my_model/server.py` is a complete template. An in-process `StreamingModel` can be served over this
protocol without writing a server: `streameval serve --model file.py:Class --port 8765`.

### Transport

- `GET /health` answers 200 with any JSON body once the model is loaded (the harness waits for it after
  `--launch`; if it is missing, an open port counts as healthy).
- `ws://<host>:<port>/session` upgrades to a WebSocket. **One connection = one session.** A server needs to serve
  one session at a time.
- JSON messages are WebSocket text frames with a `"type"` field. Media are binary frames:

```
binary message = 2-byte ASCII tag + uint64 LITTLE-ENDIAN pts_ns + payload      struct.pack("<2sQ", tag, pts_ns)
```

| tag | direction | payload | pts_ns |
|---|---|---|---|
| `SF` | client → server | JPEG source frame (editing tracks), size = `init.width/height` | source media time, `frame_index × 1e9 / fps` |
| `AI` | client → server | s16le mono 16 kHz, 10 ms (160 samples) | stream time of the first sample (0, 10 ms, 20 ms, …) |
| `VF` | server → client | JPEG output frame | server monotonic clock when generated (recorded only) |
| `AO` | server → client | s16le mono 48 kHz, any block size | server monotonic clock (recorded only) |

### Client → server

| type | fields | meaning |
|---|---|---|
| `init` | `protocol`, `track`, `case_id`, `width`, `height`, `fps`, `prompt`, `seed`, `output`, `persona?`, `image_b64?`, `image_mime?` | start of the session (first message) |
| `text` | `event_id`, `text`, `kind` | a released text event |
| `reference` | `event_id`, `op`, `slots`, `scope` or `text` | a released reference event |
| `control` | `event_id`, `action`, `params: {phase, duration_ms}` | a released game action (`action` op, opt-in); `phase` is `start` (hold until the matching `stop`, at most `duration_ms`), `stop` or `pulse` (once, for `duration_ms`) |
| `end` | — | end of the session; close the connection after it |

`init` per track:

| track | `image_b64` | `prompt` | `persona` | `width/height/fps` |
|---|---|---|---|---|
| `digital_human_audio`, `digital_human_text` | the case portrait (`portrait_policy: benchmark_portrait`; absent for `provider_stock`) | `""` | persona + known facts | registry `output` |
| `avatar_performance` | the case portrait (as above) | the case's rendering context | — | registry `output` |
| `general_interaction` | — | the opening prompt | — | registry `output` |
| `general_interaction_image` | the opening reference image | the opening prompt | — | registry `output` |
| `instruction_edit`, `reference_edit` | — | `""` | — | the pushed source: source resolution scaled down to `push.max_side` (never up), `fps = min(round(source fps), push.max_fps)` |

`output` is the registry's requested `{width, height, fps}`; `seed` is the registry seed (default 20260929). Use
it: runs record it in the frozen system configuration.

`text.kind`:

| kind | track | meaning |
|---|---|---|
| `user_prompt` | digital_human_text, avatar_performance, general_interaction(_image) | a user's chat line / a performance direction / a prompt update |
| `text_control` | instruction_edit | an editing instruction |
| `camera_control` | general_interaction(_image) | a camera direction |
| `dialogue_reply` | composition systems with `composition.input = "text"` | the reply the shared dialogue stack wrote; speak it |

`reference`:

- `slots`: `[{"slot", "image_b64", "image_mime", "scope", "asset_id"}]`; for `remove` a slot is
  `{"slot", "image_b64": null, "scope": null}` (restore the slot to the live source).
- `op`: `set` (replace the active slots), `patch` (add or replace the listed slots, keep the others),
  `remove` (restore the listed slots), `reset` (restore all).
- `scope`: the operation's protection clause (what must not change).
- On `general_interaction_image` a reference update is `op: "set"`, `slot: "subject"`, with `text` (the
  instruction that came with it) instead of `scope`.
- `reference_edit` cases start with reference A preloaded: right after `init` and **before the activity epoch** the
  client sends `{"type": "reference", "event_id": "initial", "op": "set", ...}`, which must be acked like any event.

### Server → client

| type | fields | meaning |
|---|---|---|
| `ready` | `model`, `version`, `width`, `height`, `fps`, any extra (`config`, `protocol`, …) | the session accepts inputs; recorded in the run |
| `ack` | `event_id`, `status` (`applied` / `unsupported` / `error`), `detail?` | outcome of one event; **every event needs one** |
| `error` | `message` | fatal; the session ends and the attempt fails |

### Session rules

1. **Real time.** Generate as inputs arrive. The server never sees future events or the oracle; it must not
   render offline and play back. Deliver every frame as soon as it exists: buffering delays it on the record.
2. **Readiness** = `ready` + the first `VF` frame. On avatar tracks the harness also waits until the first frame is
   ≥ 4 s old and no speech has been heard for 1.5 s (at most 25 s; the greeting of a model that greets is
   allowed to finish). Consequences:
   - an **editing server must output frames before any instruction arrives** (passing the source through is fine);
     a session backend that requires an initial instruction MUST declare `instruction_started_generation`
     and be ready to accept source frames before the epoch; this is a separate input condition;
   - an **avatar server must keep outputting frames while idle**, and should send `AO` silence when it does not
     speak (avatar audio is expected to be gapless).
3. **Audio input** (`AI`) starts after `ready` and is continuous: silence between clips, the user's (or the
   supplied speech's) audio inserted on an absolute clock.
4. **Source video** (`SF`): the first source frame is held before the epoch; from the epoch the source plays in
   real time at the push rate.
5. **Unsupported operations.** An event the server cannot express gets `ack: unsupported`; it is a failure in the
   denominator, never silently ignored. Operations the registry entry declares unsupported (`unsupported_ops`) are
   not sent at all and recorded the same way.
6. **Retries.** Only a failure before the epoch (no connection, no `ready`, no first frame within
   the shared 180-second readiness limit) is retried, as `<job_id>.retry1..3`. Any failure after the epoch counts.
7. **End.** At the case deadline the client sends `end` and closes. A server that ends the session early (closes,
   or sends `error`) fails the attempt.

### Composition systems (renderer-only avatars)

A renderer that cannot hold a conversation registers `system_kind: "composition"`. The harness runs the shared
dialogue stack (`streameval/runtime/dialogue.py`: `qwen3-asr-flash` → `qwen3.8-max-0902` →
`qwen3-tts-flash`, voice Cherry; models recorded in `system_config.components`) and sends the server what to say:

- `composition.input = "audio"`: the TTS speech is pushed as `AI` in real time;
- `composition.input = "text"`: the reply text arrives as `text` with `kind = "dialogue_reply"`.

`composition.audio_out = "server"`: the server's `AO` is the avatar's voice. `"client"`: the renderer plays no
sound and the client records the TTS speech at the moment it pushed it.

### Minimal Python server

```python
from streameval.runtime import protocol as sp

async def handler(conn: sp.ServerConnection):
    while True:
        try:
            msg = await conn.recv()            # dict (JSON) or bytes (binary media)
        except sp.ConnectionClosed:
            return
        if isinstance(msg, bytes):
            tag, pts, payload = sp.unpack(msg)
            if tag == sp.SOURCE_FRAME:
                await conn.frame(edit(payload))    # an editing model: one output frame per source frame
        elif msg["type"] == "init":
            await conn.ready("my-model", "1", msg["width"], msg["height"], msg["fps"])
        elif msg["type"] in ("text", "reference", "control"):
            apply(msg)
            await conn.ack(msg["event_id"], "applied")
        elif msg["type"] == "end":
            return

sp.serve(handler, "127.0.0.1", 8765, health=lambda: {"ok": True})
```

See `examples/my_model/server.py` for the complete version (threads, frame pacing, unsupported operations).

## Run contract: the attempt folder

The judge reads nothing but attempt folders. Any way of running a model (the harness, or a `SessionBackend` with
its own media stack) is accepted if its folders pass `streameval validate-run`
(`streameval/runtime/contract.py`). Errors make an attempt unreadable; warnings are reported and do not block
judging.

```bash
streameval validate-run runs/my_model_instruction                      # every attempt below the folder
streameval validate-run runs/.../sa.v1.001.instruction_edit.r01 --verbose
```

Words used below: a **case** is one session timeline of the benchmark; a **job** is one repeat of a case for one
system (`<case_id>.rNN`); an **attempt** is one try at running a job, in its own folder. Usually a job has one
attempt. The **activity epoch** is the instant the case clock starts, after the model is ready; an attempt
*started* if it reached it.

### Batch folder

```
<runs>/<system_id>/<population>/
    admission.json        the track's cases this system runs, and the events it cannot handle (frozen first)
    batch_plan.json       seeded job order, the frozen system configuration and its sha256, the rerun rule
    batch_log.jsonl       one line per finished attempt (exit code)
    <job_id>/             one attempt; job_id = <case_id>.rNN (NN = repeat, 3 repeats by default)
    <job_id>.retryK/      re-attempts (K = 1..3) of a job whose earlier attempts never started
    <attempt>.log         stdout and stderr of the process that ran the attempt
    server_N.log          output of the N-th model server started by --launch
<runs>/<system_id>/single/<case_id>.rNN.<time>/   one session of `streameval run` (never judged as a sample)
<runs>/preparation/<population>/   frozen metric plans and sampling jobs (made by batch)
<runs>/STOP                        if present, no new attempt starts
```

`<population>` is the roster name given with `--population` (`full` by default).

**Rerun rule.** Only a result proving that the activity epoch never started permits a retry, up to three times.
The first started attempt MUST be retained. A folder without a final result stays unknown; inactivity alone MUST
NOT classify it as a non-start.

**Sample selection.** Smoke runs and standalone `run` sessions are excluded from batch samples. For a batch job,
the first started attempt is selected regardless of provider or client diagnostics. Earlier incomplete evidence
blocks selection of a later attempt. If all attempts are documented non-starts, the job remains unknown.

All paths below describe local artifacts. Raw evidence and machine identifiers MUST NOT be included in public
aggregate exports.

### Attempt folder

```
<attempt>/
    runner.json           host and process that ran it (missing: warning)
    binding.json          what was run (written before the session)
    result.json           what happened
    browser/              the captured output
        frame_samples.jsonl  frames/s_NNNNNN.jpg   sampled output frames
        reference_frame.jpg                        last frame before the activity epoch
        remote_video_frames.jsonl                  every received frame (host time, server pts, size)
        remote_audio.s16le  remote_audio_blocks.jsonl   received audio, avatar tracks
        vad_online.jsonl                           online voice activity of the received audio
        source_frames.jsonl                        pushed source frames, editing tracks
        webrtc_stats.jsonl                         browser / RTC backends only
```

#### `binding.json`

| key | required | meaning |
|---|---|---|
| `job_id` | yes | `<case_id>.rNN` |
| `case_id`, `repeat`, `system_id` | yes | |
| `system_config` | yes | the frozen system configuration; must hold `group`, `input_profile`, `portrait_policy`, `sampling` (`frame_sample_ms`: 500 digital human, 250 otherwise) |
| `session_sha256` | yes | canonical-JSON sha256 of the case; must equal the benchmark's (the case was not modified) |
| `admission`, `assets`, `audio_durations_ns`, `smoke_test_not_a_formal_sample` | harness | admission row, sha256 of every asset read, measured audio durations, smoke flag |

The case's `(group, input_profile)` must be the system's track.

#### `result.json`

| key | required | meaning |
|---|---|---|
| `job_id`, `case_id` | yes | same as the binding |
| `status` | yes | `completed` or `failed` |
| `scheduler` | yes | the trusted scheduler's snapshot: `started_at_ns` (the activity epoch, `null` = never started), `deadline_ns`, `finished_at_ns`, `opportunities` (one row per case event: `event_id`, `status` `released` / `not_released` / `trigger_not_fired` / …, release instants) |
| `dispatch` | yes | per released event: `status` `sent` / `unsupported` / `failed`, the ack, timing |
| `unsupported_events` | yes | `[{event_id, reason}]` |
| `clock_offset_ns` | yes | page clock − scheduler clock (0 when both are the host monotonic clock) |
| `browser_start` | started | capture start; `reference_frame` `{path, sha256}` |
| `browser_finish` | | capture totals (frames, audio) |
| `readiness`, `server_ready` | harness | how readiness was reached; the model's `ready` message |
| `failure`, `cleanup_failure` | | `{type, message}` (scrubbed of credentials) |
| `termination_confirmed`, `activity_execution_complete` | harness | the session is known to have ended; every event released, dispatched and acked |
| `provider` | harness | `server_ready`, `initial_reference_ack`, `end_sent`, `missing_acks`, `receipt` (`StreamingModel.receipt()`) |
| `response_signals`, `speech_events`, `composition_turns`, `notes` | harness | avatar turn detection, composition turns, timeline notes |

Checks: every released event has a dispatch record and every dispatch record names a released event of the
case; the opportunities are exactly the case's events.

#### Captured output (started attempts)

`browser/frame_samples.jsonl`, one row per sampled frame: the **first received frame of every activity bin** of
`frame_sample_ms` (content independent).

| field | meaning |
|---|---|
| `case_ms` | presentation time on the case clock (ms since the epoch); must lie within the case |
| `sample_bin` | bin index, strictly increasing; `sample_bin × frame_sample_ms ≤ case_ms < (sample_bin+1) × frame_sample_ms` |
| `path`, `sha256` | `frames/s_NNNNNN.jpg` and its hash (checked) |
| `presentation_ms`, `presentation_basis` | page-clock time and how it was taken (`receive_callback` for the harness, `requestVideoFrameCallback` presentation time in a browser) |
| `width`, `height`, `server_pts_ns` | frame size, the model's own timestamp |

`browser/remote_audio.s16le` + `remote_audio_blocks.jsonl` (avatar tracks): the received audio as 48 kHz mono
int16 on a gapless sample clock; each block row has `samples` and `page_ms` (non-decreasing). The file must
hold exactly the listed samples. An avatar attempt without audio gets a warning.

`browser/source_frames.jsonl` (editing tracks): one row per pushed source frame with `presentation_ms` (page
clock) and `media_time` (source time). The judge pairs each output frame with the source frame shown at the
measured processing lag. Rows with neither field are annotations.

### Exit codes

| code | commands | meaning |
|---|---|---|
| 0 | all | done. `batch` returns 0 when it ran to the end, whatever the sessions did: a failed session is a result, not an error |
| 1 | `run` | the session did not complete (`status: failed`) |
| 1 | `batch`, `evaluate` | the batch halted before every job was attempted (model server down, `--budget` reached) |
| 1 | `validate`, `validate-run` | missing or altered media; an invalid attempt, or no attempt found |
| 1 | `judge` | a job or judge request failed, or replay could not find a stored request |
| 1 | `judge`, `assemble` | a missing judge key or base URL, a missing preparation |
| 2 | all | command-line usage error |
| 3 | `run`, `batch`, `evaluate` | a vendor credential is missing; nothing was sent ([add_a_model.md](add_a_model.md#accounts-cost-and-limits)) |
| 4 | `assemble`, `evaluate` | A–V Sync could not be re-decided for a system; its A–V Sync values must not be published ([evaluation.md](evaluation.md#av-sync)) |

`evaluate` stops at the first step that returns a non-zero code and returns it.

### Clocks

The harness uses one clock for everything: `time.monotonic_ns()` of the host that runs the model client, so
`clock_offset_ns = 0`. Receive time stands in for presentation time (there is no display). A backend whose
capture runs on another clock (a browser page) records `clock_offset_ns` and the pairs it was measured from.

For callback capture, audio timestamps MUST record availability at the callback. A block's duration MUST NOT
be subtracted to make its first sample appear earlier. Browser presentation timestamps and clock conversion
MUST state their measurement basis. Online conversion, transport and buffering contribute to observed latency.
