# Vidu S2 (AliRTC browser session backend)

B-layer backend (`backend.py:ViduS2`, a `SessionBackend`). Python owns session creation over the Vidu REST API,
the WebSocket control channel, the trusted scheduler and hangup; one headless Chromium page
(`streameval/providers/js/vidu_browser.mjs`) joins the AliRTC channel Vidu returns, publishes the benchmark's
synthetic tracks and records what it receives, in the run contract of `docs/protocol.md`.

| system id | track | route |
|---|---|---|
| `vidu_s2_avatar_audio` | digital_human_audio | S2 Avatar; persona and known facts in the create body, user audio as a WebAudio microphone track |
| `vidu_s2_avatar_text` | digital_human_text | S2 Avatar; user text over the WS control channel |
| `vidu_s2_avatar_performance_text` | avatar_performance | derived text route: supplied speech goes out as `LINE: <script>`, instructions as `ACTION: <text>` |
| `vidu_s2_editing_all` | reference_edit | S2 Editing; every reference condition through `mapping.py` |
| `vidu_s2_editing_t2i` | instruction_edit | composition: instruction → qwen-image reference → S2 Editing |

## Setup

```bash
pip install -e '.[providers]'                    # includes websocket-client
cd streameval/providers/js && npm ci && npx playwright install chromium && cd -
# the pinned AliRTC Web SDK 7.1.9 (not on npm); the backend refuses any other file
mkdir -p ~/.cache/streameval/providers/vidu
curl -L -o ~/.cache/streameval/providers/vidu/aliyun-rtc-sdk-7.1.9.js \
  https://g.alicdn.com/apsara-media-box/imp-web-rtc/7.1.9/aliyun-rtc-sdk.js
sha256sum ~/.cache/streameval/providers/vidu/aliyun-rtc-sdk-7.1.9.js
# 8100338351ce622a1d25e26997247e55b09135f458b90e748995b357ce26e376
```

| variable | |
|---|---|
| `VIDU_S2_API_KEY` | account key (`vda_…`), required |
| `VIDU_S2_API_BASE` | `https://api.vidu.cn` (default) or `https://api.vidu.com`; any other origin is refused |
| `STREAMEVAL_VIDU_SDK` | path of the SDK file, if not the default above |
| `DASHSCOPE_API_KEY` | `vidu_s2_editing_t2i` only (qwen-image references) |

Keys stay in the Python process; the browser receives only the RTC join token of the session it joins, which is
never written to disk and is redacted from every log line. Chromium runs with a fake capture device fed by one
second of digital silence, so no physical microphone or camera can be opened.

```bash
streameval batch --system vidu_s2_editing_all --dry-run      # jobs, dispositions, no request
streameval batch --system vidu_s2_editing_all                # paid: needs the keys, see below
```

Before a session the backend checks the credentials, the SDK hash, the key format, the origin and the account
balance (`/ent/v2/credits`, at least `VIDU_S2_CREDIT_FLOOR` credits, default 100 000); any failure refuses the session before it is created.
Vidu bills in credits and publishes no per-second price, so the dry run reports no cost estimate and `--budget` is
refused for Vidu systems.

## Edit mapping (`mapping.py`)

S2 Editing holds one reference image and one `editing_type` at a time. The newest active target wins: a target
worn by a person (its scope text matches `worn` or `draped over … shoulders/back`) goes out as `virtual_tryon`,
any other as `subject_replacement`; with no active target (after a reset or cancelled edit) S2 gets
`style_transfer` with the current source frame. Conditions S2 cannot hold (two edits at once, a restore) are still sampled
and judged.

The composed instruction route has to know which target an instruction edits, so it understands only the six
instruction templates of the v1 roster (`mapping.INSTRUCTIONS`). **The development roster's instructions are
free-form, so its instruction events are released as `unsupported`
(`instruction_outside_v1_templates`) without a provider call**: `vidu_s2_editing_t2i` scores as failures on the
development roster by construction.

## Avatar performance

The text route reads the supplied speech's transcript from the `script` field of the speech asset in
the roster's `manifest.json` and needs the case oracle's audio budgets (its `oracles.jsonl`) to pace
the scheduler; a smoke run without an oracle falls back to the case window.

## Evaluation contract

The registry declares speech-to-script, instruction-to-reference and reference-state conversions as input
conditions. These routes follow the shared readiness, retry, evidence and scoring rules. Conversion time counts
in response latency. Provider errors remain in the run result; they do not remove conditions or select a later
successful attempt.
