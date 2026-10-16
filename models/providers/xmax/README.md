# XMax X2.0 (browser SDK session backend)

B-layer backend (`backend.py:XMax`, an `EditBackend` of `stream_arena/providers/edit.py`). One headless Chromium
page (`stream_arena/providers/js/edit_page_xmax.mjs`) pushes the case's source video as a camera track through
the official XMax JS SDK, receives the edited stream and samples it; Python keeps the account key, the trusted
scheduler, the full-state planning (`stream_arena/providers/edit_state.py`) and cleanup.

| system id | track |
|---|---|
| `xmax_x2_instruction` | instruction_edit (each instruction sent verbatim as the full prompt) |
| `xmax_x2_reference` | reference_edit (one prompt from the scope texts + the reference image, uploaded at dispatch) |

## Setup

```bash
pip install -e '.[providers]'
cd stream_arena/providers/js && npm ci && npx playwright install chromium && cd -
```

| variable | |
|---|---|
| `XMAX_API_KEY` | account key, required |
| `XMAX_REGION` | `global` (default: api.xmax.ai, `@xmaxai/sdk-global`) or `cn` (api.xmaxai.com, `@xmaxai/sdk`) |

Both SDK builds are pinned at 1.1.13 by npm integrity in `package-lock.json`; the backend refuses a session if
the installed build differs. The page never sees the account key: per session the backend mints a temporary key
bounded in time (case duration + 480 s) and credits (case duration + 120).

```bash
stream-arena batch --system xmax_x2_reference --dry-run        # jobs, cost estimate, no request
stream-arena batch --system xmax_x2_reference --budget 2000    # paid; stops before exceeding 2000 credits
```

Cost estimate: 1 credit per active second plus 30 s overhead per session. Before each session the account
balance must cover the case duration + 120 credits.

## Notes

- The source is pushed at its native size, or the largest proportional 32-aligned size below 1.28 MP (the SDK's
  upload limit), at most 30 fps; it is never upscaled.
- Reference cases apply reference A in the connect context and start the activity epoch after the first
  presented output frame; instruction cases connect with `autoStart=false` and the first instruction at t=0
  starts generation.
- SDK reconnects inside a started session are recorded and never repeated by the harness.
