# Tavus CVI Full (RTC avatar, A layer)

`model.py:TavusCviFull` is a `StreamingModel` (plus `Account`) on `streameval/providers/rtc.py`: REST
conversation + Daily call, run by the ordinary harness.

| system id | track | user input |
|---|---|---|
| `tavus_cvi_full_audio` | digital_human_audio | virtual microphone of the Daily call |
| `tavus_cvi_full_text` | digital_human_text | app message `conversation.respond` |

## Setup

```bash
pip install -e '.[rtc]'          # daily-python==0.32.0, pinned SDK version
```

1. Create the frozen persona (PAL) once on your account from `pal_v1.json` → `request`:
   ```bash
   python3 -c "import json; print(json.dumps(json.load(open('models/providers/tavus/pal_v1.json'))['request']))" |
     curl -s -X POST https://tavusapi.com/v2/personas -H "x-api-key: $TAVUS_API_KEY" \
          -H "Content-Type: application/json" -d @-
   ```
2. Set `TAVUS_PAL_ID` to the returned `persona_id`.
3. `streameval batch --system tavus_cvi_full_audio --dry-run`

| variable | |
|---|---|
| `TAVUS_API_KEY` | account key, required |
| `TAVUS_PAL_ID` | your persona created from `pal_v1.json`, required |
| `TAVUS_ALLOW_PAL_DRIFT` | `1`: run even if the live persona differs from the frozen one |

## PAL drift check

Before every session the live persona is compared with the frozen one on `pal_v1.json` → `pinned_fields`
(pipeline mode, system prompt, context, greeting, replica, layers, languages, disclosures, tools, documents,
guardrails, objectives). A difference refuses the session unless `TAVUS_ALLOW_PAL_DRIFT=1`; then it is recorded
in the receipt and the run is no longer the frozen v1 configuration. Account-specific ids and credential-named
layer fields are not part of the frozen persona (`pal_v1.json` → `omitted`).

## Notes

- Faces are trained from an HTTPS image URL, which the benchmark portraits do not have, so every session uses the
  pinned stock replica `r79996137c01` ("Jamie", phoenix-4.5; the selection rule is in `pal_v1.json` →
  `replica_selection`). `portrait_policy: provider_stock`; identity metrics do not apply.
- Each conversation gets the case's `conversational_context`; dynamic greeting is off.
- `max_call_duration` above the plan limit is silently truncated by Tavus; a session that ends before the case
  deadline is a failure in the denominator.
- Cost estimate (USD): connection-to-disconnection minutes billed in 6 s increments with a 30 s minimum, at the
  0.37 USD/min Starter overage (vendor pricing as of 2026-08); included plan minutes are not subtracted.
- `close()` leaves the call, then ends the conversation (`POST /v2/conversations/{id}/end`).
