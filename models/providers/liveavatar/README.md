# HeyGen LiveAvatar FULL (RTC avatar, A layer)

`model.py:LiveAvatarFull` is a `StreamingModel` (plus `Account`) on `streameval/providers/rtc.py`: REST session
+ LiveKit room, run by the ordinary harness.

| system id | track | user input |
|---|---|---|
| `liveavatar_full_audio` | digital_human_audio | microphone track |
| `liveavatar_full_text` | digital_human_text | `avatar.speak_response` on the agent-control data topic |

## Setup

```bash
pip install -e '.[rtc]'
export LIVEAVATAR_API_KEY=...
streameval batch --system liveavatar_full_audio --dry-run
```

Per session: a context (persona + known facts, empty opening text so FULL does not greet) →
`POST /v1/sessions/token` (mode FULL) → `/v1/sessions/start` → LiveKit room. `close()` stops the session, then
leaves the room.

## Notes

- The public API has no single-image avatar creation, so every session uses the pinned public stock avatar
  "Wayne" and its English voice (`portrait_policy: provider_stock`; identity metrics do not apply).
- `session_cap_s` is the account plan's maximum session length, 1200 s on the Essential plan (`PLAN_MAX_S` in
  `model.py`; confirm it for your plan). The published v1 configuration recorded no session cap; a session the
  vendor ends before the case deadline is a failure in the denominator either way.
- Cost estimate (credits): 2 credits per started session minute in FULL mode (vendor docs as of 2026-09) over the
  case duration + setup margin; pay-as-you-go 1 credit = 0.10 USD.
