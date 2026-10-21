# Runway Characters (RTC avatar, A layer)

`model.py:RunwayCharacters` is a `StreamingModel` (plus `Account`) on `streameval/providers/rtc.py`, so it runs
under the ordinary harness: REST session + LiveKit room, avatar audio/video captured in-process, user audio
published as the room's microphone track.

| system id | track |
|---|---|
| `runway_characters_audio` | digital_human_audio (Runway has no runtime text input) |

## Setup

```bash
pip install -e '.[rtc]'          # livekit==1.1.19, pinned SDK version
export RUNWAYML_API_SECRET=...
streameval batch --system runway_characters_audio --dry-run
```

| variable | |
|---|---|
| `RUNWAYML_API_SECRET` | account key, required |
| `STREAMEVAL_RUNWAY_AVATARS` | avatar cache (default `~/.cache/streameval/providers/runway_avatars`) |

## Avatars

Runway animates an account avatar, so the backend creates one per exact benchmark portrait
(`POST /v1/avatars` with the portrait inline, voice preset `clara`, no image processing). The avatar id is cached
as `<portrait sha256>.json` in the avatar cache and reused by every later session with that portrait. Creation is
never retried: a cached avatar that is not `READY` halts the run until you resolve it (check it on your account,
then fix or delete the cache file). Note that this uploads the benchmark portraits to your Runway account.

## Limits and cost

- `maxDuration` is capped at 300 s by the vendor; a case that cannot finish inside it is kept and fails in the
  denominator.
- Cost estimate (USD): 2 credits at session start + 2 credits per started 6 s, 1 credit = 0.01 USD (vendor pricing
  as of 2026-09), over case duration + 45 s, capped at 300 s.
- `close()` deletes the session, leaves the room and reads the final status (`COMPLETED | CANCELLED | FAILED`
  confirms termination).
