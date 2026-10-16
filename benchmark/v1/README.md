# Stream Arena benchmark (v1)

The complete benchmark: **300 conditions** (60 per track, 12 per duration stratum, from 240 source families) and
**60 text-dialogue derivatives**, with their timelines, oracles and media. Results are reported on fixed rosters:

| Roster | Folder | Conditions | Reported as |
|---|---|---:|---|
| Core | [`core/`](core/) | 149 | Main results, five tracks |
| Extension | [`extension/`](extension/) | 50 | Additional conditions (10 per track) |
| Text dialogue | [`text/`](text/) | 30 | Text-input dialogue |
| Text dialogue, extra | [`text_extension/`](text_extension/) | 10 | Text-input dialogue, additional conditions |
| Development | [`development/`](development/) | 100 + 20 | Not reported; command-line default for integration |
| Excluded | [`excluded/`](excluded/) | 1 | Not scored; see [`errata.json`](errata.json) |

Conditions per track in the core roster: dialogue 30, performance 29 (see below), world generation 30,
instruction editing 30, reference editing 30. Durations are balanced over the 15, 30, 60, 120 and 300 second
strata. The 300 conditions carry 1,347 planned online events, 562 of them in the core and extension rosters.

Each roster folder holds:

| File | Contents |
|---|---|
| `cases.jsonl` | Released-input timelines, initialization, duration and source-family identifiers |
| `oracles.jsonl` | Expected outcomes, timing windows and protected content; evaluator only |
| `manifest.json` | Asset identifiers, paths under the data root, sha256, kind, role; authored speech scripts |

[`errata.json`](errata.json) records the two post-release decisions that the published results already carry:
seven avatar-performance checkpoints whose `expected` rows were shifted by one event are corrected in
`oracles.jsonl`, and `sa.v1.151.avatar_performance` is excluded from the core performance roster for every system
after a provider content-moderation refusal made it unobservable. Its 30-second stratum keeps five conditions and
its original weight, so the effective core roster is 149 conditions.

The `split` and `tier` fields are historical labels kept so that a condition stays byte-identical to the frozen
release it was scored from: `private_test` / `core`|`extension` on the reported rosters, `public_dev` / `development`
on the development roster. The development conditions use the same schema and media layout; their readiness is
`design_only` or `automated_gate_passed`, and seven text-to-video conditions also release actions.

[`ATTRIBUTION.md`](ATTRIBUTION.md) credits the 55 third-party source videos of the editing tasks and their licences.
The other 683 assets were generated for this benchmark.

## Running it

Download the media (`data/v1/`) from [Hugging Face or ModelScope](../../README.md#data), then:

```bash
export STREAM_ARENA_DATA=<path>/data
stream-arena validate      --benchmark benchmark/v1/core
stream-arena evaluate --system ID --out runs --benchmark benchmark/v1/core --population core
```

Oracles and future inputs are never passed to a participating model. Scores from a partial run are not comparable
with the published results: the published numbers use three repeats, the rosters above and the frozen judge
configuration.
