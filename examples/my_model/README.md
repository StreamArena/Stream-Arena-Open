# `examples/my_model`: a template to plug a model in

A toy live instruction editor (track `instruction_edit`) in the two shapes the harness accepts. It turns the
picture gray when an instruction asks for black and white and passes the source through otherwise, so it scores
like a model that ignores almost every instruction. Its job is to show the skeleton.

| file | shape | registered as |
|---|---|---|
| `model.py` | in-process `StreamingModel` ([docs/add_a_model.md](../../docs/add_a_model.md)) | `my_model_instruction` (`"python": "model.py:MyModel"`) |
| `server.py` | WebSocket model server ([docs/protocol.md](../../docs/protocol.md)) | `my_model_ws_instruction` (`"server": {...}`, started with `--launch`) |
| `system.json` | the registry entries of both | |

```bash
streameval list | grep my_model
streameval run --system my_model_instruction --case sa.v1.001.instruction_edit          # one session, ~20 s
# {"attempt": "runs/my_model_instruction/single/sa.v1.001.instruction_edit.r01.<time>", "status": "completed",
#  "started": true, "complete": true, "frames_received": ..., "unsupported": 0}
streameval run --system my_model_ws_instruction --case sa.v1.001.instruction_edit --launch
streameval validate-run runs/my_model_instruction
streameval evaluate --system my_model_instruction        # 5 cases x 3 repeats, then judge (needs a judge key)
```

## Making it yours

1. Copy the folder (anywhere; point `STREAMEVAL_MODELS` or `--registry` at it if it is outside `examples/`).
2. In `system.json` give both entries a new `system_id` (two entries with the same id are refused), then set
   `model_id`, `name`, `provider`, the `tracks` the model runs and, if it cannot
   carry out every operation of a track, `ops` or `unsupported_ops`. Keep one of `python` / `server` / `url`.
3. Replace `MyModel.load` and `MyModel.edit` (or `edit` in `server.py`). For other tracks the inputs differ: see
   the per-track table in [docs/add_a_model.md](../../docs/add_a_model.md#capabilities-and-operations) and
   `streameval/models/mock.py`, which covers every track.
4. Record what identifies the run: `model_version`, `weights`, `steps`, `gpu` in `system.json`, and anything else
   in `describe()` and the `ready(...)` metadata. All of it is frozen into every run.

Rules that cost points when broken: output frames from the start (before the first instruction), never buffer
output, ack every event, and ack `unsupported` rather than ignoring an instruction you cannot follow.
More in [docs/add_a_model.md](../../docs/add_a_model.md).
