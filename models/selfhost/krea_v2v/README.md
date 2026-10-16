# Krea Realtime 14B V2V — self-hosted streaming instruction editing

Krea Realtime 14B (Self-Forcing-distilled Wan2.1-T2V-14B) in upstream's **webcam mode**, the live video-to-video
path of Krea's own demo server: the live source is noised and re-generated block by block under the instruction
prompt. Served through the WebSocket protocol (`docs/protocol.md`) and the shared edit loop
`models/selfhost/edit_stream.py`.

| | |
|---|---|
| System ids | `krea_realtime_14b_v2v_instruction` (`instruction_edit`) |
| Upstream code | [krea-ai/realtime-video](https://github.com/krea-ai/realtime-video) @ `acf4b7b4e1a3049a0e04782e751d12203334038d` |
| Weights | krea/krea-realtime-video @ `9a2ceea2`; Wan-AI/Wan2.1-T2V-1.3B @ `37ec5126` (text encoder, VAE); Wan-AI/Wan2.1-T2V-14B `config.json` — the install of `models/selfhost/krea_realtime` |
| Licence | code: **CC BY-NC-SA 4.0 (non-commercial)**; weights: Apache-2.0 (see Weights and licences) |

## Setup

This model has no separate install. `setup.sh` runs `models/selfhost/krea_realtime/setup.sh` for whatever is
missing in `$KREA_ROOT` and creates two links where the harness looks for them:

```bash
bash models/selfhost/krea_v2v/setup.sh env       # krea_realtime repo + env if missing; $STREAM_ARENA_MODEL_ROOT/krea_v2v/env -> $KREA_ROOT/env
bash models/selfhost/krea_v2v/setup.sh weights   # krea_realtime weights if missing; .../krea_v2v/weights -> $KREA_ROOT/weights
bash models/selfhost/krea_v2v/setup.sh check     # krea_realtime's import check (GPU)
```

`setup.sh` (or `all`) runs the three. Prerequisites and the download variables are those of
[`krea_realtime`](../krea_realtime/README.md#setup).

| variable | default | meaning |
|---|---|---|
| `KREA_ROOT` | `$STREAM_ARENA_MODEL_ROOT/krea_realtime` | the krea_realtime install: setup.sh builds and links it, server.py imports `$KREA_ROOT/repo` and defaults `--weights` to `$KREA_ROOT/weights` |
| `KREA_V2V_ROOT` | `$STREAM_ARENA_MODEL_ROOT/krea_v2v` | folder of the two links; the harness resolves `{env}` / `{weights}` to the default location, so keep the default when using `--launch` |
| `DO_COMPILE`, `DISABLE_SAGEATTENTION`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `PYTORCH_CUDA_ALLOC_CONF` | as `krea_realtime` | set by server.py unless already set |

## Weights and licences

Shared with `krea_realtime` (`$KREA_ROOT/weights`, linked as `$STREAM_ARENA_MODEL_ROOT/krea_v2v/weights`); full
hashes in `system.json` `weights.sha256`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `checkpoints/krea-realtime-video-14b.safetensors` | krea/krea-realtime-video @ `9a2ceea280e5d584c6a28d384a9c19e38c810f8c` | `792f6042…3527eec7` | Apache-2.0 (model card) |
| `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | Wan-AI/Wan2.1-T2V-1.3B @ `37ec512624d61f7aa208f7ea8140a131f93afc9a` | `38071ab5…e442fd981` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | same | `7cace0da…745b746fd81d` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-14B/config.json` | Wan-AI/Wan2.1-T2V-14B @ `main` (v1 revision not pinned) | — | Apache-2.0 |

The code (krea-ai/realtime-video, `LICENSE.md`) is licensed **CC BY-NC-SA 4.0: non-commercial use only**,
share-alike; the weights are Apache-2.0 according to the model card. Benchmark evaluation only; nothing is
redistributed.

## Limitations

- Not a local editor: at strength 0.8 (≈5 % source signal) the whole scene is re-rendered from the instruction
  text, so neither the requested edit nor the unedited content is reliably preserved. The instruction text is not
  a scene description, the same prompt policy issue as StreamDiffusionV2.
- Landscape 832×480 only; portrait sources are centre-cropped.
- Text instructions only; no reference path.
