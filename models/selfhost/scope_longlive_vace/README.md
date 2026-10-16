# Daydream Scope (LongLive 1.3B + VACE) — self-hosted reference editing

Daydream Scope's `longlive` pipeline (LongLive-1.3B causal generator with its LoRA) with the VACE 1.3B module, which
takes live video and a reference image that can be set or changed mid-stream. Same server structure as
`selfhost/scope_vace` (StreamDiffusionV2 generator); here the generator is LongLive 1.3B. The protocol is spoken
through the shared streaming loop `models/selfhost/edit_stream.py` (single-slot reference support).

| | |
|---|---|
| System ids | `scope_longlive_vace_reference` (`reference_edit`) |
| Upstream code | [daydreamlive/scope](https://github.com/daydreamlive/scope) @ `2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0` (release v0.2.5; the checkout of `selfhost/scope_vace`) |
| Weights | daydreamlive/LongLive-1.3B @ `7735d25`; daydreamlive/WanVideo_comfy @ `441f3d5`, daydreamlive/Wan2.1-T2V-1.3B @ `d19e446` (shared with `scope_vace`) |
| Licence | code: CC BY-NC-SA 4.0; weights: LongLive-1.3B CC BY-NC-SA 4.0, VACE / Wan2.1 Apache-2.0 — non-commercial (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation.

```bash
bash models/selfhost/scope_longlive_vace/setup.sh env       # links scope_vace's env (builds it if missing)
bash models/selfhost/scope_longlive_vace/setup.sh weights   # LongLive-1.3B (8.5 GB) + links to scope_vace's weights
bash models/selfhost/scope_longlive_vace/setup.sh check     # imports the pipeline on the GPU
```

Files go to `$STREAM_ARENA_MODEL_ROOT/scope_longlive_vace` (`env` and `weights/Wan2.1-T2V-1.3B`,
`weights/WanVideo_comfy` are links into `scope_vace`).

| variable | default | meaning |
|---|---|---|
| `SCOPE_LONGLIVE_ROOT` | `$STREAM_ARENA_MODEL_ROOT/scope_longlive_vace` | this model's folder (setup.sh and server.py) |
| `SCOPE_ROOT` | `$STREAM_ARENA_MODEL_ROOT/scope_vace` | folder with the Scope checkout and env (setup.sh and server.py) |
| `SCOPE_COMMIT`, `STREAM_ARENA_PROXY`, `PIP_INDEX_URL` | see `selfhost/scope_vace` | passed on to `scope_vace/setup.sh` |

## Weights and licences

`$STREAM_ARENA_MODEL_ROOT/scope_longlive_vace/weights`, Scope's models-directory layout, downloaded from Hugging Face:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `LongLive-1.3B/models/longlive_base.pt` (5.7 GB) | daydreamlive/LongLive-1.3B @ `7735d25` | `10a2aa8f…5a208ee7` | CC BY-NC-SA 4.0 |
| `LongLive-1.3B/models/lora.pt` (2.8 GB) | same | `c4e43b87…a5ebc07a` | CC BY-NC-SA 4.0 |
| `WanVideo_comfy/Wan2_1-VACE_module_1_3B_bf16.safetensors` | daydreamlive/WanVideo_comfy @ `441f3d5` (link to `scope_vace/weights`) | `cbcab35a…56fb0574` | Apache-2.0 |
| `WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors` | same | `3fe51735…60c9ffa` | Apache-2.0 |
| `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` (+ `google/umt5-xxl` tokenizer) | daydreamlive/Wan2.1-T2V-1.3B @ `d19e446` (link) | `38071ab5…e442fd981` | Apache-2.0 |

Full hashes: `system.json` `weights.sha256` and `ready.config.weights_sha256`. Full revisions: `setup.sh`.

Code (Scope) and the LongLive-1.3B weights are licensed CC BY-NC-SA 4.0 — non-commercial use only, and adaptations
must be shared under the same licence. VACE and Wan2.1 weights are Apache-2.0. Nothing is redistributed by this
repository.

## Limitations

- VACE R2V conditions the whole frame and the reference is sent once per change (Scope's semantics), so the
  reference shows up weakly and globally rather than replacing only the target object (the same behaviour as
  `selfhost/scope_vace`).
- One reference slot only; multi-slot states and an empty state are `unsupported`.
- Output is fixed at 832x480 / 480x832 and 16 fps regardless of the source.
