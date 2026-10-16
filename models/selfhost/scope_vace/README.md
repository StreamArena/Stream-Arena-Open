# Daydream Scope (StreamDiffusionV2 + VACE) — self-hosted reference editing

Daydream Scope is a real-time pipeline runtime whose `streamdiffusionv2` pipeline takes live video and a reference
image that can be set or changed mid-stream through VACE (Wan2.1-VACE-1.3B module, R2V hints). The server loads and
drives that pipeline the way Scope's own app does and speaks the benchmark protocol through the shared streaming
loop `models/selfhost/edit_stream.py` (single-slot reference support).

| | |
|---|---|
| System ids | `scope_sdv2_vace_reference` (`reference_edit`) |
| Upstream code | [daydreamlive/scope](https://github.com/daydreamlive/scope) @ `2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0` (release v0.2.5) |
| Weights | daydreamlive/StreamDiffusionV2 @ `76bb3e6`, daydreamlive/WanVideo_comfy @ `441f3d5`, daydreamlive/Wan2.1-T2V-1.3B @ `d19e446` |
| Licence | code: CC BY-NC-SA 4.0 (non-commercial); weights: Apache-2.0 (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation. The setup script reuses the base Python packages through `--system-site-packages`; it does not replace PyTorch.

```bash
bash models/selfhost/scope_vace/setup.sh src       # daydreamlive/scope source tarball at the pinned commit (4 MB)
bash models/selfhost/scope_vace/setup.sh env       # venv: Scope's uv.lock versions of diffusers 0.36.0 / transformers 4.57.5 / peft 0.18.1, ...
bash models/selfhost/scope_vace/setup.sh weights   # Hugging Face, pinned revisions
bash models/selfhost/scope_vace/setup.sh check     # imports the pipeline on the GPU
```

Files go to `$STREAM_ARENA_MODEL_ROOT/scope_vace` (`repo/`, `env/`, `weights/` in Scope's models-directory
layout). Scope's server, WebRTC and plugin dependencies are not installed: `server.py` imports
`scope.core.pipelines` from `repo/src`. The checkout and environment are reused by `selfhost/scope_longlive_vace`.

| variable | default | meaning |
|---|---|---|
| `SCOPE_ROOT` | `$STREAM_ARENA_MODEL_ROOT/scope_vace` | this model's folder (setup.sh and server.py) |
| `SCOPE_COMMIT` | `2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0` | daydreamlive/scope commit |

## Weights and licences

`$STREAM_ARENA_MODEL_ROOT/scope_vace/weights`, Scope's models-directory layout, downloaded from Hugging Face:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `StreamDiffusionV2/wan_causal_dmd_v2v/model.pt` | daydreamlive/StreamDiffusionV2 @ `76bb3e6` | `7fab9eca…43a8930` (same as `selfhost/streamdiffusionv2`) | Apache-2.0 |
| `WanVideo_comfy/Wan2_1-VACE_module_1_3B_bf16.safetensors` | daydreamlive/WanVideo_comfy @ `441f3d5` | `cbcab35a…56fb0574` | Apache-2.0 |
| `WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors` | same | `3fe51735…60c9ffa` | Apache-2.0 |
| `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` (+ `google/umt5-xxl` tokenizer) | daydreamlive/Wan2.1-T2V-1.3B @ `d19e446` | `38071ab5…e442fd981` | Apache-2.0 |

Full hashes: `system.json` `weights.sha256` and `ready.config.weights_sha256`. Full revisions: `setup.sh`.

Code: Scope is licensed CC BY-NC-SA 4.0 — non-commercial use only, and adaptations must be shared under the same
licence. The weights are Apache-2.0. Nothing is redistributed by this repository.

## Limitations

- The reference is applied globally: VACE R2V conditions the whole frame, so the reference image's look and
  composition replace the scene instead of only the target object (with `once` it shows up weakly / transiently; with
  `every_chunk` it floods the frame). Pairing, lag (480–600 ms), aspect and resolution are correct and there are no
  black or garbage frames; this is the model's behaviour, not a pipeline fault. Inspected v1 clips show the reference
  garment or texture spread over the background and people re-rendered with a different identity.
- One reference slot only; multi-slot states and an empty state are `unsupported`.
- Output is fixed at 832x480 / 480x832 and 16 fps regardless of the source.
