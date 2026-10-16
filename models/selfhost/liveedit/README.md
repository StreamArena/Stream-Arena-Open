# LiveEdit (1.3B) — self-hosted streaming instruction editing

LiveEdit is a causal video-to-video instruction editor on Wan2.1-T2V-1.3B (4-step DMD streaming student, the
source latents concatenated to the noisy latents through a 32-channel patch embedding). Upstream ships an offline
script with a block-causal loop; `server.py` runs that loop on the live source, behind the WebSocket protocol
(`docs/protocol.md`) and the shared edit loop `models/selfhost/edit_stream.py`.

| | |
|---|---|
| System ids | `liveedit_1p3b_instruction` (`instruction_edit`) |
| Upstream code | [cp-cp/LiveEdit](https://github.com/cp-cp/LiveEdit) @ `77a9fcc6ee2e2d84d232877e4494f9e8c563514a` (2026-08-10); ECCV 2026 |
| Weights | cp-cp/LiveEdit @ `f0269e2a` (`ar-forcing_002000.pt`); Wan-AI/Wan2.1-T2V-1.3B @ `37ec5126` |
| Licence | code: Apache-2.0; weights: Apache-2.0 (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation. The setup script reuses the base Python packages through `--system-site-packages`; it does not replace PyTorch.

```bash
bash models/selfhost/liveedit/setup.sh src       # cp-cp/LiveEdit tarball at the pinned commit
bash models/selfhost/liveedit/setup.sh env       # longlive/setup.sh env, then env -> $LONGLIVE_ROOT/env
bash models/selfhost/liveedit/setup.sh weights   # checkpoint + Wan2.1-T2V-1.3B, linked into repo/
bash models/selfhost/liveedit/setup.sh check     # imports the upstream pipeline (GPU)
```

`setup.sh` (or `all`) runs the four in order. `weights/wan_models/Wan2.1-T2V-1.3B` is a link to
`$STREAM_ARENA_MODEL_ROOT/streamdiffusionv2/weights/wan_models/Wan2.1-T2V-1.3B` when that model is installed,
otherwise it is downloaded at the same revision.

| variable | default | meaning |
|---|---|---|
| `LIVEEDIT_ROOT` | `$STREAM_ARENA_MODEL_ROOT/liveedit` | this model's folder (setup.sh and server.py) |
| `LONGLIVE_ROOT` | `$STREAM_ARENA_MODEL_ROOT/longlive` | folder of the shared environment `env/` |
| `LIVEEDIT_COMMIT` | `77a9fcc6ee2e2d84d232877e4494f9e8c563514a` | cp-cp/LiveEdit commit |
| `LIVEEDIT_HF_REV` | `f0269e2a8a5e1cdd7f7831c1bdcd95291db4f6e6` | cp-cp/LiveEdit Hugging Face revision |
| `WAN13_REV` | `37ec512624d61f7aa208f7ea8140a131f93afc9a` | Wan-AI/Wan2.1-T2V-1.3B revision |
| `HF_ENDPOINT`, `HF_TOKEN` | `https://huggingface.co`, unset | Hugging Face endpoint and token |

## Weights and licences

`$LIVEEDIT_ROOT/weights`, linked into `repo/{wan_models,checkpoints}`. Full hashes are in `system.json`
`weights.sha256` and are recorded per session.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `checkpoints/liveedit/ar-forcing_002000.pt` (5.7 GB) | cp-cp/LiveEdit @ `f0269e2a8a5e1cdd7f7831c1bdcd95291db4f6e6` | `43676999…53f95` | Apache-2.0 (GitHub repo licence; the Hugging Face model card states none) |
| `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI/Wan2.1-T2V-1.3B @ `37ec512624d61f7aa208f7ea8140a131f93afc9a` | `7cace0da…fd81d` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | same | `38071ab5…fd981` | Apache-2.0 |

Weight revisions and file hashes are listed in `system.json` and `setup.sh`. Record resolved revisions and downloaded hashes for each installation.

## Limitations

- One instruction per causal session: each new instruction restarts the session and earlier edits are not kept.
- Edits leak: the edit reaches the target but often also restyles content the instruction should preserve, and
  long sessions lose detail (the checkpoint was trained on 21-latent clips).
- Step list without the training-time warp, as upstream inference.
- Text instructions only; no reference path.
