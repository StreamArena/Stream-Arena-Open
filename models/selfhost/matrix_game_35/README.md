# Matrix-Game 3.5 (distilled, first-person) — self-hosted image-to-video interaction

Matrix-Game 3.5 is a causal, camera-conditioned world model on Wan2.2-TI2V-5B with a depth-based Patch Memory.
`server.py` runs the upstream distilled rollout chunk by chunk behind the WebSocket protocol (`docs/protocol.md`),
anchored on the opening image, with a static camera and text prompts as the only live input.

| | |
|---|---|
| System ids | `matrix_game_35_i2v` (`general_interaction_image`) |
| Upstream code | [Riemann-Dynamics/Matrix-Game-3.5](https://github.com/Riemann-Dynamics/Matrix-Game-3.5) @ `fbf7def0693ae14f745ba35bf2a26215d4ef991d` |
| Weights | `RiemannDynamics/Matrix-Game-3.5-Distilled` @ `0b38ca0b` (`first-person.safetensors`); `Wan-AI/Wan2.2-TI2V-5B` @ `921dbaf3`; `depth-anything/DA3NESTED-GIANT-LARGE-1.1` @ `b2359bdf` |
| Licence | code: Apache-2.0; weights: Apache-2.0 and CC BY-NC 4.0 (DA3, non-commercial) (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model.

```
bash models/selfhost/matrix_game_35/setup.sh download   # pinned code + weights, sha256 check of first-person.safetensors
bash models/selfhost/matrix_game_35/setup.sh env        # venv over the base torch + pinned dependencies, import check
# or: bash models/selfhost/matrix_game_35/setup.sh all
```

Files go to `$STREAMEVAL_MODEL_ROOT/matrix_game_35/` (`repo/`, `weights/`, `env/`). Other models' setup scripts
(for example `longlive`) link `weights/Wan2.2-TI2V-5B` from here when it exists.

| variable | default | meaning |
|---|---|---|
| `MATRIX_GAME_35_ROOT` | `$STREAMEVAL_MODEL_ROOT/matrix_game_35` | setup.sh: this model's folder. The server does not read it: with a non-default value, pass `--weights` and set `MG_REPO` |
| `MG_ENV` | `$ROOT/env` | setup.sh: where the venv is built. The registered command uses `{env}` = `$STREAMEVAL_MODEL_ROOT/matrix_game_35/env` |
| `MG_REPO` | `$STREAMEVAL_MODEL_ROOT/matrix_game_35/repo` | server.py: the upstream code |
| `DIFFSYNTH_SKIP_DOWNLOAD` | `true` (set by server.py if unset) | upstream: do not download models at import |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` (set by server.py if unset) | PyTorch allocator setting |

## Weights and licences

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `Matrix-Game-3.5-Distilled/first-person.safetensors` | RiemannDynamics/Matrix-Game-3.5-Distilled @ `0b38ca0b0dda2bb994c570e183ad36d1acd53be2` | `de476e7fc0bdd756aafb101a2b80040f65b3ad62dafea109e299aafa599b8094` (checked by setup.sh) | Apache-2.0 |
| `Wan2.2-TI2V-5B/` (base DiT config, umT5 text encoder, Wan2.2 VAE) | Wan-AI/Wan2.2-TI2V-5B @ `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` | — | Apache-2.0 |
| `DA3NESTED-GIANT-LARGE-1.1/` (Patch Memory depth) | depth-anything/DA3NESTED-GIANT-LARGE-1.1 @ `b2359bdf726fb44ef62acca04d629dcf158053e7` | — | **CC BY-NC 4.0: non-commercial use only** |

Code: Apache-2.0. Because the Patch Memory depth model is CC BY-NC 4.0, the system as run here may be used for
non-commercial purposes only.

## Limitations

- Image-anchored only: no text-only start, so no `general_interaction` (t2v) entry.
- Static camera: free-text `camera_control` is not mapped to the model's 6-DoF pose input.
- The rollout stops producing frames after 16 blocks (≈ 80 s) by default; longer conditions see no new frames
  after that point.
