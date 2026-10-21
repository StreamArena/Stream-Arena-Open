# Causal Forcing++ (1.3B, frame-wise 2-step) — self-hosted text-to-video and image + text to video interaction

Causal Forcing++ (Tsinghua University, thu-ml) is a frame-wise autoregressive, 2-step DMD-distilled
Wan2.1-T2V-1.3B with native image start. The server runs the released checkpoint one latent frame at a time over the
WebSocket protocol ([docs/protocol.md](../../../docs/protocol.md)), with the shared live loop of
`models/selfhost/t2v_stream.py`.

| | |
|---|---|
| System ids | `causal_forcing_1p3b_t2v` (`general_interaction`), `causal_forcing_1p3b_i2v` (`general_interaction_image`) |
| Upstream code | [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) @ `8202903297918bcd12810fc4230a4db7fe065e64` |
| Weights | `zhuhz22/Causal-Forcing` @ `2f8eb8bb` (`causal-forcing++/framewise-2step.pt`); `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec5126` |
| Licence | code: Apache-2.0; weights: Apache-2.0 (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation.

```bash
bash models/selfhost/causal_forcing/setup.sh src       # upstream code at the pinned commit (network)
bash models/selfhost/causal_forcing/setup.sh env       # shared venv of longlive, env -> $LONGLIVE_ROOT/env
bash models/selfhost/causal_forcing/setup.sh weights   # checkpoint + Wan2.1-T2V-1.3B, about 23 GB (network)
bash models/selfhost/causal_forcing/setup.sh check     # imports the upstream pipeline (GPU)
```

| variable | default | meaning |
|---|---|---|
| `CAUSAL_FORCING_ROOT` | `$STREAMEVAL_MODEL_ROOT/causal_forcing` | this model's root (read by `setup.sh` and `server.py`) |
| `CAUSAL_FORCING_COMMIT` | `8202903297918bcd12810fc4230a4db7fe065e64` | upstream commit downloaded by `setup.sh src` |
| `LONGLIVE_ROOT` | `$STREAMEVAL_MODEL_ROOT/longlive` | root of the shared environment |
| `WAN21_REV` | `37ec5126…` | Wan-AI/Wan2.1-T2V-1.3B revision to download |

Wan2.1 weights are downloaded at the pinned revision.

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/causal_forcing/weights`, downloaded from Hugging Face:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `checkpoints/causal-forcing++/framewise-2step.pt` (5.7 GB) | zhuhz22/Causal-Forcing @ `2f8eb8bb` | `8f899a01…60ce42` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI/Wan2.1-T2V-1.3B @37ec5126 | `7cace0da…fd81d` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | same | `38071ab5…fd981` | Apache-2.0 |

Weight revisions and file hashes are listed in `system.json` and `setup.sh`. Record resolved revisions and downloaded hashes for each installation.

## Limitations



- No prompt-switch mechanism upstream: a new instruction only changes the text conditioning of the next block.
- No camera input (`camera_control` text is used as prompt text); mid-session reference images are not supported.
