# Self-Forcing (1.3B) — self-hosted text-to-video interaction

Self-Forcing (Adobe Research, UT Austin) is a chunk-wise autoregressive, 4-step DMD-distilled Wan2.1-T2V-1.3B. The
server runs the released checkpoint block by block over the WebSocket protocol
([docs/protocol.md](../../../docs/protocol.md)), with the shared live loop of `models/selfhost/t2v_stream.py`, and
extends it to open-ended sessions with upstream's rolling KV cache.

| | |
|---|---|
| System ids | `self_forcing_1p3b_t2v` (`general_interaction`) |
| Upstream code | [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) @ `33593df3e81fa3ec10239271dd2c100facac6de1` |
| Weights | `gdhe17/Self-Forcing` @ `2f8b7792` (`checkpoints/self_forcing_dmd.pt`); `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec5126` |
| Licence | code: Apache-2.0; weights: Apache-2.0 (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation.

```bash
bash models/selfhost/self_forcing/setup.sh src       # upstream code at the pinned commit (network)
bash models/selfhost/self_forcing/setup.sh env       # shared venv of longlive, env -> $LONGLIVE_ROOT/env
bash models/selfhost/self_forcing/setup.sh weights   # checkpoint + Wan2.1-T2V-1.3B, about 23 GB (network)
bash models/selfhost/self_forcing/setup.sh check     # imports the upstream pipeline (GPU)
```

| variable | default | meaning |
|---|---|---|
| `SELF_FORCING_ROOT` | `$STREAMEVAL_MODEL_ROOT/self_forcing` | this model's root (read by `setup.sh` and `server.py`) |
| `SELF_FORCING_COMMIT` | `33593df3e81fa3ec10239271dd2c100facac6de1` | upstream commit downloaded by `setup.sh src` |
| `LONGLIVE_ROOT` | `$STREAMEVAL_MODEL_ROOT/longlive` | root of the shared environment |
| `WAN21_REV` | `37ec5126…` | Wan-AI/Wan2.1-T2V-1.3B revision to download |

Wan2.1 weights are downloaded at the pinned revision.

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/self_forcing/weights`, downloaded from Hugging Face:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `checkpoints/self_forcing_dmd.pt` (5.7 GB) | gdhe17/Self-Forcing @ `2f8b7792` | `a0413986…8f56a3` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI/Wan2.1-T2V-1.3B @37ec5126 | `7cace0da…fd81d` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | same | `38071ab5…fd981` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` (base DiT: architecture config; every weight is overwritten by the checkpoint) | same | — | Apache-2.0 |

Weight revisions and file hashes are listed in `system.json` and `setup.sh`. Record resolved revisions and downloaded hashes for each installation.

## Limitations


- No prompt-switch mechanism upstream: a new instruction only changes the text conditioning of the next block.
- Text-to-video only; no camera input (`camera_control` text is used as prompt text).
