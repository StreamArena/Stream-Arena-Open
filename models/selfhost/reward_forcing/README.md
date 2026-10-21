# Reward Forcing (1.3B) — self-hosted text-to-video interaction

Reward Forcing (Zhejiang University, Ant Group) is a chunk-wise autoregressive, 4-step rewarded-DMD Wan2.1-T2V-1.3B
with an EMA attention sink for long rollouts. The server runs the released checkpoint block by block over the
WebSocket protocol ([docs/protocol.md](../../../docs/protocol.md)), with the shared live loop of
`models/selfhost/t2v_stream.py`.

| | |
|---|---|
| System ids | `reward_forcing_1p3b_t2v` (`general_interaction`) |
| Upstream code | [JaydenLyh/Reward-Forcing](https://github.com/JaydenLyh/Reward-Forcing) @ `4317f5a0d29d1cee6a0f2e80ed5f5724bee97c66` |
| Weights | `JaydenLu666/Reward-Forcing-T2V-1.3B` @ `fc7e9400` (`rewardforcing.pt`); `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec5126` |
| Licence | code: Apache-2.0; weights: Apache-2.0 (code repository licence; the model card states none) (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation.

```bash
bash models/selfhost/reward_forcing/setup.sh src       # upstream code at the pinned commit (network)
bash models/selfhost/reward_forcing/setup.sh env       # shared venv of longlive, env -> $LONGLIVE_ROOT/env
bash models/selfhost/reward_forcing/setup.sh weights   # checkpoint + Wan2.1-T2V-1.3B, about 23 GB (network)
bash models/selfhost/reward_forcing/setup.sh check     # imports the upstream pipeline (GPU)
```

| variable | default | meaning |
|---|---|---|
| `REWARD_FORCING_ROOT` | `$STREAMEVAL_MODEL_ROOT/reward_forcing` | this model's root (read by `setup.sh` and `server.py`) |
| `REWARD_FORCING_COMMIT` | `4317f5a0d29d1cee6a0f2e80ed5f5724bee97c66` | upstream commit downloaded by `setup.sh src` |
| `LONGLIVE_ROOT` | `$STREAMEVAL_MODEL_ROOT/longlive` | root of the shared environment |
| `WAN21_REV` | `37ec5126…` | Wan-AI/Wan2.1-T2V-1.3B revision to download |

Wan2.1 weights are downloaded at the pinned revision.

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/reward_forcing/weights`, downloaded from Hugging Face:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `Reward-Forcing-T2V-1.3B/rewardforcing.pt` (5.7 GB) | JaydenLu666/Reward-Forcing-T2V-1.3B @ `fc7e9400` | `88aebf41…23c3da` | Apache-2.0 (repository licence; the model card states none) |
| `Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI/Wan2.1-T2V-1.3B @37ec5126 | `7cace0da…fd81d` | Apache-2.0 |
| `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | same | `38071ab5…fd981` | Apache-2.0 |

Weight revisions and file hashes are listed in `system.json` and `setup.sh`. Record resolved revisions and downloaded hashes for each installation.

## Limitations

- Long sessions keep moving and stay coherent (EMA sink), but scene content drifts over minutes.
- No prompt-switch mechanism upstream: a new instruction only changes the text conditioning of the next block.
- Text-to-video only; no camera input (`camera_control` text is used as prompt text).
