# Rolling Forcing (1.3B) — self-hosted text-to-video interaction

Rolling Forcing (NTU + Tencent ARC Lab, ICLR 2026, arXiv 2509.25161) is a causal Wan2.1-T2V-1.3B distilled with DMD
that denoises a rolling window of blocks jointly. The server runs upstream's window loop live over the WebSocket
protocol ([docs/protocol.md](../../../docs/protocol.md)), with the shared live loop of
`models/selfhost/t2v_stream.py`.

| | |
|---|---|
| System ids | `rolling_forcing_1p3b_t2v` (`general_interaction`) |
| Upstream code | [TencentARC/RollingForcing](https://github.com/TencentARC/RollingForcing) @ `a1477d09e85dc759a6a6728f55f77f59342ce388`; arXiv 2509.25161 |
| Weights | `TencentARC/RollingForcing` @ `20edee70` (`checkpoints/rolling_forcing_dmd.pt`); `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec5126` |
| Licence | code: MIT; weights: MIT (see Weights and licences) |

Rolling Forcing is a Self-Forcing-family model with its own weights and training recipe:
- **Rolling-window joint denoising:** 5 blocks at staggered noise levels are denoised together in one pass, instead
  of one block at a time.
- **Attention sink:** an anchor block kept for long-term memory.
- **Training:** distilled for multi-minute rollouts.

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation.

```bash
bash models/selfhost/rolling_forcing/setup.sh src       # upstream code at the pinned commit (network)
bash models/selfhost/rolling_forcing/setup.sh env       # shared venv of longlive, env -> $LONGLIVE_ROOT/env
bash models/selfhost/rolling_forcing/setup.sh weights   # checkpoint (17 GB) + Wan2.1-T2V-1.3B, about 34 GB (network)
bash models/selfhost/rolling_forcing/setup.sh check     # imports the upstream pipeline (GPU)
```

| variable | default | meaning |
|---|---|---|
| `ROLLING_FORCING_ROOT` | `$STREAMEVAL_MODEL_ROOT/rolling_forcing` | this model's root (read by `setup.sh` and `server.py`) |
| `ROLLING_FORCING_COMMIT` | `a1477d09e85dc759a6a6728f55f77f59342ce388` | upstream commit downloaded by `setup.sh src` |
| `LONGLIVE_ROOT` | `$STREAMEVAL_MODEL_ROOT/longlive` | root of the shared environment |
| `WAN21_REV` | `37ec5126…` | Wan-AI/Wan2.1-T2V-1.3B revision to download |

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/rolling_forcing/weights`, downloaded from Hugging Face:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `checkpoints/rolling_forcing_dmd.pt` (17 GB, generator + EMA) | TencentARC/RollingForcing @ `20edee70` | `08448992…7a6d9d` | MIT |
| `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI/Wan2.1-T2V-1.3B @37ec5126 | `7cace0da…fd81d` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | same | `38071ab5…fd981` | Apache-2.0 |

Weight revisions and file hashes are listed in `system.json` and `setup.sh`. Record resolved revisions and downloaded hashes for each installation.

## Limitations

- An instruction fully conditions only the block that enters the window after it, which is output 4 blocks (2.4 s of
  video) later; the blocks in flight get only their remaining steps under the new prompt.
- Text-to-video only; no camera input (`camera_control` text is used as prompt text).
- 300 s sessions go past upstream's 1024-position RoPE table (extended with the same formula).
