# Context Forcing (1.3B) — self-hosted text-to-video interaction

Context Forcing (ICML 2026, arXiv 2602.06028; UC Merced, University of Waterloo et al.) is a chunk-wise
autoregressive, 4-step Wan2.1-T2V-1.3B with a managed long context. The server runs the released checkpoint block by
block over the WebSocket protocol ([docs/protocol.md](../../../docs/protocol.md)), with the shared live loop of
`models/selfhost/t2v_stream.py`.

| | |
|---|---|
| System ids | `context_forcing_1p3b_t2v` (`general_interaction`) |
| Upstream code | [TIGER-AI-Lab/Context-Forcing](https://github.com/TIGER-AI-Lab/Context-Forcing) @ `6dd9d9a88332a7b3f5402476895b3871306bcf88`; arXiv 2602.06028 |
| Weights | `ShuoChen20/context_forcing` @ `d44ae93b` (`model.pt`); `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec5126` |
| Licence | code: MIT; weights: MIT (model card) (see Weights and licences) |

Context Forcing belongs to the Self-Forcing family (causal Wan2.1-T2V-1.3B, few-step DMD), but it is a distinct
published model with its own weights and recipe: the student is distilled with **contextual DMD from a long-context
teacher** (a Stable-Video-Infinity-based teacher that sees the generation history, instead of a 5-second memoryless
teacher), and inference keeps a managed long context (sink + slow memory + fast window).

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation.

```bash
bash models/selfhost/context_forcing/setup.sh src       # upstream code at the pinned commit (network)
bash models/selfhost/context_forcing/setup.sh env       # shared venv of longlive, env -> $LONGLIVE_ROOT/env
bash models/selfhost/context_forcing/setup.sh weights   # checkpoint + Wan2.1-T2V-1.3B, about 23 GB (network)
bash models/selfhost/context_forcing/setup.sh check     # imports the upstream pipeline (GPU)
```

`setup.sh src` checks the codeload tarball with `gzip -t` and retries a truncated download.

| variable | default | meaning |
|---|---|---|
| `CONTEXT_FORCING_ROOT` | `$STREAMEVAL_MODEL_ROOT/context_forcing` | this model's root (read by `setup.sh` and `server.py`) |
| `CONTEXT_FORCING_COMMIT` | `6dd9d9a88332a7b3f5402476895b3871306bcf88` | upstream commit downloaded by `setup.sh src` |
| `LONGLIVE_ROOT` | `$STREAMEVAL_MODEL_ROOT/longlive` | root of the shared environment |
| `WAN21_REV` | `37ec5126…` | Wan-AI/Wan2.1-T2V-1.3B revision to download |

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/context_forcing/weights`, downloaded from Hugging Face:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `checkpoints/model.pt` (5.7 GB) | ShuoChen20/context_forcing @ `d44ae93b` | `020aacc3…5b42a7e41` | MIT (model card) |
| `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI/Wan2.1-T2V-1.3B @37ec5126 | `7cace0da…fd81d` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | same | `38071ab5…fd981` | Apache-2.0 |

Weight revisions and file hashes are listed in `system.json` and `setup.sh`. Record resolved revisions and downloaded hashes for each installation.

## Limitations

- No prompt-switch mechanism upstream: a new instruction only changes the text conditioning of the next block.
- Text-to-video only; no camera input (`camera_control` text is used as prompt text).
