# LongLive 2.0 (5B) — self-hosted text-to-video interaction

LongLive 2.0 (NVIDIA) is a causal, few-step autoregressive video generator on Wan2.2-TI2V-5B. The server runs the
released BF16 checkpoint block by block over the WebSocket protocol ([docs/protocol.md](../../../docs/protocol.md)),
with the shared live loop of `models/selfhost/t2v_stream.py`.

| | |
|---|---|
| System ids | `longlive2_5b_t2v` (`general_interaction`) |
| Upstream code | [NVlabs/LongLive](https://github.com/NVlabs/LongLive) @ `fb16a879f46e604df4a0ca48ce2bf45adfa90352`, directory `LongLive2.0/`; arXiv 2605.18739 |
| Weights | `Efficient-Large-Model/LongLive-2.0-5B` @ `8521079b` (`model_bf16.pt`); `Wan-AI/Wan2.2-TI2V-5B` @ `921dbaf3` |
| Licence | code: Apache-2.0; weights: NVIDIA Open Model License (LongLive-2.0-5B), Apache-2.0 (Wan2.2) (see Weights and licences) |

### Model configuration

This adapter loads the LongLive 2.0 BF16 checkpoint listed above. NVFP4 variants require separate CUDA
extensions and are not supported by this wrapper.

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation. The setup script reuses the base Python packages through `--system-site-packages`; it does not replace PyTorch.

```bash
bash models/selfhost/longlive/setup.sh src       # NVlabs/LongLive at the pinned commit (network)
bash models/selfhost/longlive/setup.sh env       # env/ over the base PyTorch (network for pip)
bash models/selfhost/longlive/setup.sh weights   # pinned Hugging Face revisions, about 30 GB (network)
bash models/selfhost/longlive/setup.sh check     # imports the upstream pipeline (GPU)
```

This env/ is also used by `causal_forcing`, `context_forcing`, `memflow`, `reward_forcing`, `rolling_forcing`,
`self_forcing` and `shotstream`: their `setup.sh env` runs this step and links their `env` to it.

| variable | default | meaning |
|---|---|---|
| `LONGLIVE_ROOT` | `$STREAMEVAL_MODEL_ROOT/longlive` | this model's root (read by `setup.sh` and `server.py`) |
| `LONGLIVE_COMMIT` | `fb16a879f46e604df4a0ca48ce2bf45adfa90352` | upstream commit downloaded by `setup.sh src` |

If `$STREAMEVAL_MODEL_ROOT/matrix_game_35/weights/Wan2.2-TI2V-5B` (same revision) exists, `setup.sh weights`
links it instead of downloading Wan2.2-TI2V-5B again.

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/longlive/weights`, downloaded from Hugging Face:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `LongLive-2.0-5B/model_bf16.pt` (10.0 GB) | Efficient-Large-Model/LongLive-2.0-5B @ `8521079b` | `ec9063a4…e35fd8` | NVIDIA Open Model License |
| `wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI/Wan2.2-TI2V-5B @ `921dbaf3` | see `system.json` | Apache-2.0 |
| `wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | same | see `system.json` | Apache-2.0 |
| `wan_models/Wan2.2-TI2V-5B/diffusion_pytorch_model-*.safetensors` (base DiT: architecture config; every weight is overwritten by `model_bf16.pt`) | same | — | Apache-2.0 |

Code: Apache-2.0. The benchmark only evaluates the model; nothing is redistributed.

## Limitations

- Text-to-video only; the image + text track is not registered.
- No camera input: `camera_control` text is used as prompt text.
- Sessions longer than about 170 s go past upstream's 1024-position RoPE table and past anything the model was
  trained on (the table is extended with the same formula).
