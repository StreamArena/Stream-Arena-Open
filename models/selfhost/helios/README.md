# Helios 14B — self-hosted text-to-video and image-to-video interaction

PKU-YuanGroup's real-time long-video model (Helios, a 14B autoregressive DiT built on Wan-2.1-14B), served live
through the WebSocket protocol (`docs/protocol.md`). The server unrolls upstream's chunk loop so prompts, camera
directions and the anchor image can change between chunks. Reactor also offers Helios as a hosted service; this
folder runs the open-weight release.

| | |
|---|---|
| System ids | `helios_t2v` (`general_interaction`), `helios_i2v` (`general_interaction_image`) |
| Upstream code | [PKU-YuanGroup/Helios](https://github.com/PKU-YuanGroup/Helios) @ `babed9811266e4b5b111c9c1e0977a07899066ab` (2026-08-24); arXiv 2603.04379 |
| Weights | [BestWishYsh/Helios-Distilled](https://huggingface.co/BestWishYsh/Helios-Distilled) @ `b991c0379a018f4de3227d95468237f56066f5bb` (transformer, umT5-XXL text encoder, Wan2.1 VAE, DMD scheduler; `transformer_ode/` not used) |
| Licence | code: Apache-2.0; weights: Apache-2.0 (see Weights and licences) |

## Setup

Prerequisites: git, curl, [uv](https://docs.astral.sh/uv/), an NVIDIA driver that supports CUDA 13, a GPU
with enough memory for the configured model and resolution. `setup.sh` installs its own torch (2.10.0+cu130 from the PyTorch index) into a uv
venv, so the base Python does not need PyTorch.

```
bash models/selfhost/helios/setup.sh repo      # clone PKU-YuanGroup/Helios, check out the pinned commit
bash models/selfhost/helios/setup.sh env       # Python 3.12 + venv, torch 2.10.0+cu130, cuDNN 9.26, diffusers ...
bash models/selfhost/helios/setup.sh weights   # BestWishYsh/Helios-Distilled @ pinned revision (needs env/)
bash models/selfhost/helios/setup.sh hash      # weights_hash.json
bash models/selfhost/helios/setup.sh           # = all four in order
```

`env` installs cuDNN 9.26 for the configured CUDA stack.

| variable | default | meaning |
|---|---|---|
| `HELIOS_ROOT` | `$STREAMEVAL_MODEL_ROOT/helios` | `setup.sh` only: install folder (the server and the harness's `{env}` / `{weights}` expect `<root>/helios`) |
| `HELIOS_COMMIT` | `babed9811266e4b5b111c9c1e0977a07899066ab` | `setup.sh`: upstream commit to check out |
| `HELIOS_REPO` | `$STREAMEVAL_MODEL_ROOT/helios/repo` | `server.py`: upstream checkout to import |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | `<root>/helios/inductor_cache`, `<root>/helios/triton_cache` | `server.py`: compiled-kernel caches (kept across starts) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | `server.py`: set if unset |

## Weights and licences

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `Helios-Distilled/transformer/*.safetensors` (6 shards) | BestWishYsh/Helios-Distilled @ `b991c037` | `system.json` (`weights.transformer_sha256`) | Apache-2.0 |
| `Helios-Distilled/text_encoder`, `tokenizer` (umT5-XXL) | same | `weights_hash.json` (computed by `setup.sh hash`) | Apache-2.0 |
| `Helios-Distilled/vae` (Wan2.1 VAE), `scheduler` | same | `weights_hash.json` | Apache-2.0 |

Helios code (upstream `LICENSE.txt`) and weights (Hugging Face model card): Apache-2.0. Base model Wan-2.1-14B,
umT5-XXL and the Wan2.1 VAE: Apache-2.0. Nothing is redistributed; `setup.sh` downloads from the upstream sources.

## Limitations

- Camera control is text only (no numeric camera input); a camera direction becomes part of the text condition.
- An event takes effect at the next chunk boundary (up to one chunk, ~1.4 s, after release).
- Only one reference image (the first-frame anchor): `reference` ops other than `set` are `unsupported`.
- `--max-chunks 400` caps a rollout at 550 s.
- Attention is torch SDPA instead of upstream's FlashAttention (use a compatible attention backend), and no super-resolution pass.
