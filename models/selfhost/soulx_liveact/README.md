# SoulX-LiveAct 18B — self-hosted avatar performance and composition renderer (digital human)

SoulX-LiveAct (Soul AI Lab, 2026-03) is an open-weight real-time human animation model: one portrait plus streaming
speech audio, with action / emotion text prompts that can be changed mid-stream. `server.py` runs it on 2 GPUs as a
native `avatar_performance` system (visual controls mapped onto its action / emotion prompts) and as a renderer for
the shared dialogue stack on the digital-human tracks.

| | |
|---|---|
| System ids | `soulx_liveact_performance` (`avatar_performance`), `soulx_liveact_comp_audio` (`digital_human_audio`), `soulx_liveact_comp_text` (`digital_human_text`) |
| Upstream code | [Soul-AILab/SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct) @ `ac8579b907756f228e975922a5159765c93c6a34` (2026-06-15) |
| Weights | `Soul-AILab/LiveAct` @ `1bcdef1d8941edf2a9a6e1d21f5657f0a69fcd37` (53 GB: 14B Wan2.1-I2V backbone + 4B audio module, umT5-XXL text encoder, open-CLIP, Wan2.1 VAE), `TencentGameMate/chinese-wav2vec2-base` @ `3991242c806928916fff4a8c0e4f76acf661b743` |
| Licence | code: no licence file upstream (all rights reserved by default); weights: LiveAct Apache-2.0 (model card), chinese-wav2vec2-base MIT (see Weights and licences) |

## Setup

Prerequisites: curl, tar, git, python3, `hf` (huggingface_hub CLI), [uv](https://docs.astral.sh/uv/), a C++
compiler and a CUDA 12.8 toolkit (nvcc, headers, cudart: SageAttention is built against torch 2.8.0+cu128 and needs
the 12.8 compiler; a CUDA 13 toolkit does not work). The environment installs its own PyTorch (Python 3.10, torch
2.8.0+cu128, cuDNN 9.26, vLLM 0.11.0, flash-attn 2.8.3, xfuser 0.4.5, SageAttention 2.2.0, LightX2V `8a97c759`);
`env_freeze.txt` records the result.

```
bash models/selfhost/soulx_liveact/setup.sh repo      # upstream source at the pinned commit
bash models/selfhost/soulx_liveact/setup.sh cuda      # copy nvcc / headers / cudart of the CUDA 12.8 toolkit
bash models/selfhost/soulx_liveact/setup.sh weights   # 53 GB, pinned Hugging Face revisions
bash models/selfhost/soulx_liveact/setup.sh env       # GPU machine: venv + SageAttention / LightX2V builds
bash models/selfhost/soulx_liveact/setup.sh hash      # sha256 of the loaded files -> weights_hash.json
```

`bash setup.sh` (or `all`) runs the five steps in that order.

| variable | default | meaning |
|---|---|---|
| `SOULX_LIVEACT_ROOT` | `$STREAMEVAL_MODEL_ROOT/soulx_liveact` | this model's folder (setup.sh and server.py; `{env}` in `system.json` still resolves under `STREAMEVAL_MODEL_ROOT`) |
| `SOULX_LIVEACT_COMMIT` | `ac8579b907756f228e975922a5159765c93c6a34` | upstream commit |
| `LIVEACT_REPO` | `<model folder>/repo` | upstream source tree imported by server.py |
| `CUDA128_HOME` | `/usr/local/cuda-12.8` | CUDA 12.8 toolkit the `cuda` step copies from |
| `SAGE_ARCH` | `8.9` | architecture SageAttention is built for (one per build). On the target GPU architecture it is only imported, never called; `9.0` on H100/H200 enables `SAGE_FP8_SM90` |
| `UV_CACHE_DIR` | `<model folder>/uv_cache` | uv cache |

server.py also sets `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` (default `<model folder>/inductor_cache`,
`triton_cache`) and `USE_CHANNELS_LAST_3D=1` unless already set, and points `TRITON_PTXAS_PATH` at the env's ptxas 12.9.

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/soulx_liveact/weights`. The sha256 in `system.json` equal the Hugging Face LFS oids;
`setup.sh hash` re-hashes the files into `weights_hash.json`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `LiveAct/diffusion_pytorch_model-0000{1..4}-of-00004.safetensors` | `Soul-AILab/LiveAct` @ `1bcdef1d8941edf2a9a6e1d21f5657f0a69fcd37` | `13f67d7f…`, `bb009308…`, `11dc4594…`, `4f567580…` | Apache-2.0 (model card) |
| `LiveAct/Wan2.1_VAE.pth` | same | `38071ab5…2fd981` | Apache-2.0 (model card) |
| `LiveAct/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` | same | `628c9998…158c38` | Apache-2.0 (model card) |
| `LiveAct/models_t5_umt5-xxl-enc-bf16.pth` | same | `7cace0da…6fd81d` | Apache-2.0 (model card) |
| `chinese-wav2vec2-base/pytorch_model.bin` | `TencentGameMate/chinese-wav2vec2-base` @ `3991242c806928916fff4a8c0e4f76acf661b743` | `be2da40c…c38901` | MIT |

Code: the GitHub repository has no licence file (checked at the pinned commit and on the default branch), so no
licence is granted for it. The Hugging Face model card of `Soul-AILab/LiveAct` declares Apache-2.0.

## Limitations

- No explicit per-part state: set / update / cancel controls are passed through as text, there is no left/right
  grounding beyond what the text encoder learned, and one-shot actions ("nod once") may repeat while their prompt is
  active. The judge, not the ack, decides whether a control took effect.
- Camera, scene, appearance and speech-content controls are acked `unsupported`.
- Renderer-only on the digital-human tracks: dialogue text is not used (acked `unsupported`); the avatar speaks the
  pushed TTS audio.
