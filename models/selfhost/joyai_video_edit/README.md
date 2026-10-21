# JoyAI-Video-Edit — self-hosted streaming instruction and reference editing

JoyAI-Video-Edit (JD) is a causal streaming video-to-video editor: instruction edits (V2V) and one-reference-image
edits (RV2V). `server.py` wraps the upstream streaming runtime (`xvideo.serving.joyomni_streaming`) with
`streameval.runtime.protocol.serve` (`docs/protocol.md`).

| | |
|---|---|
| System ids | `joyai_video_edit_instruction` (`instruction_edit`), `joyai_video_edit_reference` (`reference_edit`) |
| Upstream code | [jd-opensource/JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit) @ `ca17e1d1030f454cb98b0ed549b4d31a60139ceb` |
| Weights | jdopensource/JoyAI-Video-Edit @ `39491dda89fd2b535777fd8644a6631ea89b857c` (DiT 0811, VAE); XiaomiMiMo/MiMo-VL-7B-RL-2508 (text encoder, revision not pinned) |
| Licence | code: Apache-2.0; weights: Apache-2.0; text encoder: MIT (see Weights and licences) |

## Setup

Prerequisites: bash, curl, git, an NVIDIA driver supporting CUDA 12.9, about 50 GB of disk. The environment is
self-contained (micromamba Python 3.10 + conda-forge nvcc 12.9 + upstream `deploy/requirements.txt` with torch
2.9.1+cu128); no system PyTorch is used. The `hf` CLI comes from the env (or from `PATH`).

```bash
S=models/selfhost/joyai_video_edit/setup.sh
bash $S clone     # upstream repo at the pinned commit
bash $S fetch     # cutlass @ dcf215af68a2 and SageAttention @ d1a57a546c3d sources (+ upstream's cudagraph patch)
bash $S env       # micromamba env, upstream pip pins, flash-attn-4 4.0.0b13, NVRTC 12.9
bash $S weights   # DiT + VAE, MiMo-VL, YuNet; writes dit_0811.sha256 / vae.sha256 (recorded, not verified)
bash $S build     # joyomni_ops FP8 for this GPU's arch (SageAttention only on RTX 5090)  — GPU
bash $S check     # must print `nvrtc jiterator OK` and `joyomni_ops OK True`               — GPU
```

`bash $S` (or `all`) runs them in this order; `gpu` runs env, build, check. Layout under `$JOY_ROOT`:
`repo/ weights/ env/ mamba/ build/ cache_<gpu>/ logs/`.

| variable | default | meaning |
|---|---|---|
| `JOY_ROOT` | `$STREAMEVAL_MODEL_ROOT/joyai_video_edit` | this model's folder (setup.sh and server.py) |
| `JOYAI_HF_REV` | `39491dda89fd2b535777fd8644a6631ea89b857c` | jdopensource/JoyAI-Video-Edit revision |
| `MIMO_VL_REV` | `main` | XiaomiMiMo/MiMo-VL-7B-RL-2508 revision |
| `CONDA_CHANNEL` | `conda-forge` | conda channel of the CUDA toolchain |
| `HF_ENDPOINT`, `HF_TOKEN` | `https://huggingface.co`, unset | Hugging Face endpoint and token |
| `MAX_JOBS` | `8` | parallel compile jobs (build) |
| `JOYOMNI_WIDTH`, `JOYOMNI_HEIGHT`, `JOYOMNI_FPS` | `840`, `480`, `24` | defaults of `--width`, `--height`, `--fps` (server.py) |
| `JOYOMNI_FP8_IMG`, `JOYOMNI_FP8_TXT`, `JOYOMNI_CUDA_GRAPH`, `JOYOMNI_SAGE_ATTN`, `JOYOMNI_FP8_FAST_ACCUM`, `JOYOMNI_LOW_VRAM` | `1, 1, 1, 0, 0, 0` (RTX 5090: SageAttention, fast accumulation and low VRAM on; no FP8 build: FP8 off) | upstream per-GPU launch flags (DEPLOYMENT.md §4), set by server.py unless already set |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH` | `$JOY_ROOT/cache_<gpu name>/…` | per-GPU compile caches (server.py) |
| `CUDA_HOME`, `PYTORCH_CUDA_ALLOC_CONF` | `$JOY_ROOT/env`, `expandable_segments:True` | set by server.py unless already set |

## Weights and licences

`$JOY_ROOT/weights`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth` (32.5 GB) | jdopensource/JoyAI-Video-Edit @ `39491dda89fd2b535777fd8644a6631ea89b857c` | `b3904b6fda53d13b230918bb616f322d12cfb2337b0e8d9dc203cdabc36605ba` | Apache-2.0 |
| `JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors` | same | `150315748d7c3307cdae2819ee651b32d58385668ca0c4db3d3dcd6e63b77e86` | Apache-2.0 |
| `MiMo-VL-7B-RL-2508/` (16 GB) | XiaomiMiMo/MiMo-VL-7B-RL-2508 @ `main` (v1 revision not pinned) | — | MIT |
| `face_detection_yunet_2023mar.onnx` (unused: gate off) | opencv/opencv_zoo @ `main` (upstream DEPLOYMENT.md URL) | — | see opencv_zoo |

Code: Apache-2.0. Benchmark evaluation only; nothing is redistributed.

## Limitations

- Every applied instruction or reference restarts the causal session, so output pauses ~0.4–1.3 s while the new
  session prefills, and earlier edits are not kept (one prompt per session).
- One reference image: states with two or more reference slots, or an empty state, are `unsupported`.
- Output is fixed at 840×480 / 480×840 regardless of the source size.
- On the target GPU architecture the DiT uses cuDNN SDPA instead of upstream's flash-attn-4 path.
