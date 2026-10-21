# AvatarForcing 1.3B — self-hosted avatar performance and composition renderer (digital human)

AvatarForcing (Kling Team, Kuaishou; arXiv 2603.14331, "One-Step Streaming Talking Avatars via Local-Future
Sliding-Window Denoising") is a one-step streaming talking-avatar model: a causal Wan2.1-T2V-1.3B + LoRA + OmniAvatar
audio pack (in_dim 33), trained by two-stage streaming distillation (ODE backfill + DMD). `server.py` unrolls
upstream's offline sliding-window loop step by step on live audio.

| | |
|---|---|
| System ids | `avatarforcing_performance` (`avatar_performance`), `avatarforcing_comp_audio` (`digital_human_audio`), `avatarforcing_comp_text` (`digital_human_text`) |
| Upstream code | [KlingAIResearch/AvatarForcing](https://github.com/KlingAIResearch/AvatarForcing) @ `63b73e6c0f7bb42180ca6d7e1bf11c1de1a80b39` (2026-05-09); arXiv 2603.14331 |
| Weights | `lycui/AvatarForcing` @ `e2448919a7b535c29f34e07892884ae1a43c6ace` (`model.pt`); `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec5126` (T5 encoder, tokenizer, VAE); `facebook/wav2vec2-base-960h` @ `22aad52d` |
| Licence | code: conflicting — Apache-2.0 (README) vs academic use only (`LICENSE.txt`); weights: none stated; Wan2.1, wav2vec2: Apache-2.0 (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation.

```
bash models/selfhost/avatarforcing/setup.sh repo      # upstream code at the pinned commit
bash models/selfhost/avatarforcing/setup.sh env       # on the GPU machine: env/ (venv --system-site-packages)
bash models/selfhost/avatarforcing/setup.sh weights   # pinned Hugging Face revisions (needs env/)
bash models/selfhost/avatarforcing/setup.sh hash      # weights_hash.json
bash models/selfhost/avatarforcing/setup.sh all       # the four steps in order
```

`weights` copies Wan2.1-T2V-1.3B and wav2vec2-base-960h from `$STREAMEVAL_MODEL_ROOT/livetalk/weights` when that
folder holds them (the same revisions), and downloads the pinned revisions otherwise.

| variable | default | meaning |
|---|---|---|
| `AVATARFORCING_ROOT` | `$STREAMEVAL_MODEL_ROOT/avatarforcing` | this model's folder (setup.sh and server.py defaults for `--repo`, `--checkpoint`, `--wav2vec`, `--stats_dir`). `--launch` runs `$STREAMEVAL_MODEL_ROOT/avatarforcing/env/bin/python` |
| `AVATARFORCING_COMMIT` | `63b73e6c0f7bb42180ca6d7e1bf11c1de1a80b39` | upstream commit |
| `AVATARFORCING_REV` | `e2448919a7b535c29f34e07892884ae1a43c6ace` | `lycui/AvatarForcing` revision |
| `PYTHON` | `python3` | base Python for `env` |

`server.py` sets `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to `inductor_cache/` and `triton_cache/` in the
model folder unless they are already set.

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/avatarforcing/weights`; sha256 of every weight file also in `weights_hash.json` next
to it.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `AvatarForcing/model.pt` | `lycui/AvatarForcing` @ `e2448919a7b535c29f34e07892884ae1a43c6ace` | `63dd0b38…dbb2b3` | none stated on the weights repo; see below |
| `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec512624d61f7aa208f7ea8140a131f93afc9a` | `38071ab5…2fd981` | Apache-2.0 |
| `Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` (+ `google/umt5-xxl` tokenizer) | same | `7cace0da…6fd81d` | Apache-2.0 |
| `wav2vec2-base-960h/model.safetensors` | `facebook/wav2vec2-base-960h` @ `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | `8aa76ab2…17d74a` | Apache-2.0 |

The AvatarForcing licence is contradictory upstream: the README says Apache-2.0, while the repository's
`LICENSE.txt` carries Tencent's "License Terms of RollingForcing", which permit use "only for academic purposes" and
forbid "any commercial or production purposes". The Hugging Face weights repository states no licence. Treat code
and weights as academic, non-commercial use only.

## Limitations

- The 1.92 s audio look-ahead of the local-future window cannot be removed without changing the model.
- Fixed 640×624 output (the model hard-codes 1560 tokens per latent frame); the portrait is centre-cropped to it.
- Controls act only through the text prompt; camera, scene, appearance and speech-content controls are acked
  `unsupported`.
- Slow colour / texture drift appears in long sessions (≥ 45 s).
