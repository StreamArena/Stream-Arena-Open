# SoulX-FlashTalk 14B — self-hosted composition renderer (digital human)

SoulX-FlashTalk (Soul AI Lab, arXiv 2512.23379) is a real-time, infinite-streaming audio-driven avatar model:
Wan2.1-I2V-14B with the InfiniteTalk/MultiTalk audio adapter, distilled with self-correcting bidirectional
distillation to 4 steps without CFG, generating chunk-wise with motion-frame overlap. `server.py` renders the speech
of the shared dialogue stack (ASR → LLM → TTS, run by the harness) on 4 GPUs and returns frames plus the same speech.

| | |
|---|---|
| System ids | `soulx_flashtalk_audio` (`digital_human_audio`), `soulx_flashtalk_text` (`digital_human_text`) |
| Upstream code | [Soul-AILab/SoulX-FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk) @ `2ba2e1e801d59a9f526212a5aae5d374db8f2978` (2026-07-30); arXiv 2512.23379 |
| Weights | `Soul-AILab/SoulX-FlashTalk-14B` @ `f3fda6499a6022862cadd39fc41ae63b64706e1b` (DiT in 4 shards, umT5-XXL, open-CLIP, Wan2.1 VAE; 53 GB); `TencentGameMate/chinese-wav2vec2-base` @ `3991242c806928916fff4a8c0e4f76acf661b743` |
| Licence | code: Apache-2.0; weights: Apache-2.0, chinese-wav2vec2-base MIT (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation.

```
bash models/selfhost/soulx_flashtalk/setup.sh repo      # upstream code at the pinned commit
bash models/selfhost/soulx_flashtalk/setup.sh env       # on the GPU machine: env/ (venv --system-site-packages)
bash models/selfhost/soulx_flashtalk/setup.sh weights   # about 53 GB; uses env/
bash models/selfhost/soulx_flashtalk/setup.sh hash      # weights_hash.json
# or: setup.sh all (the four in order)
```

`weights` copies `chinese-wav2vec2-base` from `$STREAMEVAL_MODEL_ROOT/soulx_liveact/weights` when that exists,
otherwise downloads it at the pinned revision.

| variable | default | meaning |
|---|---|---|
| `SOULX_FLASHTALK_ROOT` | `$STREAMEVAL_MODEL_ROOT/soulx_flashtalk` | this model's folder (setup.sh and server.py: repo, weights, hashes, caches, stats) |
| `SOULX_FLASHTALK_COMMIT` | `2ba2e1e801d59a9f526212a5aae5d374db8f2978` | upstream commit |
| `SOULX_FLASHTALK_REV` | `f3fda6499a6022862cadd39fc41ae63b64706e1b` | Hugging Face revision of `SoulX-FlashTalk-14B` |
| `PYTHON` | `python3` | base Python for `env` |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | `<model folder>/inductor_cache`, `/triton_cache` | compile caches (server.py sets them if unset) |

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/soulx_flashtalk/weights`; per-file sha256 in `system.json` and `weights_hash.json`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `SoulX-FlashTalk-14B/diffusion_pytorch_model-0000{1..4}-of-00004.safetensors` | `Soul-AILab/SoulX-FlashTalk-14B` @ `f3fda6499a60` | see `system.json` | Apache-2.0 |
| `SoulX-FlashTalk-14B/models_t5_umt5-xxl-enc-bf16.pth`, `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, `Wan2.1_VAE.pth` | same | see `system.json` | Apache-2.0 |
| `chinese-wav2vec2-base/pytorch_model.bin` | `TencentGameMate/chinese-wav2vec2-base` @ `3991242c8069` | `be2da40c…c38901` | MIT |

Code: Apache-2.0 (Wan2.1 / InfiniteTalk base: Apache-2.0). Nothing is redistributed.

## Limitations

- Renderer only: what the avatar says comes from the shared dialogue stack; text events do not change the
  rendering, and the text prompt does not steer motion, so the model is not registered for `avatar_performance`.
- The registered command needs 4 GPUs (Ulysses ×4).
- Output is 512×512 rather than upstream's default 416×720.
