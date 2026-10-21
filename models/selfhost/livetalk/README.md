# LiveTalk 1.3B — self-hosted avatar performance and composition renderer (digital human)

LiveTalk (GAIR / SII / SJTU, arXiv 2512.23576) is a real-time talking-avatar model: OmniAvatar (Wan2.1-T2V-1.3B with
an audio adapter) distilled with Self-Forcing into a causal, block-wise autoregressive 4-step generator, conditioned
on one portrait, streaming speech and a text prompt. `server.py` unrolls upstream's offline inference loop block by
block on live audio.

| | |
|---|---|
| System ids | `livetalk_performance` (`avatar_performance`), `livetalk_comp_audio` (`digital_human_audio`), `livetalk_comp_text` (`digital_human_text`) |
| Upstream code | [GAIR-NLP/LiveTalk](https://github.com/GAIR-NLP/LiveTalk) @ `e0d555fc5edf8516fbed2c955f1c0f4b32476abb` (2026-01-02); arXiv 2512.23576 |
| Weights | `GAIR/LiveTalk-1.3B-V0.1` @ `be14ffdf356e73e328db88595ccf62e7e1374b74` (`model.safetensors`, 5.7 GB); `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec5126` (T5 encoder, tokenizer, VAE); `facebook/wav2vec2-base-960h` @ `22aad52d` |
| Licence | code: conflicting — Apache-2.0 (README) vs CC BY-NC-SA 4.0 (`LICENSE.md`); weights: Apache-2.0 (model card); Wan2.1, wav2vec2: Apache-2.0 (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation.

```
bash models/selfhost/livetalk/setup.sh repo      # upstream code at the pinned commit
bash models/selfhost/livetalk/setup.sh env       # on the GPU machine: env/ (venv --system-site-packages)
bash models/selfhost/livetalk/setup.sh weights   # pinned Hugging Face revisions (needs env/)
bash models/selfhost/livetalk/setup.sh hash      # weights_hash.json
bash models/selfhost/livetalk/setup.sh all       # the four steps in order
```

`weights` copies Wan2.1-T2V-1.3B from `$STREAMEVAL_MODEL_ROOT/self_forcing` and wav2vec2-base-960h from
`$STREAMEVAL_MODEL_ROOT/soulx_flashhead` when those folders already hold them, and downloads the pinned revision
otherwise. `avatarforcing` reuses this folder's `weights/` in the same way.

| variable | default | meaning |
|---|---|---|
| `LIVETALK_ROOT` | `$STREAMEVAL_MODEL_ROOT/livetalk` | this model's folder (setup.sh and server.py defaults for `--repo`, `--weights`, `--stats_dir`). `--launch` runs `$STREAMEVAL_MODEL_ROOT/livetalk/env/bin/python` |
| `LIVETALK_COMMIT` | `e0d555fc5edf8516fbed2c955f1c0f4b32476abb` | upstream commit |
| `LIVETALK_REV` | `be14ffdf356e73e328db88595ccf62e7e1374b74` | `GAIR/LiveTalk-1.3B-V0.1` revision |
| `PYTHON` | `python3` | base Python for `env` |

`server.py` sets `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to `inductor_cache/` and `triton_cache/` in the
model folder unless they are already set.

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/livetalk/weights`; sha256 also in `weights_hash.json` next to it.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `LiveTalk-1.3B-V0.1/model.safetensors` | `GAIR/LiveTalk-1.3B-V0.1` @ `be14ffdf356e73e328db88595ccf62e7e1374b74` | `fbf0e5c8…07ef50` | Apache-2.0 (model card); see below |
| `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec512624d61f7aa208f7ea8140a131f93afc9a` | `38071ab5…2fd981` | Apache-2.0 |
| `Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` (+ `google/umt5-xxl` tokenizer) | same | `7cace0da…6fd81d` | Apache-2.0 |
| `wav2vec2-base-960h/model.safetensors` | `facebook/wav2vec2-base-960h` @ `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | `8aa76ab2…17d74a` | Apache-2.0 |

The LiveTalk licence is contradictory upstream: the README and the Hugging Face model card say Apache-2.0, while the
repository's `LICENSE.md` is CC BY-NC-SA 4.0 (non-commercial, share-alike). Upstream issue #7 ("Clarify license")
asks which applies and is open without an answer. Until it is resolved, treat the code as non-commercial.

## Limitations

- 512×512 at 16 fps only (upstream's single configuration).
- Controls act through the next chunk's text prompt.
- Camera, scene, appearance and speech-content controls are acked `unsupported`.
- On the composition tracks the renderer only lip-syncs the dialogue stack's speech; dialogue text does not change
  the rendering.
