# Avatar-Forever 22B — self-hosted avatar performance and composition renderer

Avatar-Forever ("Avatar-Forever: Decoupled Parallel Training for High-Quality Real-Time Infinite Avatars", arXiv
2608.12107) turns one portrait, speech audio and a text prompt into chunk-wise autoregressive video on LTX-2.3 22B.
`server.py` drives upstream's own AR sampler one chunk at a time, live, for the avatar performance track and as the
renderer of the shared dialogue stack on the digital-human tracks.

| | |
|---|---|
| System ids | `avatar_forever_performance` (`avatar_performance`), `avatar_forever_comp_audio` (`digital_human_audio`), `avatar_forever_comp_text` (`digital_human_text`) |
| Upstream code | [leeruibin/avatarforever](https://github.com/leeruibin/avatarforever) @ `4dfc42b0e2dbbded4d148387d186219bd7601279` (2026-08-23); arXiv 2608.12107 |
| Weights | `LetsThink/AvatarForever` @ `622ea82482c09e7fb2d8e29cb7ff1d009c820f05` (`avatarforever-ltx-2.3-22b.safetensors`, 46.2 GB); `google/gemma-3-12b-it-qat-q4_0-unquantized` (gated; revision not pinned) |
| Licence | code and Avatar-Forever weights: LTX-2 Community License Agreement; Gemma: Gemma Terms of Use (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model.

```
bash models/selfhost/avatar_forever/setup.sh repo      # upstream source at the pinned commit
bash models/selfhost/avatar_forever/setup.sh env       # uv CPython 3.11 venv: torch 2.10.0+cu130, cuDNN 9.26, transformers 4.57.1, ltx-core / ltx-pipelines
bash models/selfhost/avatar_forever/setup.sh weights   # Avatar-Forever checkpoint (pinned revision) + Gemma 3 12B
bash models/selfhost/avatar_forever/setup.sh hash      # weights_hash.json (checks the checkpoint sha256)
```

`all` runs the four steps in order. Files go to `$STREAMEVAL_MODEL_ROOT/avatar_forever`.

| variable | default | meaning |
|---|---|---|
| `AF_ROOT` | `$STREAMEVAL_MODEL_ROOT/avatar_forever` | this model's folder (setup.sh and server.py) |
| `AF_COMMIT` | `4dfc42b0e2dbbded4d148387d186219bd7601279` | upstream commit |
| `AF_HF_REV` | `622ea82482c09e7fb2d8e29cb7ff1d009c820f05` | `LetsThink/AvatarForever` revision |
| `HF_TOKEN` | — | Hugging Face token (Gemma is gated) |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | `$AF_ROOT/inductor_cache`, `$AF_ROOT/triton_cache` | compiler caches (server.py, only if unset) |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | server.py, only if unset |
| `HF_HUB_OFFLINE` | `1` | server.py loads local files only, unless set otherwise |

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/avatar_forever/weights`; `setup.sh hash` writes the sha256 of every file to
`weights_hash.json`, which the server reports in `ready.config`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `avatarforever-ltx-2.3-22b.safetensors` (46.2 GB) | `LetsThink/AvatarForever` @ `622ea82482c09e7fb2d8e29cb7ff1d009c820f05` | `08dea67e…c3b145` (checked by `setup.sh hash`) | LTX-2 Community License Agreement |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | `google/gemma-3-12b-it-qat-q4_0-unquantized` (gated; revision not pinned, the repository has not changed since 2025-04-15) | in `weights_hash.json` | Gemma Terms of Use |

Code: the upstream repository's `LICENSE` is the LTX-2 Community License Agreement. The Hugging Face weights
repository has no model card; the checkpoint is an LTX-2.3 derivative, so the same agreement applies. Under that
agreement, entities with annual revenues of at least US$10,000,000 need a separate paid commercial licence from
Lightricks to use LTX-2 and its derivatives; it also lists use restrictions (Attachment A).

## Limitations

- Visual controls act only through the text prompt: no per-part state, no left/right grounding, onset up to one
  chunk (1.28 s) + compute. Camera, scene, lighting, clothing, style and speech-content controls are acked
  `unsupported`.
- Each chunk is conditioned on audio up to 0.28 s past its last frame (upstream's audio-to-chunk alignment), so a
  chunk starts only after that audio has arrived; output lags the speech by about 2.5 s.
- The checkpoint's first-frame channel condition is inert (zero weights, upstream issue #5); identity comes from the
  frame-0 latent.
- Renderer only on the digital-human tracks: dialogue text is not used and is acked `unsupported`.
