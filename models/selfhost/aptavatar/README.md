# AptAvatar 14B — self-hosted audio-driven avatar (performance and composition renderer)

AptAvatar (TaoLiveAIGC, Taobao Live, Alibaba; arXiv 2607.24013) is a fast long-form audio-driven avatar model:
Wan2.1-I2V-14B with the InfiniteTalk audio adapter, distilled to 2 steps without CFG (endpoint-anchored distribution
distillation, self-generated history replay). `server.py` runs upstream's chunk-wise generation loop live on 4 GPUs
(xfuser Ulysses sequence parallel) behind the WebSocket protocol (`docs/protocol.md`).

| | |
|---|---|
| System ids | `aptavatar_performance` (`avatar_performance`), `aptavatar_comp_audio` (`digital_human_audio`), `aptavatar_comp_text` (`digital_human_text`) |
| Upstream code | [TaoLiveAIGC/AptAvatar](https://github.com/TaoLiveAIGC/AptAvatar) @ `c3cc19ca4481d0aafa35998fe6a538003e5cef61` (2026-08-03); arXiv 2607.24013 |
| Weights | `TaoLiveAIGC/AptAvatar` @ `e5ae7ce74e1766693ed19473c3b51247351a0c78` (DiT in 8 shards, umT5-XXL, open-CLIP, Wan2.1 VAE; 54 GB); `TencentGameMate/chinese-wav2vec2-base` @ `3991242c806928916fff4a8c0e4f76acf661b743` |
| Licence | code: Apache-2.0; weights: Apache-2.0; chinese-wav2vec2-base: MIT (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation. The setup script reuses the base Python packages through `--system-site-packages`; it does not replace PyTorch.

```
bash models/selfhost/aptavatar/setup.sh all    # or the steps one by one:
bash models/selfhost/aptavatar/setup.sh repo     # network: upstream code at the pinned commit
bash models/selfhost/aptavatar/setup.sh env      # network (PyPI), GPU machine: env/ over the base torch
bash models/selfhost/aptavatar/setup.sh weights  # network (Hugging Face, ~54 GB); copies chinese-wav2vec2-base from soulx_flashtalk when present
bash models/selfhost/aptavatar/setup.sh hash     # sha256 of every weight file into weights_hash.json (sent in ready.config)
```

Files go to `$STREAMEVAL_MODEL_ROOT/aptavatar/` (`repo/`, `env/`, `weights/`, `weights_hash.json`,
`logs/sessions/`).

| variable | default | meaning |
|---|---|---|
| `APTAVATAR_ROOT` | `$STREAMEVAL_MODEL_ROOT/aptavatar` | this model's folder (setup and server) |
| `APTAVATAR_COMMIT` | `c3cc19ca4481d0aafa35998fe6a538003e5cef61` | upstream code commit |
| `APTAVATAR_REV` | `e5ae7ce74e1766693ed19473c3b51247351a0c78` | Hugging Face revision of `TaoLiveAIGC/AptAvatar` |
| `PYTHON` | `python3` | base interpreter the venv is built over |
| `HF_ENDPOINT`, `HF_TOKEN` | `https://huggingface.co`, unset | Hugging Face endpoint and token |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | `<model folder>/inductor_cache`, `<model folder>/triton_cache` | compile caches (server) |

## Weights and licences

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `AptAvatar/diffusion_pytorch_model-0000{1..8}-of-00008.safetensors`, `models_t5_umt5-xxl-enc-bf16.pth`, `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, `Wan2.1_VAE.pth` | `TaoLiveAIGC/AptAvatar` @ `e5ae7ce74e1766693ed19473c3b51247351a0c78` | `system.json` | Apache-2.0 (the HF repository ships the same LICENSE as the code); Wan2.1 / InfiniteTalk base: Apache-2.0 |
| `chinese-wav2vec2-base/pytorch_model.bin` | `TencentGameMate/chinese-wav2vec2-base` @ `3991242c806928916fff4a8c0e4f76acf661b743` | `system.json` | MIT |

Code: Apache-2.0. Nothing is redistributed here.

## Limitations

- Lips lag speech by about one chunk (2.56 s) plus compute, because each chunk needs the audio of its whole 73-frame
  window.
- Only body, gaze and expression controls can be expressed (through the per-chunk text prompt, from the next chunk
  on); camera, scene, lighting, clothing, style and speech-content controls are not.
- Output is 512×512, not upstream's 704×1280.
