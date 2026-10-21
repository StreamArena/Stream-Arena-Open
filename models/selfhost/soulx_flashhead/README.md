# SoulX-FlashHead (Lite) — self-hosted composition renderer (digital human)

SoulX-FlashHead is a 1.3B streaming talking-head model driven by speech audio. `server.py` renders the speech of the
shared dialogue stack (ASR → LLM → TTS, run by the harness) with the `Model_Lite` variant and returns frames plus the
same speech as the avatar's audio.

| | |
|---|---|
| System ids | `soulx_flashhead_audio` (`digital_human_audio`), `soulx_flashhead_text` (`digital_human_text`) |
| Upstream code | [Soul-AILab/SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) @ `9bc03de06bb0de82cd6bc477804512ae06144bf2`; technical report arXiv 2602.07449 |
| Weights | `Soul-AILab/SoulX-FlashHead-1_3B` (`Model_Lite`, `VAE_LTX`; Hugging Face revision not pinned, sha256 in `system.json`), `facebook/wav2vec2-base-960h` @ `22aad52d435e` |
| Licence | code: Apache-2.0; weights: DiT Apache-2.0, `VAE_LTX` OpenRAIL-M, wav2vec2 Apache-2.0 (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation.

```
bash models/selfhost/soulx_flashhead/setup.sh fetch   # source at the pinned commit, weights, weights_sha256.txt
bash models/selfhost/soulx_flashhead/setup.sh env     # on the GPU machine: env/ (venv --system-site-packages)
```

| variable | default | meaning |
|---|---|---|
| `SOULX_FLASHHEAD_ROOT` | `$STREAMEVAL_MODEL_ROOT/soulx_flashhead` | this model's folder (setup.sh; server.py's `--repo` default). `--launch` always passes `--weights $STREAMEVAL_MODEL_ROOT/soulx_flashhead/weights` |
| `SOULX_FLASHHEAD_COMMIT` | `9bc03de06bb0de82cd6bc477804512ae06144bf2` | upstream commit |
| `PYTHON` | `python3` | base Python for `env` |

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/soulx_flashhead/weights`; sha256 also in `weights_sha256.txt` next to it.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `SoulX-FlashHead-1_3B/Model_Lite/diffusion_pytorch_model.safetensors` | `Soul-AILab/SoulX-FlashHead-1_3B` (revision not pinned; record the downloaded revision) | `aaf1cde6…ccdf6e` | Apache-2.0 |
| `SoulX-FlashHead-1_3B/VAE_LTX/diffusion_pytorch_model.safetensors` | same | `265ca87c…e2010` | OpenRAIL-M (LTX-Video 0.9 VAE) |
| `wav2vec2-base-960h/model.safetensors` | `facebook/wav2vec2-base-960h` @ `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | `8aa76ab2…17d74a` | Apache-2.0 |

Code: Apache-2.0. The model card of `SoulX-FlashHead-1_3B` lists component licences: the DiT weights and the Wan2.1
VAE are Apache-2.0, the LTX-Video 0.9 VAE (`VAE_LTX`) is OpenRAIL-M; the Lite variant used here requires
complying with the OpenRAIL-M terms for that component.

## Limitations

- Renderer only: what the avatar says comes from the shared dialogue stack; text events do not change the
  rendering.
- The portrait is not face-cropped by default; a portrait whose face is small in the frame is rendered as given.
- The `SoulX-FlashHead-1_3B` Hugging Face revision is not pinned; compare the downloaded files with the
  sha256 in `system.json`.
