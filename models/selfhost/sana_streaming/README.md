# SANA-Streaming — self-hosted streaming instruction editing

SANA-Streaming (NVlabs) is a streaming DiT for instruction-conditioned video-to-video editing on the LTX-2 VAE.
Upstream publishes the model and an offline `long_streaming` script, not a live server; `server.py` runs the same
components incrementally on the pushed source stream, behind the WebSocket protocol (`docs/protocol.md`) and the
shared edit loop `models/selfhost/edit_stream.py`.

| | |
|---|---|
| System ids | `sana_streaming_2b_instruction` (`instruction_edit`) |
| Upstream code | [NVlabs/Sana](https://github.com/NVlabs/Sana) @ `f9178744c096dcf2a2ea773da183e341bcbeb044` (upstream notes: `docs/sana_streaming.md` in that repository) |
| Weights | Efficient-Large-Model/SANA-Streaming @ `616778bc`; Lightricks/LTX-2 @ `dfcc2108` (VAE); Efficient-Large-Model/gemma-2-2b-it @ `569d9809` |
| Licence | code: Apache-2.0; weights: Apache-2.0 (SANA-Streaming), LTX-2 Community License (VAE), Gemma Terms of Use (text encoder) (see Weights and licences) |

## Setup

Prerequisites: bash, curl, tar, [uv](https://docs.astral.sh/uv/), an NVIDIA driver supporting CUDA 12.8, ~25 GB of
disk. The venv is self-contained (uv CPython 3.12, torch 2.9.1+cu128 from download.pytorch.org); no system PyTorch.

```bash
bash models/selfhost/sana_streaming/setup.sh src       # NVlabs/Sana tarball at the pinned commit
bash models/selfhost/sana_streaming/setup.sh python    # uv CPython 3.12 + venv env/
bash models/selfhost/sana_streaming/setup.sh env       # torch 2.9.1+cu128, inference subset of upstream deps, mmcv 1.7.2 without ops
bash models/selfhost/sana_streaming/setup.sh weights   # SANA-Streaming, LTX-2 VAE, gemma-2-2b-it
```

`setup.sh` (no argument) runs the four in this order. The `hf` CLI is taken from `PATH` or from `env/bin/hf`.

| variable | default | meaning |
|---|---|---|
| `SANA_ROOT` | `$STREAMEVAL_MODEL_ROOT/sana_streaming` | this model's folder (setup.sh and server.py) |
| `SANA_COMMIT` | `f9178744c096dcf2a2ea773da183e341bcbeb044` | NVlabs/Sana commit |
| `SANA_STREAMING_REV` | `616778bcee475e314177bb4bd81be2b50af69b26` | Efficient-Large-Model/SANA-Streaming revision |
| `LTX2_REV` | `dfcc2108383fe1aaa0584bdf55d368a4bdadd90c` | Lightricks/LTX-2 revision |
| `GEMMA_REV` | `569d9809d0c8b6722d4d31b5a77a2ec7a400650a` | Efficient-Large-Model/gemma-2-2b-it revision |
| `UV_BIN`, `HF_BIN` | `uv`, `hf` on `PATH` (`hf`: else `env/bin/hf`) | executables |
| `HF_ENDPOINT`, `HF_TOKEN` | `https://huggingface.co`, unset | Hugging Face endpoint and token |
| `USE_CHUNKWISE_GDN`, `HF_HUB_OFFLINE` | `1`, `1` | set by server.py unless already set (`DISABLE_XFORMERS=1` and `TOKENIZERS_PARALLELISM=false` are always set) |

## Weights and licences

`$STREAMEVAL_MODEL_ROOT/sana_streaming/weights`. Full hashes are in `system.json` `weights.sha256` and are
reported by the server in `ready.config.weights_sha256`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `SANA-Streaming/dit/sana_streaming_ar.pth` | Efficient-Large-Model/SANA-Streaming @ `616778bcee475e314177bb4bd81be2b50af69b26` | `971839a0…fc5fa` | Apache-2.0 |
| `SANA-Streaming/ltx2_causal_vae_0516/diffusion_pytorch_model.safetensors` | same | `3694526c…7cc4a` | Apache-2.0 |
| `LTX-2/vae/diffusion_pytorch_model.safetensors` | Lightricks/LTX-2 @ `dfcc2108383fe1aaa0584bdf55d368a4bdadd90c` | `107cc359…14bfb` | LTX-2 Community License |
| `gemma-2-2b-it/model-0000{1,2}-of-00002.safetensors` | Efficient-Large-Model/gemma-2-2b-it @ `569d9809d0c8b6722d4d31b5a77a2ec7a400650a` | `532d792c…1bf`, `6d6d9ce8…20` | Gemma Terms of Use |

The LTX-2 Community License requires entities with annual revenue of at least USD 10 million to obtain a paid
commercial licence from Lightricks. gemma-2-2b-it is taken from the SANA authors' own Hugging Face organisation
(as upstream's documentation specifies); Google's Gemma Terms of Use apply to it. Code: Apache-2.0. Benchmark
evaluation only; nothing is redistributed.

## Limitations

- An instruction takes effect only at the next 24-frame chunk boundary.
- BF16 only; the upstream quantized (MPQ) deployment path is not used.
- Text instructions only; no reference path.
