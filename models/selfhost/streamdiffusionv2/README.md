# StreamDiffusionV2 (1.3B) — self-hosted streaming instruction editing

StreamDiffusionV2 is a prompt-conditioned streaming video-to-video model (a stylization model, not a local editor)
on Wan2.1-T2V-1.3B. The server runs upstream's single-GPU stream-batch pipeline behind the WebSocket protocol
(`docs/protocol.md`) and the shared edit loop `models/selfhost/edit_stream.py`.

| | |
|---|---|
| System ids | `streamdiffusionv2_1p3b_instruction` (`instruction_edit`) |
| Upstream code | [chenfengxu714/StreamDiffusionV2](https://github.com/chenfengxu714/StreamDiffusionV2) @ `6961a5cf2045d1dda05a04ef229698bdc04e873a` |
| Weights | jerryfeng/StreamDiffusionV2 @ `2373eb2b` (`wan_causal_dmd_v2v`); Wan-AI/Wan2.1-T2V-1.3B @ `37ec5126` |
| Licence | code: Apache-2.0; weights: Apache-2.0 (see Weights and licences) |

## Setup

Prerequisites: bash, curl, tar, [uv](https://docs.astral.sh/uv/), an NVIDIA driver supporting CUDA 12.8, ~30 GB of
disk. The venv is self-contained (uv CPython 3.10, torch 2.8.0+cu128 from download.pytorch.org); no system PyTorch.

```bash
bash models/selfhost/streamdiffusionv2/setup.sh src       # upstream tarball at the pinned commit
bash models/selfhost/streamdiffusionv2/setup.sh python    # uv CPython 3.10 + venv env/
bash models/selfhost/streamdiffusionv2/setup.sh env       # torch 2.8.0+cu128 and upstream deps
bash models/selfhost/streamdiffusionv2/setup.sh weights   # Wan2.1-T2V-1.3B, wan_causal_dmd_v2v, taew2_1.pth; linked into repo/
```

`setup.sh` (no argument) runs the four in this order. The `hf` CLI of the weights step is taken from `PATH` or
from `env/bin/hf` (installed by the env step).

| variable | default | meaning |
|---|---|---|
| `SDV2_ROOT` | `$STREAM_ARENA_MODEL_ROOT/streamdiffusionv2` | this model's folder (setup.sh and server.py) |
| `SDV2_COMMIT` | `6961a5cf2045d1dda05a04ef229698bdc04e873a` | upstream commit |
| `SDV2_HF_REV` | `2373eb2b39278b3a1aa174964a724ee78ead96f0` | jerryfeng/StreamDiffusionV2 revision |
| `WAN13_REV` | `37ec512624d61f7aa208f7ea8140a131f93afc9a` | Wan-AI/Wan2.1-T2V-1.3B revision |
| `UV_BIN`, `HF_BIN` | `uv`, `hf` on `PATH` (`hf`: else `env/bin/hf`) | executables |
| `HF_ENDPOINT`, `HF_TOKEN` | `https://huggingface.co`, unset | Hugging Face endpoint and token |

Other models reuse these weights: `editalive` links the umT5-XXL encoder and `krea_realtime` the Wan2.1-T2V-1.3B
files from `$STREAM_ARENA_MODEL_ROOT/streamdiffusionv2/weights` when present.

## Weights and licences

`$STREAM_ARENA_MODEL_ROOT/streamdiffusionv2/weights`, linked into `repo/{wan_models,ckpts}`. Full hashes are in
`system.json` `weights.sha256` and are reported by the server in `ready.config.weights_sha256`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `ckpts/wan_causal_dmd_v2v/model.pt` | jerryfeng/StreamDiffusionV2 @ `2373eb2b39278b3a1aa174964a724ee78ead96f0` | `7fab9eca…43a8930` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` | Wan-AI/Wan2.1-T2V-1.3B @ `37ec512624d61f7aa208f7ea8140a131f93afc9a` | `96b6b242…d957e8f` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | same | `7cace0da…745b746fd81d` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | same | `38071ab5…e442fd981` | Apache-2.0 |
| `ckpts/taew2_1.pth` (only with `--use_taehv`) | madebyollin/taehv @ `main` (unpinned) | — | MIT |

Weight revisions and file hashes are listed in `system.json` and `setup.sh`. Record resolved revisions and downloaded hashes for each installation.

## Limitations

- A stylization model: the instruction restyles the whole frame; it is not a local editor.
- Every instruction restarts the stream session (≈1.4 s), and earlier instructions are not kept.
- Text instructions only; no reference path.
