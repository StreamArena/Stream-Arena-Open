# StreamDiffusion (SD-Turbo img2img) — self-hosted streaming instruction editing

The original StreamDiffusion, a per-frame streaming img2img pipeline, as a classic baseline: SD-Turbo with TAESD,
configured as upstream's real-time img2img demo and served through the WebSocket protocol (`docs/protocol.md`)
and the shared edit loop `models/selfhost/edit_stream.py`.

| | |
|---|---|
| System ids | `streamdiffusion_sdturbo_instruction` (`instruction_edit`) |
| Upstream code | [cumulo-autumn/StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) @ `b623251dc055e1fd858d53509aa43e09dfc5cdc0` (2024-12-04) |
| Weights | stabilityai/sd-turbo @ `b261bac6`; madebyollin/taesd @ `614f7681` |
| Licence | code: Apache-2.0; weights: sd-turbo **Stability AI Community License** (commercial use restricted), TAESD MIT (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. The setup script reuses the base Python packages through `--system-site-packages`; it does not replace PyTorch.

```bash
bash models/selfhost/streamdiffusion/setup.sh src       # upstream python files at the pinned commit
bash models/selfhost/streamdiffusion/setup.sh env       # venv + diffusers 0.24.0, huggingface_hub 0.25.2, transformers 4.44.2, accelerate 0.34.2
bash models/selfhost/streamdiffusion/setup.sh weights   # sd-turbo, taesd
bash models/selfhost/streamdiffusion/setup.sh check     # imports the wrapper (GPU)
```

`setup.sh` (or `all`) runs the four in order.

| variable | default | meaning |
|---|---|---|
| `SD1_ROOT` | `$STREAM_ARENA_MODEL_ROOT/streamdiffusion` | this model's folder (setup.sh and server.py) |
| `SD1_COMMIT` | `b623251dc055e1fd858d53509aa43e09dfc5cdc0` | upstream commit |
| `HF_ENDPOINT`, `HF_TOKEN` | `https://huggingface.co`, unset | Hugging Face endpoint and token |
| `HF_HUB_OFFLINE` | `1` | set by server.py unless already set (local files only) |

## Weights and licences

`$SD1_ROOT/weights`. Full hashes are in `system.json` `weights.sha256` (also `ready.config.weights_sha256`).

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `sd-turbo/unet/diffusion_pytorch_model.safetensors` | stabilityai/sd-turbo @ `b261bac6fd2cf515557d5d0707481eafa0485ec2` | `438da6db…f0c40532` | Stability AI Community License |
| `sd-turbo/text_encoder/model.safetensors` | same | `67e01354…48959d15` | same |
| `sd-turbo/vae/diffusion_pytorch_model.safetensors` (loaded, replaced by TAESD) | same | `2aa1f430…25683ec` | same |
| `taesd/diffusion_pytorch_model.safetensors` | madebyollin/taesd @ `614f76814bbe30edbe2e627ace1c2234c81a2c0e` | `db169d69…25513549` | MIT |

sd-turbo at this revision ships the Stability AI Community License (`LICENSE.md`, July 2024): free for research and
non-commercial use; commercial use is free only for individuals and organisations with less than USD 1 million
annual revenue, above that it needs an enterprise licence from Stability AI. Code: Apache-2.0. Benchmark evaluation
only; nothing is redistributed.

## Limitations

- Instructions are used as generation prompts. The adapter does not add an instruction-following model.
- Source frames may be dropped when generation cannot keep up with the configured input rate.
- No reference-image editing route is registered.
- CLIP truncates instructions beyond 77 tokens; frames use a per-frame SD-Turbo pipeline.
