# StreamV2V — self-hosted instruction editing

StreamV2V (streaming video-to-video translation with feature banks, SD 1.5 + LCM-LoRA) configured as its upstream
live-camera demo and wrapped as a WebSocket protocol server (`docs/protocol.md`) on the shared streaming loop
`models/selfhost/edit_stream.py`.

| | |
|---|---|
| System ids | `streamv2v_sd15_instruction` (`instruction_edit`) |
| Upstream code | [Jeff-LiangF/streamv2v](https://github.com/Jeff-LiangF/streamv2v) @ `b35f3877e3bc21e960dfe2e4ca54eb5b15bc1650`; "Looking Backward: Streaming Video-to-Video Translation with Feature Banks" (ICLR 2025) |
| Weights | Jiali/stable-diffusion-1.5 @ `f94658e`, latent-consistency/lcm-lora-sdv1-5 @ `cf2fced`, madebyollin/taesd @ `614f768` |
| Licence | code: UT Austin Research License (non-commercial); weights: CreativeML OpenRAIL-M (SD 1.5), OpenRAIL++ (LCM-LoRA), MIT (TAESD) (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Install a matching TensorRT build. The setup script reuses the base Python packages through `--system-site-packages`; it does not replace PyTorch.

```
bash models/selfhost/streamv2v/setup.sh src       # upstream code at the pinned commit (network)
bash models/selfhost/streamv2v/setup.sh env       # venv (network, base torch)
bash models/selfhost/streamv2v/setup.sh weights   # HF weights at pinned revisions (network)
bash models/selfhost/streamv2v/setup.sh check     # import check (GPU)
```

Without an argument all four steps run in order. The `weights` step nulls the safety checker entry in the SD 1.5
`model_index.json` (the demo runs without it).

| variable | default | meaning |
|---|---|---|
| `SV2V_ROOT` | `$STREAM_ARENA_MODEL_ROOT/streamv2v` | this model's folder: `repo/`, `src_diffusers/`, `env/`, `weights/` (setup.sh and server.py) |
| `SV2V_COMMIT` | `b35f3877e3bc21e960dfe2e4ca54eb5b15bc1650` | upstream commit |

## Weights and licences

Files under `$SV2V_ROOT/weights`, downloaded from Hugging Face:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `stable-diffusion-1.5/unet/diffusion_pytorch_model.safetensors` | Jiali/stable-diffusion-1.5 @ `f94658e` | `19da7aaa…c8ffae9f1` | CreativeML OpenRAIL-M |
| `stable-diffusion-1.5/text_encoder/model.safetensors` | same | `d008943c…faad88334` | same |
| `lcm-lora-sdv1-5/pytorch_lora_weights.safetensors` | latent-consistency/lcm-lora-sdv1-5 @ `cf2fced` | `8f90d840…72333afe4` | OpenRAIL++ |
| `taesd/diffusion_pytorch_model.safetensors` | madebyollin/taesd @ `614f768` | `db169d69…25513549` | MIT |

Full hashes: `system.json` `weights.sha256`. Jiali/stable-diffusion-1.5 is a third-party copy of SD 1.5, used
because the upstream demo loads it.

The StreamV2V code is under the UT Austin Research License: free for academic, research, experimental or personal
use only; commercial use needs a separate licence from The University of Texas at Austin. The SD 1.5 and LCM-LoRA
weights carry the use restrictions of their OpenRAIL licences.

## Limitations

- Instructions are used as generation prompts. The adapter does not add an instruction-following model.
- Source frames may be dropped when generation cannot keep up with the configured input rate.
- No reference-image editing route is registered.
- The pipeline uses cached SDPA attention and temporal feature banks; style LoRAs are not loaded.
