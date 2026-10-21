# HY-World 1.5 WorldPlay — self-hosted image-to-video interaction

Tencent's real-time interactive world model (WorldPlay: a HunyuanVideo-1.5 8B DiT with action/camera conditioning,
autoregressive and distilled to 4 steps), served live chunk by chunk through the WebSocket protocol
(`docs/protocol.md`) on one GPU, or with the DiT sequence-parallel over two GPUs.

| | |
|---|---|
| System ids | `hy_worldplay_i2v` (`general_interaction_image`) |
| Upstream code | [Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) @ `1588e1336e842b03b0a7860c654ebd7c46bb065e` |
| Weights | `tencent/HY-WorldPlay` @ `f4c29235` (`ar_distilled_action_model/model.safetensors`, HY-World1.5-Autoregressive-480P-I2V-distill) + HunyuanVideo-1.5, Qwen2.5-VL-7B-Instruct, byt5-small, Glyph-SDXL-v2, FLUX.1-Redux-dev image encoder (about 58 GB in total) |
| Licence | code and weights: Tencent HY-WorldPlay Community License (not licensed in the EU, UK and South Korea); SigLIP image encoder: FLUX.1 [dev] Non-Commercial License (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model.

```bash
bash models/selfhost/hy_worldplay/setup.sh download   # network: code at the pinned commit + about 58 GB of weights
bash models/selfhost/hy_worldplay/setup.sh env        # on the GPU machine: venv at $STREAMEVAL_MODEL_ROOT/hy_worldplay/env
```

`env` installs diffusers 0.35.0, transformers 4.56.0, peft 0.17.0 and the other inference dependencies from pip.
flash-attn is optional (torch SDPA fallback). An unknown step prints the usage and exits with status 2.

| variable | default | meaning |
|---|---|---|
| `HYWP_ROOT` | `$STREAMEVAL_MODEL_ROOT/hy_worldplay` | setup.sh only; the harness launch uses `<root>/hy_worldplay`, so with another folder pass `--weights` and `HYWP_REPO` to the server yourself |
| `HYWP_COMMIT` | `1588e1336e842b03b0a7860c654ebd7c46bb065e` | upstream commit checked out by setup.sh |
| `HYWP_REPO` | `$STREAMEVAL_MODEL_ROOT/hy_worldplay/repo` | server.py: where the upstream code is imported from |
| `HF_TOKEN` | — | Hugging Face token; required for the gated FLUX.1-Redux-dev files |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` (set by server.py if unset) | PyTorch allocator setting |
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | set by `torch.distributed.run`; single GPU: 0, 0, 1, 127.0.0.1, a free port | server.py process group |

## Weights and licences

Downloaded by `setup.sh download` into `$STREAMEVAL_MODEL_ROOT/hy_worldplay/weights`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `HY-WorldPlay/ar_distilled_action_model/model.safetensors` (34 GB) | `tencent/HY-WorldPlay` @ `f4c29235647707b571479a69b569e4166f9f5bf8` | `2642ca972e49355e1a00d18dbc9847a2d8c1e707a6a9abd0994d4949258cc963` (= HF LFS oid) | Tencent HY-WorldPlay Community License |
| `HunyuanVideo-1.5/{config.json,vae,scheduler,transformer/480p_i2v/config.json}` | `tencent/HunyuanVideo-1.5` @ `9b49404b3f5df2a8f0b31df27a0c7ab872e7b038` | — | Tencent Hunyuan Community License |
| `HunyuanVideo-1.5/text_encoder/llm` | `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5` | — | Apache-2.0 |
| `HunyuanVideo-1.5/text_encoder/byt5-small` | `google/byt5-small` @ `68377bdc18a2ffec8a0533fef03b1c513a4dd49d` | — | Apache-2.0 |
| `HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2/{assets/*.json,checkpoints/byt5_model.pt}` | ModelScope `AI-ModelScope/Glyph-SDXL-v2` (the source upstream's README names; single `master` branch, not pinned) | — | not stated on the ModelScope page |
| `HunyuanVideo-1.5/vision_encoder/siglip/{image_encoder,feature_extractor}` | `black-forest-labs/FLUX.1-Redux-dev` (gated; not pinned, latest revision) | — | FLUX.1 [dev] Non-Commercial License |

Licence terms stated plainly:

- **Territory.** The Tencent HY-WorldPlay Community License (code and the action checkpoint) and the Tencent Hunyuan
  Community License (HunyuanVideo-1.5) do not apply in the European Union, the United Kingdom and South Korea: use
  there is not licensed. Elsewhere they allow use within their terms, including an acceptable-use policy, a ban on
  using the outputs to improve other AI models, and a separate licence from Tencent above 1 million monthly active
  users.
- **Non-commercial.** The FLUX.1-Redux-dev image encoder is under the FLUX.1 [dev] Non-Commercial License, so the
  system as wrapped here is for non-commercial use only. The licence must be accepted on Hugging Face before the
  download works.

Nothing is redistributed by this repository.

## Other layouts

Run under `torch.distributed.run --standalone --nproc_per_node 2`, the server uses upstream's 2-GPU sequence
parallelism.

## Limitations

- Image-to-video only: the released checkpoint is 480p I2V; a text-only start is rejected (or degenerates with
  `--t2v`), so the t2v track is not registered.
- Camera directions are mapped by keywords onto translation / yaw / pitch; phrases outside that space (crane,
  pedestal, "return to front view") are acked `unsupported`. Zoom is approximated by moving forward.
- A reference image changes only the SigLIP semantic condition and the prompt; the pixel anchor stays the opening image.
- `reference` ops other than `set` are not supported.
- The rollout is capped by `--max-chunks` (default 60 chunks = 40 s of video); at ~7.7 s per chunk the cap is not reached
  within the 300 s sessions on one GPU.
