# Krea Realtime 14B — self-hosted text / image + text interaction

Krea Realtime 14B is a Self-Forcing-distilled Wan2.1-T2V-14B that generates causal 3-latent blocks live with
prompt changes. `server.py` imports Krea's own streaming server code (`release_server.py`) unchanged and drives its
`GenerationSession` block by block, behind the WebSocket protocol (`docs/protocol.md`) and the shared live loop
`models/selfhost/t2v_stream.py`.

| | |
|---|---|
| System ids | `krea_realtime_14b_t2v` (`general_interaction`), `krea_realtime_14b_i2v` (`general_interaction_image`) |
| Upstream code | [krea-ai/realtime-video](https://github.com/krea-ai/realtime-video) @ `acf4b7b4e1a3049a0e04782e751d12203334038d` (last push 2025-11-13) |
| Weights | [krea/krea-realtime-video](https://huggingface.co/krea/krea-realtime-video) @ `9a2ceea2` (`krea-realtime-video-14b.safetensors`, 14.3 B params, fp16, ungated); Wan-AI/Wan2.1-T2V-1.3B @ `37ec5126` (text encoder, VAE); Wan-AI/Wan2.1-T2V-14B `config.json` |
| Licence | code: **CC BY-NC-SA 4.0 (non-commercial)**; weights: Apache-2.0 (see Weights and licences) |


## Setup

Prerequisites: bash, curl, tar, [uv](https://docs.astral.sh/uv/), an NVIDIA driver supporting CUDA 13, ~45 GB of
disk. The environment is self-contained (uv CPython 3.11, torch 2.10.0+cu130 from download.pytorch.org); no
system PyTorch is used.

```bash
bash models/selfhost/krea_realtime/setup.sh repo      # krea-ai/realtime-video tarball at the pinned commit
bash models/selfhost/krea_realtime/setup.sh env       # uv CPython 3.11 + venv: torch 2.10.0+cu130, upstream inference pins, cuDNN 9.26
bash models/selfhost/krea_realtime/setup.sh weights   # Krea checkpoint, Wan2.1-T2V-1.3B files, umT5 .pth -> .safetensors, Wan2.1-T2V-14B config.json
bash models/selfhost/krea_realtime/setup.sh check     # imports upstream release_server (GPU)
```

`setup.sh` (or `all`) runs the four in order. Wan2.1 components are downloaded at the pinned revision. The umT5 `.pth` is converted to the `.safetensors` name upstream loads.

| variable | default | meaning |
|---|---|---|
| `KREA_ROOT` | `$STREAMEVAL_MODEL_ROOT/krea_realtime` | this model's folder (setup.sh and server.py) |
| `WAN13_REV` | `37ec512624d61f7aa208f7ea8140a131f93afc9a` | Wan-AI/Wan2.1-T2V-1.3B revision |
| `WAN14_REV` | `main` | Wan-AI/Wan2.1-T2V-14B revision of `config.json` |
| `HF_ENDPOINT`, `HF_TOKEN` | `https://huggingface.co`, unset | Hugging Face endpoint and token |
| `DO_COMPILE`, `DISABLE_SAGEATTENTION` | `true` (`false` with `--no-compile`), `1` | upstream switches, set by server.py unless already set |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `PYTORCH_CUDA_ALLOC_CONF` | `$KREA_ROOT/inductor_cache`, `$KREA_ROOT/triton_cache`, `expandable_segments:True` | set by server.py unless already set |

## Weights and licences

`$KREA_ROOT/weights`. The sha256 of the used files is recorded per session in `ready.system_config.weights_sha256`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `checkpoints/krea-realtime-video-14b.safetensors` (28.6 GB) | krea/krea-realtime-video @ `9a2ceea280e5d584c6a28d384a9c19e38c810f8c` | `792f6042a645e227f5d71851845dc4098a7338fec9ec55279f88ebbc3527eec7` | Apache-2.0 (model card) |
| `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | Wan-AI/Wan2.1-T2V-1.3B @ `37ec512624d61f7aa208f7ea8140a131f93afc9a` | `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` (+ converted `.safetensors`), `google/` tokenizer | same | `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-14B/config.json` | Wan-AI/Wan2.1-T2V-14B @ `main` (v1 revision not pinned) | — | Apache-2.0 |

The code (krea-ai/realtime-video, `LICENSE.md`) is licensed **CC BY-NC-SA 4.0: non-commercial use only**, share-alike.
The weights are Apache-2.0 according to the Hugging Face model card. The two licences differ; anyone running this
server runs Krea's code and is bound by the non-commercial terms. Benchmark evaluation only; nothing is
redistributed.

## Limitations

- No mid-session image input: reference updates on the image track are unsupported and count as failures.
- Prompt changes are blended over 4 blocks, so an instruction takes full effect about 4 blocks (~3 s) after the ack.
- Fixed 832×480 output.
