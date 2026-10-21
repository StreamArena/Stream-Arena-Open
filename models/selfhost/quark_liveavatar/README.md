# Quark Live Avatar 14B — self-hosted avatar performance and composition renderer (digital human)

Live Avatar (Alibaba Quark / USTC, "Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite
Length", arXiv 2512.04677) generates an audio-driven avatar video from one portrait, streaming speech audio and a
text prompt. `server.py` drives the upstream model classes block by block over 5 GPUs (upstream's timestep-forcing
pipeline parallelism) and serves it natively on `avatar_performance` and as a renderer for the shared dialogue stack
on the digital-human tracks.

Not to be confused with HeyGen's hosted product "LiveAvatar" (`liveavatar_full` on the board); this system is named
**Quark Live Avatar 14B** everywhere.

| | |
|---|---|
| System ids | `quark_liveavatar_performance` (`avatar_performance`), `quark_liveavatar_comp_audio` (`digital_human_audio`), `quark_liveavatar_comp_text` (`digital_human_text`) |
| Upstream code | [Alibaba-Quark/LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar) @ `c3c47d031d8bf3247428333c1fe610579c71a551` (2026-08-24); arXiv 2512.04677 |
| Weights | `Wan-AI/Wan2.2-S2V-14B` @ `dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b` (14B DiT, umT5-XXL, Wan2.1 VAE, wav2vec2-large-xlsr-53-english); `Quark-Vision/Live-Avatar` @ `92cdccd12a91e8a63767a7c821b7c75e51d5a172` (`liveavatar.safetensors`, rank-128 LoRA, 4-step DMD distillation, merged at load) |
| Licence | code: Apache-2.0; weights: Apache-2.0 (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Use a compatible FlashAttention build or the adapter's documented SDPA implementation.

```
bash models/selfhost/quark_liveavatar/setup.sh repo      # source at the pinned commit
bash models/selfhost/quark_liveavatar/setup.sh env       # uv venv
bash models/selfhost/quark_liveavatar/setup.sh weights   # 35 GB, pinned Hugging Face revisions
bash models/selfhost/quark_liveavatar/setup.sh hash      # weights_hash.json
```

`setup.sh` (no argument) runs all four. If `models/selfhost/soulx_liveact`'s weights are already under the model
root, the byte-identical umT5-XXL encoder and Wan2.1 VAE are copied from there instead of downloaded.

| variable | default | meaning |
|---|---|---|
| `QUARK_LIVEAVATAR_ROOT` | `$STREAMEVAL_MODEL_ROOT/quark_liveavatar` | this model's folder (setup.sh and server.py defaults: weights, `repo_commit.txt`, `weights_hash.json`, inductor / triton caches, session logs) |
| `QUARK_LIVEAVATAR_COMMIT` | `c3c47d031d8bf3247428333c1fe610579c71a551` | upstream commit (setup.sh) |
| `LIVEAVATAR_REPO` | `<model folder>/repo` | upstream source tree imported by server.py |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | `<model folder>/inductor_cache`, `<model folder>/triton_cache` | compile caches (server.py sets them if unset) |

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/quark_liveavatar/weights`; `setup.sh hash` re-hashes every loaded file into
`weights_hash.json` (sent in `ready.config`). The sha256 in `system.json` equal the Hugging Face LFS object ids.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `Wan2.2-S2V-14B/diffusion_pytorch_model-0000{1..4}-of-00004.safetensors` (32.6 GB) | `Wan-AI/Wan2.2-S2V-14B` @ `dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b` | see `system.json` | Apache-2.0 |
| `Wan2.2-S2V-14B/models_t5_umt5-xxl-enc-bf16.pth` (11.4 GB) | same | `7cace0da…fd81d` | Apache-2.0 |
| `Wan2.2-S2V-14B/Wan2.1_VAE.pth` | same | `38071ab5…fd981` | Apache-2.0 |
| `Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors` | same | `6144f846…19875` | Apache-2.0 (as part of the Wan2.2-S2V-14B repository) |
| `LiveAvatar/liveavatar.safetensors` (1.35 GB) | `Quark-Vision/Live-Avatar` @ `92cdccd12a91e8a63767a7c821b7c75e51d5a172` | `fc96b161…448f9` | Apache-2.0 |

Code: Apache-2.0 (`LICENSE` in the repository). The `Quark-Vision/Live-Avatar` and `Wan-AI/Wan2.2-S2V-14B` model cards
declare Apache-2.0. Upstream's README describes the project as a research preview.

## Other layouts

Besides the registered 5-GPU layout, `torchrun --nproc_per_node=3 server.py --dit_ranks 2 …` (2 steps per DiT rank
+ 1 VAE rank) and plain `python server.py` (one GPU; about 0.9 s per 0.48 s block, not real time) also work.

## Limitations

- The registered pipeline uses five GPUs; adjust the launch configuration only with a separately recorded system configuration.
- The text prompt steers upper-body motion and expression only; camera, setting, clothing and speech content cannot
  be controlled (acked `unsupported`). Applied controls are passed as text: no explicit per-part state, no
  guaranteed left/right grounding.
- Renderer only on the digital-human tracks: what the avatar says comes from the shared dialogue stack, and dialogue
  text events are acked `unsupported`.
- Deviations from upstream: audio features are computed per block from a 4 s left window plus 0.1 s look-ahead
  instead of over the whole file; 512×512 instead of 720×400; bf16 instead of FP8; FlashAttention 3 not used.
