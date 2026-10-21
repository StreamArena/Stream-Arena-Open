# ShotStream (1.3B) — self-hosted text-to-video interaction

ShotStream (ECCV 2026, arXiv 2603.25746; CUHK MMLab + Kling Team, Kuaishou) is a causal multi-shot Wan2.1-T2V-1.3B
that generates a story shot by shot. The server streams upstream's multi-shot rollout block by block over the
WebSocket protocol ([docs/protocol.md](../../../docs/protocol.md)) and maps each instruction to a new shot, using the
shared live loop of `models/selfhost/t2v_stream.py`.

| | |
|---|---|
| System ids | `shotstream_1p3b_t2v` (`general_interaction`) |
| Upstream code | [KlingAIResearch/ShotStream](https://github.com/KlingAIResearch/ShotStream) @ `c6a11719ee8ac203ee20afc27056446fe25cef86`; arXiv 2603.25746 |
| Weights | `KlingTeam/ShotStream` @ `225806ba` (`shotstream_merged.pt`, `shotstream.yaml`, `default_config.yaml`); `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec5126` |
| Licence | code: Apache-2.0; weights: Apache-2.0 (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model.

```bash
bash models/selfhost/shotstream/setup.sh src       # upstream code at the pinned commit (network; large tarball)
bash models/selfhost/shotstream/setup.sh env       # shared venv of longlive, env -> $LONGLIVE_ROOT/env
bash models/selfhost/shotstream/setup.sh weights   # checkpoint + configs + Wan2.1-T2V-1.3B, about 23 GB (network)
bash models/selfhost/shotstream/setup.sh check     # imports the upstream pipeline with the shims (GPU)
```

The upstream repository tarball is large (demo data), so `setup.sh src` can take many minutes.

| variable | default | meaning |
|---|---|---|
| `SHOTSTREAM_ROOT` | `$STREAMEVAL_MODEL_ROOT/shotstream` | this model's root (read by `setup.sh` and `server.py`) |
| `SHOTSTREAM_COMMIT` | `c6a11719ee8ac203ee20afc27056446fe25cef86` | upstream commit downloaded by `setup.sh src` |
| `SHOTSTREAM_HF_REV` | `225806baab3c54e94351474e976948eb5c61b6fe` | KlingTeam/ShotStream revision |
| `LONGLIVE_ROOT` | `$STREAMEVAL_MODEL_ROOT/longlive` | root of the shared environment |
| `WAN21_REV` | `37ec5126…` | Wan-AI/Wan2.1-T2V-1.3B revision to download |

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/shotstream/weights`, downloaded from Hugging Face:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `ShotStream/shotstream_merged.pt` (5.7 GB) | KlingTeam/ShotStream @ `225806ba` | `91ee3403…8d2b50` | Apache-2.0 |
| `ShotStream/{shotstream,default_config}.yaml` | same | | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI/Wan2.1-T2V-1.3B @37ec5126 | `7cace0da…fd81d` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | same | `38071ab5…fd981` | Apache-2.0 |

Weight revisions and file hashes are listed in `system.json` and `setup.sh`. Record resolved revisions and downloaded hashes for each installation.

## Limitations

- Every new shot is a cut: an instruction, and every 5 s without one, starts a new shot, so the stream is not one
  continuous take.
- Sessions longer than about 30 s have more shots than upstream's demos (at most 6); the context is drawn from the
  last 5 shots and re-indexed to stay inside that range.
- Text-to-video only; no camera input (`camera_control` text is the shot caption).
