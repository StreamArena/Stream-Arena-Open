# MemFlow (1.3B) — self-hosted text-to-video interaction

MemFlow (HKU / Kling Team, arXiv 2512.14699) is a LongLive-style causal, 4-step Wan2.1-T2V-1.3B with a
prompt-retrieved memory bank and an interactive prompt switch. The server runs the released checkpoint block by block
over the WebSocket protocol ([docs/protocol.md](../../../docs/protocol.md)), with the shared live loop of
`models/selfhost/t2v_stream.py`.

| | |
|---|---|
| System ids | `memflow_1p3b_t2v` (`general_interaction`) |
| Upstream code | [KlingAIResearch/MemFlow](https://github.com/KlingAIResearch/MemFlow) @ `7ed514771f5add2af7cc60889e8310ce2b70ab1e`; arXiv 2512.14699 |
| Weights | `KlingTeam/MemFlow` @ `c41043fd` (`base.pt`, `lora.pt`); `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec5126` |
| Licence | code: Apache-2.0, with LongLive-derived files under CC BY-NC-SA 4.0 headers; weights: no licence stated (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model.

```bash
bash models/selfhost/memflow/setup.sh src       # upstream code at the pinned commit (network)
bash models/selfhost/memflow/setup.sh env       # shared venv of longlive, env -> $LONGLIVE_ROOT/env
bash models/selfhost/memflow/setup.sh weights   # base.pt + lora.pt + Wan2.1-T2V-1.3B, about 26 GB (network)
bash models/selfhost/memflow/setup.sh check     # imports the upstream pipeline (GPU)
```

| variable | default | meaning |
|---|---|---|
| `MEMFLOW_ROOT` | `$STREAMEVAL_MODEL_ROOT/memflow` | this model's root (read by `setup.sh` and `server.py`) |
| `MEMFLOW_COMMIT` | `7ed514771f5add2af7cc60889e8310ce2b70ab1e` | upstream commit downloaded by `setup.sh src` |
| `MEMFLOW_HF_REV` | `c41043fd7df4576e62603d3eb40b25aaa5542022` | KlingTeam/MemFlow revision |
| `LONGLIVE_ROOT` | `$STREAMEVAL_MODEL_ROOT/longlive` | root of the shared environment |
| `WAN21_REV` | `37ec5126…` | Wan-AI/Wan2.1-T2V-1.3B revision to download |

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/memflow/weights`, downloaded from Hugging Face:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `MemFlow/base.pt` (5.7 GB) | KlingTeam/MemFlow @ `c41043fd` | `10a2aa8f…208ee7` | not stated on the model card |
| `MemFlow/lora.pt` (2.8 GB) | same | `1dec690a…5f781a` | same |
| `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI/Wan2.1-T2V-1.3B @37ec5126 | `7cace0da…fd81d` | Apache-2.0 |
| `wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | same | `38071ab5…fd981` | Apache-2.0 |

Weight revisions and file hashes are listed in `system.json` and `setup.sh`. Record resolved revisions and downloaded hashes for each installation.

Code: Apache-2.0 repository licence; source files derived from LongLive carry CC BY-NC-SA 4.0 headers
(non-commercial, share-alike). The MemFlow weights have no stated licence, so no usage rights beyond what the
authors grant are implied.

## Limitations

- Text-to-video only; no camera input (`camera_control` text is used as prompt text).
- 300 s sessions go past upstream's 1024-position RoPE table (extended with the same formula).
