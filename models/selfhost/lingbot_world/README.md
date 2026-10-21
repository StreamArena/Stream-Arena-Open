# LingBot-World 2.0 1.3B causal-fast — self-hosted image + text interaction

LingBot-World 2.0 (LingBot-World-Infinity) is Robbyant's image-to-video world model on Wan 1.3B with a camera-pose
input, distilled to 4 steps. `server.py` runs upstream's causal-fast rollout live, chunk by chunk, behind the
WebSocket protocol (`docs/protocol.md`), using the shared live loop `models/selfhost/t2v_stream.py` with its session
subclassed here for camera events.

| | |
|---|---|
| System ids | `lingbot_world_v2_1p3b_i2v` (`general_interaction_image`) |
| Upstream code | [Robbyant/lingbot-world-v2](https://github.com/Robbyant/lingbot-world-v2) @ `1895d300d8ac936401689b26389f51cbd36530eb` |
| Weights | `robbyant/lingbot-world-v2-1.3b-causal-fast` @ `7e36a5f919f86cb4255cc9bfc30adb44963fbde1` (DiT shards); Wan2.1 umT5-XXL encoder, VAE and tokenizer from `Wan-AI/Wan2.1-T2V-1.3B` |
| Licence | code: CC BY-NC-SA 4.0; weights: CC BY-NC-SA 4.0, Wan2.1 encoder and VAE Apache-2.0 — **non-commercial** (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. The setup script reuses the base Python packages through `--system-site-packages`; it does not replace PyTorch.

```
bash models/selfhost/lingbot_world/setup.sh src       # network: upstream code at the pinned commit
bash models/selfhost/lingbot_world/setup.sh env       # network: builds the longlive venv, links env/ to it
bash models/selfhost/lingbot_world/setup.sh weights   # network: Hugging Face downloads
bash models/selfhost/lingbot_world/setup.sh check     # GPU: imports the upstream pipeline
```

Wan2.1 weights are downloaded at the pinned revision.

| variable | default | meaning |
|---|---|---|
| `LINGBOT_WORLD_ROOT` | `$STREAMEVAL_MODEL_ROOT/lingbot_world` | this model's folder (setup.sh and server.py) |
| `LINGBOT_WORLD_COMMIT` | `1895d300d8ac936401689b26389f51cbd36530eb` | upstream commit fetched by `src` |
| `LONGLIVE_ROOT` | `$STREAMEVAL_MODEL_ROOT/longlive` | folder of the shared venv |

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/lingbot_world/weights`; full sha256 values are in `system.json`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `lingbot-world-v2-1.3b-causal-fast/model-0000{1..6}-of-00006.safetensors` | robbyant/lingbot-world-v2-1.3b-causal-fast @ `7e36a5f9` | see `system.json` | CC BY-NC-SA 4.0 |
| `Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI/Wan2.1-T2V-1.3B (no revision pinned; identified by sha256) | `7cace0da…6fd81d` | Apache-2.0 |
| `Wan2.1-T2V-1.3B/Wan2.1_VAE.pth` | same | `38071ab5…442fd981` | Apache-2.0 |
| `Wan2.1-T2V-1.3B/google/umt5-xxl/` (tokenizer) | same | — | Apache-2.0 |

The upstream code and the LingBot-World weights are licensed CC BY-NC-SA 4.0: **non-commercial use only**, with
attribution, and derivative works must be shared under the same licence. Benchmark evaluation only; nothing is
redistributed.

## Limitations

- Image-to-video only: a session without an opening image fails, so the text-only track is not registered.
- Mid-session reference images have no model input (`reference_update` unsupported).
- 300 s sessions use RoPE positions past upstream's 1024-latent table (extended with the same formula), beyond what
  the model was trained on.
