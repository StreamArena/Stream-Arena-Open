# AlayaWorld — self-hosted image-to-video interaction

AlayaWorld (AlayaLab) is an interactive long-horizon world model built on LTX-2.3. The server drives upstream's own
interactive adapter (`reactor/alayaworld.py`, the model class behind AlayaLab's browser demo) chunk by chunk over the
benchmark protocol; the shared live loop is `models/selfhost/t2v_stream.py`.

| | |
|---|---|
| System ids | `alayaworld_i2v` (`general_interaction_image`) |
| Upstream code | [AlayaLab/AlayaWorld](https://github.com/AlayaLab/AlayaWorld) @ `7e7dcfe5b82419d665ffdd458815553e459a8a39` (main on 2026-10-04; contains `reactor/`, the interactive serving adapter released 2026-08-20) |
| Weights | [AlayaLab/AlayaWorld](https://huggingface.co/AlayaLab/AlayaWorld) `merged_infer.safetensors` @ `f9699009`; Gemma 3 12B text encoder; Depth-Anything-3 DA3NESTED-GIANT-LARGE-1.1; TAEHV tiny decoder — all at the revisions pinned in upstream `reactor/alayaworld.yaml` |
| Licence | code and model weights: LTX-2 Community License Agreement; Gemma: Gemma Terms of Use; DA3NESTED weights: CC BY-NC 4.0, **non-commercial** (see Weights and licences) |

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/), git, curl, an NVIDIA driver that supports CUDA 13 (the env step
installs its own torch 2.10.0+cu130), about 70 GB of disk for the weights, and a Hugging Face account that has
accepted the Gemma licence on [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
(set `HF_TOKEN`).

```
bash models/selfhost/alayaworld/setup.sh repo      # upstream code at the pinned commit
bash models/selfhost/alayaworld/setup.sh env       # uv Python 3.12 + venv
bash models/selfhost/alayaworld/setup.sh weights   # assets at the revisions pinned in reactor/alayaworld.yaml
bash models/selfhost/alayaworld/setup.sh check     # GPU: import the adapter
```

`bash setup.sh` (or `all`) runs the four steps in order. `weights` downloads the merged checkpoint from Hugging Face,
fetches the pinned TAEHV commit with one shallow fetch, and lets the adapter's own `prepare_runtime_assets` fetch and
verify the rest (Gemma, Depth-Anything-3 source and weights) at their pinned revisions.

| variable | default | meaning |
|---|---|---|
| `ALAYA_ROOT` | `$STREAMEVAL_MODEL_ROOT/alayaworld` | this model's folder (setup.sh and server.py) |
| `ALAYA_COMMIT` | `7e7dcfe5b82419d665ffdd458815553e459a8a39` | upstream commit |
| `HF_TOKEN` | — | Hugging Face token (Gemma is gated) |

`server.py` sets these if they are not already set: `REACTOR_WEIGHTS_PATH` (`$ALAYA_ROOT/checkpoints`, overridden by
`--weights`), `PYTORCH_ALLOC_CONF=expandable_segments:True`, `HF_HUB_OFFLINE=1` (the assets are prepared by
`setup.sh`), `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` (`$ALAYA_ROOT/inductor_cache`, `triton_cache`).

## Weights and licences

Files go to `$ALAYA_ROOT/checkpoints`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `merged_infer.safetensors` (33.6 GB: AR + DMD DiT on LTX-2.3 22B, VAE, history encoder) | AlayaLab/AlayaWorld @ `f9699009a00f93214d8d2942d6546f3fa5398bea` (ungated) | recorded per session in `ready.system_config.weights_sha256` | LTX-2 Community License Agreement |
| `gemma-3-12b-it-qat-q4_0-unquantized/` (text encoder) | google/gemma-3-12b-it-qat-q4_0-unquantized @ `68f7ee4fbd59087436ada77ed2d62f373fdd4482` (gated) | — | Gemma Terms of Use |
| `hf_cache/…/DA3NESTED-GIANT-LARGE-1.1` (spatial memory depth) | depth-anything/DA3NESTED-GIANT-LARGE-1.1 @ `b2359bdf726fb44ef62acca04d629dcf158053e7` | — | CC BY-NC 4.0 |
| `Depth-Anything-3/` (source) | ByteDance-Seed/Depth-Anything-3 @ `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | — | Apache-2.0 |
| `taehv/taeltx2_3_wide.pth` (tiny decoder) | madebyollin/taehv @ `32ac0146b11007cda5a57b60a3b35653361fb8a4` | — | MIT |

- The LTX-2 Community License Agreement (AlayaWorld code and weights) requires entities with annual revenues of at
  least USD 10 million to obtain a paid commercial licence from Lightricks.
- The DA3NESTED-GIANT-LARGE-1.1 weights are CC BY-NC 4.0: **non-commercial use only**. The model cannot run without
  them (spatial memory), so this system as a whole is usable for non-commercial purposes only.

Benchmark evaluation only; nothing is redistributed.

## Limitations

- No text-only start: the interactive path needs an opening image, so the text-to-video track is not registered.
- The six native camera axes are held at zero; camera directions are only prompt text.
- No mid-session reference image input (`reference_update` unsupported).
