# Yume 1.5 (5B, 720p) — self-hosted interactive world model, text or image + text start

Yume 1.5 is a segment-wise autoregressive world model (32-frame segments, each conditioned on its own caption,
long history compressed in-model). `server.py` runs upstream's long-video loop live, one segment at a time, behind
the WebSocket protocol ([docs/protocol.md](../../../docs/protocol.md)), using the shared live loop
`models/selfhost/t2v_stream.py`.

| | |
|---|---|
| System ids | `yume_5b_t2v` (`general_interaction`), `yume_5b_i2v` (`general_interaction_image`) |
| Upstream code | [stdstu12/YUME](https://github.com/stdstu12/YUME), @ `111c3fab7fb020d1e261a68be6ec78a3fecc8d5b` (the head of `main` since 2026-01-14; see Setup); Yume 1.5, arXiv 2512.22096 (CVPR 2026) |
| Weights | [stdstu123/Yume-5B-720P](https://huggingface.co/stdstu123/Yume-5B-720P) @ `b117c778405246a0b932d7ec5873b01a4840da3a`: DiT, Wan2.2 VAE, umT5-XXL text encoder, tokenizer |
| Licence | code: Apache-2.0; weights: Apache-2.0 (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model.

```
bash models/selfhost/yume/setup.sh repo      # network: stdstu12/YUME code tarball (media / demo data skipped)
bash models/selfhost/yume/setup.sh env       # network: venv with torch 2.10.0+cu130, upstream pins for
                                             #   diffusers 0.32.0 / transformers 4.46.1 / peft 0.13.2, cuDNN >= 9.26
bash models/selfhost/yume/setup.sh weights   # network: Yume-5B-720P (DiT 20.8 GB, umT5-XXL 11.4 GB, Wan2.2 VAE)
bash models/selfhost/yume/setup.sh check     # GPU: imports the upstream model code
bash models/selfhost/yume/setup.sh all       # the four steps in order
```

Files go to `$STREAMEVAL_MODEL_ROOT/yume/` (`repo/`, `py/`, `env/`, `weights/Yume-5B-720P/`).

Weight revisions and file hashes are listed in `system.json` and `setup.sh`. Record resolved revisions and downloaded hashes for each installation.

| variable | default | meaning |
|---|---|---|
| `YUME_ROOT` | `$STREAMEVAL_MODEL_ROOT/yume` | this model's folder (setup.sh and server.py) |
| `YUME_COMMIT` | `111c3fab…` | upstream commit to fetch |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` (set by server.py if unset) | PyTorch allocator settings |

## Weights and licences

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `Yume-5B-720P/diffusion_pytorch_model.safetensors` (DiT, 20.8 GB) | stdstu123/Yume-5B-720P @ `b117c778` | recorded per session in `ready.system_config.weights_sha256` | Apache-2.0 |
| `Yume-5B-720P/models_t5_umt5-xxl-enc-bf16.pth` (umT5-XXL, 11.4 GB) | same | same | Apache-2.0 |
| `Yume-5B-720P/Wan2.2_VAE.pth` | same | same | Apache-2.0 |
| `Yume-5B-720P/google/umt5-xxl/` (tokenizer) | same | — | Apache-2.0 |

The weights repository is ungated. Code: Apache-2.0. Nothing is redistributed here.

## Limitations

- No mid-session image input: the reference update of the image track is unsupported.
- Instructions and camera moves are caption text only; they apply from the next segment (about 2.1–2.4 s median
  after release).
- The upstream commit used for v1 is unknown (see Setup).
