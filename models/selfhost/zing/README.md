# Zing-0.5 (5B) — self-hosted world model, text or image start

Zing-0.5 (Seedleap.ai) is a 5B causal world model on Wan2.2-TI2V-5B. The server unrolls upstream's autoregressive
rollout so that every causal block is generated, decoded and sent before the next one starts.

| | |
|---|---|
| System ids | `zing_0p5_5b_t2v` (`general_interaction`), `zing_0p5_5b_i2v` (`general_interaction_image`) |
| Upstream code | [seedleap/zing-world-model](https://github.com/seedleap/zing-world-model) @ `11212da06f63290f4d354cdc0324241ba6eac356`; paper arXiv 2609.17909 (Zing Team, Seedleap.ai) |
| Weights | [seedleap/zing-0.5](https://huggingface.co/seedleap/zing-0.5) @ `b9ca1ed1de750bf19a0e6562c6cd71fa7ef35648` (34 GB) |
| Licence | code: Apache-2.0; weights: Apache-2.0 (see Weights and licences) |

The server is started with `{env}/bin/python {model_dir}/server.py --port {port}` (`system.json`). The shared live
loop (block scheduling, prompt policy, acks, frame pacing) is `models/selfhost/t2v_stream.py`.

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. The setup script reuses the base Python packages through `--system-site-packages`; it does not replace PyTorch.

```
bash models/selfhost/zing/setup.sh src       # upstream code tarball at the pinned commit (network)
bash models/selfhost/zing/setup.sh env       # env/: upstream pins diffusers 0.35.0, transformers 4.56.0, accelerate 1.13.0, peft 0.17.1 (network)
bash models/selfhost/zing/setup.sh weights   # seedleap/zing-0.5 at the pinned revision (network)
bash models/selfhost/zing/setup.sh check     # imports the upstream pipeline and prints the GPU (GPU)
```

`bash models/selfhost/zing/setup.sh` (or `all`) runs the four steps in order; each skips what is already in place.
flash-attn is not installed; the torch SDPA shim is used instead.

| variable | default | meaning |
|---|---|---|
| `ZING_ROOT` | `$STREAMEVAL_MODEL_ROOT/zing` | this model's folder (read by `setup.sh` and `server.py`; the harness's `{env}` placeholder always uses `$STREAMEVAL_MODEL_ROOT/zing/env`) |
| `ZING_COMMIT` | `11212da06f63290f4d354cdc0324241ba6eac356` | upstream commit downloaded by `setup.sh src` |

## Weights and licences

Under `$STREAMEVAL_MODEL_ROOT/zing/weights/Zing-0.5`:

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `generator/model.pt` | seedleap/zing-0.5 @ `b9ca1ed1` | `e14ca90b…3e85e4` (full values in `system.json`) | Apache-2.0 |
| `pretrained/text_encoder/model-0000{1,2,3}-of-00003.safetensors` (UMT5-XXL) | same | in `system.json` | Apache-2.0 |
| `pretrained/tokenizer` | same | — | Apache-2.0 |
| `pretrained/vae/diffusion_pytorch_model.safetensors` (Wan2.2 VAE) | same | in `system.json` | Apache-2.0 |

The sha256 values of the generator, text-encoder shards and VAE are also recorded per session in
`ready.system_config.weights_sha256`. Code: Apache-2.0. No non-commercial terms apply.

## Limitations

- No mid-session image input: `reference` events on the i2v track are unsupported.
- No action (keyboard) input is used; camera directions are given only as prompt text.
- A prompt change takes effect at the next block (at most one 16-frame block of compute plus the frames already
  waiting to be sent).
- Image start center-crops the opening image to 832:480 (upstream resizes without cropping).
