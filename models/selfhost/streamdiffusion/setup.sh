#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
BASE=${STREAM_ARENA_MODEL_ROOT:-$HOME/.cache/stream_arena/models}
ROOT=${SD1_ROOT:-$BASE/streamdiffusion}
COMMIT=${SD1_COMMIT:-b623251dc055e1fd858d53509aa43e09dfc5cdc0}      # cumulo-autumn/StreamDiffusion main, 2024-12-04
TURBO_REV=b261bac6fd2cf515557d5d0707481eafa0485ec2                  # stabilityai/sd-turbo
TAESD_REV=614f76814bbe30edbe2e627ace1c2234c81a2c0e                  # madebyollin/taesd
PROXY=${STREAM_ARENA_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
case $STEP in all|src|env|weights|check) ;; *) echo "usage: $0 [all|src|env|weights|check]" >&2; exit 2;; esac
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/utils/wrapper.py ]]; then
  # the codeload tarball is 124 MB, mostly demo assets: fetch only the python files the server needs
  rm -rf repo && mkdir -p repo
  for f in setup.py LICENSE utils/__init__.py utils/wrapper.py demo/realtime-img2img/img2img.py \
      src/streamdiffusion/__init__.py src/streamdiffusion/pipeline.py src/streamdiffusion/image_filter.py \
      src/streamdiffusion/image_utils.py src/streamdiffusion/pip_utils.py src/streamdiffusion/acceleration/__init__.py; do
    mkdir -p "repo/$(dirname $f)"
    https_proxy=$PROXY curl -sfL --max-time 300 \
      "https://raw.githubusercontent.com/cumulo-autumn/StreamDiffusion/$COMMIT/$f" -o "repo/$f"
  done
  echo "$COMMIT" > repo_commit.txt
fi

# Upstream pins diffusers==0.24.0 (needs huggingface_hub < 0.26 for cached_download). torch comes from the base
# Python (a CUDA build for your GPU). No xformers / TensorRT (optional accelerators upstream).
if [[ $STEP == all || $STEP == env ]]; then
  if ! $PY -c "import torch, diffusers, transformers, accelerate, fire, omegaconf" 2>/dev/null; then
    python3 -c "import torch" 2>/dev/null || {
      echo "python3 cannot import torch: the env needs a base Python with a CUDA build of PyTorch (see header)" >&2; exit 1; }
    python3 -m venv --system-site-packages env
    PKGS="diffusers==0.24.0 huggingface_hub==0.25.2 transformers==4.44.2 accelerate==0.34.2 fire omegaconf"
    https_proxy=$PROXY $PY -m pip install -q $PKGS
  fi
  $PY -c "import torch, diffusers, transformers, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), 'diffusers', diffusers.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__)"
  $PY -m pip freeze > env_freeze.txt 2>/dev/null || true
fi

if [[ $STEP == all || $STEP == weights ]]; then
  mkdir -p weights
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  https_proxy=$PROXY $DL - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/sd-turbo", revision="$TURBO_REV", local_dir="weights/sd-turbo", max_workers=4,
                  allow_patterns=["model_index.json", "*/config.json", "scheduler/*", "tokenizer/*",
                                  "unet/diffusion_pytorch_model.safetensors", "text_encoder/model.safetensors",
                                  "vae/diffusion_pytorch_model.safetensors"])
snapshot_download("madebyollin/taesd", revision="$TAESD_REV", local_dir="weights/taesd",
                  allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
print("downloaded sd-turbo $TURBO_REV, taesd $TAESD_REV")
EOF
  ls -la weights/*
fi

if [[ $STEP == all || $STEP == check ]]; then
  (cd repo && PYTHONPATH=.:src ../$PY -c "
import torch
from utils.wrapper import StreamDiffusionWrapper
print('IMPORT_OK streamdiffusion on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))")
fi
echo "SD1_SETUP_DONE ($STEP)"
