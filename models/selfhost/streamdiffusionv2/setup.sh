#!/usr/bin/env bash
# StreamDiffusionV2 (1.3B): download code and weights and build the Python environment for
# models/selfhost/streamdiffusionv2/server.py. Idempotent; each step skips what is already in place.
#
#   bash models/selfhost/streamdiffusionv2/setup.sh [all|src|python|env|weights]
#
# Steps (all need network only; no GPU is needed to install, the env step prints whether CUDA is visible):
#   src      upstream source at a pinned commit (codeload tarball)
#   python   uv-managed CPython 3.10 and the venv env/
#   env      torch 2.8.0+cu128 and upstream's inference dependencies (~6 GB)
#   weights  Wan2.1-T2V-1.3B, the causal DMD v2v checkpoint and (optional) the TAEHV decoder (~23 GB)
#
# Layout under $ROOT (default $STREAM_ARENA_MODEL_ROOT/streamdiffusionv2; not in git):
#   repo/     upstream source at $COMMIT; repo/{wan_models,ckpts} -> weights/
#   py/       uv-managed CPython 3.10
#   env/      venv (torch 2.8.0+cu128, upstream deps; server.py puts repo/ on sys.path)
#   weights/  wan_models/Wan2.1-T2V-1.3B, ckpts/wan_causal_dmd_v2v, ckpts/taew2_1.pth
#
# Prerequisites: bash, curl, tar, uv (https://docs.astral.sh/uv/), the Hugging Face CLI `hf` (on PATH or
# installed in env/ by the env step), an NVIDIA driver supporting CUDA 12.8. The venv does not use a system
# PyTorch: the env step installs torch from download.pytorch.org.
#
# Environment variables:
#   STREAM_ARENA_MODEL_ROOT  model root (default ~/.cache/stream_arena/models)
#   SDV2_ROOT                this model's folder (default $STREAM_ARENA_MODEL_ROOT/streamdiffusionv2; server.py too)
#   SDV2_COMMIT              chenfengxu714/StreamDiffusionV2 commit (pinned default)
#   SDV2_HF_REV              jerryfeng/StreamDiffusionV2 revision (pinned default)
#   WAN13_REV                Wan-AI/Wan2.1-T2V-1.3B revision (default 37ec5126..., whose files match the recorded sha256)
#   UV_BIN, HF_BIN           uv and hf executables (default: on PATH; hf also env/bin/hf)
#   PIP_INDEX_URL            package index for the non-torch packages (default PyPI)
#   STREAM_ARENA_PROXY       optional HTTPS proxy for downloads (default $https_proxy)
#   HF_ENDPOINT, HF_TOKEN    Hugging Face endpoint and token
set -euo pipefail
STEP=${1:-all}
BASE=${STREAM_ARENA_MODEL_ROOT:-$HOME/.cache/stream_arena/models}
ROOT=${SDV2_ROOT:-$BASE/streamdiffusionv2}
COMMIT=${SDV2_COMMIT:-6961a5cf2045d1dda05a04ef229698bdc04e873a}
SDV2_REV=${SDV2_HF_REV:-2373eb2b39278b3a1aa174964a724ee78ead96f0}       # jerryfeng/StreamDiffusionV2
WAN13_REV=${WAN13_REV:-37ec512624d61f7aa208f7ea8140a131f93afc9a}       # Wan-AI/Wan2.1-T2V-1.3B
PROXY=${STREAM_ARENA_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
case $STEP in all|src|python|env|weights) ;; *) echo "usage: $0 [all|src|python|env|weights]" >&2; exit 2;; esac
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python
pipx() { https_proxy=$PROXY $PY -m pip install -q --no-cache-dir "$@"; }

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/pyproject.toml ]]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL "https://codeload.github.com/chenfengxu714/StreamDiffusionV2/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1
  echo "$COMMIT" > repo_commit.txt
fi

if [[ $STEP == all || $STEP == python ]] && [[ ! -x $PY ]]; then
  UV=${UV_BIN:-$(command -v uv || true)}
  [[ -n $UV ]] || { echo "uv not found: install it (https://docs.astral.sh/uv/) or set UV_BIN" >&2; exit 1; }
  UV_PYTHON_INSTALL_DIR=$ROOT/py https_proxy=$PROXY "$UV" python install --no-bin 3.10
  "$UV" venv --seed --python "$ROOT/py/cpython-3.10-linux-x86_64-gnu/bin/python3.10" env
fi

# Use the dependencies and installation requirements documented in README.md.
if [[ $STEP == all || $STEP == env ]]; then
  $PY -c "import torch; assert torch.__version__.startswith('2.8.0')" 2>/dev/null || \
    pipx --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchvision==0.23.0+cu128
  if ! $PY -c "import diffusers, transformers, omegaconf, PIL" 2>/dev/null; then
    pipx torch==2.8.0 torchvision==0.23.0 \
      accelerate==1.10.1 diffusers==0.35.1 einops==0.8.1 ftfy==6.3.1 imageio==2.37.0 numpy==1.24.4 \
      omegaconf==2.3.0 Pillow==10.1.0 regex==2024.11.6 sentencepiece==0.2.0 transformers==4.54.0 safetensors
  fi
  PYTHONPATH=repo $PY -c "import torch, streamv2v; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
fi

if [[ $STEP == all || $STEP == weights ]]; then
  HF=${HF_BIN:-$(command -v hf || echo "$ROOT/env/bin/hf")}
  [[ -x $HF ]] || { echo "hf not found: pip install huggingface_hub, run the env step, or set HF_BIN" >&2; exit 1; }
  mkdir -p weights
  [[ -f weights/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors ]] || \
    https_proxy=$PROXY $HF download Wan-AI/Wan2.1-T2V-1.3B --revision "$WAN13_REV" --local-dir weights/wan_models/Wan2.1-T2V-1.3B
  [[ -f weights/ckpts/wan_causal_dmd_v2v/model.pt ]] || \
    https_proxy=$PROXY $HF download jerryfeng/StreamDiffusionV2 --revision "$SDV2_REV" --include "wan_causal_dmd_v2v/*" \
      --local-dir weights/ckpts
# Use the dependencies and installation requirements documented in README.md.
  [[ -f weights/ckpts/taew2_1.pth ]] || \
    https_proxy=$PROXY curl -sfL https://github.com/madebyollin/taehv/raw/main/taew2_1.pth -o weights/ckpts/taew2_1.pth
  ln -sfn "$ROOT/weights/wan_models" repo/wan_models      # upstream resolves these relative to the repo root
  ln -sfn "$ROOT/weights/ckpts" repo/ckpts
fi
echo "SDV2_SETUP_DONE ($STEP)"
