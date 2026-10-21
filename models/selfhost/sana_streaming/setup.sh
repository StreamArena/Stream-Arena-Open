#!/usr/bin/env bash
# SANA-Streaming (NVlabs, streaming DiT): download code and weights and build the Python environment for
# models/selfhost/sana_streaming/server.py. Idempotent; each step skips what is already in place.
#
#   bash models/selfhost/sana_streaming/setup.sh [all|src|python|env|weights]
#
# Steps (all need network only; no GPU is needed to install, the env step prints whether CUDA is visible):
#   src      NVlabs/Sana at a pinned commit (codeload tarball)
#   python   uv-managed CPython 3.12 (upstream needs >= 3.11) and the venv env/
#   env      torch 2.9.1+cu128 and the inference subset of upstream's pyproject (+ mmcv 1.7.2 without CUDA ops)
#   weights  SANA-Streaming DiT and causal decoder, LTX-2 VAE, gemma-2-2b-it (~17 GB)
#
# Layout under $ROOT (default $STREAMEVAL_MODEL_ROOT/sana_streaming; not in git):
#   repo/     NVlabs/Sana at $COMMIT
#   py/       uv-managed CPython 3.12
#   env/      venv: torch 2.9.1+cu128 and the inference subset of upstream pyproject
#   weights/  SANA-Streaming/{dit/sana_streaming_ar.pth, ltx2_causal_vae_0516}, LTX-2/vae, gemma-2-2b-it
#
# Prerequisites: bash, curl, tar, uv (https://docs.astral.sh/uv/), the Hugging Face CLI `hf` (on PATH or
# installed in env/ by the env step), an NVIDIA driver supporting CUDA 12.8. The venv does not use a system
# PyTorch: the env step installs torch from download.pytorch.org. Upstream's setup notes are NVlabs/Sana
# docs/sana_streaming.md (in repo/ after the src step).
#
# Environment variables:
#   STREAMEVAL_MODEL_ROOT  model root (default ~/.cache/streameval/models)
#   SANA_ROOT                this model's folder (default $STREAMEVAL_MODEL_ROOT/sana_streaming; server.py too)
#   SANA_COMMIT              NVlabs/Sana commit (pinned default)
#   SANA_STREAMING_REV, LTX2_REV, GEMMA_REV   Hugging Face revisions (pinned defaults)
#   UV_BIN, HF_BIN           uv and hf executables (default: on PATH; hf also env/bin/hf)
#   PIP_INDEX_URL            package index for the non-torch packages (default PyPI)
#   STREAMEVAL_PROXY       optional HTTPS proxy for downloads (default $https_proxy)
#   HF_ENDPOINT, HF_TOKEN    Hugging Face endpoint and token
set -euo pipefail
STEP=${1:-all}
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${SANA_ROOT:-$BASE/sana_streaming}
COMMIT=${SANA_COMMIT:-f9178744c096dcf2a2ea773da183e341bcbeb044}
SANA_REV=${SANA_STREAMING_REV:-616778bcee475e314177bb4bd81be2b50af69b26}   # Efficient-Large-Model/SANA-Streaming
LTX2_REV=${LTX2_REV:-dfcc2108383fe1aaa0584bdf55d368a4bdadd90c}             # Lightricks/LTX-2
GEMMA_REV=${GEMMA_REV:-569d9809d0c8b6722d4d31b5a77a2ec7a400650a}           # Efficient-Large-Model/gemma-2-2b-it
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
case $STEP in all|src|python|env|weights) ;; *) echo "usage: $0 [all|src|python|env|weights]" >&2; exit 2;; esac
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python
pipx() { https_proxy=$PROXY $PY -m pip install -q --no-cache-dir "$@"; }

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/pyproject.toml ]]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL "https://codeload.github.com/NVlabs/Sana/tar.gz/$COMMIT" | tar xz -C repo --strip-components=1
  echo "$COMMIT" > repo_commit.txt
fi

if [[ $STEP == all || $STEP == python ]] && [[ ! -x $PY ]]; then
  UV=${UV_BIN:-$(command -v uv || true)}
  [[ -n $UV ]] || { echo "uv not found: install it (https://docs.astral.sh/uv/) or set UV_BIN" >&2; exit 1; }
  UV_PYTHON_INSTALL_DIR=$ROOT/py https_proxy=$PROXY "$UV" python install --no-bin 3.12
  "$UV" venv --seed --python "$ROOT/py/cpython-3.12-linux-x86_64-gnu/bin/python3.12" env
fi

# Inference subset of upstream pyproject (training / eval extras such as xformers, hpsv2, wandb, CLIP are not
# needed; mmcv 1.7.2 is, pure-Python, for its Registry); versions follow upstream pyproject and repo/docs/sana_streaming.md.
if [[ $STEP == all || $STEP == env ]]; then
  $PY -c "import torch; assert torch.__version__.startswith('2.9.1')" 2>/dev/null || \
    pipx --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1+cu128 torchvision==0.24.1+cu128
  if ! $PY -c "import fla, diffusers, pyrallis, transformers, peft, timm, mmcv, pytz, qwen_vl_utils" 2>/dev/null; then
    pipx torch==2.9.1 torchvision==0.24.1 \
      huggingface-hub==0.36.0 'accelerate>=1.3' transformers==4.57.3 'diffusers>=0.37.0' einops ftfy psutil ninja \
      omegaconf opencv-python-headless peft==0.18.0 protobuf regex sentencepiece timm==0.6.13 termcolor pyrallis \
      'imageio[pyav,ffmpeg]' 'flash-linear-attention>=0.4.2' Pillow matplotlib tqdm beautifulsoup4 came-pytorch fire \
      pytz qwen-vl-utils addict yapf
    # mmcv 1.7.2 sdist: setup.py needs pkg_resources (setuptools < 81), no CUDA ops; it pulls opencv-python (libGL)
    pipx 'setuptools<70' wheel
    MMCV_WITH_OPS=0 pipx --no-build-isolation mmcv==1.7.2
    pipx --no-deps --force-reinstall opencv-python-headless
  fi
  (cd repo && ../$PY -c "import torch, diffusion.model.builder; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
fi

if [[ $STEP == all || $STEP == weights ]]; then
  HF=${HF_BIN:-$(command -v hf || echo "$ROOT/env/bin/hf")}
  [[ -x $HF ]] || { echo "hf not found: pip install huggingface_hub, run the env step, or set HF_BIN" >&2; exit 1; }
  mkdir -p weights
  [[ -f weights/SANA-Streaming/dit/sana_streaming_ar.pth ]] || https_proxy=$PROXY $HF download \
    Efficient-Large-Model/SANA-Streaming --revision "$SANA_REV" \
    --include "dit/sana_streaming_ar.pth" "ltx2_causal_vae_0516/*" --local-dir weights/SANA-Streaming
  [[ -f weights/LTX-2/vae/diffusion_pytorch_model.safetensors ]] || \
    https_proxy=$PROXY $HF download Lightricks/LTX-2 --revision "$LTX2_REV" --include "vae/*" --local-dir weights/LTX-2
  [[ -f weights/gemma-2-2b-it/config.json ]] || \
    https_proxy=$PROXY $HF download Efficient-Large-Model/gemma-2-2b-it --revision "$GEMMA_REV" --local-dir weights/gemma-2-2b-it
fi
echo "SANA_SETUP_DONE ($STEP)"
