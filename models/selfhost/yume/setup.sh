#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${YUME_ROOT:-$BASE/yume}
COMMIT=${YUME_COMMIT:-111c3fab7fb020d1e261a68be6ec78a3fecc8d5b}   # stdstu12/YUME main since 2026-01-14
REV=b117c778405246a0b932d7ec5873b01a4840da3a      # stdstu123/Yume-5B-720P (Hugging Face, 2026-10-04)
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
INDEX=${PIP_INDEX_URL:-https://pypi.org/simple}
STEP=${1:-all}
case $STEP in all|repo|env|weights|check) ;; *) echo "usage: bash setup.sh [all|repo|env|weights|check]" >&2; exit 2 ;; esac
mkdir -p "$ROOT"/{logs,weights}
UV=${UV_BIN:-$(command -v uv || true)}
[[ -n $UV ]] || { echo "uv not found: install it (https://docs.astral.sh/uv/) or set UV_BIN" >&2; exit 1; }
PY=$ROOT/env/bin/python
export UV_PYTHON_INSTALL_DIR=$ROOT/py UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DISABLE_XET=1

repo() {
  if [[ ! -f $ROOT/repo/wan23/textimage2video.py ]]; then
    local sha
    sha=$(https_proxy=$PROXY curl -sf https://api.github.com/repos/stdstu12/YUME/commits/$COMMIT | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")
    mkdir -p "$ROOT/repo"
    https_proxy=$PROXY curl -sfL "https://codeload.github.com/stdstu12/YUME/tar.gz/$sha" \
      | tar xz -C "$ROOT/repo" --strip-components=1 --exclude='*.mp4' --exclude='*.gif' --exclude='*.png' \
          --exclude='*.jpg' --exclude='*.pyc' --exclude='demo' --exclude='assets' --exclude='val' --exclude='test_video'
    echo "$sha" > "$ROOT/repo_commit.txt"
  fi
}

env_() {
  [[ -x $PY ]] || { https_proxy=$PROXY "$UV" python install 3.11 && "$UV" venv -p 3.11 "$ROOT/env"; }
  if ! "$PY" -c "import torch, diffusers, transformers, peft, easydict, ftfy, einops, setuptools; assert torch.__version__.startswith('2.10.0')" 2>/dev/null; then
    https_proxy=$PROXY "$UV" pip install -p "$PY" torch==2.10.0 torchvision==0.25.0 triton==3.6.0 \
      --index-url https://download.pytorch.org/whl/cu130
    # inference subset of upstream requirements.txt
    https_proxy=$PROXY "$UV" pip install -p "$PY" torch==2.10.0 torchvision==0.25.0 triton==3.6.0 \
      diffusers==0.32.0 transformers==4.46.1 peft==0.13.2 accelerate==1.0.1 easydict==1.13 einops==0.8.0 ftfy==6.3.0 \
      regex sentencepiece protobuf safetensors imageio imageio-ffmpeg "numpy<2" scipy tqdm pillow setuptools \
      huggingface_hub==0.26.1 \
      --index-url "$INDEX" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130
  fi
  # cuDNN 9.26 or newer (the v1 environment's version), installed over the one torch brings when older
  "$PY" -c "import torch, sys; sys.exit(torch.backends.cudnn.version() < 92600)" 2>/dev/null || \
    https_proxy=$PROXY "$UV" pip install -p "$PY" --index-url "$INDEX" --no-deps "nvidia-cudnn-cu13==9.26.0.51"
  "$PY" -c "
import torch, diffusers, transformers
print('env ok torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)
print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None"
  "$UV" pip freeze -p "$PY" > "$ROOT/env_freeze.txt"
}

weights() {
  local D=$ROOT/weights/Yume-5B-720P
  if [[ ! -f $D/.complete ]]; then
    https_proxy=$PROXY "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('stdstu123/Yume-5B-720P', revision='$REV', local_dir='$D', max_workers=8, ignore_patterns=['*.mp4'])"
    echo "stdstu123/Yume-5B-720P@$REV" > "$D/.complete"
  fi
  ls -la "$D"
}

check() {
  (cd "$ROOT/repo" && PYTHONPATH=$HERE/../_shims:$ROOT/repo "$PY" -c "
import torch, flash_attn, wan23
from wan23.configs import WAN_CONFIGS
from wan23.utils.utils import masks_like
print('IMPORT_OK yume on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), sorted(WAN_CONFIGS))")
}

case $STEP in
  repo) repo ;;
  env) env_ ;;
  weights) weights ;;
  check) check ;;
  all) repo; env_; weights; check ;;
  *) echo "usage: bash setup.sh [all|repo|env|weights|check]" >&2; exit 2 ;;
esac
echo "YUME_SETUP_DONE ($STEP)"
