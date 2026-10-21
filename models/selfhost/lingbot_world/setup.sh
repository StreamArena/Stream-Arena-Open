#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
case $STEP in all|src|env|weights|check) ;; *) echo "usage: $0 [all|src|env|weights|check]" >&2; exit 2 ;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${LINGBOT_WORLD_ROOT:-$BASE/lingbot_world}
COMMIT=${LINGBOT_WORLD_COMMIT:-1895d300d8ac936401689b26389f51cbd36530eb}   # Robbyant/lingbot-world-v2 main, 2026-09-10
HF_REV=7e36a5f919f86cb4255cc9bfc30adb44963fbde1                             # robbyant/lingbot-world-v2-1.3b-causal-fast
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/generate.py ]]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL "https://codeload.github.com/Robbyant/lingbot-world-v2/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1 --exclude='*.png' --exclude='*.jpg' --exclude='*.gif' --exclude='*.mp4' \
      --exclude='*.pdf'
  echo "$COMMIT" > repo_commit.txt
fi

if [[ $STEP == all || $STEP == env ]]; then
  if [[ ! -x $PY ]] && ! python3 -c "import torch" 2>/dev/null; then
    echo "python3 cannot import torch: the env step needs a base Python with a CUDA build of PyTorch" >&2; exit 1
  fi
  bash "$HERE/../longlive/setup.sh" env
  [[ -e env ]] || ln -sfn "${LONGLIVE_ROOT:-$BASE/longlive}/env" env
  $PY -c "import scipy, easydict, ftfy, einops, regex, safetensors; print('env ok')"
fi

if [[ $STEP == all || $STEP == weights ]]; then
  mkdir -p weights/lingbot-world-v2-1.3b-causal-fast
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  [[ -f weights/lingbot-world-v2-1.3b-causal-fast/model-00006-of-00006.safetensors ]] || $DL -c "
from huggingface_hub import snapshot_download
snapshot_download('robbyant/lingbot-world-v2-1.3b-causal-fast', revision='$HF_REV', local_dir='weights/lingbot-world-v2-1.3b-causal-fast', allow_patterns=['model-*.safetensors', 'model.safetensors.index.json', 'README.md'], max_workers=4)"
  if [[ ! -e weights/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth ]]; then
      $DL -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', revision='37ec512624d61f7aa208f7ea8140a131f93afc9a', local_dir='weights/Wan2.1-T2V-1.3B', allow_patterns=['Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth', 'google/*'], max_workers=4)"
  fi
  ls -la weights/lingbot-world-v2-1.3b-causal-fast weights/
fi

if [[ $STEP == all || $STEP == check ]]; then
  (cd repo && PYTHONPATH="$HERE/../_shims" "$ROOT/$PY" -c "
import torch
from wan.image2video import WanI2VCausal
print('IMPORT_OK lingbot_world on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))")
fi
echo "LINGBOT_WORLD_SETUP_DONE ($STEP)"
