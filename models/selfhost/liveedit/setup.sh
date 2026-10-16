#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${STREAM_ARENA_MODEL_ROOT:-$HOME/.cache/stream_arena/models}
ROOT=${LIVEEDIT_ROOT:-$BASE/liveedit}
COMMIT=${LIVEEDIT_COMMIT:-77a9fcc6ee2e2d84d232877e4494f9e8c563514a}       # cp-cp/LiveEdit main, 2026-08-10
HF_REV=${LIVEEDIT_HF_REV:-f0269e2a8a5e1cdd7f7831c1bdcd95291db4f6e6}  # cp-cp/LiveEdit (HF)
WAN13_REV=${WAN13_REV:-37ec512624d61f7aa208f7ea8140a131f93afc9a}   # Wan-AI/Wan2.1-T2V-1.3B
export LONGLIVE_ROOT=${LONGLIVE_ROOT:-$BASE/longlive}
PROXY=${STREAM_ARENA_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
case $STEP in all|src|env|weights|check) ;; *) echo "usage: $0 [all|src|env|weights|check]" >&2; exit 2;; esac
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/inference-mm.py ]]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL --max-time 900 "https://codeload.github.com/cp-cp/LiveEdit/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1 --exclude='*.mp4' --exclude='*.gif'
  echo "$COMMIT" > repo_commit.txt
fi

if [[ $STEP == all || $STEP == env ]]; then
  python3 -c "import torch" 2>/dev/null || [[ -x $LONGLIVE_ROOT/env/bin/python ]] || {
    echo "python3 cannot import torch: the env step needs a base Python with a CUDA build of PyTorch (see header)" >&2; exit 1; }
  bash "$HERE/../longlive/setup.sh" env
  [[ -e env ]] || ln -sfn "$LONGLIVE_ROOT/env" env
fi

if [[ $STEP == all || $STEP == weights ]]; then
  mkdir -p weights/wan_models weights/checkpoints/liveedit
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  [[ -f weights/checkpoints/liveedit/ar-forcing_002000.pt ]] || https_proxy=$PROXY $DL -c "
from huggingface_hub import hf_hub_download
hf_hub_download('cp-cp/LiveEdit', 'ar-forcing_002000.pt', revision='$HF_REV', local_dir='weights/checkpoints/liveedit')"
  if [[ ! -e weights/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth ]]; then
    SHARED=$BASE/streamdiffusionv2/weights/wan_models/Wan2.1-T2V-1.3B   # same revision, if already downloaded
    if [[ -f $SHARED/Wan2.1_VAE.pth && -f $SHARED/diffusion_pytorch_model.safetensors ]]; then
      ln -sfn "$SHARED" weights/wan_models/Wan2.1-T2V-1.3B
    else
      https_proxy=$PROXY $DL -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', revision='$WAN13_REV', local_dir='weights/wan_models/Wan2.1-T2V-1.3B', ignore_patterns=['assets/*', 'examples/*'], max_workers=4)"
    fi
  fi
  ln -sfn "$ROOT/weights/wan_models" repo/wan_models                  # upstream resolves both relative to the repo
  ln -sfn "$ROOT/weights/checkpoints" repo/checkpoints
  ls -la weights/checkpoints/liveedit weights/wan_models
fi

if [[ $STEP == all || $STEP == check ]]; then
  (cd repo && "$ROOT/$PY" -c "
import torch
from pipeline.causal_inference import CausalInferencePipeline
print('IMPORT_OK liveedit on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))")
fi
echo "LIVEEDIT_SETUP_DONE ($STEP)"
