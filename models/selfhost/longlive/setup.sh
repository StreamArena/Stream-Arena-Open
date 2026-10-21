#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
case $STEP in
  all|src|env|weights|check) ;;
  *) echo "usage: bash $0 [all|src|env|weights|check]" >&2; exit 2 ;;
esac
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${LONGLIVE_ROOT:-$BASE/longlive}
COMMIT=${LONGLIVE_COMMIT:-fb16a879f46e604df4a0ca48ce2bf45adfa90352}      # NVlabs/LongLive main, 2026-09-30
LL2_REV=8521079b863720a57c1a8d9b19c8d9e6ccb04c0f                         # Efficient-Large-Model/LongLive-2.0-5B
WAN22_REV=921dbaf3f1674a56f47e83fb80a34bac8a8f203e                       # Wan-AI/Wan2.2-TI2V-5B
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/LongLive2.0/inference.py ]]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL "https://codeload.github.com/NVlabs/LongLive/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1 --exclude='*.png' --exclude='*.jpg' --exclude='*.gif' --exclude='*.mp4'
  echo "$COMMIT" > repo_commit.txt
fi

# transformers < 5 is upstream's pin (causal_model imports x_clip_loss).
if [[ $STEP == all || $STEP == env ]]; then
  if ! $PY -c "import torch, diffusers, transformers, omegaconf, easydict, ftfy, sentencepiece, einops, peft" 2>/dev/null; then
    python3 -c "import torch, torchvision" 2>/dev/null || {
      echo "python3 cannot import torch/torchvision: env/ is built over a base Python that already provides a CUDA" \
           "build of PyTorch for your GPU (see the header of this script)" >&2; exit 1; }
    python3 -m venv --system-site-packages env
    PKGS="diffusers==0.31.0 transformers==4.57.3 accelerate==1.10.1 omegaconf==2.3.0 easydict==1.13 ftfy==6.3.1
      regex sentencepiece==0.2.1 safetensors einops tqdm imageio imageio-ffmpeg scipy peft==0.17.1 pillow
      huggingface_hub opencv-python-headless"
    https_proxy=$PROXY $PY -m pip install -q $PKGS
  fi
  $PY -c "import torch, torchvision, diffusers, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), 'torchvision', torchvision.__version__, 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)"
  $PY -m pip freeze > env_freeze.txt 2>/dev/null || true
fi

if [[ $STEP == all || $STEP == weights ]]; then
  mkdir -p weights/wan_models weights/LongLive-2.0-5B
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  [[ -f weights/LongLive-2.0-5B/model_bf16.pt ]] || https_proxy=$PROXY $DL -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Efficient-Large-Model/LongLive-2.0-5B', 'model_bf16.pt', revision='$LL2_REV', local_dir='weights/LongLive-2.0-5B')"
  if [[ ! -e weights/wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth ]]; then
    SHARED=$BASE/matrix_game_35/weights/Wan2.2-TI2V-5B         # same HF revision, if already downloaded
    if [[ -f $SHARED/Wan2.2_VAE.pth && -f $SHARED/diffusion_pytorch_model-00001-of-00003.safetensors ]]; then
      ln -sfn "$SHARED" weights/wan_models/Wan2.2-TI2V-5B
    else
      https_proxy=$PROXY $DL -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.2-TI2V-5B', revision='$WAN22_REV', local_dir='weights/wan_models/Wan2.2-TI2V-5B', ignore_patterns=['assets/*', 'examples/*'], max_workers=4)"
    fi
  fi
  ln -sfn "$ROOT/weights/wan_models" repo/LongLive2.0/wan_models      # upstream resolves it relative to the repo
  ls -la weights/LongLive-2.0-5B weights/wan_models
fi

if [[ $STEP == all || $STEP == check ]]; then
  (cd repo/LongLive2.0 && "$ROOT/$PY" -c "
import torch, torchvision.io as tvio
from pipeline import CausalDiffusionInferencePipeline
from utils.nvfp4_checkpoint import unwrap_generator_state_dict
print('IMPORT_OK longlive2 on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))")
fi
echo "LONGLIVE_SETUP_DONE ($STEP)"
