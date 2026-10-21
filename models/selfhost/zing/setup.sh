#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
case $STEP in all|src|env|weights|check) ;; *) echo "usage: bash $0 [all|src|env|weights|check]" >&2; exit 2;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${ZING_ROOT:-$BASE/zing}
COMMIT=${ZING_COMMIT:-11212da06f63290f4d354cdc0324241ba6eac356}   # seedleap/zing-world-model main, 2026-09-17
HF_REV=b9ca1ed1de750bf19a0e6562c6cd71fa7ef35648                   # seedleap/zing-0.5
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=8
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/src/zing_v0_5/pipeline.py ]]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL "https://codeload.github.com/seedleap/zing-world-model/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1 --exclude='*.png' --exclude='*.jpg' --exclude='*.gif' --exclude='*.mp4' \
          --exclude='*.pdf'
  echo "$COMMIT" > repo_commit.txt
fi

if [[ $STEP == all || $STEP == env ]]; then
  if ! $PY -c "import torch, diffusers, transformers, yaml, sentencepiece, accelerate, peft; assert diffusers.__version__ == '0.35.0'" 2>/dev/null; then
    python3 -c "import torch" 2>/dev/null || { echo "python3 has no PyTorch: install a CUDA build of torch for your GPU in the base Python first (see the header)" >&2; exit 1; }
    python3 -m venv --system-site-packages env
    PKGS="diffusers==0.35.0 transformers==4.56.0 accelerate==1.13.0 safetensors==0.4.5 sentencepiece==0.2.1
      PyYAML==6.0.3 peft==0.17.1 tqdm huggingface_hub pillow"   # peft: diffusers 0.35 refuses an older peft (0.9) in the base environment
    https_proxy=$PROXY $PY -m pip install -q $PKGS
  fi
  $PY -c "import torch, diffusers, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)"
  $PY -m pip freeze > env_freeze.txt 2>/dev/null || true
fi

if [[ $STEP == all || $STEP == weights ]]; then
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  [[ -f weights/Zing-0.5/.complete_$HF_REV ]] || { https_proxy=$PROXY $DL -c "
from huggingface_hub import snapshot_download
snapshot_download('seedleap/zing-0.5', revision='$HF_REV', local_dir='weights/Zing-0.5', max_workers=8)" \
    && touch weights/Zing-0.5/.complete_$HF_REV; }
  du -sh weights/Zing-0.5; ls -la weights/Zing-0.5 weights/Zing-0.5/generator weights/Zing-0.5/pretrained
fi

if [[ $STEP == all || $STEP == check ]]; then
  PYTHONPATH=$HERE/../_shims:$ROOT/repo/src "$ROOT/$PY" -c "
import torch, flash_attn
from zing_v0_5.pipeline import InferencePipeline
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
print('IMPORT_OK zing on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'flash_attn', flash_attn.__version__)"
fi
echo "ZING_SETUP_DONE ($STEP)"
