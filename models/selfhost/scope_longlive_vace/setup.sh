#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
case $STEP in all|env|weights|check) ;; *) echo "usage: $0 [all|env|weights|check]" >&2; exit 2;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${STREAM_ARENA_MODEL_ROOT:-$HOME/.cache/stream_arena/models}
ROOT=${SCOPE_LONGLIVE_ROOT:-$BASE/scope_longlive_vace}
SCOPE=${SCOPE_ROOT:-$BASE/scope_vace}
LL_REV=7735d25c7944e4497fa5882aaf7a3bdc9c1fda32                     # daydreamlive/LongLive-1.3B
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python

if [[ $STEP == all || $STEP == env ]]; then
  [[ -f $SCOPE/repo/src/scope/core/pipelines/longlive/pipeline.py ]] || bash "$HERE/../scope_vace/setup.sh" src
  bash "$HERE/../scope_vace/setup.sh" env
  [[ -e env ]] || ln -sfn "$SCOPE/env" env
fi

if [[ $STEP == all || $STEP == weights ]]; then
  mkdir -p weights
  for d in Wan2.1-T2V-1.3B WanVideo_comfy; do
    [[ -f $SCOPE/weights/$d/config.json ]] || bash "$HERE/../scope_vace/setup.sh" weights
    [[ -e weights/$d ]] || ln -sfn "$SCOPE/weights/$d" weights/$d
  done
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  [[ -f weights/LongLive-1.3B/models/lora.pt && -f weights/LongLive-1.3B/models/longlive_base.pt ]] || $DL -c "
from huggingface_hub import snapshot_download
snapshot_download('daydreamlive/LongLive-1.3B', revision='$LL_REV', local_dir='weights/LongLive-1.3B',
                  allow_patterns=['config.json', 'models/longlive_base.pt', 'models/lora.pt'], max_workers=4)"
  ls -la weights/ weights/LongLive-1.3B/models
fi

if [[ $STEP == all || $STEP == check ]]; then
  PYTHONPATH=$SCOPE/repo/src $PY -c "
import torch
from scope.core.pipelines.longlive.pipeline import LongLivePipeline
print('IMPORT_OK scope longlive on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
fi
echo "SCOPE_LONGLIVE_SETUP_DONE ($STEP)"
