#!/usr/bin/env bash
# Rolling Forcing (Wan2.1-T2V-1.3B) self-host setup. Idempotent: each step skips what is already in place.
#
#   bash models/selfhost/rolling_forcing/setup.sh [all|src|env|weights|check]
#
# Steps:
#   src      download TencentARC/RollingForcing at the pinned commit (small codeload tarball; network, no GPU)
#   env      run `longlive/setup.sh env` (builds the shared causal Wan venv if missing) and link env -> it
#   weights  download the Rolling Forcing checkpoint (17 GB, pinned revision) and Wan2.1-T2V-1.3B (about 34 GB; network)
#   check    import the upstream pipeline and print the GPU (GPU)
#   all      every step above, in order (default)
#
# Prerequisite: the environment is the venv of models/selfhost/longlive (one environment for the causal Wan
# models), built by `longlive/setup.sh env` over a base Python whose site-packages provide a CUDA build of
# PyTorch for your GPU.
# See the header of longlive/setup.sh.
#
# Layout under the model root $ROOT (not in git):
#   repo/      TencentARC/RollingForcing at the pinned commit (media files skipped); repo/wan_models links to
#              weights/wan_models
#   env        link to $LONGLIVE_ROOT/env
#   weights/   checkpoints/rolling_forcing_dmd.pt, wan_models/Wan2.1-T2V-1.3B
#
# Environment variables:
#   STREAMEVAL_MODEL_ROOT  parent of all model roots (default ~/.cache/streameval/models)
#   ROLLING_FORCING_ROOT     this model's root (default $STREAMEVAL_MODEL_ROOT/rolling_forcing)
#   ROLLING_FORCING_COMMIT   upstream commit to download (pinned default)
#   LONGLIVE_ROOT            root of the shared environment (default $STREAMEVAL_MODEL_ROOT/longlive)
#   WAN21_REV                Wan-AI/Wan2.1-T2V-1.3B revision (default 37ec5126..., the head since 2025-03, pinned source revision)
#   STREAMEVAL_PROXY       optional HTTPS proxy for downloads (default $https_proxy, else none)
#   HF_ENDPOINT              Hugging Face endpoint (default https://huggingface.co)
set -euo pipefail
STEP=${1:-all}
case $STEP in
  all|src|env|weights|check) ;;
  *) echo "usage: bash $0 [all|src|env|weights|check]" >&2; exit 2 ;;
esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${ROLLING_FORCING_ROOT:-$BASE/rolling_forcing}
LL_ROOT=${LONGLIVE_ROOT:-$BASE/longlive}
COMMIT=${ROLLING_FORCING_COMMIT:-a1477d09e85dc759a6a6728f55f77f59342ce388}  # TencentARC/RollingForcing main, 2025-10-31
HF_REV=20edee70781b8586d03f9a9a42b613f8fa8f556a                            # TencentARC/RollingForcing (HF)
WAN21_REV=${WAN21_REV:-37ec512624d61f7aa208f7ea8140a131f93afc9a}                                            # Wan-AI/Wan2.1-T2V-1.3B
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/inference.py ]]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL "https://codeload.github.com/TencentARC/RollingForcing/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1 --exclude='*.png' --exclude='*.jpg' --exclude='*.gif' --exclude='*.mp4'
  echo "$COMMIT" > repo_commit.txt
fi

if [[ $STEP == all || $STEP == env ]]; then
  bash "$HERE/../longlive/setup.sh" env
  [[ -e env ]] || ln -sfn "$LL_ROOT/env" env
fi

if [[ $STEP == all || $STEP == weights ]]; then
  mkdir -p weights/wan_models weights/checkpoints
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  [[ -f weights/checkpoints/rolling_forcing_dmd.pt ]] || https_proxy=$PROXY $DL -c "
from huggingface_hub import hf_hub_download
hf_hub_download('TencentARC/RollingForcing', 'checkpoints/rolling_forcing_dmd.pt', revision='$HF_REV', local_dir='weights')"
  if [[ ! -e weights/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth ]]; then
    https_proxy=$PROXY $DL -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', revision='$WAN21_REV', local_dir='weights/wan_models/Wan2.1-T2V-1.3B', ignore_patterns=['assets/*', 'examples/*'], max_workers=4)"
  fi
  ln -sfn "$ROOT/weights/wan_models" repo/wan_models                  # upstream resolves it relative to the repo
  ls -la weights/checkpoints weights/wan_models
fi

if [[ $STEP == all || $STEP == check ]]; then
  (cd repo && "$ROOT/$PY" -c "
import torch
from pipeline import CausalInferencePipeline
print('IMPORT_OK rolling_forcing on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))")
fi
echo "ROLLING_FORCING_SETUP_DONE ($STEP)"
