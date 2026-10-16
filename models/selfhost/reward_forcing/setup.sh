#!/usr/bin/env bash
# Reward Forcing (Wan2.1-T2V-1.3B) self-host setup. Idempotent: each step skips what is already in place.
#
#   bash models/selfhost/reward_forcing/setup.sh [all|src|env|weights|check]
#
# Steps:
#   src      download JaydenLyh/Reward-Forcing at the pinned commit (small codeload tarball; network, no GPU)
#   env      run `longlive/setup.sh env` (builds the shared causal Wan venv if missing) and link env -> it
#   weights  download the Reward Forcing checkpoint (pinned revision) and Wan2.1-T2V-1.3B (about 23 GB; network)
#   check    import the upstream pipeline and print the GPU (GPU)
#   all      every step above, in order (default)
#
# Prerequisite: the environment is the venv of models/selfhost/longlive (one environment for the causal Wan
# models), built by `longlive/setup.sh env` over a base Python whose site-packages provide a CUDA build of
# PyTorch for your GPU.
# See the header of longlive/setup.sh.
#
# Layout under the model root $ROOT (not in git):
#   repo/      JaydenLyh/Reward-Forcing at the pinned commit (media files skipped)
#   env        link to $LONGLIVE_ROOT/env
#   weights/   Reward-Forcing-T2V-1.3B/rewardforcing.pt, Wan2.1-T2V-1.3B; this directory takes the place of
#              upstream's checkpoints/ (server.py points the config's checkpoint_path here)
#
# Environment variables:
#   STREAM_ARENA_MODEL_ROOT  parent of all model roots (default ~/.cache/stream_arena/models)
#   REWARD_FORCING_ROOT      this model's root (default $STREAM_ARENA_MODEL_ROOT/reward_forcing)
#   REWARD_FORCING_COMMIT    upstream commit to download (pinned default)
#   LONGLIVE_ROOT            root of the shared environment (default $STREAM_ARENA_MODEL_ROOT/longlive)
#   WAN21_REV                Wan-AI/Wan2.1-T2V-1.3B revision (default 37ec5126..., the head since 2025-03, pinned source revision)
#   STREAM_ARENA_PROXY       optional HTTPS proxy for downloads (default $https_proxy, else none)
#   HF_ENDPOINT              Hugging Face endpoint (default https://huggingface.co)
#   If $STREAM_ARENA_MODEL_ROOT/streamdiffusionv2/weights/wan_models/Wan2.1-T2V-1.3B exists, it is linked as
#   weights/Wan2.1-T2V-1.3B instead of downloading Wan2.1-T2V-1.3B again.
set -euo pipefail
STEP=${1:-all}
case $STEP in
  all|src|env|weights|check) ;;
  *) echo "usage: bash $0 [all|src|env|weights|check]" >&2; exit 2 ;;
esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${STREAM_ARENA_MODEL_ROOT:-$HOME/.cache/stream_arena/models}
ROOT=${REWARD_FORCING_ROOT:-$BASE/reward_forcing}
LL_ROOT=${LONGLIVE_ROOT:-$BASE/longlive}
COMMIT=${REWARD_FORCING_COMMIT:-4317f5a0d29d1cee6a0f2e80ed5f5724bee97c66}  # JaydenLyh/Reward-Forcing main, 2025-12-15
HF_REV=fc7e9400d1cf0855431d7a65d946e0921effc7e4                           # JaydenLu666/Reward-Forcing-T2V-1.3B
WAN21_REV=${WAN21_REV:-37ec512624d61f7aa208f7ea8140a131f93afc9a}                                            # Wan-AI/Wan2.1-T2V-1.3B
PROXY=${STREAM_ARENA_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/inference.py ]]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL "https://codeload.github.com/JaydenLyh/Reward-Forcing/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1 --exclude='*.png' --exclude='*.jpg' --exclude='*.gif' --exclude='*.mp4'
  echo "$COMMIT" > repo_commit.txt
fi

if [[ $STEP == all || $STEP == env ]]; then
  bash "$HERE/../longlive/setup.sh" env
  [[ -e env ]] || ln -sfn "$LL_ROOT/env" env
fi

if [[ $STEP == all || $STEP == weights ]]; then
  mkdir -p weights/Reward-Forcing-T2V-1.3B
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  [[ -f weights/Reward-Forcing-T2V-1.3B/rewardforcing.pt ]] || https_proxy=$PROXY $DL -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JaydenLu666/Reward-Forcing-T2V-1.3B', 'rewardforcing.pt', revision='$HF_REV', local_dir='weights/Reward-Forcing-T2V-1.3B')"
  if [[ ! -e weights/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth ]]; then
    SHARED=$BASE/streamdiffusionv2/weights/wan_models/Wan2.1-T2V-1.3B   # if already downloaded
    if [[ -f $SHARED/Wan2.1_VAE.pth && -f $SHARED/diffusion_pytorch_model.safetensors ]]; then
      ln -sfn "$SHARED" weights/Wan2.1-T2V-1.3B
    else
      https_proxy=$PROXY $DL -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', revision='$WAN21_REV', local_dir='weights/Wan2.1-T2V-1.3B', ignore_patterns=['assets/*', 'examples/*'], max_workers=4)"
    fi
  fi
  ls -la weights/Reward-Forcing-T2V-1.3B weights/
fi

if [[ $STEP == all || $STEP == check ]]; then
  (cd repo && "$ROOT/$PY" -c "
import torch
from pipeline import CausalInferencePipeline
print('IMPORT_OK reward_forcing on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))")
fi
echo "REWARD_FORCING_SETUP_DONE ($STEP)"
