#!/usr/bin/env bash
# MemFlow (Wan2.1-T2V-1.3B causal T2V with a memory bank) self-host setup. Idempotent: each step skips what is already in place.
#
#   bash models/selfhost/memflow/setup.sh [all|src|env|weights|check]
#
# Steps:
#   src      download KlingAIResearch/MemFlow at the pinned commit (1.4 MB codeload tarball; network, no GPU)
#   env      run `longlive/setup.sh env` (builds the shared causal Wan venv if missing) and link env -> it
#   weights  download MemFlow base.pt + lora.pt (pinned revision) and Wan2.1-T2V-1.3B (about 26 GB; network)
#   check    import the upstream pipeline and print the GPU (GPU)
#   all      every step above, in order (default)
#
# Prerequisite: the environment is the venv of models/selfhost/longlive (one environment for the causal Wan
# models), built by `longlive/setup.sh env` over a base Python whose site-packages provide a CUDA build of
# PyTorch for your GPU.
# See the header of longlive/setup.sh.
#
# Layout under the model root $ROOT (not in git):
#   repo/      KlingAIResearch/MemFlow at the pinned commit (media skipped); repo/wan_models links to weights/wan_models
#   env        link to $LONGLIVE_ROOT/env
#   weights/   MemFlow/{base,lora}.pt, wan_models/Wan2.1-T2V-1.3B
#
# Environment variables:
#   STREAMEVAL_MODEL_ROOT  parent of all model roots (default ~/.cache/streameval/models)
#   MEMFLOW_ROOT             this model's root (default $STREAMEVAL_MODEL_ROOT/memflow)
#   MEMFLOW_COMMIT           upstream commit to download (pinned default)
#   MEMFLOW_HF_REV           KlingTeam/MemFlow revision (pinned default)
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
ROOT=${MEMFLOW_ROOT:-$BASE/memflow}
LL_ROOT=${LONGLIVE_ROOT:-$BASE/longlive}
COMMIT=${MEMFLOW_COMMIT:-7ed514771f5add2af7cc60889e8310ce2b70ab1e}       # KlingAIResearch/MemFlow main, 2025-12-29
HF_REV=${MEMFLOW_HF_REV:-c41043fd7df4576e62603d3eb40b25aaa5542022}  # KlingTeam/MemFlow (HF)
WAN21_REV=${WAN21_REV:-37ec512624d61f7aa208f7ea8140a131f93afc9a}                                            # Wan-AI/Wan2.1-T2V-1.3B
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/interactive_inference.py ]]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL --max-time 900 "https://codeload.github.com/KlingAIResearch/MemFlow/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1 --exclude='*.mp4' --exclude='*.gif'
  echo "$COMMIT" > repo_commit.txt
fi

if [[ $STEP == all || $STEP == env ]]; then
  bash "$HERE/../longlive/setup.sh" env
  [[ -e env ]] || ln -sfn "$LL_ROOT/env" env
fi

if [[ $STEP == all || $STEP == weights ]]; then
  mkdir -p weights/wan_models weights/MemFlow
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  for f in base.pt lora.pt; do
    [[ -f weights/MemFlow/$f ]] || https_proxy=$PROXY $DL -c "
from huggingface_hub import hf_hub_download
hf_hub_download('KlingTeam/MemFlow', '$f', revision='$HF_REV', local_dir='weights/MemFlow')"
  done
  if [[ ! -e weights/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth ]]; then
    https_proxy=$PROXY $DL -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', revision='$WAN21_REV', local_dir='weights/wan_models/Wan2.1-T2V-1.3B', ignore_patterns=['assets/*', 'examples/*'], max_workers=4)"
  fi
  [[ -d repo ]] || { echo "memflow: run 'setup.sh src' before 'weights'" >&2; exit 1; }
  ln -sfn "$ROOT/weights/wan_models" repo/wan_models                  # upstream resolves it relative to the repo
  ls -la weights/MemFlow weights/wan_models
fi

if [[ $STEP == all || $STEP == check ]]; then
  (cd repo && "$ROOT/$PY" -c "
import torch
import peft
from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
print('IMPORT_OK memflow on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))")
fi
echo "MEMFLOW_SETUP_DONE ($STEP)"
