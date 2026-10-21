#!/usr/bin/env bash
# ShotStream (Wan2.1-T2V-1.3B causal multi-shot T2V) self-host setup. Idempotent: each step skips what is already in place.
#
#   bash models/selfhost/shotstream/setup.sh [all|src|env|weights|check]
#
# Steps:
#   src      download KlingAIResearch/ShotStream at the pinned commit (a large codeload tarball because of demo data;
#            media files are skipped when unpacking; network, no GPU)
#   env      run `longlive/setup.sh env` (builds the shared causal Wan venv if missing) and link env -> it
#   weights  download ShotStream checkpoint + configs (pinned revision) and Wan2.1-T2V-1.3B (about 23 GB; network)
#   check    import the upstream pipeline and print the GPU (GPU)
#   all      every step above, in order (default)
#
# Prerequisite: the environment is the venv of models/selfhost/longlive (one environment for the causal Wan
# models), built by `longlive/setup.sh env` over a base Python whose site-packages provide a CUDA build of
# PyTorch for your GPU.
# See the header of longlive/setup.sh.
#
# Layout under the model root $ROOT (not in git):
#   repo/      KlingAIResearch/ShotStream at the pinned commit (media skipped); repo/ckpts links to weights/ShotStream
#              and repo/wan_models to weights/wan_models (upstream resolves both relative to the repo)
#   env        link to $LONGLIVE_ROOT/env
#   weights/   ShotStream/{shotstream_merged.pt,shotstream.yaml,default_config.yaml}, wan_models/Wan2.1-T2V-1.3B
# `check` puts models/selfhost/_shims on PYTHONPATH (flash_attn API on torch SDPA) as server.py does.
#
# Environment variables:
#   STREAMEVAL_MODEL_ROOT  parent of all model roots (default ~/.cache/streameval/models)
#   SHOTSTREAM_ROOT          this model's root (default $STREAMEVAL_MODEL_ROOT/shotstream)
#   SHOTSTREAM_COMMIT        upstream commit to download (pinned default)
#   SHOTSTREAM_HF_REV        KlingTeam/ShotStream revision (pinned default)
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
ROOT=${SHOTSTREAM_ROOT:-$BASE/shotstream}
LL_ROOT=${LONGLIVE_ROOT:-$BASE/longlive}
COMMIT=${SHOTSTREAM_COMMIT:-c6a11719ee8ac203ee20afc27056446fe25cef86}      # KlingAIResearch/ShotStream main, 2026-09-16
HF_REV=${SHOTSTREAM_HF_REV:-225806baab3c54e94351474e976948eb5c61b6fe}      # KlingTeam/ShotStream (HF), 2026-09-16
WAN21_REV=${WAN21_REV:-37ec512624d61f7aa208f7ea8140a131f93afc9a}                                            # Wan-AI/Wan2.1-T2V-1.3B
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/Inference_Causal.py ]]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL --max-time 3600 "https://codeload.github.com/KlingAIResearch/ShotStream/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1 --exclude='*.mp4' --exclude='*.gif' --exclude='*.png' --exclude='*.jpg'
  echo "$COMMIT" > repo_commit.txt
fi

if [[ $STEP == all || $STEP == env ]]; then
  bash "$HERE/../longlive/setup.sh" env
  [[ -e env ]] || ln -sfn "$LL_ROOT/env" env
fi

if [[ $STEP == all || $STEP == weights ]]; then
  mkdir -p weights/wan_models weights/ShotStream
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  for f in shotstream_merged.pt shotstream.yaml default_config.yaml; do
    [[ -f weights/ShotStream/$f ]] || https_proxy=$PROXY $DL -c "
from huggingface_hub import hf_hub_download
hf_hub_download('KlingTeam/ShotStream', '$f', revision='$HF_REV', local_dir='weights/ShotStream')"
  done
  if [[ ! -e weights/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth ]]; then
    https_proxy=$PROXY $DL -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', revision='$WAN21_REV', local_dir='weights/wan_models/Wan2.1-T2V-1.3B', ignore_patterns=['assets/*', 'examples/*'], max_workers=4)"
  fi
  ln -sfn "$ROOT/weights/wan_models" repo/wan_models
  ln -sfn "$ROOT/weights/ShotStream" repo/ckpts
  ls -la weights/ShotStream weights/wan_models
fi

if [[ $STEP == all || $STEP == check ]]; then
  (cd repo && PYTHONPATH="$HERE/../_shims" "$ROOT/$PY" -c "
import sys, types, torch
import flash_attn
sys.modules.setdefault('flask', types.SimpleNamespace(config=None))   # model/base.py imports flask.config, unused
import transformers.utils.import_utils                                # probes decord before the stub below
sys.modules.setdefault('decord', types.SimpleNamespace())             # utils/dataset.py (video reading), unused
from pipeline.causal_inference_ar import CausalInferenceArPipeline
print('IMPORT_OK shotstream on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flash_attn.__version__)")
fi
echo "SHOTSTREAM_SETUP_DONE ($STEP)"
