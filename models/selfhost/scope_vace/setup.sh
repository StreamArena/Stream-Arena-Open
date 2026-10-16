#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
case $STEP in all|src|env|weights|check) ;; *) echo "usage: $0 [all|src|env|weights|check]" >&2; exit 2;; esac
BASE=${STREAM_ARENA_MODEL_ROOT:-$HOME/.cache/stream_arena/models}
ROOT=${SCOPE_ROOT:-$BASE/scope_vace}
COMMIT=${SCOPE_COMMIT:-2aced4ded3513a76cd35f0dbfb42fbfbf5e98ab0}   # daydreamlive/scope v0.2.5, 2026-05-15
WAN_REV=d19e4465081f6ad9b1f61166e287453e4b8453f3                    # daydreamlive/Wan2.1-T2V-1.3B
COMFY_REV=441f3d5bd87f43de634698000157d893ddc99684                  # daydreamlive/WanVideo_comfy
SDV2_REV=76bb3e67e4ad5dcc1d91095cc54b7dd770240d04                   # daydreamlive/StreamDiffusionV2
PROXY=${STREAM_ARENA_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/src/scope/core/pipelines/streamdiffusionv2/pipeline.py ]]; then
  rm -rf repo && mkdir -p repo
  https_proxy=$PROXY curl -sfL --max-time 1800 "https://codeload.github.com/daydreamlive/scope/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1 --exclude='*/frontend' --exclude='*/app' --exclude='*/e2e' \
        --exclude='*.mp4' --exclude='*.png' --exclude='*.jpg'
  echo "$COMMIT" > repo_commit.txt
fi

# scope's uv.lock: diffusers 0.36.0, transformers 4.57.5, peft 0.18.1, accelerate 1.12.0, huggingface-hub 0.36.0.
# torch / torchvision / flash-attn / triton / torchao come from the base environment (see the header).
# sageattention is not installed (scope probes it and falls back).
if [[ $STEP == all || $STEP == env ]]; then
  if ! $PY -c "import torch, diffusers, transformers, peft, omegaconf, ftfy, einops, pluggy, flash_attn" 2>/dev/null; then
    python3 -c "import torch" 2>/dev/null || {
      echo "python3 has no torch: use a base Python whose site-packages provide a CUDA build of PyTorch (see the header)" >&2
      exit 1; }
    python3 -m venv --system-site-packages env
    PKGS="diffusers==0.36.0 transformers==4.57.5 accelerate==1.12.0 peft==0.18.1 huggingface_hub==0.36.0
      einops==0.8.1 omegaconf==2.3.0 ftfy==6.3.1 safetensors==0.7.0 pluggy easydict lmdb imageio regex
      sentencepiece protobuf pydantic"
    https_proxy=$PROXY $PY -m pip install -q $PKGS
  fi
  $PY -c "import torch, diffusers, transformers, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), 'diffusers', diffusers.__version__, 'transformers', transformers.__version__, 'flash_attn', flash_attn.__version__)"
  $PY -m pip freeze > env_freeze.txt 2>/dev/null || true
fi

if [[ $STEP == all || $STEP == weights ]]; then
  mkdir -p weights
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  $DL - <<EOF
from huggingface_hub import snapshot_download
for repo, rev, allow in [
        ("daydreamlive/Wan2.1-T2V-1.3B", "$WAN_REV", ["config.json", "Wan2.1_VAE.pth", "google/*"]),
        ("daydreamlive/WanVideo_comfy", "$COMFY_REV",
         ["config.json", "umt5-xxl-enc-fp8_e4m3fn.safetensors", "Wan2_1-VACE_module_1_3B_bf16.safetensors"]),
        ("daydreamlive/StreamDiffusionV2", "$SDV2_REV", ["config.json", "wan_causal_dmd_v2v/model.pt"])]:
    snapshot_download(repo, revision=rev, allow_patterns=allow, local_dir="weights/" + repo.split("/")[1], max_workers=4)
    print("downloaded", repo, rev)
EOF
  ls -la weights/*
fi

if [[ $STEP == all || $STEP == check ]]; then
  PYTHONPATH=repo/src $PY -c "
import torch
from scope.core.pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline
print('IMPORT_OK scope streamdiffusionv2 on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
fi
echo "SCOPE_SETUP_DONE ($STEP)"
