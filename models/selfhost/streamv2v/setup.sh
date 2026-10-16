#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
case $STEP in
  all|src|env|weights|check) ;;
  *) echo "usage: bash setup.sh [all|src|env|weights|check]" >&2; exit 2 ;;
esac
BASE=${STREAM_ARENA_MODEL_ROOT:-$HOME/.cache/stream_arena/models}
ROOT=${SV2V_ROOT:-$BASE/streamv2v}
COMMIT=${SV2V_COMMIT:-b35f3877e3bc21e960dfe2e4ca54eb5b15bc1650}     # Jeff-LiangF/streamv2v main, 2025-12-29
DIFFUSERS_FORK=3e3b72f557e91546894340edabc845e894f00922             # varshith15/diffusers (upstream setup.py)
SD15_REV=f94658e11151229e2626d5f375450a3412bf6766                   # Jiali/stable-diffusion-1.5
LCM_REV=cf2fced511dbe7e26c8d1d397e728fbab875db4b                    # latent-consistency/lcm-lora-sdv1-5
TAESD_REV=614f76814bbe30edbe2e627ace1c2234c81a2c0e                  # madebyollin/taesd
PROXY=${STREAM_ARENA_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_MAX_WORKERS=4
mkdir -p "$ROOT"
cd "$ROOT"
PY=env/bin/python
pipi() { https_proxy=$PROXY $PY -m pip install -q "$@"; }   # PIP_INDEX_URL is honoured by pip itself

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/utils/wrapper.py ]]; then
  rm -rf repo && mkdir -p repo
  https_proxy=$PROXY curl -sfL --max-time 1800 "https://codeload.github.com/Jeff-LiangF/streamv2v/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1 --exclude='*/frontend' --exclude='*.png' --exclude='*.jpg' \
        --exclude='*.gif' --exclude='*.mp4'
  echo "$COMMIT" > repo_commit.txt
fi

if [[ $STEP == all || $STEP == env ]]; then
  if ! $PY -c "import torch, diffusers, transformers, peft, polygraphy, onnx_graphsurgeon, tensorrt, fire, einops; from cuda import cudart" 2>/dev/null; then
    python3 -c "import torch" 2>/dev/null || {
      echo "python3 has no torch: env needs a base Python whose site-packages provide a CUDA build of PyTorch" >&2
      exit 1; }
    python3 -m venv --system-site-packages env
    if [[ ! -f src_diffusers/setup.py ]]; then
      rm -rf src_diffusers && mkdir -p src_diffusers
      https_proxy=$PROXY curl -sfL --max-time 1800 "https://codeload.github.com/varshith15/diffusers/tar.gz/$DIFFUSERS_FORK" \
        | tar xz -C src_diffusers --strip-components=1 --exclude='*/docs' --exclude='*/tests' --exclude='*/examples'
    fi
    pipi --no-deps ./src_diffusers
    pipi "transformers==4.57.3" "accelerate>=1.3" "peft==0.17.1" "huggingface_hub>=0.34,<1" importlib_metadata \
      fire omegaconf einops polygraphy onnx-graphsurgeon "cuda-python>=12.6,<13"
    $PY -c "import tensorrt" 2>/dev/null || pipi "tensorrt==10.9.0.34"   # import-time only (upstream extra pin)
  fi
# Use the dependencies and installation requirements documented in README.md.
  $PY -c "import peft, sys; sys.exit(peft.__version__ != '0.17.1')" 2>/dev/null || pipi "peft==0.17.1"
  $PY -c "import torch, diffusers, transformers, tensorrt; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), 'diffusers', diffusers.__version__, 'transformers', transformers.__version__, 'tensorrt', tensorrt.__version__)"
  $PY -m pip freeze > env_freeze.txt 2>/dev/null || true
fi

if [[ $STEP == all || $STEP == weights ]]; then
  mkdir -p weights
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  $DL - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("Jiali/stable-diffusion-1.5", revision="$SD15_REV", local_dir="weights/stable-diffusion-1.5",
                  max_workers=4, allow_patterns=["model_index.json", "*/config.json", "*/*_config.json",
                  "scheduler/*", "tokenizer/*", "feature_extractor/*", "unet/diffusion_pytorch_model.safetensors",
                  "text_encoder/model.safetensors", "vae/diffusion_pytorch_model.safetensors"])
snapshot_download("latent-consistency/lcm-lora-sdv1-5", revision="$LCM_REV", local_dir="weights/lcm-lora-sdv1-5",
                  allow_patterns=["pytorch_lora_weights.safetensors", "*.json"])
snapshot_download("madebyollin/taesd", revision="$TAESD_REV", local_dir="weights/taesd",
                  allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
print("downloaded SD1.5 $SD15_REV, LCM-LoRA $LCM_REV, taesd $TAESD_REV")
EOF
  # the demo loads SD 1.5 without its safety checker (use_safety_checker=False); drop it from the pipeline index
  $DL - <<'EOF'
import json
p = "weights/stable-diffusion-1.5/model_index.json"
d = json.load(open(p))
d["safety_checker"] = [None, None]
d["requires_safety_checker"] = False
json.dump(d, open(p, "w"), indent=2)
EOF
  ls -la weights/*
fi

if [[ $STEP == all || $STEP == check ]]; then
  (cd repo && PYTHONPATH=.:src ../$PY -c "
import torch
from utils.wrapper import StreamV2VWrapper
print('IMPORT_OK streamv2v on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))")
fi
echo "SV2V_SETUP_DONE ($STEP)"
