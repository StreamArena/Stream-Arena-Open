#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${KREA_ROOT:-$BASE/krea_realtime}
COMMIT=acf4b7b4e1a3049a0e04782e751d12203334038d   # krea-ai/realtime-video main (last push 2025-11-13)
REV=9a2ceea280e5d584c6a28d384a9c19e38c810f8c      # krea/krea-realtime-video
WAN13_REV=${WAN13_REV:-37ec512624d61f7aa208f7ea8140a131f93afc9a}   # Wan-AI/Wan2.1-T2V-1.3B
WAN14_REV=${WAN14_REV:-main}                                       # unpinned: the v1 revision was not recorded
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
STEP=${1:-all}
case $STEP in all|repo|env|weights|check) ;; *) echo "usage: $0 [all|repo|env|weights|check]" >&2; exit 2;; esac
mkdir -p "$ROOT"/{logs,weights/checkpoints,weights/wan_models}
PY=$ROOT/env/bin/python
export UV_PYTHON_INSTALL_DIR=$ROOT/py UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DISABLE_XET=1

repo() {
  if [[ ! -f $ROOT/repo/release_server.py ]]; then
    mkdir -p "$ROOT/repo"
    https_proxy=$PROXY curl -sfL "https://codeload.github.com/krea-ai/realtime-video/tar.gz/$COMMIT" \
      | tar xz -C "$ROOT/repo" --strip-components=1 --exclude='*.whl' --exclude='*.mp4' --exclude='*.ipynb'
  fi
  echo "$COMMIT" > "$ROOT/repo_commit.txt"
}

env_() {
  UV=${UV_BIN:-$(command -v uv || true)}
  [[ -n $UV ]] || { echo "uv not found: install it (https://docs.astral.sh/uv/) or set UV_BIN" >&2; exit 1; }
  [[ -x $PY ]] || { https_proxy=$PROXY "$UV" python install 3.11 && "$UV" venv -p 3.11 "$ROOT/env"; }
  if ! "$PY" -c "import torch, diffusers, transformers, fastapi, multipart, msgpack, cv2, dotenv, omegaconf, setuptools; assert torch.__version__.startswith('2.10.0')" 2>/dev/null; then
    https_proxy=$PROXY "$UV" pip install -p "$PY" torch==2.10.0 torchvision==0.25.0 triton==3.6.0 \
      --index-url https://download.pytorch.org/whl/cu130
    # inference subset of upstream pyproject.toml pins (training / jupyter / tensorrt / dashscope not needed)
    https_proxy=$PROXY "$UV" pip install -p "$PY" torch==2.10.0 torchvision==0.25.0 triton==3.6.0 \
      diffusers==0.31.0 transformers==4.54.1 accelerate==1.9.0 omegaconf==2.3.0 easydict==1.13 einops==0.8.1 \
      ftfy==6.3.1 regex sentencepiece protobuf safetensors==0.5.3 tqdm imageio==2.37.0 imageio-ffmpeg==0.6.0 \
      opencv-python-headless numpy==2.1.2 scipy fastapi==0.116.1 uvicorn==0.35.0 pydantic msgpack==1.1.1 \
      python-dotenv==1.1.1 python-multipart setuptools requests av huggingface_hub==0.34.3 pillow \
      --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130
  fi
# Use the dependencies and installation requirements documented in README.md.
  "$PY" -c "import torch, sys; sys.exit(torch.backends.cudnn.version() < 92600)" 2>/dev/null || \
    https_proxy=$PROXY "$UV" pip install -p "$PY" --no-deps "nvidia-cudnn-cu13==9.26.0.51"
  "$PY" - <<'EOF'
import torch, diffusers, transformers
print("env ok torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
      "diffusers", diffusers.__version__, "transformers", transformers.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
EOF
  "$UV" pip freeze -p "$PY" > "$ROOT/env_freeze.txt"
}

weights() {
  local W=$ROOT/weights
  if [[ ! -f $W/checkpoints/.complete ]]; then
    https_proxy=$PROXY "$PY" -c "
from huggingface_hub import hf_hub_download
hf_hub_download('krea/krea-realtime-video', 'krea-realtime-video-14b.safetensors', revision='$REV', local_dir='$W/checkpoints')"
    echo "krea/krea-realtime-video@$REV" > "$W/checkpoints/.complete"
  fi
  local D=$W/wan_models/Wan2.1-T2V-1.3B
  mkdir -p "$D"
  if [[ -e $D/models_t5_umt5-xxl-enc-bf16.pth ]]; then
    :
  else
    https_proxy=$PROXY "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', revision='$WAN13_REV', local_dir='$D', max_workers=4,
                  allow_patterns=['Wan2.1_VAE.pth', 'google/*', 'config.json', 'models_t5_umt5-xxl-enc-bf16.pth'])"
  fi
  if [[ ! -f $D/models_t5_umt5-xxl-enc-bf16.safetensors ]]; then
    "$PY" - "$D" <<'EOF'
import sys, torch
from safetensors.torch import save_file
d = sys.argv[1]
sd = torch.load(f"{d}/models_t5_umt5-xxl-enc-bf16.pth", map_location="cpu", weights_only=True)
save_file({k: v.contiguous() for k, v in sd.items()}, f"{d}/models_t5_umt5-xxl-enc-bf16.safetensors.part")
import os; os.rename(f"{d}/models_t5_umt5-xxl-enc-bf16.safetensors.part", f"{d}/models_t5_umt5-xxl-enc-bf16.safetensors")
print("t5 converted", len(sd), "tensors")
EOF
  fi
  mkdir -p "$W/wan_models/Wan2.1-T2V-14B"
  [[ -s $W/wan_models/Wan2.1-T2V-14B/config.json ]] || https_proxy=$PROXY "$PY" -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Wan-AI/Wan2.1-T2V-14B', 'config.json', revision='$WAN14_REV', local_dir='$W/wan_models/Wan2.1-T2V-14B')"
  cat "$W/wan_models/Wan2.1-T2V-14B/config.json"; echo
  ls -la "$W/checkpoints" "$D"
}

check() {
  (cd "$ROOT/repo" && MODEL_FOLDER=$ROOT/weights/wan_models DISABLE_SAGEATTENTION=1 "$PY" -c "
import torch, release_server, pipeline
from wan.modules.causal_model import CausalWanModel
c = CausalWanModel.load_config('$ROOT/weights/wan_models/Wan2.1-T2V-14B')
print('IMPORT_OK krea on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), {k: c[k] for k in ('dim', 'num_layers', 'num_heads')})")
}

case $STEP in
  repo) repo ;;
  env) env_ ;;
  weights) weights ;;
  check) check ;;
  all) repo; env_; weights; check ;;
esac
echo "KREA_SETUP_DONE ($STEP)"
