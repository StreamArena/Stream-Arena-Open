#!/bin/bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-fetch}
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${SOULX_FLASHHEAD_ROOT:-$BASE/soulx_flashhead}
COMMIT=${SOULX_FLASHHEAD_COMMIT:-9bc03de06bb0de82cd6bc477804512ae06144bf2}   # Soul-AILab/SoulX-FlashHead
W2V_REV=22aad52d435eb6dbaf354bdad9b0da84ce7d6156                             # facebook/wav2vec2-base-960h (unchanged since 2022-11-14)
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
PYTHON=${PYTHON:-python3}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
case $STEP in
  fetch|env) ;;
  *) echo "usage: bash setup.sh [fetch|env]" >&2; exit 2 ;;
esac
mkdir -p "$ROOT/weights"
cd "$ROOT"

if [[ $STEP == env ]]; then
  $PYTHON -c "import torch" 2>/dev/null || { echo "$PYTHON has no torch: this step needs a base Python whose site-packages provide a CUDA build of PyTorch " >&2; exit 1; }
  export PIP_CACHE_DIR=$ROOT/pip_cache
  rm -rf env
  $PYTHON -m venv --system-site-packages env
  $PYTHON -m pip list --format=freeze 2>/dev/null | grep -i -E \
    "^(torch|torchvision|torchaudio|triton|pytorch-triton|numpy|tensorrt|flash.attn|nvidia-|cuda-|apex|transformer.engine|onnx==)" \
    > env/constraints.txt || true
  P="env/bin/python -m pip install -q -c env/constraints.txt"
  $P "diffusers>=0.34.0" "transformers==4.57.3" "tokenizers>=0.20.3" "accelerate>=1.8.1" "peft>=0.17" easydict ftfy imageio \
     imageio-ffmpeg scikit-image loguru pyloudnorm librosa soundfile opencv-python-headless pillow pyyaml
  $P --no-deps "xfuser==0.7.0" "yunchang==0.6.4" || echo "xfuser unavailable: server.py stubs it (single GPU)"
  env/bin/python -c "import torch, flash_attn, diffusers, transformers, cv2, librosa
print('env ok', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
  exit 0
fi

if [ ! -f repo/generate_video.py ]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL "https://codeload.github.com/Soul-AILab/SoulX-FlashHead/tar.gz/$COMMIT" | tar xz -C repo --strip-components=1
  echo "$COMMIT" > repo_commit.txt
fi
# Use the dependencies and installation requirements documented in README.md.
[ -d weights/SoulX-FlashHead-1_3B/Model_Lite ] || https_proxy=$PROXY hf download Soul-AILab/SoulX-FlashHead-1_3B --local-dir weights/SoulX-FlashHead-1_3B
[ -f weights/wav2vec2-base-960h/config.json ] || https_proxy=$PROXY hf download facebook/wav2vec2-base-960h --revision "$W2V_REV" --local-dir weights/wav2vec2-base-960h
[ -s weights_sha256.txt ] || (cd weights && sha256sum SoulX-FlashHead-1_3B/Model_Lite/diffusion_pytorch_model.safetensors \
  SoulX-FlashHead-1_3B/VAE_LTX/diffusion_pytorch_model.safetensors wav2vec2-base-960h/model.safetensors > ../weights_sha256.txt)
echo "source + weights ready in $ROOT; now build env/ on the GPU machine: setup.sh env"
