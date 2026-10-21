#!/bin/bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
mode=${1:-download}
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${MATRIX_GAME_35_ROOT:-$BASE/matrix_game_35}
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
CODE_SHA=fbf7def0693ae14f745ba35bf2a26215d4ef991d   # Riemann-Dynamics/Matrix-Game-3.5 main, 2026-09-07
MG_REV=0b38ca0b0dda2bb994c570e183ad36d1acd53be2     # RiemannDynamics/Matrix-Game-3.5-Distilled
DA3_REV=b2359bdf726fb44ef62acca04d629dcf158053e7    # depth-anything/DA3NESTED-GIANT-LARGE-1.1
WAN_REV=921dbaf3f1674a56f47e83fb80a34bac8a8f203e    # Wan-AI/Wan2.2-TI2V-5B
CKPT_SHA256=de476e7fc0bdd756aafb101a2b80040f65b3ad62dafea109e299aafa599b8094  # first-person.safetensors

download() {
  if [ ! -f "$ROOT/repo/infer_distilled.py" ]; then
    mkdir -p "$ROOT"
    https_proxy=$PROXY curl -sfL -o "$ROOT/repo.tar.gz" https://codeload.github.com/Riemann-Dynamics/Matrix-Game-3.5/tar.gz/$CODE_SHA
    rm -rf "$ROOT/repo" "$ROOT/Matrix-Game-3.5-$CODE_SHA"
    tar xzf "$ROOT/repo.tar.gz" -C "$ROOT" && mv "$ROOT/Matrix-Game-3.5-$CODE_SHA" "$ROOT/repo"
  fi
  export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_WORKERS=4
  W=$ROOT/weights
  mkdir -p "$W"
  fetch() {   # three tries; a download that still fails stops the script
    local i
    for i in 1 2 3; do https_proxy=$PROXY hf download "$@" && return 0; sleep 10; done
    echo "download failed: $1" >&2; return 1
  }
  fetch RiemannDynamics/Matrix-Game-3.5-Distilled --revision $MG_REV --local-dir "$W/Matrix-Game-3.5-Distilled"
  fetch depth-anything/DA3NESTED-GIANT-LARGE-1.1 --revision $DA3_REV --local-dir "$W/DA3NESTED-GIANT-LARGE-1.1"
  fetch Wan-AI/Wan2.2-TI2V-5B --revision $WAN_REV --exclude "assets/*" "examples/*" --local-dir "$W/Wan2.2-TI2V-5B"
  echo "$CKPT_SHA256  Matrix-Game-3.5-Distilled/first-person.safetensors" > "$W/weights.sha256"
  (cd "$W" && sha256sum -c weights.sha256)
}

env_build() {
# Use the dependencies and installation requirements documented in README.md.
  python3 -c "import torch" 2>/dev/null || {
    echo "python3 cannot import torch: env needs a base Python with a CUDA build of PyTorch (see the header)" >&2
    exit 1; }
  ENV=${MG_ENV:-$ROOT/env}
  python3 -m venv --system-site-packages "$ENV"
  PY=$ENV/bin/python
  PKGS="accelerate==1.15.0 transformers==5.17.0 sentencepiece==0.2.2 peft==0.21.0 modelscope==1.40.1
    huggingface_hub==1.32.0 safetensors==0.8.0 omegaconf==2.3.1 opencv-python-headless==5.0.0.93 OpenEXR==3.5.1
    imageio==2.38.0 imageio-ffmpeg==0.6.0 moviepy==1.0.3 pycolmap==4.2.0 trimesh==5.1.0 ftfy==6.3.1 decord==0.6.0
    einops PyYAML tqdm pandas pillow pycocotools"
  https_proxy=$PROXY $PY -m pip install -q $PKGS
  $PY -c "import torch, transformers, accelerate; print('torch', torch.__version__, torch.cuda.is_available())"
  cd "$ROOT/repo" && PYTHONPATH=.:third_party/depth-anything-3/src $PY -c "
import examples.wanvideo.pipeline.mosaic.causal_inference, examples.wanvideo.pipeline.mosaic.datasets, depth_anything_3.api
print('IMPORT_OK')"
}

case $mode in
  download) download ;;
  env) env_build ;;
  all) download; env_build ;;
  *) echo "usage: bash setup.sh [download|env|all]" >&2; exit 2 ;;
esac
