#!/bin/bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail

BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
JOY_ROOT=${JOY_ROOT:-$BASE/joyai_video_edit}
REPO_URL=https://github.com/jd-opensource/JoyAI-Video-Edit
REPO_COMMIT=ca17e1d1030f454cb98b0ed549b4d31a60139ceb
SAGE_COMMIT=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
CUTLASS_COMMIT=dcf215af68a2d08d305076c152a06f201728cd53
JOYAI_REV=${JOYAI_HF_REV:-39491dda89fd2b535777fd8644a6631ea89b857c}   # jdopensource/JoyAI-Video-Edit
MIMO_REV=${MIMO_VL_REV:-main}                                        # unpinned: the v1 revision was not recorded
CF=${CONDA_CHANNEL:-conda-forge}
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
ENV=$JOY_ROOT/env
MAMBA=$JOY_ROOT/mamba/bin/micromamba
export MAMBA_ROOT_PREFIX=$JOY_ROOT/mamba/root
export PIP_CACHE_DIR=$JOY_ROOT/pipcache
mkdir -p "$JOY_ROOT/logs"

with_proxy() { https_proxy=$PROXY "$@"; }

do_clone() {
  pin "$JOY_ROOT/repo" "$REPO_URL" "$REPO_COMMIT"
}

do_weights() {
  local W=$JOY_ROOT/weights
  mkdir -p "$W"
  local HF; HF=$(command -v hf || echo "$ENV/bin/hf")
  with_proxy "$HF" download jdopensource/JoyAI-Video-Edit --repo-type model --revision "$JOYAI_REV" \
    --local-dir "$W/JoyAI-Video-Edit" --include "dit/joyai_video_edit_dit_0811.pth" "vae/*"
  with_proxy "$HF" download XiaomiMiMo/MiMo-VL-7B-RL-2508 --repo-type model --revision "$MIMO_REV" --local-dir "$W/MiMo-VL-7B-RL-2508"
  # YuNet: upstream's face-presence gate (DEPLOYMENT.md, opencv_zoo main); the gate is off in server.py
  [ -s "$W/face_detection_yunet_2023mar.onnx" ] || with_proxy curl -sfL -o "$W/face_detection_yunet_2023mar.onnx" \
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  # yolov8n person gate is optional and unused by server.py (gates are off).
  (cd "$W" && sha256sum JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth > dit_0811.sha256 \
     && sha256sum JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors > vae.sha256)
}

do_env() {
  if [ ! -x "$MAMBA" ]; then
    mkdir -p "$JOY_ROOT/mamba/bin"
    with_proxy curl -sfL -o "$MAMBA" \
      https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
    chmod +x "$MAMBA"
  fi
# Use the dependencies and installation requirements documented in README.md.
  local PKGS="cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-version=12.9 libcublas-dev libcusparse-dev libcusolver-dev cuda-profiler-api"
  if [ ! -x "$ENV/bin/python" ]; then
    with_proxy "$MAMBA" create -y -p "$ENV" --override-channels -c $CF python=3.10 git $PKGS
  elif ! "$ENV/bin/nvcc" --version | grep -q "release 12.9"; then
    with_proxy "$MAMBA" install -y -p "$ENV" --override-channels -c $CF $PKGS
  fi
  # PyPI (or $PIP_INDEX_URL) + download.pytorch.org (named in upstream's requirements); uv when available
  local IDX=(); [ -n "${PIP_INDEX_URL:-}" ] && IDX=(--index-url "$PIP_INDEX_URL")
  pipi() {
    if command -v uv >/dev/null; then
      with_proxy env UV_CACHE_DIR=$JOY_ROOT/uvcache uv pip install -p "$ENV/bin/python" \
        --index-strategy unsafe-best-match ${IDX[@]+"${IDX[@]}"} "$@"
    else
      with_proxy "$ENV/bin/python" -m pip install ${IDX[@]+"${IDX[@]}"} "$@"
    fi
  }
  pipi -r "$JOY_ROOT/repo/deploy/requirements.txt" setuptools wheel
  # FA4 (B200/B300 attention path; JIT kernels, no build), pinned as in DEPLOYMENT.md
  pipi flash-attn-4==4.0.0b13 nvidia-cutlass-dsl==4.5.1 quack-kernels==0.4.1 apache-tvm-ffi==0.1.12 \
    || echo "flash-attn-4 unavailable -> cuDNN SDPA"
# Use the dependencies and installation requirements documented in README.md.
  pipi --no-deps nvidia-cuda-nvrtc-cu12==12.9.86
}

pin() {   # pin <dir> <url> <commit>: clone once, then force the checkout (repairs interrupted clones)
  [ -d "$1/.git" ] || with_proxy git clone "$2" "$1"
  rm -f "$1/.git/index.lock"
  [ "$(git -C "$1" rev-parse HEAD)" = "$(git -C "$1" rev-parse "$3^{commit}")" ] && [ -z "$(git -C "$1" status --porcelain -uno)" ] \
    && return 0
  git -C "$1" checkout -q -f "$3"
  git -C "$1" clean -qfdx
}

do_fetch() {
  local T=$JOY_ROOT/build
  mkdir -p "$T"
  pin "$T/cutlass" https://github.com/NVIDIA/cutlass.git "$CUTLASS_COMMIT"
  if ! git -C "$T/SageAttention" apply -R --check "$JOY_ROOT/repo/deploy/sageattention-cudagraph-stream.patch" 2>/dev/null; then
    pin "$T/SageAttention" https://github.com/thu-ml/SageAttention.git "$SAGE_COMMIT"
    git -C "$T/SageAttention" apply "$JOY_ROOT/repo/deploy/sageattention-cudagraph-stream.patch"
  fi
}

do_build() {
  # Needs the target GPU. The conda toolchain keeps CUDA headers under targets/.
  export PATH=$ENV/bin:$PATH CUDA_HOME=$ENV CPATH=$ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}
  export LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
  local T=$JOY_ROOT/build CAP JOBS=${MAX_JOBS:-8}
  do_fetch
  CAP=$("$ENV/bin/python" -c "import torch; c=torch.cuda.get_device_capability(0); print(f'{c[0]}.{c[1]}')")
  echo "build for compute capability $CAP"
  # SageAttention only pays off on GeForce RTX 5090 (DEPLOYMENT.md); B-class GPUs use FA4 / cuDNN.
  if [ "$CAP" = "12.0" ] && ! "$ENV/bin/python" -c "import sageattention" 2>/dev/null; then
    (cd "$T/SageAttention" && TORCH_CUDA_ARCH_LIST=$CAP EXT_PARALLEL=2 NVCC_APPEND_FLAGS="--threads 2" MAX_JOBS=$JOBS \
      "$ENV/bin/python" setup.py install)
  fi
  # (re)build unless an FP8-capable joyomni_ops is already installed
  if ! "$ENV/bin/python" -c "import sys, joyomni_ops; sys.exit(0 if joyomni_ops.has_fp8() else 1)" 2>/dev/null; then
    local ARCH=${CAP/./}a
    if ! JOYOMNI_OPS_CUDA_ARCHS=$ARCH JOYOMNI_OPS_CUTLASS_DIR=$T/cutlass MAX_JOBS=$JOBS \
        "$ENV/bin/python" -m pip install --no-build-isolation "$JOY_ROOT/repo/deploy/joyomni_ops"; then
      echo "FP8 build for sm_$ARCH failed -> light joyomni_ops (bf16 Linears; server sets JOYOMNI_FP8_*=0)"
      JOYOMNI_OPS_NO_FP8=1 JOYOMNI_OPS_CUDA_ARCHS=${CAP/./} MAX_JOBS=$JOBS \
        "$ENV/bin/python" -m pip install --no-build-isolation "$JOY_ROOT/repo/deploy/joyomni_ops"
    fi
  fi
}

do_check() {
  "$ENV/bin/python" - <<'PY'
import torch, cv2, av, transformers, diffusers
jit = torch.cuda.jiterator._create_jit_fn("template <typename T> T f(T x) { return x * x; }")
print("nvrtc jiterator OK", jit(torch.full((4,), 3.0, device="cuda")).tolist())
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
for m in ("sageattention", "joyomni_ops", "flash_attn.cute"):
    try:
        mod = __import__(m, fromlist=["_"]); print(m, "OK", getattr(mod, "has_fp8", lambda: "")())
    except Exception as e:
        print(m, "MISSING", e)
PY
}

case "${1:-all}" in
  all) do_clone; do_fetch; do_env; do_weights; do_build; do_check ;;
  clone) do_clone ;;
  weights) do_weights ;;
  env) do_env ;;
  fetch) do_fetch ;;
  build) do_build ;;
  check) do_check ;;
  gpu) do_env; do_build; do_check ;;
  *) echo "usage: $0 {all|clone|weights|fetch|env|build|check|gpu}" >&2; exit 2 ;;
esac
