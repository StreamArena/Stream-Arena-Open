#!/usr/bin/env bash
# SoulX-LiveAct 18B self-host setup: source tree, CUDA 12.8 toolkit subset, Python environment, weights and weight
# hashes for the server in this folder (docs/protocol.md). Idempotent: each step skips when its output exists.
#
#   bash models/selfhost/soulx_liveact/setup.sh [all|repo|cuda|weights|env|hash|env+hash]
#
#   repo      network: Soul-AILab/SoulX-LiveAct at the pinned commit (codeload tarball)
#   cuda      local copy only: nvcc, nvvm, headers and cudart of a CUDA 12.8 toolkit (427 MB) into cuda-12.8/
#   weights   network, 53 GB: Soul-AILab/LiveAct and TencentGameMate/chinese-wav2vec2-base at pinned revisions
#   env       network + GPU machine: uv venv (Python 3.10, torch 2.8.0+cu128), builds SageAttention and LightX2V
#             (several CPU cores, some minutes); needs the `cuda` step
#   hash      sha256 of every file the server loads -> weights_hash.json (reads 53 GB)
#   all       repo, cuda, weights, env, hash
#
# Prerequisites: curl, tar, git, python3; the Hugging Face CLI (`hf`, from huggingface_hub) for `weights`; uv
# (https://docs.astral.sh/uv/) and a C++ compiler for `env`; a CUDA 12.8 toolkit for `cuda` (building the
# SageAttention extension against torch 2.8.0+cu128 needs the CUDA 12.8 nvcc, headers and cudart; a CUDA 13 toolkit
# does not work). The environment installs its own PyTorch; no base PyTorch is needed.
#
# Layout under $ROOT:
#   repo/               upstream source; repo_commit.txt records the commit
#   cuda-12.8/          bin/ (nvcc), nvvm/, include/, lib64/ (cudart, stubs)
#   env/                the venv (py/ holds the uv-managed Python 3.10, uv_cache/ the uv cache); env_freeze.txt
#   build/              SageAttention and LightX2V checkouts; logs/ their build logs (and the server's session logs)
#   weights/            LiveAct/, chinese-wav2vec2-base/
#   weights_hash.json   written by `hash`, sent by the server in ready.config
#
# Environment variables:
#   STREAMEVAL_MODEL_ROOT   model root (default ~/.cache/streameval/models)
#   SOULX_LIVEACT_ROOT        this model's folder (default $STREAMEVAL_MODEL_ROOT/soulx_liveact); server.py reads it too
#   SOULX_LIVEACT_COMMIT      upstream commit (default ac8579b907756f228e975922a5159765c93c6a34, the v1 code)
#   CUDA128_HOME              CUDA 12.8 toolkit to copy from (default /usr/local/cuda-12.8)
#   SAGE_ARCH                 CUDA architecture SageAttention is built for (default 8.9; 9.0 on H100/H200); see env_
#   UV_BIN                    uv executable (default: uv on PATH)
#   UV_CACHE_DIR              uv cache (default $ROOT/uv_cache)
#   STREAMEVAL_PROXY        optional HTTPS proxy for the downloads (default: $https_proxy)
#   HF_ENDPOINT               Hugging Face endpoint (default https://huggingface.co)
set -euo pipefail
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${SOULX_LIVEACT_ROOT:-$BASE/soulx_liveact}
COMMIT=${SOULX_LIVEACT_COMMIT:-ac8579b907756f228e975922a5159765c93c6a34}   # Soul-AILab/SoulX-LiveAct, 2026-06-15
LIVEACT_REV=1bcdef1d8941edf2a9a6e1d21f5657f0a69fcd37   # Soul-AILab/LiveAct
W2V_REV=3991242c806928916fff4a8c0e4f76acf661b743       # TencentGameMate/chinese-wav2vec2-base
SAGE=v2.2.0                                            # thu-ml/SageAttention tag
LIGHTX2V=8a97c7591d7252ef491392e83e1eb18617ac9368      # ModelTC/LightX2V, 2026-09-29 (LightVAE)
CUDA128_HOME=${CUDA128_HOME:-/usr/local/cuda-12.8}
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
STEP=${1:-all}
case "$STEP" in
  all|repo|cuda|weights|env|hash|env+hash) ;;
  *) echo "usage: bash setup.sh [all|repo|cuda|weights|env|hash|env+hash]" >&2; exit 2 ;;
esac
mkdir -p "$ROOT"/{weights,logs,build}

repo() {
  [ -f "$ROOT/repo/generate.py" ] && return
  mkdir -p "$ROOT/repo"
  https_proxy=$PROXY curl -sfL "https://codeload.github.com/Soul-AILab/SoulX-LiveAct/tar.gz/$COMMIT" \
    | tar xz -C "$ROOT/repo" --strip-components=1
  echo "$COMMIT" > "$ROOT/repo_commit.txt"
}

cuda() {
  # Building the SageAttention extension against torch 2.8.0+cu128 needs the CUDA 12.8 compiler, headers and
  # cudart. Copy that subset (427 MB) from the toolkit at $CUDA128_HOME so that the env step finds it under $ROOT.
  [ -x "$ROOT/cuda-12.8/bin/nvcc" ] && return
  local C=$CUDA128_HOME INC LIB
  [ -x "$C/bin/nvcc" ] && "$C/bin/nvcc" --version | grep -q "release 12\." || {
    echo "no CUDA 12.x nvcc at $C/bin/nvcc: install the CUDA 12.8 toolkit or set CUDA128_HOME to it" >&2; exit 1; }
  if [ -d "$C/targets/x86_64-linux/include" ]; then INC=$C/targets/x86_64-linux/include LIB=$C/targets/x86_64-linux/lib
  else INC=$C/include LIB=$C/lib64; fi
  mkdir -p "$ROOT/cuda-12.8/lib64" "$ROOT/cuda-12.8/include"
  cp -a "$C/bin" "$C/nvvm" "$ROOT/cuda-12.8/"
  cp -a "$INC/." "$ROOT/cuda-12.8/include/"
  cp -a "$LIB"/libcudart* "$LIB/stubs" "$ROOT/cuda-12.8/lib64/"
}

env_() {
  local UVX=${UV_BIN:-$(command -v uv || true)}
  [[ -n $UVX ]] || { echo "uv not found: install it (https://docs.astral.sh/uv/) or set UV_BIN" >&2; exit 1; }
  [ -x "$ROOT/cuda-12.8/bin/nvcc" ] || { echo "run the cuda step first: $ROOT/cuda-12.8/bin/nvcc is missing" >&2; exit 1; }
  export UV_CACHE_DIR=${UV_CACHE_DIR:-$ROOT/uv_cache} UV_PYTHON_INSTALL_DIR=$ROOT/py
  export UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
  UV() { https_proxy=$PROXY "$UVX" "$@"; }
  [ -x "$ROOT/env/bin/python" ] || UV venv -p 3.10 "$ROOT/env"
  local PY=$ROOT/env/bin/python
  if ! "$PY" -c "import torch, vllm, xfuser, flash_attn"; then
    UV pip install -p "$PY" torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128
    # upstream requirements.txt (minus the gradio/dashscope demo deps) + vllm fp8 GEMM kernels
    UV pip install -p "$PY" "opencv-python>=4.11.0.86" "diffusers>=0.31.0" "transformers==4.57.1" \
        "tokenizers>=0.20.3" "accelerate>=1.12.1" tqdm imageio easydict ftfy imageio-ffmpeg scikit-image loguru \
        numpy xfuser==0.4.5 pyloudnorm moviepy==1.0.3 decord einops sentencepiece safetensors soundfile \
        vllm==0.11.0 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 setuptools wheel ninja \
        --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
    UV pip install -p "$PY" \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
    # vllm pulls xformers, which rejects flash-attn 2.8.3 and then breaks `import diffusers` (xfuser); LiveAct only
    # uses vllm's fp8 GEMM ops, so drop it (flash-attn 2 is needed by the Wan CLIP encoder; no 2.8.2 torch2.8 wheel)
    UV pip uninstall -p "$PY" xformers
  fi
# Use the dependencies and installation requirements documented in README.md.
  [ -x "$ROOT/env/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas" ] || \
    UV pip install -p "$PY" "nvidia-cuda-nvcc-cu12==12.9.86"
# Use the dependencies and installation requirements documented in README.md.
  "$PY" -c "import torch, sys; sys.exit(torch.backends.cudnn.version() < 92600)" 2>/dev/null || \
    UV pip install -p "$PY" --no-deps "nvidia-cudnn-cu12==9.26.0.51"
  export CUDA_HOME=$ROOT/cuda-12.8 PATH=$ROOT/cuda-12.8/bin:$ROOT/env/bin:$PATH
# Use the dependencies and installation requirements documented in README.md.
  export TORCH_CUDA_ARCH_LIST=${SAGE_ARCH:-8.9} MAX_JOBS=8 EXT_PARALLEL=2 NVCC_APPEND_FLAGS=--threads=2
  if ! "$PY" -c "import sageattention" 2>/dev/null; then
    rm -rf "$ROOT/build/SageAttention"
    https_proxy=$PROXY git clone -q https://github.com/thu-ml/SageAttention.git "$ROOT/build/SageAttention"
    (cd "$ROOT/build/SageAttention" && git checkout -q $SAGE && "$PY" setup.py install) > "$ROOT/logs/build_sage.log" 2>&1
  fi
  # LightVAE (upstream: `python setup_vae.py install`, whose easy_install dependency resolution breaks on sdists):
  # build the pure-python wheel, install it without deps, then add whatever `import lightx2v` pulls in.
  local VAE=lightx2v.models.video_encoders.hf.wan.vae
  if ! "$PY" -c "import $VAE" 2>/dev/null; then
    [ -d "$ROOT/build/LightX2V/.git" ] || https_proxy=$PROXY git clone -q https://github.com/ModelTC/LightX2V "$ROOT/build/LightX2V"
    git -C "$ROOT/build/LightX2V" checkout -q $LIGHTX2V
    (cd "$ROOT/build/LightX2V" && rm -rf build dist && "$PY" setup_vae.py -q bdist_wheel) > "$ROOT/logs/build_lightx2v.log" 2>&1
    UV pip install -p "$PY" --no-deps "$ROOT"/build/LightX2V/dist/lightx2v-*.whl
    for _ in $(seq 30); do
      err=$("$PY" -c "import $VAE" 2>&1) && break
      mod=$(echo "$err" | grep -oP "No module named '\K[^'.]+" | tail -1)
      [ -n "$mod" ] || { echo "$err"; return 1; }
      case $mod in cv2) mod=opencv-python ;; yaml) mod=pyyaml ;; PIL) mod=pillow ;; zmq) mod=pyzmq ;; esac
      UV pip install -p "$PY" "$mod" torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match
    done
  fi
  "$PY" - <<'EOF'
import torch, sageattention, vllm, xfuser, flash_attn
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
print("env ok", torch.__version__, torch.version.cuda, vllm.__version__, flash_attn.__version__)
EOF
  UV pip freeze -p "$PY" > "$ROOT/env_freeze.txt"
}

weights() {
  export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
  # everything the model loads is outside assets/
  [ -f "$ROOT/weights/LiveAct/models_t5_umt5-xxl-enc-bf16.pth" ] || https_proxy=$PROXY hf download Soul-AILab/LiveAct \
    --revision "$LIVEACT_REV" --max-workers 4 --exclude 'assets/*' '.ipynb_checkpoints/*' --local-dir "$ROOT/weights/LiveAct"
  [ -f "$ROOT/weights/chinese-wav2vec2-base/pytorch_model.bin" ] || https_proxy=$PROXY hf download \
    TencentGameMate/chinese-wav2vec2-base --revision "$W2V_REV" --max-workers 4 --local-dir "$ROOT/weights/chinese-wav2vec2-base"
}

hash_() {
  [ -f "$ROOT/weights_hash.json" ] && return
  # full sha256 of every file the server loads (streamed, 8 MiB buffer); the server copies it into ready.config
  python3 - "$ROOT/weights" > "$ROOT/weights_hash.json" <<'EOF'
import hashlib, json, os, sys
root = sys.argv[1]
out = {}
for rel in sorted(["LiveAct/diffusion_pytorch_model.safetensors.index.json", "LiveAct/config.json",
                   "LiveAct/Wan2.1_VAE.pth", "LiveAct/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                   "LiveAct/models_t5_umt5-xxl-enc-bf16.pth", "chinese-wav2vec2-base/pytorch_model.bin"]
                  + [f"LiveAct/diffusion_pytorch_model-0000{i}-of-00004.safetensors" for i in range(1, 5)]):
    h = hashlib.sha256()
    with open(os.path.join(root, rel), "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    out[rel] = {"bytes": os.path.getsize(os.path.join(root, rel)), "sha256": h.hexdigest()}
print(json.dumps(out, indent=1))
EOF
  cat "$ROOT/weights_hash.json"
}

case "$STEP" in
  repo) repo ;;
  cuda) cuda ;;
  env) env_ ;;
  weights) weights ;;
  hash) hash_ ;;
  env+hash) env_; hash_ ;;
  all) repo; cuda; weights; env_; hash_ ;;
esac
