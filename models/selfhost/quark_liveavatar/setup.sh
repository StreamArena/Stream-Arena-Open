#!/usr/bin/env bash
# Quark Live Avatar 14B (Alibaba-Quark/LiveAvatar): source tree, Python environment and weights for the self-hosted
# server (docs/protocol.md). Idempotent; each step skips what is already in place. Everything goes under $ROOT.
#
#   bash models/selfhost/quark_liveavatar/setup.sh [all|repo|env|weights|hash|env+weights]
#
#   repo      network: upstream source at a pinned commit (a 50 MB codeload tarball)
#   env       network + a GPU machine: uv venv with Python 3.10, torch 2.8.0+cu128, flash-attn 2.8.3, cuDNN 9.26,
#             ptxas 12.9 and the inference dependencies (several GB)
#   weights   network: 35 GB from Hugging Face (Wan-AI/Wan2.2-S2V-14B and Quark-Vision/Live-Avatar, pinned revisions)
#   hash      sha256 of every weight file the server loads -> weights_hash.json (reported in ready.config)
#   all       repo, env, weights, hash (default); env+weights: env, weights, hash
#
# Prerequisites: curl, tar, python3 and uv (https://docs.astral.sh/uv/). Unlike most self-hosted models here, `env`
# installs its own PyTorch (torch 2.8.0+cu128 from download.pytorch.org), so the NVIDIA driver must support CUDA
# 12.8. The server needs 5 GPUs (see README).
#
# Layout under $ROOT:
#   repo/                 Alibaba-Quark/LiveAvatar at $COMMIT; repo_commit.txt records it
#   env/                  the venv (env_freeze.txt: its pip freeze); py/ the uv-managed Python 3.10
#   weights/Wan2.2-S2V-14B/          DiT shards, umT5-XXL, Wan2.1 VAE, wav2vec2-large-xlsr-53-english, google/umt5-xxl
#   weights/LiveAvatar/liveavatar.safetensors   the LoRA
#   weights_hash.json     {file: {bytes, sha256}}
#   logs/                 per-session statistics written by server.py
#
# Environment variables:
#   STREAMEVAL_MODEL_ROOT   model root (default ~/.cache/streameval/models)
#   QUARK_LIVEAVATAR_ROOT     this model's folder (default $STREAMEVAL_MODEL_ROOT/quark_liveavatar)
#   QUARK_LIVEAVATAR_COMMIT   upstream commit (default c3c47d031d8bf3247428333c1fe610579c71a551, the v1 code)
#   UV_BIN                    uv binary (default: uv on PATH)
#   STREAMEVAL_PROXY        optional HTTPS proxy for the downloads (default: $https_proxy)
#   HF_ENDPOINT               Hugging Face endpoint (default https://huggingface.co)
# The umT5-XXL encoder and the Wan2.1 VAE are byte-identical to files of models/selfhost/soulx_liveact; if that
# model's weights are already present under the model root they are copied instead of downloaded.
set -euo pipefail
STEP=${1:-all}
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${QUARK_LIVEAVATAR_ROOT:-$BASE/quark_liveavatar}
LA=$BASE/soulx_liveact                                   # same umT5 / Wan2.1 VAE files (sha256)
COMMIT=${QUARK_LIVEAVATAR_COMMIT:-c3c47d031d8bf3247428333c1fe610579c71a551}   # Alibaba-Quark/LiveAvatar, 2026-08-24
WAN_REV=dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b       # Wan-AI/Wan2.2-S2V-14B
LORA_REV=92cdccd12a91e8a63767a7c821b7c75e51d5a172      # Quark-Vision/Live-Avatar
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
case $STEP in
  all|repo|env|weights|hash|env+weights) ;;
  *) echo "usage: bash setup.sh [all|repo|env|weights|hash|env+weights]" >&2; exit 2 ;;
esac
mkdir -p "$ROOT"/{weights,logs}

uv_() {
  UV=${UV_BIN:-$(command -v uv || true)}
  [[ -n $UV ]] || { echo "uv not found: install it (https://docs.astral.sh/uv/) or set UV_BIN" >&2; exit 1; }
  https_proxy=$PROXY HTTPS_PROXY=$PROXY "$UV" "$@"
}

repo() {
  [ -f "$ROOT/repo/minimal_inference/s2v_streaming_interact.py" ] && return
  mkdir -p "$ROOT/repo"
  https_proxy=$PROXY curl -sfL "https://codeload.github.com/Alibaba-Quark/LiveAvatar/tar.gz/$COMMIT" \
    | tar xz -C "$ROOT/repo" --strip-components=1
  echo "$COMMIT" > "$ROOT/repo_commit.txt"
}

env_() {
  export UV_PYTHON_INSTALL_DIR=$ROOT/py UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
  [ -x "$ROOT/env/bin/python" ] || uv_ venv -p 3.10 "$ROOT/env"
  local PY=$ROOT/env/bin/python
  if ! "$PY" -c "import torch, flash_attn, peft, diffusers, transformers, librosa, cv2" 2>/dev/null; then
    uv_ pip install -p "$PY" torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128
    # the part of upstream requirements.txt that the streaming inference path imports (no gradio / TTS / SAM2 /
    # training extras)
    uv_ pip install -p "$PY" "opencv-python>=4.9.0.80" "diffusers>=0.31.0" "transformers==4.51.3" \
        "tokenizers>=0.20.3" "accelerate>=1.1.1" peft==0.17.1 tqdm "imageio[ffmpeg]" easydict ftfy "numpy<2" \
        omegaconf einops decord librosa regex safetensors sentencepiece pyyaml scipy requests pillow dashscope \
        "huggingface_hub[cli]<1.0" torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 setuptools wheel \
        --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
    uv_ pip install -p "$PY" \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
  fi
# Use the dependencies and installation requirements documented in README.md.
  [ -x "$ROOT/env/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas" ] || \
    uv_ pip install -p "$PY" "nvidia-cuda-nvcc-cu12==12.9.86"
# Use the dependencies and installation requirements documented in README.md.
  "$PY" -c "import torch, sys; sys.exit(torch.backends.cudnn.version() < 92600)" 2>/dev/null || \
    uv_ pip install -p "$PY" --no-deps "nvidia-cudnn-cu12==9.26.0.51"
  # whatever else the upstream modules pull in at import time
  local MODS="liveavatar.models.wan.causal_model_s2v, liveavatar.models.wan.wan_2_2.modules.vae_streaming, liveavatar.models.wan.wan_2_2.modules.t5, liveavatar.models.wan.causal_audio_encoder, liveavatar.utils.fp8_linear, liveavatar.utils.load_weight_utils"
  for _ in $(seq 30); do
    err=$(cd "$ROOT/repo" && ENABLE_COMPILE=false "$PY" -c "import $MODS" 2>&1) && break
    mod=$(echo "$err" | grep -oP "No module named '\K[^'.]+" | tail -1)
    [ -n "$mod" ] || { echo "$err"; return 1; }
    case $mod in cv2) mod=opencv-python ;; yaml) mod=pyyaml ;; PIL) mod=pillow ;; git) mod=GitPython ;; esac
    uv_ pip install -p "$PY" "$mod" torch==2.8.0 "numpy<2" "transformers==4.51.3" \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  done
  (cd "$ROOT/repo" && ENABLE_COMPILE=false "$PY" - <<'EOF'
import torch, flash_attn, peft, diffusers, transformers
from liveavatar.models.wan.causal_model_s2v import CausalWanModel_S2V
print("env ok", torch.__version__, torch.version.cuda, "cudnn", torch.backends.cudnn.version(), "peft", peft.__version__,
      "transformers", transformers.__version__, "diffusers", diffusers.__version__)
EOF
  )
  uv_ pip freeze -p "$PY" > "$ROOT/env_freeze.txt"
}

weights() {
  export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DOWNLOAD_TIMEOUT=120
  local HF="$ROOT/env/bin/hf"; [ -x "$HF" ] || HF="$ROOT/env/bin/huggingface-cli"; [ -x "$HF" ] || HF=$(command -v hf)
  local W=$ROOT/weights/Wan2.2-S2V-14B
  mkdir -p "$W"
  # umT5-XXL encoder and the Wan2.1 VAE are byte-identical to the files SoulX-LiveAct holds (same sha256): copied
  # when present; otherwise the download below fetches them
  for f in models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth; do
    [ -f "$W/$f" ] || { [ -f "$LA/weights/LiveAct/$f" ] && cp --reflink=auto "$LA/weights/LiveAct/$f" "$W/$f.tmp" && mv "$W/$f.tmp" "$W/$f"; } || true
  done
  for try in 1 2 3 4 5; do
    https_proxy=$PROXY "$HF" download Wan-AI/Wan2.2-S2V-14B --revision $WAN_REV --max-workers 4 --local-dir "$W" \
      --include 'config.json' 'diffusion_pytorch_model*' 'Wan2.1_VAE.pth' 'models_t5_umt5-xxl-enc-bf16.pth' \
                'google/umt5-xxl/*' 'wav2vec2-large-xlsr-53-english/*.json' \
                'wav2vec2-large-xlsr-53-english/model.safetensors' && break
    sleep 20
  done
  for try in 1 2 3 4 5; do
    https_proxy=$PROXY "$HF" download Quark-Vision/Live-Avatar --revision $LORA_REV --local-dir "$ROOT/weights/LiveAvatar" \
      --include 'liveavatar.safetensors' 'README.md' && break
    sleep 20
  done
  [ -f "$ROOT/weights/LiveAvatar/liveavatar.safetensors" ] && [ -f "$W/diffusion_pytorch_model-00004-of-00004.safetensors" ]
}

hash_() {
  [ -f "$ROOT/weights_hash.json" ] && return
  # full sha256 per file the server loads (streamed, 8 MiB buffer); copied into ready.config
  python3 - "$ROOT/weights" > "$ROOT/weights_hash.json.tmp" <<'EOF'
import hashlib, json, os, sys
root = sys.argv[1]
out = {}
W = "Wan2.2-S2V-14B/"
for rel in sorted([W + "config.json", W + "diffusion_pytorch_model.safetensors.index.json", W + "Wan2.1_VAE.pth",
                   W + "models_t5_umt5-xxl-enc-bf16.pth", W + "wav2vec2-large-xlsr-53-english/model.safetensors",
                   "LiveAvatar/liveavatar.safetensors"]
                  + [W + f"diffusion_pytorch_model-0000{i}-of-00004.safetensors" for i in range(1, 5)]):
    h = hashlib.sha256()
    with open(os.path.join(root, rel), "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    out[rel] = {"bytes": os.path.getsize(os.path.join(root, rel)), "sha256": h.hexdigest()}
print(json.dumps(out, indent=1))
EOF
  mv "$ROOT/weights_hash.json.tmp" "$ROOT/weights_hash.json"
  cat "$ROOT/weights_hash.json"
}

case "$STEP" in
  repo) repo ;;
  env) env_ ;;
  weights) weights ;;
  hash) hash_ ;;
  env+weights) repo; env_; weights; hash_ ;;
  all) repo; env_; weights; hash_ ;;
esac
