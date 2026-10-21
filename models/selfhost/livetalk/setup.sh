#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
B=${LIVETALK_ROOT:-$BASE/livetalk}
COMMIT=${LIVETALK_COMMIT:-e0d555fc5edf8516fbed2c955f1c0f4b32476abb}     # GAIR-NLP/LiveTalk main, 2026-01-02
LT_REV=${LIVETALK_REV:-be14ffdf356e73e328db88595ccf62e7e1374b74}         # GAIR/LiveTalk-1.3B-V0.1
WAN21_REV=37ec512624d61f7aa208f7ea8140a131f93afc9a    # Wan-AI/Wan2.1-T2V-1.3B (unchanged since 2025-03-01; sha256 match system.json)
W2V_REV=22aad52d435eb6dbaf354bdad9b0da84ce7d6156      # facebook/wav2vec2-base-960h (unchanged since 2022-11-14)
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
PYTHON=${PYTHON:-python3}
case $STEP in
  all|repo|env|weights|hash) ;;
  *) echo "usage: bash setup.sh [all|repo|env|weights|hash]" >&2; exit 2 ;;
esac
mkdir -p "$B"/{weights,logs}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DISABLE_XET=1

repo() {
  [[ -f $B/repo/model/causal_wan_video_dit_omniavatar.py ]] && return
  mkdir -p "$B/repo"
  https_proxy=$PROXY python3 "$(dirname "${BASH_SOURCE[0]}")/../_fetch_github.py" GAIR-NLP/LiveTalk "$COMMIT" "$B/repo"
  echo "$COMMIT" > "$B/repo_commit.txt"
}

env_() {
  local PY=$B/env/bin/python
  if ! "$PY" -c "import torch, flash_attn, diffusers, transformers, xfuser, librosa, cv2, huggingface_hub, einops, ftfy" 2>/dev/null; then
    $PYTHON -c "import torch" 2>/dev/null || { echo "$PYTHON has no torch: this step needs a base Python whose site-packages provide a CUDA build of PyTorch " >&2; exit 1; }
    (
      export PIP_CACHE_DIR=$B/pip_cache
      [[ -x $PY ]] || $PYTHON -m venv --system-site-packages "$B/env"
      $PYTHON -m pip list --format=freeze 2>/dev/null | grep -i -E \
        "^(torch|torchvision|torchaudio|triton|pytorch-triton|numpy|tensorrt|flash.attn|nvidia-|cuda-|apex|transformer.engine|onnx==)" \
        > "$B/env/constraints.txt"
      P="env https_proxy=$PROXY $PY -m pip install -q -c $B/env/constraints.txt"
      $P "diffusers>=0.31.0" "transformers==4.57.3" "tokenizers>=0.20.3" "accelerate>=1.1.1" easydict ftfy imageio \
         imageio-ffmpeg einops omegaconf safetensors "librosa==0.10.2.post1" soundfile opencv-python-headless pillow \
         pyyaml "huggingface_hub[cli]" sentencepiece regex "peft>=0.17"
      $P --no-deps "xfuser==0.4.1" || $P --no-deps "xfuser==0.7.0"
      $P --no-deps "yunchang==0.6.4" || true
      $P distvae || true
    )
  fi
  "$PY" -c "import torch, flash_attn, diffusers, transformers, xfuser, cv2, librosa
print('env ok', torch.__version__, torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'flash_attn', flash_attn.__version__, 'diffusers', diffusers.__version__)
print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None"
  "$PY" -m pip freeze > "$B/env_freeze.txt"
}

weights() {
  local PY=$B/env/bin/python
  if [[ ! -f $B/weights/LiveTalk-1.3B-V0.1/model.safetensors ]]; then
    https_proxy=$PROXY "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('GAIR/LiveTalk-1.3B-V0.1', revision='$LT_REV', local_dir='$B/weights/LiveTalk-1.3B-V0.1')"
  fi
  local W=$B/weights/Wan2.1-T2V-1.3B S=$BASE/self_forcing/weights/wan_models/Wan2.1-T2V-1.3B
  if [[ ! -f $W/Wan2.1_VAE.pth || ! -f $W/models_t5_umt5-xxl-enc-bf16.pth || ! -d $W/google/umt5-xxl ]]; then
    mkdir -p "$W"
    if [[ -f $S/Wan2.1_VAE.pth && -f $S/models_t5_umt5-xxl-enc-bf16.pth && -d $S/google/umt5-xxl ]]; then
      cp -r "$S/Wan2.1_VAE.pth" "$S/models_t5_umt5-xxl-enc-bf16.pth" "$S/google" "$S/config.json" "$W/"
      echo "copied from $S" > "$W/.source"
    else
      https_proxy=$PROXY "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', revision='$WAN21_REV', local_dir='$W', allow_patterns=['Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth', 'google/*', 'config.json'])"
    fi
  fi
  local A=$BASE/soulx_flashhead/weights/wav2vec2-base-960h
  if [[ ! -f $B/weights/wav2vec2-base-960h/config.json ]]; then
    if [[ -f $A/config.json ]]; then cp -r "$A" "$B/weights/"; else
      https_proxy=$PROXY "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/wav2vec2-base-960h', revision='$W2V_REV', local_dir='$B/weights/wav2vec2-base-960h')"
    fi
  fi
}

hash_() {
  [[ -s $B/weights_hash.json ]] && return
  python3 - "$B/weights" > "$B/weights_hash.json.tmp" <<'EOF'
import hashlib, json, os, sys
root = sys.argv[1]
out = {}
for rel in ["LiveTalk-1.3B-V0.1/model.safetensors", "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", "wav2vec2-base-960h/model.safetensors",
            "wav2vec2-base-960h/pytorch_model.bin"]:
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(8 << 20), b""):
            h.update(blk)
    out[rel] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
print(json.dumps(out, indent=1))
EOF
  mv "$B/weights_hash.json.tmp" "$B/weights_hash.json"
  cat "$B/weights_hash.json"
}

case "$STEP" in
  repo) repo ;;
  env) env_ ;;
  weights) weights ;;
  hash) hash_ ;;
  all) repo; env_; weights; hash_ ;;
esac
echo "LIVETALK_SETUP_DONE ($STEP)"
