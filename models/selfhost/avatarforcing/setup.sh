#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
B=${AVATARFORCING_ROOT:-$BASE/avatarforcing}
COMMIT=${AVATARFORCING_COMMIT:-63b73e6c0f7bb42180ca6d7e1bf11c1de1a80b39}   # KlingAIResearch/AvatarForcing main, 2026-05-09
REV=${AVATARFORCING_REV:-e2448919a7b535c29f34e07892884ae1a43c6ace}         # lycui/AvatarForcing
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
  if [[ ! -f $B/repo/pipeline/avatar_forcing_inference.py ]]; then
    mkdir -p "$B/repo"
    https_proxy=$PROXY python3 "$(dirname "${BASH_SOURCE[0]}")/../_fetch_github.py" KlingAIResearch/AvatarForcing "$COMMIT" "$B/repo"
    echo "$COMMIT" > "$B/repo_commit.txt"
  fi
  mkdir -p "$B/repo/wan_models"
  ln -sfn "$B/weights/Wan2.1-T2V-1.3B" "$B/repo/wan_models/Wan2.1-T2V-1.3B"
  ln -sfn "$B/weights/wav2vec2-base-960h" "$B/repo/wan_models/wav2vec2-base-960h"
}

env_() {
  local PY=$B/env/bin/python
  if ! "$PY" -c "import torch, flash_attn, diffusers, transformers, omegaconf, peft, einops, librosa, cv2, huggingface_hub, ftfy" 2>/dev/null; then
    $PYTHON -c "import torch" 2>/dev/null || { echo "$PYTHON has no torch: this step needs a base Python whose site-packages provide a CUDA build of PyTorch " >&2; exit 1; }
    (
      export PIP_CACHE_DIR=$B/pip_cache
      [[ -x $PY ]] || $PYTHON -m venv --system-site-packages "$B/env"
      $PYTHON -m pip list --format=freeze 2>/dev/null | grep -i -E \
        "^(torch|torchvision|torchaudio|triton|pytorch-triton|numpy|tensorrt|flash.attn|nvidia-|cuda-|apex|transformer.engine|onnx==)" \
        > "$B/env/constraints.txt" || true
      P="env https_proxy=$PROXY $PY -m pip install -q -c $B/env/constraints.txt"
      $P "diffusers==0.35.2" "transformers==4.49.0" "accelerate>=1.8.1" omegaconf "peft==0.17.1" einops ftfy easydict \
         imageio imageio-ffmpeg librosa soundfile opencv-python-headless pillow pyyaml "huggingface_hub[cli]" \
         sentencepiece lmdb av tqdm
    )
  fi
  "$PY" -c "import torch, flash_attn, diffusers, transformers, peft
print('env ok', torch.__version__, torch.version.cuda, 'flash_attn', flash_attn.__version__, 'transformers', transformers.__version__)
print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None"
  "$PY" -m pip freeze > "$B/env_freeze.txt"
}

weights() {
  local PY=$B/env/bin/python
  if [[ ! -f $B/weights/AvatarForcing/.complete ]]; then
    https_proxy=$PROXY "$PY" - "$B" "$REV" <<'EOF'
import sys
from huggingface_hub import snapshot_download
b, rev = sys.argv[1], sys.argv[2]
print("downloaded", snapshot_download("lycui/AvatarForcing", revision=rev, local_dir=f"{b}/weights/AvatarForcing", max_workers=4))
EOF
    echo "$REV" > "$B/weights/AvatarForcing/.complete"
  fi
  # Wan2.1 T5 / VAE and wav2vec2: copied from livetalk's download of the same revisions when present, else downloaded
  local L=$BASE/livetalk/weights W=$B/weights/Wan2.1-T2V-1.3B
  if [[ ! -e $W/.copied ]]; then
    if [[ -f $L/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth && -f $L/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth && -d $L/Wan2.1-T2V-1.3B/google/umt5-xxl ]]; then
      rm -rf "$W"; cp -r "$L/Wan2.1-T2V-1.3B" "$W"
    else
      https_proxy=$PROXY "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', revision='$WAN21_REV', local_dir='$W', allow_patterns=['Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth', 'google/*', 'config.json'])"
    fi
    touch "$W/.copied"
  fi
  local A=$B/weights/wav2vec2-base-960h
  if [[ ! -e $A/.copied ]]; then
    if [[ -f $L/wav2vec2-base-960h/config.json ]]; then
      rm -rf "$A"; cp -r "$L/wav2vec2-base-960h" "$A"
    else
      https_proxy=$PROXY "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/wav2vec2-base-960h', revision='$W2V_REV', local_dir='$A')"
    fi
    touch "$A/.copied"
  fi
}

hash_() {
  [[ -s $B/weights_hash.json ]] && return
  python3 - "$B/weights" > "$B/weights_hash.json.tmp" <<'EOF'
import hashlib, json, os, sys
root = sys.argv[1]
out = {}
for d in ("AvatarForcing", "Wan2.1-T2V-1.3B", "wav2vec2-base-960h"):
    for dp, _, fs in os.walk(os.path.join(root, d)):
        if "/.cache" in dp:
            continue
        for f in fs:
            p = os.path.join(dp, f)
            if not f.endswith((".safetensors", ".pth", ".bin", ".pt")):
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for blk in iter(lambda: fh.read(8 << 20), b""):
                    h.update(blk)
            out[os.path.relpath(p, root)] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
print(json.dumps(dict(sorted(out.items())), indent=1))
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
echo "AVATARFORCING_SETUP_DONE ($STEP)"
