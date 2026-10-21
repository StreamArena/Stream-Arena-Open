#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${APTAVATAR_ROOT:-$BASE/aptavatar}
COMMIT=${APTAVATAR_COMMIT:-c3cc19ca4481d0aafa35998fe6a538003e5cef61}   # TaoLiveAIGC/AptAvatar main, 2026-08-03
REV=${APTAVATAR_REV:-e5ae7ce74e1766693ed19473c3b51247351a0c78}         # HF TaoLiveAIGC/AptAvatar
W2V_REV=3991242c806928916fff4a8c0e4f76acf661b743                       # HF TencentGameMate/chinese-wav2vec2-base
PYTHON=${PYTHON:-python3}
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
STEP=${1:-all}
case "$STEP" in
  all|repo|env|weights|hash) ;;
  *) echo "usage: bash $0 [all|repo|env|weights|hash]" >&2; exit 2 ;;
esac
mkdir -p "$ROOT"/{weights,logs}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DISABLE_XET=1

repo() {
  [[ -f $ROOT/repo/AptAvatar/src/pipeline/AptAvatar_pipeline.py ]] && return
  mkdir -p "$ROOT/repo"
  https_proxy=$PROXY python3 "$(dirname "${BASH_SOURCE[0]}")/../_fetch_github.py" TaoLiveAIGC/AptAvatar "$COMMIT" "$ROOT/repo"
  echo "$COMMIT" > "$ROOT/repo_commit.txt"
}

env_() {
  local PY=$ROOT/env/bin/python
  if ! "$PY" -c "import torch, flash_attn, diffusers, transformers, xfuser, yunchang, librosa, pyloudnorm, cv2, huggingface_hub, optimum.quanto, yaml; import importlib.metadata as m; assert m.version('xfuser') == '0.4.5'" 2>/dev/null; then
    "$PYTHON" -c "import torch" 2>/dev/null || { echo "$PYTHON has no torch: this step needs a base Python whose site-packages provide a CUDA build of PyTorch " >&2; exit 1; }
    [[ -x $PY ]] || "$PYTHON" -m venv --system-site-packages "$ROOT/env"
    # keep the base environment's torch / CUDA stack: pip may not replace it
    "$PYTHON" -m pip list --format=freeze 2>/dev/null | grep -i -E \
      "^(torch|torchvision|torchaudio|triton|pytorch-triton|numpy|tensorrt|flash.attn|nvidia-|cuda-|apex|transformer.engine|onnx==)" \
      > "$ROOT/env/constraints.txt" || true
    P="$PY -m pip install -q -c $ROOT/env/constraints.txt"
    https_proxy=$PROXY $P "diffusers>=0.34.0" "transformers==4.57.3" "tokenizers>=0.20.3" "accelerate>=1.8.1" easydict \
       ftfy imageio imageio-ffmpeg scikit-image loguru pyloudnorm librosa soundfile opencv-python-headless pillow pyyaml \
       einops decord "huggingface_hub[cli]" sentencepiece "peft>=0.17" "optimum-quanto==0.2.6"
    https_proxy=$PROXY $P --no-deps "xfuser==0.4.5" "yunchang==0.6.4"
    https_proxy=$PROXY $P distvae || true
  fi
  "$PY" -c "import torch, flash_attn, diffusers, transformers, xfuser, cv2, librosa
print('env ok', torch.__version__, torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'flash_attn', flash_attn.__version__)
print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None"
  "$PY" -m pip freeze > "$ROOT/env_freeze.txt"
}

weights() {
  local PY=$ROOT/env/bin/python
  if [[ ! -f $ROOT/weights/AptAvatar/.complete ]]; then
    https_proxy=$PROXY "$PY" - "$ROOT" "$REV" <<'EOF'
import sys
from huggingface_hub import snapshot_download
b, rev = sys.argv[1], sys.argv[2]
p = snapshot_download("TaoLiveAIGC/AptAvatar", revision=rev, local_dir=f"{b}/weights/AptAvatar", max_workers=8,
                      ignore_patterns=["assets/*"])
print("downloaded", p)
EOF
    echo "$REV" > "$ROOT/weights/AptAvatar/.complete"
  fi
  # soulx_flashtalk uses the same chinese-wav2vec2-base revision: copy it when present, else download it
  local W=$BASE/soulx_flashtalk/weights/chinese-wav2vec2-base
  if [[ ! -f $ROOT/weights/chinese-wav2vec2-base/pytorch_model.bin ]]; then
    if [[ -f $W/pytorch_model.bin ]]; then cp -r "$W" "$ROOT/weights/"; else
      https_proxy=$PROXY "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('TencentGameMate/chinese-wav2vec2-base', revision='$W2V_REV', local_dir='$ROOT/weights/chinese-wav2vec2-base')"
    fi
  fi
}

hash_() {
  [[ -s $ROOT/weights_hash.json ]] && return
  python3 - "$ROOT/weights" > "$ROOT/weights_hash.json.tmp" <<'EOF'
import hashlib, json, os, sys
root = sys.argv[1]
out = {}
for d in ("AptAvatar", "chinese-wav2vec2-base"):
    for dp, _, fs in os.walk(os.path.join(root, d)):
        if "/.cache" in dp:
            continue
        for f in fs:
            p = os.path.join(dp, f)
            if not f.endswith((".safetensors", ".pth", ".bin")):
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for blk in iter(lambda: fh.read(8 << 20), b""):
                    h.update(blk)
            out[os.path.relpath(p, root)] = {"bytes": os.path.getsize(p), "sha256": h.hexdigest()}
print(json.dumps(dict(sorted(out.items())), indent=1))
EOF
  mv "$ROOT/weights_hash.json.tmp" "$ROOT/weights_hash.json"
  cat "$ROOT/weights_hash.json"
}

case "$STEP" in
  repo) repo ;;
  env) env_ ;;
  weights) weights ;;
  hash) hash_ ;;
  all) repo; env_; weights; hash_ ;;
esac
