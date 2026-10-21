#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
B=${SOULX_FLASHTALK_ROOT:-$BASE/soulx_flashtalk}
COMMIT=${SOULX_FLASHTALK_COMMIT:-2ba2e1e801d59a9f526212a5aae5d374db8f2978}   # Soul-AILab/SoulX-FlashTalk main, 2026-07-30
FT_REV=${SOULX_FLASHTALK_REV:-f3fda6499a6022862cadd39fc41ae63b64706e1b}      # Soul-AILab/SoulX-FlashTalk-14B
W2V_REV=3991242c806928916fff4a8c0e4f76acf661b743                             # TencentGameMate/chinese-wav2vec2-base
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
PYTHON=${PYTHON:-python3}
STEP=${1:-all}
case $STEP in
  all|repo|env|weights|hash) ;;
  *) echo "usage: bash setup.sh [all|repo|env|weights|hash]" >&2; exit 2 ;;
esac
mkdir -p "$B"/{weights,logs}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DISABLE_XET=1

repo() {
  [[ -f $B/repo/flash_talk/inference.py ]] && return
  mkdir -p "$B/repo"
  https_proxy=$PROXY python3 "$(dirname "${BASH_SOURCE[0]}")/../_fetch_github.py" Soul-AILab/SoulX-FlashTalk "$COMMIT" "$B/repo"
  echo "$COMMIT" > "$B/repo_commit.txt"
}

env_() {
  local PY=$B/env/bin/python
  if ! "$PY" -c "import torch, flash_attn, diffusers, transformers, xfuser, yunchang, librosa, pyloudnorm, cv2, huggingface_hub, optimum.quanto; import importlib.metadata as m; assert m.version('xfuser') == '0.4.5'" 2>/dev/null; then
    $PYTHON -c "import torch" 2>/dev/null || { echo "$PYTHON has no torch: this step needs a base Python whose site-packages provide a CUDA build of PyTorch " >&2; exit 1; }
    (
      export PIP_CACHE_DIR=$B/pip_cache
      [[ -x $PY ]] || $PYTHON -m venv --system-site-packages "$B/env"
      $PYTHON -m pip list --format=freeze 2>/dev/null | grep -i -E \
        "^(torch|torchvision|torchaudio|triton|pytorch-triton|numpy|tensorrt|flash.attn|nvidia-|cuda-|apex|transformer.engine|onnx==)" \
        > "$B/env/constraints.txt" || true
      P="env https_proxy=$PROXY $PY -m pip install -q -c $B/env/constraints.txt"
      $P "diffusers>=0.34.0" "transformers==4.57.3" "tokenizers>=0.20.3" "accelerate>=1.8.1" easydict ftfy imageio \
         imageio-ffmpeg scikit-image loguru pyloudnorm librosa soundfile opencv-python-headless pillow pyyaml einops \
         decord "huggingface_hub[cli]" sentencepiece "peft>=0.17" "optimum-quanto==0.2.6"
      $P --no-deps "xfuser==0.4.5" "yunchang==0.6.4"   # as soulx_liveact; xfuser 0.7 needs an unreleased yunchang
      $P distvae || true
    )
  fi
  "$PY" -c "import torch, flash_attn, diffusers, transformers, xfuser, cv2, librosa
print('env ok', torch.__version__, torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'flash_attn', flash_attn.__version__)
print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None"
  "$PY" -m pip freeze > "$B/env_freeze.txt"
}

weights() {
  local PY=$B/env/bin/python
  if [[ ! -f $B/weights/SoulX-FlashTalk-14B/.complete ]]; then
    https_proxy=$PROXY "$PY" - "$B" "$FT_REV" <<'EOF'
import sys
from huggingface_hub import snapshot_download
b, rev = sys.argv[1], sys.argv[2]
p = snapshot_download("Soul-AILab/SoulX-FlashTalk-14B", revision=rev, local_dir=f"{b}/weights/SoulX-FlashTalk-14B",
                      max_workers=8)
print("downloaded", p)
EOF
    https_proxy=$PROXY "$PY" -c "from huggingface_hub import HfApi; print(HfApi().model_info('Soul-AILab/SoulX-FlashTalk-14B', revision='$FT_REV').sha)" \
      > "$B/weights/SoulX-FlashTalk-14B/.complete"
  fi
  local W=$BASE/soulx_liveact/weights/chinese-wav2vec2-base   # same revision when soulx_liveact is set up; optional
  if [[ ! -f $B/weights/chinese-wav2vec2-base/pytorch_model.bin ]]; then
    if [[ -f $W/pytorch_model.bin ]]; then cp -r "$W" "$B/weights/"; else
      https_proxy=$PROXY "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('TencentGameMate/chinese-wav2vec2-base', revision='$W2V_REV', local_dir='$B/weights/chinese-wav2vec2-base')"
    fi
  fi
}

hash_() {
  [[ -s $B/weights_hash.json ]] && return
  python3 - "$B/weights" > "$B/weights_hash.json.tmp" <<'EOF'
import hashlib, json, os, sys
root = sys.argv[1]
out = {}
for d in ("SoulX-FlashTalk-14B", "chinese-wav2vec2-base"):
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
