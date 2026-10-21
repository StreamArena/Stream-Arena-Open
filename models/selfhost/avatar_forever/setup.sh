#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${AF_ROOT:-$BASE/avatar_forever}
COMMIT=${AF_COMMIT:-4dfc42b0e2dbbded4d148387d186219bd7601279}   # leeruibin/avatarforever main, 2026-08-23
HF_REV=${AF_HF_REV:-622ea82482c09e7fb2d8e29cb7ff1d009c820f05}  # LetsThink/AvatarForever main, 2026-08-23
CKPT=avatarforever-ltx-2.3-22b.safetensors
CKPT_SHA=08dea67ed3d35a75da49b056ce2ca59a0e9ce538b429143fb9022c9bd5c3b145   # HF LFS oid
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
case $STEP in
  all|repo|env|weights|hash) ;;
  *) echo "usage: bash setup.sh [all|repo|env|weights|hash]" >&2; exit 2 ;;
esac
UV=${UV_BIN:-$(command -v uv || true)}
[[ -n $UV ]] || { echo "uv not found: install it (https://docs.astral.sh/uv/) or set UV_BIN" >&2; exit 1; }
mkdir -p "$ROOT"/{logs,weights}
PY=$ROOT/env/bin/python
export UV_PYTHON_INSTALL_DIR=$ROOT/py UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DISABLE_XET=1

repo() {
  if [[ ! -f $ROOT/repo/inference.py ]]; then
    rm -rf "$ROOT/repo.tmp" && mkdir -p "$ROOT/repo.tmp"
    https_proxy=$PROXY curl -sfL "https://codeload.github.com/leeruibin/avatarforever/tar.gz/$COMMIT" \
      | tar xz -C "$ROOT/repo.tmp" --strip-components=1 --exclude='*.mp4' --exclude='*.gif'
    mv "$ROOT/repo.tmp" "$ROOT/repo"
    echo "$COMMIT" > "$ROOT/repo_commit.txt"
  fi
  ls "$ROOT/repo"
}

env_() {
  [[ -x $PY ]] || { HTTPS_PROXY=$PROXY "$UV" python install 3.11 && "$UV" venv -p 3.11 "$ROOT/env"; }
  if ! "$PY" -c "import torch, transformers, av, ltx_core, ltx_pipelines; assert torch.__version__.startswith('2.10.0')" 2>/dev/null; then
    HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY "$UV" pip install -p "$PY" torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
      triton==3.6.0 --index-url https://download.pytorch.org/whl/cu130
    HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY "$UV" pip install -p "$PY" "transformers==4.57.1" "accelerate>=1.10" safetensors \
      einops numpy "scipy>=1.14" av tqdm pillow opencv-python-headless sentencepiece protobuf huggingface_hub \
      torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 triton==3.6.0 \
      --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130
    "$UV" pip install -p "$PY" --no-deps -e "$ROOT/repo/packages/ltx-core" -e "$ROOT/repo/packages/ltx-pipelines"
  fi
# Use the dependencies and installation requirements documented in README.md.
  "$PY" -c "import torch, sys; sys.exit(torch.backends.cudnn.version() < 92600)" 2>/dev/null || \
    HTTPS_PROXY=$PROXY HTTP_PROXY=$PROXY "$UV" pip install -p "$PY" --no-deps "nvidia-cudnn-cu13==9.26.0.51"
  "$PY" -c "
import torch, transformers, ltx_core, ltx_pipelines
print('env ok torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'transformers', transformers.__version__)
print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None"
  "$UV" pip freeze -p "$PY" > "$ROOT/env_freeze.txt"
}

weights() {
  local G=$ROOT/weights/gemma-3-12b-it-qat-q4_0-unquantized
# Use the dependencies and installation requirements documented in README.md.
  if [[ ! -f $G/config.json ]]; then
    https_proxy=$PROXY "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('google/gemma-3-12b-it-qat-q4_0-unquantized', local_dir='$G')"
  fi
  if [[ ! -f $ROOT/weights/.ckpt_ok ]]; then
    https_proxy=$PROXY "$PY" -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('LetsThink/AvatarForever', '$CKPT', revision='$HF_REV', local_dir='$ROOT/weights'))"
    echo "$HF_REV" > "$ROOT/weights/.ckpt_ok"
  fi
  du -shL "$ROOT/weights"/* 2>/dev/null || true
}

hash_() {
  [[ -f $ROOT/weights_hash.json ]] && return
  "$PY" - "$ROOT/weights" "$CKPT" "$CKPT_SHA" > "$ROOT/weights_hash.json.tmp" <<'EOF'
import glob, hashlib, json, os, sys
root, ckpt, want = sys.argv[1:4]
out = {}
files = [ckpt] + sorted(os.path.relpath(p, root) for p in glob.glob(f"{root}/gemma-3-12b-it-qat-q4_0-unquantized/*.safetensors"))
for rel in files:
    h = hashlib.sha256()
    with open(os.path.join(root, rel), "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    out[rel] = {"bytes": os.path.getsize(os.path.join(root, rel)), "sha256": h.hexdigest()}
assert out[ckpt]["sha256"] == want, f"checkpoint sha256 mismatch: {out[ckpt]['sha256']} != {want}"
print(json.dumps(out, indent=1))
EOF
  mv "$ROOT/weights_hash.json.tmp" "$ROOT/weights_hash.json"
  cat "$ROOT/weights_hash.json"
}

case $STEP in
  repo) repo ;;
  env) env_ ;;
  weights) weights ;;
  hash) hash_ ;;
  all) repo; env_; weights; hash_ ;;
esac
echo "AF_SETUP_DONE ($STEP)"
