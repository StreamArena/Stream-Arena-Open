#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${ALAYA_ROOT:-$BASE/alayaworld}
COMMIT=${ALAYA_COMMIT:-7e7dcfe5b82419d665ffdd458815553e459a8a39}   # AlayaLab/AlayaWorld main, 2026-10-04
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
case $STEP in all|repo|env|weights|check) ;; *) echo "usage: bash setup.sh [all|repo|env|weights|check]" >&2; exit 2 ;; esac
mkdir -p "$ROOT"/{logs,checkpoints}
UV=${UV_BIN:-$(command -v uv || true)}
[[ -n $UV ]] || { echo "uv not found: install it (https://docs.astral.sh/uv/) or set UV_BIN" >&2; exit 1; }
PY=$ROOT/env/bin/python
export UV_PYTHON_INSTALL_DIR=$ROOT/py UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
if [[ -n ${PIP_INDEX_URL:-} ]]; then export UV_INDEX_URL=$PIP_INDEX_URL; fi
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DISABLE_XET=1

repo() {
  if [[ ! -f $ROOT/repo/reactor/alayaworld.py ]]; then
    mkdir -p "$ROOT/repo"
    https_proxy=$PROXY curl -sfL "https://codeload.github.com/AlayaLab/AlayaWorld/tar.gz/$COMMIT" \
      | tar xz -C "$ROOT/repo" --strip-components=1 --exclude='*.mp4' --exclude='*.gif' --exclude='*.mov'
    echo "$COMMIT" > "$ROOT/repo_commit.txt"
  fi
  ls "$ROOT/repo/playground"
}

env_() {
  [[ -x $PY ]] || { https_proxy=$PROXY "$UV" python install 3.12 && "$UV" venv -p 3.12 "$ROOT/env"; }
  if ! "$PY" -c "import torch, diffusers, transformers, reactor_runtime, flash_attn.cute, setuptools; assert torch.__version__.startswith('2.10.0')" 2>/dev/null; then
    https_proxy=$PROXY "$UV" pip install -p "$PY" torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
      triton==3.6.0 --index-url https://download.pytorch.org/whl/cu130
    grep -v -E '^(--extra-index-url|torch==|torchvision==|torchaudio==)' "$ROOT/repo/reactor/requirements.txt" > "$ROOT/requirements.serve.txt"
    https_proxy=$PROXY "$UV" pip install -p "$PY" -r "$ROOT/requirements.serve.txt" torch==2.10.0 \
      torchvision==0.25.0 torchaudio==2.10.0 triton==3.6.0 setuptools \
      --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130
    https_proxy=$PROXY "$UV" pip install -p "$PY" -r "$ROOT/repo/reactor/requirements-flash-attn.txt" \
      torch==2.10.0 --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130
    https_proxy=$PROXY "$UV" pip install -p "$PY" --no-deps "protobuf==7.35.1"
  fi
  "$PY" -c "import torch, sys; sys.exit(torch.backends.cudnn.version() < 92600)" 2>/dev/null || \
    https_proxy=$PROXY "$UV" pip install -p "$PY" --no-deps "nvidia-cudnn-cu13==9.26.0.51"
  "$PY" -c "
import torch, diffusers, transformers
print('env ok torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)
print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None"
  "$UV" pip freeze -p "$PY" > "$ROOT/env_freeze.txt"
}

weights() {
  # the 33 GB merged checkpoint first, on its own, at the revision pinned in reactor/alayaworld.yaml, with the
  # adapter's completion marker
  if [[ ! -f $ROOT/checkpoints/.merged_infer.safetensors.reactor-revision ]]; then
    https_proxy=$PROXY "$PY" - "$ROOT" <<'PYEOF'
import sys, yaml
from pathlib import Path
from huggingface_hub import hf_hub_download
b = Path(sys.argv[1]); a = yaml.safe_load(open(b / "repo/reactor/alayaworld.yaml"))["assets"]["model"]
hf_hub_download(a["repo_id"], a["path"], revision=a["revision"], local_dir=b / "checkpoints")
(b / "checkpoints" / f".{a['path']}.reactor-revision").write_text(a["revision"] + "\n")
print("merged checkpoint ok", a["revision"])
PYEOF
  fi
  # TAEHV source: one shallow fetch of the pinned commit (the adapter's blob-less clone + checkout fetches blob by
  # blob, which is slow); same commit, so the adapter's revision check passes
  local T=$ROOT/checkpoints/taehv
  if [[ ! -d $T/.git ]]; then
    local TURL TREV
    TURL=$("$PY" -c "import yaml; print(yaml.safe_load(open('$ROOT/repo/reactor/alayaworld.yaml'))['assets']['taehv_source']['url'])")
    TREV=$("$PY" -c "import yaml; print(yaml.safe_load(open('$ROOT/repo/reactor/alayaworld.yaml'))['assets']['taehv_source']['revision'])")
    rm -rf "$T.tmp" "$ROOT"/checkpoints/.reactor-download-*
    git init -q "$T.tmp" && git -C "$T.tmp" remote add origin "$TURL"
    https_proxy=$PROXY git -C "$T.tmp" fetch -q --depth 1 origin "$TREV"
    git -C "$T.tmp" checkout -q --detach FETCH_HEAD && mv "$T.tmp" "$T"
  fi
  # everything else (Gemma from Hugging Face, Depth-Anything-3 source and weights) through the adapter's own pinned
  # downloads, which verify each revision
  (cd "$ROOT/repo" && REACTOR_WEIGHTS_PATH=$ROOT/checkpoints https_proxy=$PROXY "$PY" -c "
from pathlib import Path
from reactor.alayaworld_assets import read_config, prepare_runtime_assets, validate_runtime_paths
c = read_config(Path('reactor/alayaworld.yaml'))
prepare_runtime_assets(c); validate_runtime_paths(c)
print('ASSETS_OK', c.model.path, c.gemma.path)")
  du -sh "$ROOT/checkpoints"/* 2>/dev/null || true
}

check() {
  (cd "$ROOT/repo" && REACTOR_WEIGHTS_PATH=$ROOT/checkpoints "$PY" -c "
import torch, flash_attn.cute, reactor.alayaworld as aw
print('IMPORT_OK alayaworld on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))")
}

case $STEP in
  repo) repo ;;
  env) env_ ;;
  weights) weights ;;
  check) check ;;
  all) repo; env_; weights; check ;;
  *) echo "usage: bash setup.sh [all|repo|env|weights|check]" >&2; exit 2 ;;
esac
echo "ALAYA_SETUP_DONE ($STEP)"
