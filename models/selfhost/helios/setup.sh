#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${HELIOS_ROOT:-$BASE/helios}
COMMIT=${HELIOS_COMMIT:-babed9811266e4b5b111c9c1e0977a07899066ab}   # PKU-YuanGroup/Helios main, 2026-08-24
REPO_ID=BestWishYsh/Helios-Distilled
REV=b991c0379a018f4de3227d95468237f56066f5bb      # Hugging Face revision, 2026-10-03
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
case $STEP in
  all|repo|env|weights|hash) ;;
  *) echo "usage: bash setup.sh [all|repo|env|weights|hash]" >&2; exit 2 ;;
esac
UV=${UV_BIN:-$(command -v uv || true)}
[[ -n $UV ]] || { echo "uv not found: install it (https://docs.astral.sh/uv/) or set UV_BIN" >&2; exit 1; }
mkdir -p "$ROOT"/{logs,weights}
PY=$ROOT/env/bin/python
export UV_PYTHON_INSTALL_DIR=$ROOT/py UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy

repo() {
  if [[ ! -f $ROOT/repo/infer_helios.py ]]; then
    https_proxy=$PROXY git clone -q https://github.com/PKU-YuanGroup/Helios.git "$ROOT/repo"
  fi
  git -C "$ROOT/repo" checkout -q "$COMMIT"
  git -C "$ROOT/repo" rev-parse HEAD > "$ROOT/repo_commit.txt"
}

env_() {
  [[ -x $PY ]] || { https_proxy=$PROXY "$UV" python install 3.12 && "$UV" venv -p 3.12 "$ROOT/env"; }
  if ! "$PY" -c "import torch, diffusers, transformers, triton, ftfy, click, typer; assert torch.__version__.startswith('2.10.0')" 2>/dev/null; then
    https_proxy=$PROXY "$UV" pip install -p "$PY" torch==2.10.0 torchvision==0.25.0 \
      --index-url https://download.pytorch.org/whl/cu130
    # inference subset of upstream requirements.txt (training / gradio / wandb / deepspeed not needed)
    https_proxy=$PROXY "$UV" pip install -p "$PY" torch==2.10.0 torchvision==0.25.0 triton==3.6.0 \
      diffusers==0.38.0 transformers==5.3.0 accelerate==1.12.0 "huggingface-hub[cli]==1.4.1" "numpy<2.0.0" \
      ftfy regex sentencepiece protobuf einops imageio imageio-ffmpeg pillow safetensors loguru click typer \
      ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
      --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130
  fi
# Use the dependencies and installation requirements documented in README.md.
  "$PY" -c "import torch, sys; sys.exit(torch.backends.cudnn.version() < 92600)" 2>/dev/null || \
    https_proxy=$PROXY "$UV" pip install -p "$PY" --no-deps ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
      "nvidia-cudnn-cu13==9.26.0.51"
  "$PY" - <<'EOF'
import torch, diffusers, transformers, triton
print("env ok torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
      "diffusers", diffusers.__version__, "transformers", transformers.__version__, "triton", triton.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); print("matmul ok", float((x @ x).float().abs().mean()))
EOF
  "$UV" pip freeze -p "$PY" > "$ROOT/env_freeze.txt"
}

weights() {
  local D=$ROOT/weights/Helios-Distilled
  if [[ ! -f $D/.complete ]]; then
    https_proxy=$PROXY HF_HUB_DISABLE_XET=1 "$ROOT/env/bin/hf" download $REPO_ID --revision $REV \
      --exclude "transformer_ode/*" --max-workers 8 --local-dir "$D"
    echo "$REPO_ID@$REV" > "$D/.complete"
  fi
}

hash_() {
  [[ -f $ROOT/weights_hash.json ]] && return
  # full sha256 per loaded file (6 parallel readers), checked against the HF LFS oids where the hub reports them
  (cd "$ROOT/weights" && find Helios-Distilled -type f \( -name "*.safetensors" -o -name "*.json" -o -name "spiece.model" \) \
     -not -path "*/.cache/*" | sort | xargs -P 6 -n 1 sha256sum) | sort -k2 > "$ROOT/weights.sha256"
  "$PY" - "$ROOT" <<'EOF'
import json, os, sys
b = sys.argv[1]
rows = {}
for line in open(f"{b}/weights.sha256"):
    h, rel = line.split()
    rows[rel] = {"sha256": h, "bytes": os.path.getsize(f"{b}/weights/{rel}")}
import hashlib
combined = hashlib.sha256("".join(f"{k} {v['sha256']}\n" for k, v in sorted(rows.items())
                                  if k.startswith("Helios-Distilled/transformer/")).encode()).hexdigest()
json.dump({"files": rows, "transformer_combined_sha256": combined}, open(f"{b}/weights_hash.json", "w"), indent=1)
print("transformer_combined_sha256", combined, "files", len(rows))
EOF
}

case $STEP in
  repo) repo ;;
  env) env_ ;;
  weights) weights ;;
  hash) hash_ ;;
  all) repo; env_; weights; hash_ ;;
esac
echo "HELIOS_SETUP_DONE ($STEP)"
