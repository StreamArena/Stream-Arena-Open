#!/bin/bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${HYWP_ROOT:-$BASE/hy_worldplay}
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
REPO_URL=https://github.com/Tencent-Hunyuan/HY-WorldPlay.git
COMMIT=${HYWP_COMMIT:-1588e1336e842b03b0a7860c654ebd7c46bb065e}   # Tencent-Hunyuan/HY-WorldPlay, 2026-06-10
HYWP_REV=f4c29235647707b571479a69b569e4166f9f5bf8                  # tencent/HY-WorldPlay
HYV15_REV=9b49404b3f5df2a8f0b31df27a0c7ab872e7b038                 # tencent/HunyuanVideo-1.5
QWEN_REV=cc594898137f460bfe9f0759e9844b3ce807cfb5                  # Qwen/Qwen2.5-VL-7B-Instruct
BYT5_REV=68377bdc18a2ffec8a0533fef03b1c513a4dd49d                  # google/byt5-small
W=$ROOT/weights
H=$W/HunyuanVideo-1.5
mode=${1:-all}

usage() { echo "usage: bash setup.sh [all|download|env]" >&2; exit 2; }
retry() { for i in 1 2 3 4 5; do https_proxy=$PROXY "$@" && return 0; sleep 20; done; return 1; }

download() {
  for c in git hf modelscope; do
    command -v $c >/dev/null || { echo "$c not found: install it (hf: pip install -U huggingface_hub; modelscope: pip install modelscope)" >&2; exit 1; }
  done
  if [ ! -d "$ROOT/repo/hyvideo" ]; then
    https_proxy=$PROXY git clone "$REPO_URL" "$ROOT/repo" && git -C "$ROOT/repo" checkout "$COMMIT"
  fi
  export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_ENABLE_HF_TRANSFER=0
  mkdir -p "$H/text_encoder" "$H/vision_encoder/siglip"
  retry hf download tencent/HY-WorldPlay ar_distilled_action_model/model.safetensors README.md License.txt \
    --revision $HYWP_REV --max-workers 4 --local-dir "$W/HY-WorldPlay"
  retry hf download tencent/HunyuanVideo-1.5 --include "vae/*" "scheduler/*" "transformer/480p_i2v/config.json" \
    config.json LICENSE --revision $HYV15_REV --max-workers 4 --local-dir "$H"
  retry hf download Qwen/Qwen2.5-VL-7B-Instruct --revision $QWEN_REV --max-workers 4 --local-dir "$H/text_encoder/llm"
  retry hf download google/byt5-small --exclude "*.h5" "*.msgpack" --revision $BYT5_REV --max-workers 4 \
    --local-dir "$H/text_encoder/byt5-small"
  # Glyph-SDXL-v2: the ModelScope source upstream's README names (it has a single `master` branch, no revision pin)
  retry modelscope download --model AI-ModelScope/Glyph-SDXL-v2 assets/color_idx.json \
    assets/multilingual_10-lang_idx.json checkpoints/byt5_model.pt --local_dir "$H/text_encoder/Glyph-SDXL-v2"
  # gated: accept the FLUX.1 [dev] Non-Commercial License on Hugging Face first and export HF_TOKEN
  retry hf download black-forest-labs/FLUX.1-Redux-dev image_encoder/config.json image_encoder/model.safetensors \
    feature_extractor/preprocessor_config.json --max-workers 4 --local-dir "$H/vision_encoder/siglip"
  (cd "$W" && sha256sum HY-WorldPlay/ar_distilled_action_model/model.safetensors > weights.sha256)
}

env_build() {
  python3 -c "import torch" 2>/dev/null || { echo "python3 cannot import torch: env needs a base Python with a CUDA build of PyTorch (see the header)" >&2; exit 1; }
  # reuse the base Python's CUDA torch/torchvision (system site packages)
  python3 -m venv --system-site-packages "$ROOT/env"
  . "$ROOT/env/bin/activate"
  pip install -q "diffusers==0.35.0" "transformers[accelerate,tiktoken]==4.56.0" "peft==0.17.0" einops loguru \
    "safetensors>=0.4.5" "omegaconf>=2.3.0" imageio imageio-ffmpeg "moviepy==1.0.3" "qwen-vl-utils==0.0.8" ftfy scipy \
    pandas protobuf pillow sentencepiece
  python -c "import torch, diffusers, transformers; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" || echo "no flash_attn (sdpa fallback)"
}

case $mode in
  download) download ;;
  env) env_build ;;
  all) download; env_build ;;
  *) usage ;;
esac
