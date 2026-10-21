#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-all}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${EDITALIVE_ROOT:-$BASE/editalive}
COMMIT=${EDITALIVE_COMMIT:-9fe7fdb4ce3ff69305f026e8dec265aaf7a4196f}   # GVCLab/EditaLive main, 2026-09-14
WAN_REV=cb93a225fbaf1ca100f54e79da8f994995b689b3                       # Wan-AI/Wan2.2-Animate-14B
EL_REV=${EDITALIVE_HF_REV:-main}                                       # unpinned: the v1 revision was not recorded
FV_REV=${FLASH_VAED_REV:-4e329b3b17871056cca1ec18e235eab2b17907e6}     # Aoko955/Flash-VAED
T5_SHA256=7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_MAX_WORKERS=8
case $STEP in all|src|env|weights|check) ;; *) echo "usage: $0 [all|src|env|weights|check]" >&2; exit 2;; esac
mkdir -p "$ROOT/weights"
cd "$ROOT"
PY=$ROOT/env/bin/python
retry() { for t in 1 2 3 4; do "$@" && return 0; echo "retry $t: $*" >&2; sleep 60; done; return 1; }
export UV_PYTHON_INSTALL_DIR=$ROOT/py UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy

if [[ $STEP == all || $STEP == src ]] && [[ ! -f repo/edit_webcam.py ]]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL --max-time 900 "https://codeload.github.com/GVCLab/EditaLive/tar.gz/$COMMIT" \
    | tar xz -C repo --strip-components=1 --exclude='*.mp4' --exclude='*.gif'
  echo "$COMMIT" > repo_commit.txt
fi

if [[ $STEP == all || $STEP == env ]]; then
  UV=${UV_BIN:-$(command -v uv || true)}
  [[ -n $UV ]] || { echo "uv not found: install it (https://docs.astral.sh/uv/) or set UV_BIN" >&2; exit 1; }
  [[ -x $PY ]] || { https_proxy=$PROXY "$UV" python install 3.11 && "$UV" venv -p 3.11 "$ROOT/env"; }
  if ! "$PY" -c "import torch, diffusers, transformers, decord, onnxruntime, tensorrt_libs, setuptools, cv2, easydict, einops; assert torch.__version__.startswith('2.10.0')" 2>/dev/null; then
    retry env https_proxy=$PROXY "$UV" pip install -p "$PY" torch==2.10.0 torchvision==0.25.0 triton==3.6.0 \
      --index-url https://download.pytorch.org/whl/cu130
    # upstream requirements.txt pins (torch replaced above; moviepy / imageio only serve the offline script's file I/O)
    retry env https_proxy=$PROXY "$UV" pip install -p "$PY" torch==2.10.0 torchvision==0.25.0 triton==3.6.0 \
      accelerate==1.12.0 diffusers==0.37.0 transformers==4.57.3 safetensors==0.7.0 huggingface-hub==0.36.2 \
      sentencepiece==0.2.1 einops==0.8.1 easydict==1.13 numpy==1.26.4 scipy==1.15.3 Pillow==11.3.0 \
      opencv-python-headless==4.9.0.80 decord==0.6.0 "imageio[ffmpeg]==2.37.2" tqdm==4.67.1 packaging==25.0 \
      onnxruntime-gpu==1.30.0 tensorrt-cu13==11.3.0.99 onnx==1.23.2 loguru==0.7.3 matplotlib==3.10.8 ftfy regex protobuf pydantic setuptools \
      --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130
  fi
# Use the dependencies and installation requirements documented in README.md.
  "$PY" -c "import torch, sys; sys.exit(torch.backends.cudnn.version() < 92600)" 2>/dev/null || \
    https_proxy=$PROXY "$UV" pip install -p "$PY" --no-deps "nvidia-cudnn-cu13==9.26.0.51"
  "$PY" - <<'EOF'
import torch, diffusers, transformers, onnxruntime
print("env ok torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
      "diffusers", diffusers.__version__, "transformers", transformers.__version__,
      "onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
EOF
  "$UV" pip freeze -p "$PY" > "$ROOT/env_freeze.txt"
fi

if [[ $STEP == all || $STEP == weights ]]; then
  DL=$([[ -x $PY ]] && echo $PY || echo python3)
  W=$ROOT/weights
  mkdir -p "$W/Wan-Animate"
  if [[ ! -f $W/Wan-Animate/.complete ]]; then
    $DL -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.2-Animate-14B', revision='$WAN_REV', local_dir='$W/Wan-Animate', max_workers=8,
                  ignore_patterns=['relighting_lora/*', 'relighting_lora.ckpt', 'process_checkpoint/sam2/*',
                                   'xlm-roberta-large/*', 'assets/*', 'models_t5_umt5-xxl-enc-bf16.pth'])"
    echo "Wan-AI/Wan2.2-Animate-14B@$WAN_REV" > "$W/Wan-Animate/.complete"
  fi
  [[ -e $W/Wan-Animate/models_t5_umt5-xxl-enc-bf16.pth ]] || $DL -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Wan-AI/Wan2.2-Animate-14B', 'models_t5_umt5-xxl-enc-bf16.pth', revision='$WAN_REV',
                local_dir='$W/Wan-Animate')   # sha256 $T5_SHA256"
  $DL -c "
from huggingface_hub import hf_hub_download
import os, shutil
for repo, rev, name, dst in [('huaichang/EditaLive', '$EL_REV', 'weights/editalive_edit.safetensors', 'EditaLive/editalive_edit.safetensors'),
                             ('huaichang/EditaLive', '$EL_REV', 'weights/editalive_streaming.safetensors', 'EditaLive/editalive_streaming.safetensors'),
                             ('huaichang/EditaLive', '$EL_REV', 'weights/lightx2v.safetensors', 'EditaLive/lightx2v.safetensors'),
                             ('Aoko955/Flash-VAED', '$FV_REV', 'models/wan/Flash_VAED_Wan.pth', 'Flash-VAED/Flash_VAED_Wan.pth')]:
    out = os.path.join('$W', dst)
    if os.path.isfile(out):
        continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p = hf_hub_download(repo, name, revision=rev, local_dir='$W/_dl')
    shutil.move(p, out)
    print('downloaded', repo, name)
"
  ln -sfn "$W" "$ROOT/repo/weights"                       # upstream defaults resolve ./weights
  ls -la "$W/Wan-Animate" "$W/Wan-Animate/process_checkpoint"/* "$W/EditaLive" "$W/Flash-VAED"
fi

if [[ $STEP == all || $STEP == check ]]; then
  (cd repo && PYTHONPATH="$HERE/../_shims" "$PY" -c "
import torch, onnxruntime
from editalive.pipeline.editalive_streaming import EditaLiveStreamingPipeline
print('IMPORT_OK editalive on', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), onnxruntime.get_available_providers())")
fi
echo "EDITALIVE_SETUP_DONE ($STEP)"
