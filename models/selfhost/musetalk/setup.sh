#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${MUSETALK_ROOT:-$BASE/musetalk}
COMMIT=${MUSETALK_COMMIT:-0a89dec45a0192b824e3cf4daf96c239440c5ed8}      # TMElyralab/MuseTalk main, 2025-09-26
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
case "${1:-}" in
  ""|env) ;;
  *) echo "usage: bash setup.sh [env]   (no argument: source + weights; env: build env/)" >&2; exit 2 ;;
esac
mkdir -p "$ROOT/weights"
cd "$ROOT"

if [[ "${1:-}" == "env" ]]; then
  python3 -c "import torch" 2>/dev/null || {
    echo "python3 has no torch: env/ is built over a base Python that provides a CUDA build of PyTorch" >&2; exit 1; }
  export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$ROOT/pip_cache}
  [[ -x env/bin/python ]] || python3 -m venv --system-site-packages env
  # keep the base environment's torch / CUDA stack: pip may not replace it
  python3 -m pip list --format=freeze 2>/dev/null | grep -i -E \
    "^(torch|torchvision|torchaudio|triton|pytorch-triton|numpy|tensorrt|flash.attn|nvidia-|cuda-|apex|transformer.engine|onnx==)" \
    > env/constraints.txt
  P="env/bin/python -m pip install -q -c env/constraints.txt"
  https_proxy=$PROXY $P "diffusers>=0.30.2,<0.36" "transformers>=4.39.2,<4.58" "accelerate>=0.28" omegaconf einops \
     librosa soundfile opencv-python-headless pillow "huggingface_hub<1" scipy matplotlib json_tricks munkres \
     "mmengine==0.10.7" addict yapf rich termcolor shapely terminaltables pycocotools \
     "peft>=0.17"   # diffusers needs peft >= 0.17; an older peft in the base environment is upgraded
  https_proxy=$PROXY $P --no-deps "mmcv-lite==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"
  https_proxy=$PROXY $P xtcocotools || echo "xtcocotools unavailable (no cp312 wheel): server.py stubs it (dataset/eval code only)"
  env/bin/python -c "import torch, diffusers, transformers, mmcv, mmengine, cv2, librosa
print('env ok', torch.__version__, diffusers.__version__, transformers.__version__, mmcv.__version__,
      torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
  exit 0
fi

if [ ! -f repo/scripts/realtime_inference.py ]; then
  rm -rf repo && mkdir -p repo
  https_proxy=$PROXY curl -sfL --retry 3 "https://codeload.github.com/TMElyralab/MuseTalk/tar.gz/$COMMIT" -o repo.tar.gz
  tar xzf repo.tar.gz -C repo --strip-components=1 && rm repo.tar.gz
  echo "$COMMIT" > repo_commit.txt
fi
W=weights
get() {  # repo revision local_dir files...
  local repo=$1 r=$2 dir=$3; shift 3
  for f in "$@"; do
    [ -s "$dir/$f" ] || https_proxy=$PROXY hf download "$repo" "$f" --revision "$r" --local-dir "$dir" > /dev/null
  done
  rec "$repo@$r"
}
rec() { touch weights_revision.txt; grep -qF "$1" weights_revision.txt || echo "$1" >> weights_revision.txt; }
get TMElyralab/MuseTalk 2bcb936e2fddb4d86db4c62fd45b387d0c061571 $W musetalkV15/musetalk.json musetalkV15/unet.pth
get stabilityai/sd-vae-ft-mse 31f26fdeee1355a5c34592e401dd41e45d25a493 $W/sd-vae config.json diffusion_pytorch_model.bin
get openai/whisper-tiny 169d4a4341b33bc18d8881c4b69c2e104e1cc0af $W/whisper config.json pytorch_model.bin preprocessor_config.json
get yzd-v/DWPose 1a7144101628d69ee7a3768d1ee3a094070dc388 $W/dwpose dw-ll_ucoco_384.pth
# face-parse-bisent: upstream downloads 79999_iter.pth from Google Drive (gdown id 154JgKpz...) and resnet18 from
# download.pytorch.org; try those first, fall back to an HF mirror with the identical file (sha256 checked below)
mkdir -p $W/face-parse-bisent
FP_SHA=468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567
if [ ! -s $W/face-parse-bisent/79999_iter.pth ]; then
  https_proxy=$PROXY curl -sfL "https://drive.usercontent.google.com/download?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&export=download&confirm=t" \
    -o $W/face-parse-bisent/79999_iter.pth.part || true
  if [ "$(sha256sum < $W/face-parse-bisent/79999_iter.pth.part 2>/dev/null | cut -c1-64)" == "$FP_SHA" ]; then
    mv $W/face-parse-bisent/79999_iter.pth.part $W/face-parse-bisent/79999_iter.pth
    echo "face-parse-bisent/79999_iter.pth: Google Drive 154JgKpzCPW82qINcVieuPH3fZ2e0P812" >> weights_revision.txt
  fi
  rm -f $W/face-parse-bisent/79999_iter.pth.part
fi
[ -s $W/face-parse-bisent/79999_iter.pth ] || { https_proxy=$PROXY hf download ManyOtherFunctions/face-parse-bisent \
  79999_iter.pth --revision 0073b233a5a3c4b1377d4dbf49245017938a72b5 --local-dir $W/face-parse-bisent > /dev/null
  echo "face-parse-bisent/79999_iter.pth: hf ManyOtherFunctions/face-parse-bisent@0073b233a5a3c4b1377d4dbf49245017938a72b5" >> weights_revision.txt; }
[ -s $W/face-parse-bisent/resnet18-5c106cde.pth ] || { https_proxy=$PROXY curl -sfL \
  https://download.pytorch.org/models/resnet18-5c106cde.pth -o $W/face-parse-bisent/resnet18-5c106cde.pth
  rec "face-parse-bisent/resnet18-5c106cde.pth: download.pytorch.org"; }
echo "$FP_SHA  $W/face-parse-bisent/79999_iter.pth" | sha256sum -c -
# S3FD face detector of face_detection (upstream fetches it at first use from adrianbulat.com via torch.hub); when
# that host does not answer, the identical file from an HF mirror is used (sha256 checked below)
mkdir -p $W/s3fd
S3=$W/s3fd/s3fd-619a316812.pth
if [ ! -s $S3 ]; then
  https_proxy=$PROXY curl -sfL -m 300 https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -o $S3.part \
    && mv $S3.part $S3 && echo "s3fd/s3fd-619a316812.pth: adrianbulat.com/downloads/python-fan" >> weights_revision.txt || true
  rm -f $S3.part
fi
if [ ! -s $S3 ]; then
  https_proxy=$PROXY hf download n0x1103/s3fd s3fd-619a316812.pth --revision 7cebac68bfcdf83590c457e8aac5e0ac4b6e3983 \
    --local-dir $W/s3fd > /dev/null
  echo "s3fd/s3fd-619a316812.pth: hf n0x1103/s3fd@7cebac68bfcdf83590c457e8aac5e0ac4b6e3983 (identical to adrianbulat.com s3fd-619a316812.pth)" >> weights_revision.txt
fi
echo "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543  $S3" | sha256sum -c -
ln -sfn "$ROOT/weights" repo/models
ln -sfn "$ROOT/weights/s3fd/s3fd-619a316812.pth" repo/musetalk/utils/face_detection/detection/sfd/s3fd.pth
[ -s weights_sha256.txt ] || (cd weights && sha256sum musetalkV15/unet.pth sd-vae/diffusion_pytorch_model.bin \
  whisper/pytorch_model.bin dwpose/dw-ll_ucoco_384.pth face-parse-bisent/79999_iter.pth \
  face-parse-bisent/resnet18-5c106cde.pth s3fd/s3fd-619a316812.pth > ../weights_sha256.txt)
echo "source + weights ready in $ROOT; now build env/ on the GPU machine: setup.sh env"
