#!/usr/bin/env bash
# Use the dependencies and installation requirements documented in README.md.
set -euo pipefail
STEP=${1:-download}
BASE=${STREAMEVAL_MODEL_ROOT:-$HOME/.cache/streameval/models}
ROOT=${AVTR1_ROOT:-$BASE/avtr1}
COMMIT=${AVTR1_COMMIT:-eb1e5a89ed7bec3debe712b090efd676b3a420bd}
AVTR1_REV=${AVTR1_HF_REVISION:-4c9bd5550f2617d0409ac602f75378502f675731}
DITTO_REV=${DITTO_HF_REVISION:-e4a2f60328ee7c32af585ac4b3cce299e4c8e254}
PROXY=${STREAMEVAL_PROXY:-${https_proxy:-}}
HERE=$(cd "$(dirname "$0")" && pwd)
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

case $STEP in
  download|env) ;;
  *) echo "usage: bash $0 [download|env]" >&2; exit 2 ;;
esac

mkdir -p "$ROOT/weights"
cd "$ROOT"

if [[ $STEP == env ]]; then
  python3 -c "import torch, tensorrt" 2>/dev/null || {
    echo "python3 cannot import torch and tensorrt: env/ is built over a base Python that provides a CUDA build of" \
         "PyTorch and TensorRT for your GPU (see the header of this script)" >&2; exit 1; }
  export PIP_CACHE_DIR=$ROOT/pip_cache
  [[ -x env/bin/python ]] || python3 -m venv --system-site-packages env
  python3 -m pip list --format=freeze 2>/dev/null | grep -i -E \
    "^(torch|torchvision|torchaudio|triton|pytorch-triton|numpy|tensorrt|flash.attn|nvidia-|cuda-|apex|transformer.engine|onnx==)" \
    > env/constraints.txt || true
  P="env/bin/python -m pip install -q -c env/constraints.txt"
  https_proxy=$PROXY $P "anyio" "einops>=0.8.1" "imageio>=2.37" "imageio-ffmpeg>=0.6" "opencv-python-headless>=4.11" tqdm attrs tenacity \
     "kornia>=0.8.1,<0.9" "scikit-image>=0.25" "soxr>=0.4" "roma>=1.5.4" "av>=15" soundfile "safetensors>=0.5" \
     "huggingface_hub>=0.30,<1" onnx-graphsurgeon onnxruntime pillow
  https_proxy=$PROXY $P --no-deps -e repo
  env/bin/python -c "import torch, tensorrt, onnxruntime, kornia, roma, soxr, cv2, avtr1_renderer.pipeline
print('env ok', torch.__version__, tensorrt.__version__, onnxruntime.get_available_providers(),
      torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
  env/bin/python "$HERE/server.py" --weights "$ROOT/weights" --prepare     # TensorRT engines -> storage_sm<cc>
  exit 0
fi

if [ ! -f repo/pixi.toml ]; then
  mkdir -p repo
  https_proxy=$PROXY curl -sfL "https://codeload.github.com/avaturn-live/avtr-1/tar.gz/$COMMIT" | tar xz -C repo --strip-components=1
  echo "$COMMIT" > repo_commit.txt
fi

D=weights/avtr-1
[ -s $D/build_artifacts/avtr1.scripted.pt ] || \
  https_proxy=$PROXY hf download avaturn-live/avtr-1 --revision "$AVTR1_REV" --local-dir $D
[ -f weights/ditto-talkinghead/ditto_onnx/decoder.onnx ] || \
  https_proxy=$PROXY hf download digital-avatar/ditto-talkinghead --revision "$DITTO_REV" --include 'ditto_onnx/*' \
    --local-dir weights/ditto-talkinghead
[ -s weights_sha256.txt ] || (cd weights && sha256sum avtr-1/build_artifacts/* avtr-1/renderer_runtime_artifacts/* \
  ditto-talkinghead/ditto_onnx/* > ../weights_sha256.txt)

# storage layout of avtr1_renderer (revision "main"); server.py links it into storage_sm<cc> next to the engines
S=storage/main
mkdir -p $S/renderer_runtime_artifacts $S/build_artifacts $S/avatars_artifacts
for n in appearance_extractor motion_extractor landmark106 landmark203 insightface_det blaze_face face_mesh; do
  ln -sfn $ROOT/weights/ditto-talkinghead/ditto_onnx/$n.onnx $S/renderer_runtime_artifacts/$n.onnx
done
ln -sfn $ROOT/$D/renderer_runtime_artifacts/libgrid_sample_3d_plugin.so $S/renderer_runtime_artifacts/libgrid_sample_3d_plugin.so
for n in decoder warp_network warp_network_ori stitch_network; do
  ln -sfn $ROOT/weights/ditto-talkinghead/ditto_onnx/$n.onnx $S/build_artifacts/$n.onnx
done
for n in modnet.onnx hubert-lbs-avtr1.onnx avtr1.scripted.pt; do
  ln -sfn $ROOT/$D/build_artifacts/$n $S/build_artifacts/$n
done
for n in backgrounds reference_frames pasteback_mask.png; do
  ln -sfn $ROOT/$D/avatars_artifacts/$n $S/avatars_artifacts/$n
done
echo "source + weights ready in $ROOT; now build env/ and the TensorRT engines on the GPU machine: setup.sh env"
