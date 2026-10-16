#!/usr/bin/env bash
# Krea Realtime 14B V2V (upstream webcam mode): set up models/selfhost/krea_v2v/server.py. This model has no
# install of its own: it uses the code, environment and weights of models/selfhost/krea_realtime, built by
# krea_realtime/setup.sh (called from here when missing), and links them where the harness looks for them.
# Idempotent.
#
#   bash models/selfhost/krea_v2v/setup.sh [all|env|weights|check]
#
# Steps:
#   env       krea_realtime repo + env (network), then $ROOT/env -> $KREA_ROOT/env
#   weights   krea_realtime weights (network, ~30 GB), then $ROOT/weights -> $KREA_ROOT/weights
#             (krea-realtime-video-14b.safetensors, 28.6 GB, is not duplicated)
#   check     krea_realtime's import check (GPU)
#
# Layout: $ROOT (default $STREAM_ARENA_MODEL_ROOT/krea_v2v) holds only the links env/ and weights/. The harness
# resolves {env} and {weights} of system.json to $STREAM_ARENA_MODEL_ROOT/krea_v2v/{env,weights}, so keep the
# default ROOT when running under the harness. server.py itself imports the upstream code from $KREA_ROOT/repo.
#
# Prerequisites: those of krea_realtime/setup.sh (bash, curl, tar, uv, an NVIDIA driver supporting CUDA 13).
#
# Environment variables:
#   STREAM_ARENA_MODEL_ROOT  model root (default ~/.cache/stream_arena/models)
#   KREA_V2V_ROOT            this model's folder of links (default $STREAM_ARENA_MODEL_ROOT/krea_v2v)
#   KREA_ROOT                the krea_realtime install (default $STREAM_ARENA_MODEL_ROOT/krea_realtime); passed on to
#                            krea_realtime/setup.sh and read by server.py
#   plus the variables of krea_realtime/setup.sh (UV_BIN, WAN13_REV, WAN14_REV, STREAM_ARENA_PROXY, HF_ENDPOINT, ...)
set -euo pipefail
STEP=${1:-all}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${STREAM_ARENA_MODEL_ROOT:-$HOME/.cache/stream_arena/models}
ROOT=${KREA_V2V_ROOT:-$BASE/krea_v2v}
export KREA_ROOT=${KREA_ROOT:-$BASE/krea_realtime}
KREA=$KREA_ROOT
case $STEP in all|env|weights|check) ;; *) echo "usage: $0 [all|env|weights|check]" >&2; exit 2;; esac
mkdir -p "$ROOT"
if [[ $STEP == all || $STEP == env ]]; then
  [[ -f $KREA/repo/release_server.py ]] || bash "$HERE/../krea_realtime/setup.sh" repo
  [[ -x $KREA/env/bin/python ]] || bash "$HERE/../krea_realtime/setup.sh" env
  [[ -e $ROOT/env ]] || ln -sfn "$KREA/env" "$ROOT/env"
fi
if [[ $STEP == all || $STEP == weights ]]; then
  # krea_realtime's weights step downloads with its own environment
  [[ -f $KREA/repo/release_server.py ]] || bash "$HERE/../krea_realtime/setup.sh" repo
  [[ -x $KREA/env/bin/python ]] || bash "$HERE/../krea_realtime/setup.sh" env
  [[ -f $KREA/weights/checkpoints/.complete ]] || bash "$HERE/../krea_realtime/setup.sh" weights
  [[ -e $ROOT/weights ]] || ln -sfn "$KREA/weights" "$ROOT/weights"
fi
if [[ $STEP == all || $STEP == check ]]; then
  bash "$HERE/../krea_realtime/setup.sh" check
fi
echo "KREA_V2V_SETUP_DONE ($STEP)"
