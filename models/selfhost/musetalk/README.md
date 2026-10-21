# MuseTalk 1.5 — self-hosted composition renderer (digital human)

MuseTalk 1.5 is a lip-sync model that regenerates the mouth and jaw region of a reference video from speech. The
server runs it in upstream's real-time mode on a one-frame reference video made of the case portrait; the shared
dialogue stack (ASR → LLM → TTS, run by the harness) streams the TTS speech to it as `audio_in`, and the server
returns frames plus the same speech as `audio_out`.

| | |
|---|---|
| System ids | `musetalk_audio` (`digital_human_audio`), `musetalk_text` (`digital_human_text`) — `system_kind: composition`, `composition.input = audio`, `audio_out = server` |
| Upstream code | [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) @ `0a89dec45a0192b824e3cf4daf96c239440c5ed8` |
| Weights | [TMElyralab/MuseTalk](https://huggingface.co/TMElyralab/MuseTalk) @ `2bcb936e2fddb4d86db4c62fd45b387d0c061571` (`musetalkV15/unet.pth`, `musetalk.json`) + sd-vae-ft-mse, whisper-tiny, DWPose, face-parse-bisent, S3FD (see Weights and licences) |
| Licence | code: MIT; weights: MIT (components MIT / Apache-2.0 / BSD-3-Clause; see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. The setup script reuses the base Python packages through `--system-site-packages`; it does not replace PyTorch.

```bash
bash models/selfhost/musetalk/setup.sh        # source tree at the pinned commit, weights, weight hashes
bash models/selfhost/musetalk/setup.sh env    # env/ (on the GPU machine)
```

`setup.sh env` adds diffusers 0.35, transformers 4.57, peft ≥ 0.17 (diffusers' minimum), mmengine 0.10.7,
mmdet 3.1.0, mmpose 1.1.0 and **mmcv-lite 2.0.1** (the same release without compiled ops), constrained to the base
environment's torch / CUDA packages. DWPose's config builds its CSPNeXt backbone from the mmdet registry, whose
import pulls in `mmcv.ops`; top-down pose inference never calls those ops, so `server.py` serves `mmcv.ops.*` as
placeholders that raise if used (`ready.config.mmcv_ops`). `xtcocotools` has no cp312 wheel; when it is missing,
`server.py` registers an inert stub (it is only imported by dataset / evaluation code). Checkpoints are loaded with
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (set by `server.py` unless already set; torch ≥ 2.6 defaults to
`weights_only=True`, which rejects mmengine checkpoints).

| variable | default | meaning |
|---|---|---|
| `MUSETALK_ROOT` | `$STREAMEVAL_MODEL_ROOT/musetalk` | this model's folder |
| `MUSETALK_COMMIT` | `0a89dec45a0192b824e3cf4daf96c239440c5ed8` | upstream commit |
| `PIP_CACHE_DIR` | `$MUSETALK_ROOT/pip_cache` | pip cache of `setup.sh env` |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | `1` (set by `server.py`) | allow loading the mmengine checkpoints |

## Weights and licences

Files under `$STREAMEVAL_MODEL_ROOT/musetalk/weights` (upstream's `models/` tree); sha256 as in `system.json`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `musetalkV15/unet.pth`, `musetalkV15/musetalk.json` | TMElyralab/MuseTalk @ `2bcb936e` | `7ebf6c98…787007` | MIT |
| `sd-vae/diffusion_pytorch_model.bin` (+ `config.json`) | stabilityai/sd-vae-ft-mse @ `31f26fde` | `1b4889b6…377ddc` | MIT |
| `whisper/pytorch_model.bin` (+ configs) | openai/whisper-tiny @ `169d4a43` | `9607f98a…8aac1d` | MIT (whisper; the HF model card states Apache-2.0) |
| `dwpose/dw-ll_ucoco_384.pth` | yzd-v/DWPose @ `1a714410` | `0d9408b1…8c7a07` | Apache-2.0 |
| `face-parse-bisent/79999_iter.pth` | upstream's Google Drive id `154JgKpz…`, else HF ManyOtherFunctions/face-parse-bisent @ `0073b233` (identical file, sha256 checked) | `468e13ca…b26567` | MIT (face-parsing.PyTorch) |
| `face-parse-bisent/resnet18-5c106cde.pth` | download.pytorch.org | `5c106cde…0613f8` | BSD-3-Clause (torchvision) |
| `s3fd/s3fd-619a316812.pth` | adrianbulat.com (upstream's torch.hub source), else HF n0x1103/s3fd @ `7cebac68` (identical file, sha256 checked) | `619a3168…b33543` | BSD-3-Clause (face-alignment) |

MuseTalk code and weights are MIT (`LICENSE` in the GitHub and HF repositories); the component licences above are
Weight revisions and hashes are listed in the table and `system.json`.

## Limitations

- Lip sync only: there is **no head motion, no blinking, no gaze or expression change and no body motion** — only
  the lips and jaw move, driven by the speech.
- If the face detector finds no face, the adapter streams the unchanged portrait with the speech and records
  `animated: false` with a reason. The output is scored by the same rules as every other session.
- Deviations from upstream's environment: torch 2.10 instead of 2.0.1, mmcv-lite with placeholder `mmcv.ops`, an
  `xtcocotools` stub when the wheel is unavailable (see Setup).
