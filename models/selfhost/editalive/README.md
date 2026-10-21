# EditaLive (14B) — self-hosted streaming instruction editing

EditaLive re-renders a person in a live video from a reference frame, driven by per-frame pose and face crops, on
Wan2.2-Animate-14B with three LoRAs. The server runs upstream's live webcam path with the benchmark source in place
of the camera, behind the WebSocket protocol (`docs/protocol.md`) and the shared edit loop
`models/selfhost/edit_stream.py`.

| | |
|---|---|
| System ids | `editalive_14b_instruction` (`instruction_edit`) |
| Upstream code | [GVCLab/EditaLive](https://github.com/GVCLab/EditaLive) @ `9fe7fdb4ce3ff69305f026e8dec265aaf7a4196f` (2026-09-14); arXiv 2608.27123 |
| Weights | huaichang/EditaLive (revision not pinned, see Weights and licences); Wan-AI/Wan2.2-Animate-14B @ `cb93a225`; Aoko955/Flash-VAED @ `4e329b3b` |
| Licence | code and LoRAs: Apache-2.0, upstream README: academic research only; Wan2.2-Animate-14B: Apache-2.0; Flash-VAED: none stated (see Weights and licences) |

## Setup

Prerequisites: bash, curl, tar, [uv](https://docs.astral.sh/uv/), an NVIDIA driver supporting CUDA 13, about 50 GB
of disk for weights. The environment does not reuse a system PyTorch: `setup.sh env` installs torch 2.10.0+cu130.

```bash
bash models/selfhost/editalive/setup.sh src       # GVCLab/EditaLive tarball at the pinned commit
bash models/selfhost/editalive/setup.sh env       # uv CPython 3.11 + venv (torch 2.10.0+cu130, upstream pins, TensorRT 11, cuDNN 9.26)
bash models/selfhost/editalive/setup.sh weights   # Wan2.2-Animate-14B, EditaLive LoRAs, Flash-VAED
bash models/selfhost/editalive/setup.sh check     # imports the streaming pipeline (GPU)
```

`setup.sh` (no argument) runs all four. The umT5-XXL text encoder is downloaded from Wan-AI/Wan2.2-Animate-14B at the pinned revision.

| variable | default | meaning |
|---|---|---|
| `EDITALIVE_ROOT` | `$STREAMEVAL_MODEL_ROOT/editalive` | this model's folder: `repo/`, `env/`, `weights/`, caches (setup.sh and server.py) |
| `EDITALIVE_COMMIT` | `9fe7fdb4ce3ff69305f026e8dec265aaf7a4196f` | GVCLab/EditaLive commit |
| `EDITALIVE_HF_REV` | `main` | huaichang/EditaLive revision |
| `FLASH_VAED_REV` | `4e329b3b17871056cca1ec18e235eab2b17907e6` | Aoko955/Flash-VAED revision |
| `HF_ENDPOINT`, `HF_TOKEN` | `https://huggingface.co`, unset | Hugging Face endpoint and token |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | `$EDITALIVE_ROOT/compile_cache/...`, `$EDITALIVE_ROOT/triton_cache` | compile caches (server.py) |
| `WAN_ORT_INTRA_THREADS` | `4` | ONNX Runtime threads of the pose extractor (server.py) |
| `HF_HUB_OFFLINE` | `1` | server.py loads local files only |

## Weights and licences

`$STREAMEVAL_MODEL_ROOT/editalive/weights`, from Hugging Face. The sha256 of every file is recorded per session in
`ready.system_config.weights_sha256`.

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `EditaLive/{editalive_edit,lightx2v,editalive_streaming}.safetensors` | huaichang/EditaLive @ `main` (revision not pinned; record the downloaded revision) | per session | Apache-2.0 licence file; the upstream README restricts use to **academic research only** |
| `Wan-Animate/` DiT shards, `Wan2.1_VAE.pth`, CLIP, `process_checkpoint/{det,pose2d}` | Wan-AI/Wan2.2-Animate-14B @ `cb93a225fbaf1ca100f54e79da8f994995b689b3` | per session | Apache-2.0 |
| `Wan-Animate/models_t5_umt5-xxl-enc-bf16.pth` | same repo and revision | `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d` | Apache-2.0 |
| `Flash-VAED/Flash_VAED_Wan.pth` | Aoko955/Flash-VAED @ `4e329b3b17871056cca1ec18e235eab2b17907e6` | per session | no licence stated on the model card |

EditaLive is for academic research only according to its authors, regardless of the Apache-2.0 file. Benchmark
evaluation only; nothing is redistributed.

## Limitations

- EditaLive is trained for person edits (add / remove accessories, garment and hair colour, ~20 whole-subject
  styles; `repo/docs/SUPPORTED_EDITS.md`). Benchmark instructions are passed verbatim whatever their target. The edits
  that land are person edits (garment and headwear); scene and object targets are re-rendered unedited, and a scene
  without a person is re-rendered close to the source without the edit.
- One instruction per session, as upstream: a new instruction starts a new session from a fresh reference frame, so
  an earlier edit is gone once the next instruction applies.
- Upstream's face-box helper (`get_face_bboxes`) raises on NaN face keypoints when ViTPose finds no face; this ends
  the session as it ends upstream's webcam session. Such failures remain in the evaluation denominator.
- Landscape output only (672×384); portrait sources are centre-cropped.
- Dense attention instead of upstream's default sparse VSA.
