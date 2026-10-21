# AVTR-1 — self-hosted digital-human renderer (composition)

AVTR-1 is a flow-matching autoregressive talking-portrait model (Avaturn / Goodsize Inc.) that renders lip-synced
speech and active listening from a portrait and dual-stream audio. It is wrapped as a renderer-only composition
system: the shared v1 dialogue stack produces the speech, and `server.py` drives the AVTR-1 renderer pipeline
(`avtr1_renderer.pipeline.Pipeline`, TensorRT) with it.

| | |
|---|---|
| System ids | `avtr1_audio` (`digital_human_audio`), `avtr1_text` (`digital_human_text`) |
| Upstream code | [avaturn-live/avtr-1](https://github.com/avaturn-live/avtr-1) @ `eb1e5a89ed7bec3debe712b090efd676b3a420bd` |
| Weights | [avaturn-live/avtr-1](https://huggingface.co/avaturn-live/avtr-1) (gated) @ `4c9bd555`; [digital-avatar/ditto-talkinghead](https://huggingface.co/digital-avatar/ditto-talkinghead) `ditto_onnx/` @ `e4a2f603` |
| Licence | code: PolyForm Noncommercial 1.0.0 (renderer), AVTR-1 Community License (scripts); weights: AVTR-1 Community License; InsightFace models non-commercial research only (see Weights and licences) |

## Setup

Provide a CUDA build of PyTorch compatible with your GPU and the upstream model. Install a matching TensorRT build. The setup script reuses the base Python packages through `--system-site-packages`; it does not replace PyTorch.

```bash
bash models/selfhost/avtr1/setup.sh            # download: source, weights, weight hashes, storage layout
bash models/selfhost/avtr1/setup.sh env        # on the GPU machine: env/ + TensorRT engines
```

1. `download` (network, no GPU): the upstream source at the pinned commit (`repo/`), `avaturn-live/avtr-1` into
   `weights/avtr-1/`, `digital-avatar/ditto-talkinghead` `ditto_onnx/*` into `weights/ditto-talkinghead/`, the
   sha256 of every weight file into `weights_sha256.txt`, and the artifact layout of `avtr1_renderer`
   (`storage/main/`, links into `weights/`).
2. `env` (GPU): a venv over the base Python with the inference dependencies (base torch / TensorRT / numpy pinned
   as constraints), CPU onnxruntime and upstream installed editable; then the TensorRT engines are built
   (`server.py --prepare`, upstream `scripts/build_engines.py`; about 5.5 min once per GPU architecture, kept in
   `storage_sm<cc>/`). If they are missing the server builds them on its first start.

| variable | default | meaning |
|---|---|---|
| `AVTR1_ROOT` | `$STREAMEVAL_MODEL_ROOT/avtr1` | this model's folder; `--launch` expects the default location |
| `AVTR1_COMMIT` | `eb1e5a89ed7bec3debe712b090efd676b3a420bd` | upstream commit |
| `AVTR1_HF_REVISION` | `4c9bd555…` | revision of `avaturn-live/avtr-1` |
| `DITTO_HF_REVISION` | `e4a2f603…` | revision of `digital-avatar/ditto-talkinghead` |
| `HF_TOKEN` | — | Hugging Face token of an account that accepted the AVTR-1 licence |

`server.py` sets `AVTR1_LOCAL_STORAGE` (upstream's artifact root) itself to `storage_sm<cc>/` under the model folder.

## Weights and licences

| file | source @ revision | sha256 | licence |
|---|---|---|---|
| `avtr-1/build_artifacts/avtr1.scripted.pt` | avaturn-live/avtr-1 @ `4c9bd555` | `c345fb1e…72aed3` | AVTR-1 Community License |
| `avtr-1/build_artifacts/hubert-lbs-avtr1.onnx` | same | `c6c44581…2b968d` | AVTR-1 Community License; HuBERT Apache-2.0 |
| `avtr-1/build_artifacts/modnet.onnx` | same | `244c7acc…876bbd` | Apache-2.0 (MODNet) |
| `avtr-1/renderer_runtime_artifacts/libgrid_sample_3d_plugin.so`, `avatars_artifacts/` | same | `weights_sha256.txt` | Apache-2.0 (grid_sample plugin); AVTR-1 Community License |
| `ditto-talkinghead/ditto_onnx/decoder.onnx`, `warp_network.onnx`, other extractor / stitch graphs | digital-avatar/ditto-talkinghead @ `e4a2f603` | `94fca15f…890be`, `88f82907…52dba9` (full values in `system.json`) | LivePortrait-derived (MIT) |
| `ditto-talkinghead/ditto_onnx/insightface_det.onnx`, `landmark106.onnx` | same | `weights_sha256.txt` | InsightFace: **non-commercial research use only** |

Weight revisions and file hashes are listed in `system.json` and `setup.sh`. Record resolved revisions and downloaded hashes for each installation.

Upstream's `LICENSE.md` describes three separately licensed parts:

- weights and `scripts/`: **AVTR-1 Community License** (Goodsize Inc.) — use, including commercial use, by entities
  under USD 10M annual revenue; a separate commercial agreement above that; use restrictions in its Attachment A
  (for example no impersonation without consent);
- `src/avtr1_renderer/` (the inference pipeline used here) and `src/avaturn_live_streamer/`: **PolyForm
  Noncommercial 1.0.0** — **no commercial use of any kind**, regardless of revenue;
- the bundled InsightFace SCRFD detector and 2D106 landmarks: **non-commercial research use only**; HuBERT, MODNet
  and the grid_sample plugin are Apache-2.0, LivePortrait is MIT.

Running the renderer is therefore limited to non-commercial use. The weights may not be redistributed outside the
Community License; a commercial deployment needs licences from Goodsize Inc. and InsightFace. Nothing is
redistributed by this repository.

## Limitations

- Renderer only: AVTR-1's own conversation backend is replaced by the shared dialogue stack, so speech content and
  voice come from that stack, not from AVTR-1.
- The "listen" audio stream is silence, so active-listening motion is driven only by the absence of the avatar's
  own speech, not by the user's audio.
- Every chunk waits for 205 ms of look-ahead audio, which adds to the renderer latency.
- Deviations from upstream: torch 2.10 / TensorRT 10.14 instead of the pinned torch 2.7.1 / TensorRT 10.12, and the
  portrait-registration ONNX models on CPU instead of onnxruntime's CUDA provider.
- Portraits are center-cropped to square; non-square portraits lose their edges.
