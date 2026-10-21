# Model adapters

Registered model adapters for streaming evaluation:

* `selfhost/<model>/`: an open model wrapped as a server of the [WebSocket protocol](../docs/protocol.md)
  (`server.py`), its registry entry (`system.json`), a script that fetches the upstream code and weights
  (`setup.sh`) and a README.
* `providers/<vendor>/`: an adapter for a commercial service, used with your own account
  ([docs/add_a_model.md](../docs/add_a_model.md#vendor-systems)).

`streameval list` prints every system id with its track. A system id is `<system_id of the entry>_<track
suffix>`; the tables list available adapters.

**This repository contains wrapper code only.** No upstream code or weights are included; `setup.sh` downloads
them from their publishers, and they stay under their own licences. Several are non-commercial or research-only,
and some are gated on Hugging Face (accept the licence there and set `HF_TOKEN`). Read the "Weights and licences"
section of a model's README before using it; the *restrictions* column below is a summary, not legal advice.

## Self-hosted models

| model | system ids | upstream code | licences | restrictions |
|---|---|---|---|---|
| [alayaworld](selfhost/alayaworld/) | `alayaworld_i2v` | [AlayaLab/AlayaWorld](https://github.com/AlayaLab/AlayaWorld) @ `7e7dcfe5` | LTX-2 Community; Gemma Terms; DA3 weights CC BY-NC 4.0 | non-commercial |
| [aptavatar](selfhost/aptavatar/) | `aptavatar_performance`<br>`aptavatar_comp_audio`<br>`aptavatar_comp_text` | [TaoLiveAIGC/AptAvatar](https://github.com/TaoLiveAIGC/AptAvatar) @ `c3cc19ca` | Apache-2.0 | – |
| [avatar_forever](selfhost/avatar_forever/) | `avatar_forever_performance`<br>`avatar_forever_comp_audio`<br>`avatar_forever_comp_text` | [leeruibin/avatarforever](https://github.com/leeruibin/avatarforever) @ `4dfc42b0` | LTX-2 Community; Gemma Terms | paid licence above USD 10M revenue |
| [avatarforcing](selfhost/avatarforcing/) | `avatarforcing_performance`<br>`avatarforcing_comp_audio`<br>`avatarforcing_comp_text` | [KlingAIResearch/AvatarForcing](https://github.com/KlingAIResearch/AvatarForcing) @ `63b73e6c` | upstream states both Apache-2.0 and academic use only | treat as academic use only |
| [avtr1](selfhost/avtr1/) | `avtr1_audio`<br>`avtr1_text` | [avaturn-live/avtr-1](https://github.com/avaturn-live/avtr-1) @ `eb1e5a89` | PolyForm Noncommercial 1.0.0; AVTR-1 Community License; InsightFace models | non-commercial |
| [causal_forcing](selfhost/causal_forcing/) | `causal_forcing_1p3b_t2v`<br>`causal_forcing_1p3b_i2v` | [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) @ `82029032` | Apache-2.0 | – |
| [context_forcing](selfhost/context_forcing/) | `context_forcing_1p3b_t2v` | [TIGER-AI-Lab/Context-Forcing](https://github.com/TIGER-AI-Lab/Context-Forcing) @ `6dd9d9a8` | MIT | – |
| [editalive](selfhost/editalive/) | `editalive_14b_instruction` | [GVCLab/EditaLive](https://github.com/GVCLab/EditaLive) @ `9fe7fdb4` | Apache-2.0 files; upstream README: academic research only | treat as academic research only |
| [helios](selfhost/helios/) | `helios_t2v`<br>`helios_i2v` | [PKU-YuanGroup/Helios](https://github.com/PKU-YuanGroup/Helios) @ `babed981` | Apache-2.0 | – |
| [hy_worldplay](selfhost/hy_worldplay/) | `hy_worldplay_i2v` | [Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) @ `1588e133` | Tencent HY-WorldPlay / HunyuanVideo-1.5 Community; FLUX.1 [dev] Non-Commercial | non-commercial; not licensed in the EU, UK, South Korea |
| [joyai_video_edit](selfhost/joyai_video_edit/) | `joyai_video_edit_instruction`<br>`joyai_video_edit_reference` | [jd-opensource/JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit) @ `ca17e1d1` | Apache-2.0; MiMo-VL MIT | – |
| [krea_realtime](selfhost/krea_realtime/) | `krea_realtime_14b_t2v`<br>`krea_realtime_14b_i2v` | [krea-ai/realtime-video](https://github.com/krea-ai/realtime-video) @ `acf4b7b4` | code CC BY-NC-SA 4.0; weights Apache-2.0 | non-commercial |
| [lingbot_world](selfhost/lingbot_world/) | `lingbot_world_v2_1p3b_i2v` | [Robbyant/lingbot-world-v2](https://github.com/Robbyant/lingbot-world-v2) @ `1895d300` | CC BY-NC-SA 4.0 | non-commercial |
| [livetalk](selfhost/livetalk/) | `livetalk_performance`<br>`livetalk_comp_audio`<br>`livetalk_comp_text` | [GAIR-NLP/LiveTalk](https://github.com/GAIR-NLP/LiveTalk) @ `e0d555fc` | upstream states both Apache-2.0 and CC BY-NC-SA 4.0 | treat as non-commercial |
| [longlive](selfhost/longlive/) | `longlive2_5b_t2v` | [NVlabs/LongLive](https://github.com/NVlabs/LongLive) @ `fb16a879` | code Apache-2.0; weights NVIDIA Open Model License | – |
| [matrix_game_35](selfhost/matrix_game_35/) | `matrix_game_35_i2v` | [Riemann-Dynamics/Matrix-Game-3.5](https://github.com/Riemann-Dynamics/Matrix-Game-3.5) @ `fbf7def0` | Apache-2.0; DA3 weights CC BY-NC 4.0 | non-commercial (depth weights) |
| [memflow](selfhost/memflow/) | `memflow_1p3b_t2v` | [KlingAIResearch/MemFlow](https://github.com/KlingAIResearch/MemFlow) @ `7ed51477` | Apache-2.0, some files CC BY-NC-SA 4.0; weights: none stated | partly non-commercial |
| [musetalk](selfhost/musetalk/) | `musetalk_audio`<br>`musetalk_text` | [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) @ `0a89dec4` | MIT | – |
| [quark_liveavatar](selfhost/quark_liveavatar/) | `quark_liveavatar_performance`<br>`quark_liveavatar_comp_audio`<br>`quark_liveavatar_comp_text` | [Alibaba-Quark/LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar) @ `c3c47d03` | Apache-2.0 | – |
| [reward_forcing](selfhost/reward_forcing/) | `reward_forcing_1p3b_t2v` | [JaydenLyh/Reward-Forcing](https://github.com/JaydenLyh/Reward-Forcing) @ `4317f5a0` | Apache-2.0 | – |
| [rolling_forcing](selfhost/rolling_forcing/) | `rolling_forcing_1p3b_t2v` | [TencentARC/RollingForcing](https://github.com/TencentARC/RollingForcing) @ `a1477d09` | MIT | – |
| [sana_streaming](selfhost/sana_streaming/) | `sana_streaming_2b_instruction` | [NVlabs/Sana](https://github.com/NVlabs/Sana) @ `f9178744` | Apache-2.0; LTX-2 Community; Gemma Terms | paid licence above USD 10M revenue |
| [self_forcing](selfhost/self_forcing/) | `self_forcing_1p3b_t2v` | [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) @ `33593df3` | Apache-2.0 | – |
| [shotstream](selfhost/shotstream/) | `shotstream_1p3b_t2v` | [KlingAIResearch/ShotStream](https://github.com/KlingAIResearch/ShotStream) @ `c6a11719` | Apache-2.0 | – |
| [soulx_flashhead](selfhost/soulx_flashhead/) | `soulx_flashhead_audio`<br>`soulx_flashhead_text` | [Soul-AILab/SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) @ `9bc03de0` | Apache-2.0; VAE_LTX OpenRAIL-M | use restrictions (OpenRAIL-M) |
| [soulx_flashtalk](selfhost/soulx_flashtalk/) | `soulx_flashtalk_audio`<br>`soulx_flashtalk_text` | [Soul-AILab/SoulX-FlashTalk](https://github.com/Soul-AILab/SoulX-FlashTalk) @ `2ba2e1e8` | Apache-2.0 | – |
| [soulx_liveact](selfhost/soulx_liveact/) | `soulx_liveact_performance`<br>`soulx_liveact_comp_audio`<br>`soulx_liveact_comp_text` | [Soul-AILab/SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct) @ `ac8579b9` | code: no licence file upstream; weights Apache-2.0 | code has no licence |
| [yume](selfhost/yume/) | `yume_5b_t2v`<br>`yume_5b_i2v` | [stdstu12/YUME](https://github.com/stdstu12/YUME) @ `111c3fab` | Apache-2.0 | – |
| [zing](selfhost/zing/) | `zing_0p5_5b_t2v`<br>`zing_0p5_5b_i2v` | [seedleap/zing-world-model](https://github.com/seedleap/zing-world-model) @ `11212da0` | Apache-2.0 | – |

### Running one

```bash
export STREAMEVAL_MODEL_ROOT=/big/disk/streameval_models        # default ~/.cache/streameval/models
bash models/selfhost/longlive/setup.sh                              # steps differ per model: see its README
streameval run --system longlive2_5b_t2v --case sa.v1.043.general_interaction --launch --smoke
streameval batch --system longlive2_5b_t2v --launch               # all applicable full-benchmark jobs for this track
```

`--launch` starts the `server.cmd` of the entry, waits for `GET /health` (weight loading can take many minutes)
and stops the server afterwards. Each model lives in `$STREAMEVAL_MODEL_ROOT/<model>/` (`repo/`, `env/`,
`weights/`); nothing is written into this repository. The launch command and its health timeout are in the
model's `system.json`, and `python server.py --help` lists the server's flags. Composition systems (renderers
driven by the shared dialogue stack) also need `DASHSCOPE_API_KEY`.

Installation requirements:

* Use CUDA, PyTorch and attention kernels compatible with the target GPU. Some adapters provide SDPA fallbacks.
* Some setup scripts reuse a base CUDA Python environment through `--system-site-packages`; others use `uv`.
  Follow the individual model's setup instructions and allocate the GPU count in its registry entry.
* Pin and record downloaded source and weight revisions. Upstream code and weights retain their own licences.
* Evaluation defaults to the complete benchmark (`benchmark/v1/full`); the development roster remains
  available for integration checks (see [benchmark/v1](../benchmark/v1/README.md)).

Environment variables shared by the scripts (a model's README lists only its own):

| variable | default | meaning |
|---|---|---|
| `STREAMEVAL_MODEL_ROOT` | `~/.cache/streameval/models` | where code, environments and weights go; also read by the harness for `{env}` and `{weights}` in `server.cmd` |
| `<MODEL>_ROOT` | `$STREAMEVAL_MODEL_ROOT/<model>` | one model's folder (the harness still expects the default location) |
| `STREAMEVAL_PROXY` | `$https_proxy` | optional HTTPS proxy for downloads |
| `HF_ENDPOINT`, `HF_TOKEN` | `https://huggingface.co`, – | Hugging Face endpoint, and the token for gated repositories |
| `PIP_INDEX_URL` | PyPI | package index |
| `UV_BIN` | `uv` on `PATH` | the `uv` executable, for the scripts that use it |

Some models reuse another model's download when it is present (the Wan2.1 base weights, for example) and fetch
the same pinned revision themselves otherwise.

### Shared files

| file | what it is |
|---|---|
| `selfhost/t2v_stream.py` | the live loop shared by the block-wise generation servers: block prompts, paced sending, acks, weight hashing |
| `selfhost/edit_stream.py` | the same for the editing servers: newest-frame source intake, paced sending, acks |
| `selfhost/_fetch_github.py` | downloads an upstream repository at a commit as a tarball |
| `selfhost/_shims/`, `selfhost/soulx_flashtalk/shims/` | small stand-ins, written for this repository, for the parts of `flash_attn` and `xformers` that upstream code imports, implemented with torch's scaled-dot-product attention. They contain no code of those projects and are put on `sys.path` only by the servers that need them |

## Vendor adapters

| folder | system ids | service |
|---|---|---|
| [vidu](providers/vidu/) | `vidu_s2_avatar_audio`<br>`vidu_s2_avatar_text`<br>`vidu_s2_avatar_performance_text`<br>`vidu_s2_editing_all`<br>`vidu_s2_editing_t2i` | Vidu S2 avatar and editing |
| [xmax](providers/xmax/) | `xmax_x2_instruction`<br>`xmax_x2_reference` | XMax X2.0 editing |
| [runway](providers/runway/) | `runway_characters_audio` | Runway Characters |
| [liveavatar](providers/liveavatar/) | `liveavatar_full_audio`<br>`liveavatar_full_text` | HeyGen LiveAvatar (stock avatar) |
| [tavus](providers/tavus/) | `tavus_cvi_full_audio`<br>`tavus_cvi_full_text` | Tavus CVI |
| [vivix](providers/vivix/) | `vivix_a1_audio` | Vivix A1 |

Setup, accounts, cost estimates and dry runs: [docs/add_a_model.md](../docs/add_a_model.md#vendor-systems). An adapter uses the
vendor's official API or SDK with your own key, under that vendor's terms of service.

## Adding a model

A new model does not have to live here: see [examples/my_model](../examples/my_model) and
[docs/add_a_model.md](../docs/add_a_model.md). A self-hosted model contributed to this folder follows the layout
above and the README template of the existing ones (system ids, upstream commit and weight revisions, licences,
setup, weights and licences, limitations).
