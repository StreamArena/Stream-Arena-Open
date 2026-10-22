<div align="left">

<h1 align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/streameval-wordmark-dark.png" />
    <img src="assets/streameval-wordmark.png" alt="StreamEval" width="175" />
  </picture>
</h1>

**交互式流式视频生成模型评测基准**

[![Hugging Face](https://img.shields.io/badge/Data-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/StreamEval/StreamEval)
[![ModelScope](https://img.shields.io/badge/Data-ModelScope-624AFF)](https://www.modelscope.ai/datasets/StreamEval/StreamEval)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2610.28402-B31B1B?logo=arxiv)](https://arxiv.org/pdf/2610.28402)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-StreamEval-2563EB)](https://streameval.ai/)

[English](README.md) | **简体中文**

</div>

StreamEval 评测交互式流式视频生成，覆盖五项任务和 15 至 300 秒的会话。输入逐步到达，用于检验系统执行新请求和保持上下文的能力。评测框架记录输出与时间戳，并进行自动评分。

## 任务与指标

| 任务 | 系统接收的输入 | 评分指标 |
|---|---|---|
| 数字人对话 | 人像与人设；用户的语音（或文本）轮次 | Quality、A–V Sync、Stability、Dialogue、Identity |
| 数字人表演 | 人像；需要演绎的语音与实时动作指令 | Quality、A–V Sync、Stability、Following、Identity |
| 世界生成 | 开场提示（或图像）；实时提示与相机控制 | Quality、Stability、Following、Continuity（图像条件输入另加 Reference） |
| 指令编辑 | 实时源视频；文本编辑指令 | Quality、Stability、Following、Edit Success、Preservation |
| 参考编辑 | 实时源视频；需要应用、修补、移除或重置的参考图 | Quality、Stability、Reference、Edit Success、Preservation |

各指标为 0–100 分。任务的 Overall 是其适用内容指标的等权平均；不同任务之间不合并总分。[指标定义](docs/evaluation.md)。

## 基准

![基准构建与校验](assets/construction.jpg)

<sub>模型辅助构建与分模态校验。</sub>

![基准构成](assets/statistics.png)

<sub>基准子集：任务构成、输入类型、输入数量与时间间隔。</sub>

完整基准位于 [`benchmark/v1/full`](benchmark/v1/full)，覆盖五项任务与各时长层。每个评测条件集合目录包含 `cases.jsonl`（输入时间轴）、`oracles.jsonl`（期望结果，仅裁判读取）和 `manifest.json`（媒体路径与 SHA-256）。文件格式见[基准说明](benchmark/v1/README.md)。

## 快速开始

需要 Linux 或 macOS 上的 Python ≥ 3.10。以下步骤均在代码仓库根目录、同一个已激活的 Python 环境中运行。

### 安装与 CPU 示例

此示例使用本地生成的媒体，不需要 API key、GPU 或下载基准数据。

```bash
git clone https://github.com/StreamEval/StreamEval.git streameval && cd streameval
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[runtime,judge,dev]"
python -m streameval.benchmark.synthetic --out .smoke
streameval run --model streameval.models.mock:MockModel --track instruction_edit \
  --case sa.v1.901.instruction_edit --benchmark .smoke/set --data .smoke/data --out .smoke/attempt
streameval validate-run .smoke/attempt --benchmark .smoke/set --data .smoke/data
pytest -q
```

### 下载与校验

[Hugging Face](https://huggingface.co/datasets/StreamEval/StreamEval) 和 [ModelScope](https://www.modelscope.ai/datasets/StreamEval/StreamEval) 提供相同的基准定义与媒体（约 3.1 GB）。

```bash
python -m pip install -U huggingface_hub modelscope
```

选择一个平台下载。从 Hugging Face 下载：

```python
from huggingface_hub import snapshot_download
path = snapshot_download("StreamEval/StreamEval", repo_type="dataset")
print(path)
```

也可从 ModelScope 下载：
```python
from modelscope import dataset_snapshot_download
path = dataset_snapshot_download(
    "StreamEval/StreamEval",
    endpoint="https://www.modelscope.ai",
)
print(path)
```

将 `<path>` 替换为下载命令返回的目录。
```bash
export STREAMEVAL_DATA="<path>/data"
streameval validate --benchmark benchmark/v1/full
```

### 真实模型评测

按[模型适配器说明](models/README.md)安装所选模型的依赖，并创建凭据文件：

```bash
cp .env.example .env
```

在 `.env` 中填写模型与裁判凭据。从 `streameval list` 选择 `ID`，然后运行注册的自托管模型：

```bash
streameval list
streameval evaluate --system ID --launch --out runs_full \
  --benchmark benchmark/v1/full --population full
```

注册的自托管服务使用 `--launch`；连接已启动的服务时，将它替换为 `--url ws://HOST:PORT/session`。厂商服务按对应适配器说明配置。

结果保存在 `runs_full/`。结果比较所需的设置见[评测协议](docs/evaluation.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/add_a_model.md](docs/add_a_model.md) | 模型接口、`system.json`、服务端、组合式数字人、厂商适配器 |
| [docs/protocol.md](docs/protocol.md) | WebSocket 协议与每场会话的产物 |
| [docs/evaluation.md](docs/evaluation.md) | 指标、聚合与裁判 |
| [docs/reproduce.md](docs/reproduce.md) | 运行一个集合、分阶段运行、裁判回放、结果文件 |
| [benchmark/v1/README.md](benchmark/v1/README.md) | 基准与各评测条件集合 |
| [models/README.md](models/README.md) | v1 系统：上游代码、权重与许可 |

## 引用

```bibtex
@article{streameval2026,
  title        = {StreamEval: Evaluating Interactive Streaming Video Generation Models},
  author       = {{StreamEval Team}},
  journal      = {arXiv preprint arXiv:2610.28402},
  year         = {2026}
}
```

## 许可

- **代码**：[Apache License 2.0](LICENSE)。被封装的第三方模型保留其原许可，其中部分为非商用
  （[models/README.md](models/README.md)）；厂商适配器受各厂商服务条款约束。
- **基准数据**（时间轴、oracle、生成媒体）：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。
  编辑任务的源视频节选自第三方素材，保留其原许可（CC BY、CC BY-SA、CC0 或公有领域）；使用时（包括发布编辑后的输出）
  请按 [benchmark/v1/ATTRIBUTION.md](benchmark/v1/ATTRIBUTION.md) 署名。详见 [NOTICE](NOTICE)。
