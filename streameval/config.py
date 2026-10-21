"""Paths and endpoints, from the environment only (optionally a git-ignored .env file in the current directory or
in the checkout).

No credential is ever read from, or written to, a tracked file.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from . import REPO


def _dotenv(path: Path) -> dict:
    values = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def env(key: str, default: str | None = None) -> str | None:
    """Process environment first, then ./.env; empty values count as unset."""
    value = os.environ.get(key) or _dotenv(Path.cwd() / ".env").get(key) or _dotenv(REPO / ".env").get(key)
    return value or default


def data_root() -> Path:
    """The media root: STREAMEVAL_DATA, else data/ in the checkout (assets are at <root>/<manifest path>)."""
    return Path(env("STREAMEVAL_DATA", str(REPO / "data"))).expanduser().resolve()


@dataclass(frozen=True)
class Endpoint:
    base_url: str | None
    model: str
    api_key: str | None

    def require(self, what: str) -> "Endpoint":
        if not self.base_url or not self.api_key:
            raise SystemExit(f"{what}: no API key or base URL; set DASHSCOPE_API_KEY (or the STREAMEVAL_*_API_KEY "
                             "and _BASE_URL of this endpoint) in the environment or in .env (see .env.example)")
        return self


def judge_endpoint() -> Endpoint:
    """Vision/text judge (OpenAI-compatible chat completions). Default: qwen3.8-max-0902 on DashScope."""
    return Endpoint(env("STREAMEVAL_JUDGE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                    env("STREAMEVAL_JUDGE_MODEL", "qwen3.8-max-0902"),
                    env("STREAMEVAL_JUDGE_API_KEY") or env("DASHSCOPE_API_KEY"))


def av_judge_endpoint() -> Endpoint:
    """A-V Sync judge: rates lip opening on face crops (it never hears the audio); OpenAI Responses API, the URL is the
    API base (".../v1") or the full ".../responses" endpoint. Default: gpt-6-astra."""
    return Endpoint(env("STREAMEVAL_AV_JUDGE_BASE_URL"), env("STREAMEVAL_AV_JUDGE_MODEL", "gpt-6-astra"),
                    env("STREAMEVAL_AV_JUDGE_API_KEY"))


def asr_endpoint() -> Endpoint:
    return Endpoint(env("STREAMEVAL_ASR_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                    env("STREAMEVAL_ASR_MODEL", "qwen3-asr-flash"),
                    env("STREAMEVAL_ASR_API_KEY") or env("DASHSCOPE_API_KEY"))
