"""Stream-Arena: benchmark, streaming-session harness, judge and scoring for interactive streaming video generation."""
from __future__ import annotations

import os
from pathlib import Path

__version__ = "1.0.0"
DATASET_VERSION = "stream_arena_v1.0"
# The repository checkout: benchmark/, evaluation/, leaderboard/, models/ and examples/ live next to the package and are
# not installed with it, so the package is used from a checkout (pip install -e .). STREAM_ARENA_ROOT names the
# checkout when the package itself is installed elsewhere.
REPO = Path(os.environ.get("STREAM_ARENA_ROOT") or Path(__file__).resolve().parents[1]).expanduser().resolve()
