"""StreamEval: benchmark, streaming-session harness, judge and scoring for interactive streaming video generation."""
from __future__ import annotations

import os
from pathlib import Path

__version__ = "1.0.0"
DATASET_VERSION = "streameval_v1.0"
# The repository checkout: benchmark/, evaluation/, models/ and examples/ live next to the package and are
# not installed with it, so the package is used from a checkout (pip install -e .). STREAMEVAL_ROOT names the
# checkout when the package itself is installed elsewhere.
REPO = Path(os.environ.get("STREAMEVAL_ROOT") or Path(__file__).resolve().parents[1]).expanduser().resolve()
