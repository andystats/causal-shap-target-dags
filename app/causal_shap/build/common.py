"""Shared paths and boilerplate for the build stages.

This is the single home for the ``APP_DIR``/``PROJECT_DIR``/``BUNDLES_DIR`` path
block that every former numbered script recomputed, plus the JSON status print
and the one-time Agg matplotlib backend selection (figures render headless).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# causal_shap/build/common.py -> parents[2] is the app/ directory.
APP_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = APP_DIR.parent
BUNDLES_DIR = APP_DIR / "bundles"


def print_status(payload: object) -> None:
    """Emit a stage's JSON summary exactly as the numbered scripts did."""
    print(json.dumps(payload, indent=2))
