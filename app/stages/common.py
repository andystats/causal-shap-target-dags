"""Shared rendering helpers for the ladder stages.

Pure functions returning HTML strings, mirroring the app's original renderer
style. Glossary terms become hover popovers fed by a pre-rendered JSON artifact
(built from the canonical site/data/glossary.yml); a small fallback keeps the
app working before that artifact is built.
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
GLOSSARY_PATH = APP_DIR / "assets" / "glossary.json"

_FALLBACK_GLOSSARY = {
    "collider": "A node caused by two or more variables; conditioning on it opens a spurious path.",
    "proxy": "A variable predictive of the outcome but with no causal effect on it.",
    "CPDAG": "Completed Partially Directed Acyclic Graph: an equivalence class of DAGs.",
    "PBI": "Proximity Bias Index: how much attribution mass concentrates near the outcome.",
    "ATE": "Average Treatment Effect across the whole population.",
    "ATT": "Average Treatment effect on the Treated.",
    "estimand": "The causal quantity you intend to estimate, defined before any model.",
}


@lru_cache(maxsize=1)
def glossary() -> dict[str, str]:
    if GLOSSARY_PATH.exists():
        return json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    return dict(_FALLBACK_GLOSSARY)


@lru_cache(maxsize=64)
def image_uri(path_text: str) -> str:
    path = Path(path_text)
    mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@lru_cache(maxsize=32)
def csv_data(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text)


def image_html(path: Path, alt: str) -> str:
    return f'<img class="figure" src="{image_uri(str(path))}" alt="{alt}">'


def term(word: str, label: str | None = None) -> str:
    definition = glossary().get(word, "")
    text = label or word
    if not definition:
        return text
    return f'<span class="term" title="{definition}">{text}</span>'


def metric_cards(items: list[tuple[str, str, str | None]]) -> str:
    cards = []
    for label, value, detail in items:
        detail_html = f'<div class="metric-detail">{detail}</div>' if detail else ""
        cards.append(
            f'<div class="metric"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>{detail_html}</div>'
        )
    return f'<div class="metrics">{"".join(cards)}</div>'


def table_html(frame: pd.DataFrame, *, index: bool = False) -> str:
    return f'<div class="table-wrap">{frame.to_html(index=index, classes="data-table", border=0, float_format=lambda v: f"{v:.3f}")}</div>'


def callout(text: str, kind: str = "callout") -> str:
    return f'<div class="{kind}">{text}</div>'
