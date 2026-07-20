"""Ground-truth estimands from potential outcomes.

The layered spec produces clean potential outcomes Y(0), Y(1), so several
estimands are known exactly (up to Monte Carlo error) for the same synthetic
dataset — the "several parameters beyond a single ATE" deliverable.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _summarize(effect: np.ndarray, label: str) -> dict[str, object]:
    n = len(effect)
    mean = float(np.mean(effect)) if n else float("nan")
    se = float(np.std(effect, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"estimand": label, "value": mean, "mc_std_error": se, "n": n}


def true_estimands(
    y0: np.ndarray,
    y1: np.ndarray,
    treatment: np.ndarray,
    subgroups: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Return a tidy table of ATE, ATT, ATC, and any subgroup CATEs with MC SEs."""
    y0 = np.asarray(y0, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    treatment = np.asarray(treatment)
    effect = y1 - y0

    rows = [
        _summarize(effect, "ATE"),
        _summarize(effect[treatment == 1], "ATT"),
        _summarize(effect[treatment == 0], "ATC"),
    ]
    for name, mask in (subgroups or {}).items():
        rows.append(_summarize(effect[np.asarray(mask, dtype=bool)], f"CATE[{name}]"))
    return pd.DataFrame(rows)


def naive_difference_in_means(y_obs: np.ndarray, treatment: np.ndarray) -> float:
    """The confounded estimate a naive analyst would report."""
    y_obs = np.asarray(y_obs, dtype=float)
    treatment = np.asarray(treatment)
    return float(y_obs[treatment == 1].mean() - y_obs[treatment == 0].mean())
