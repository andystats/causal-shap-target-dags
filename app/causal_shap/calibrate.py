"""Fit linear-logistic structural equations to data on a given topology.

Discovery yields a graph; the structural engines need a parameterized
``LinearLogisticSCM``. This module supplies the missing fitter: OLS for
continuous nodes, logistic regression for binary ones, parents taken from the
graph as handed in.

The result is graded **replay**: it supports abduct-then-simulate, which is all
the intervention-selection machinery in ``policy`` needs, and it supports it
exactly — abduction absorbs each root's location into the recovered exogenous
draws, so replaying them reproduces the observed continuous columns to
numerical precision. Fresh generation from scratch is NOT supported, because
``LinearLogisticSCM`` ignores the intercept of parentless continuous nodes
(``structural_value.py:151``); a freshly simulated root would sit at zero
regardless of what the data said. Rather than half-support generation, v1 says
so out loud and records each root's observed mean in the diagnostics.

Binary columns must be coded 0/1. Other two-value encodings such as {-1, 1}
are rejected rather than guessed at: ``recover_exogenous`` treats only the
literal value 1 as positive, so any other coding would calibrate cleanly and
then replay wrongly, which is the worst possible failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import networkx as nx
import numpy as np
import pandas as pd

from .structural_value import LinearLogisticSCM, NodeSpec

GRADE_REPLAY = "replay"

ASSUMPTIONS = (
    "linear structural equations with additive Gaussian noise (continuous nodes)",
    "logistic structural equations (binary nodes)",
    "the supplied topology is correct and causally sufficient",
    "replay-grade: abduct-then-simulate only; fresh generation is unsupported",
)

_MIN_NOISE_SD = 1e-9


@dataclass(frozen=True)
class CalibratedSCM:
    """A fitted structural model plus the honesty metadata that must travel with it."""

    scm: LinearLogisticSCM
    diagnostics: pd.DataFrame
    grade: str = GRADE_REPLAY
    assumptions: tuple[str, ...] = ASSUMPTIONS
    n_undirected_pairs: int = 0
    warnings: tuple[str, ...] = ()
    root_means: Mapping[str, float] = field(default_factory=dict)


def fit_linear_logistic_scm(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    *,
    seed: int,
    n_undirected_pairs: int = 0,
) -> CalibratedSCM:
    """Fit one ``NodeSpec`` per graph node from observed data.

    ``seed`` is recorded for provenance; the fit itself is deterministic
    (least squares, and a fixed-seed logistic solver).
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Calibration requires an acyclic graph")
    missing = sorted(set(graph.nodes) - set(data.columns))
    if missing:
        raise ValueError(f"Data are missing graph nodes: {missing[:5]}")

    specs: list[NodeSpec] = []
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    root_means: dict[str, float] = {}

    for node in nx.topological_sort(graph):
        observed = data[node].to_numpy(dtype=float)
        parents = tuple(sorted(graph.predecessors(node)))
        kind = _node_kind(node, observed)

        if kind == "binary" and not parents:
            probability = float(np.clip(observed.mean(), 1e-9, 1 - 1e-9))
            specs.append(NodeSpec(node, "binary", root_probability=probability))
            rows.append(_row(node, kind, 0, "rate", probability))
        elif kind == "binary":
            intercept, coefficients, auc = _fit_logistic(data, node, parents, seed)
            specs.append(
                NodeSpec(node, "binary", parents=parents, coefficients=coefficients,
                         intercept=intercept)
            )
            rows.append(_row(node, kind, len(parents), "auc", auc))
        elif not parents:
            noise_sd = float(observed.std())
            if noise_sd < _MIN_NOISE_SD:
                noise_sd = _MIN_NOISE_SD
                warnings.append(f"{node}: constant column; noise floored")
            root_means[node] = float(observed.mean())
            specs.append(NodeSpec(node, "continuous", noise_sd=noise_sd))
            rows.append(_row(node, kind, 0, "sd", noise_sd))
        else:
            intercept, coefficients, r2, resid_sd = _fit_ols(data, node, parents)
            if resid_sd < _MIN_NOISE_SD:
                resid_sd = _MIN_NOISE_SD
                warnings.append(f"{node}: deterministic in its parents; noise floored")
            specs.append(
                NodeSpec(node, "continuous", parents=parents,
                         coefficients=coefficients, intercept=intercept,
                         noise_sd=resid_sd)
            )
            rows.append(_row(node, kind, len(parents), "r2", r2))

    return CalibratedSCM(
        scm=LinearLogisticSCM(specs),
        diagnostics=pd.DataFrame(rows),
        n_undirected_pairs=n_undirected_pairs,
        warnings=tuple(warnings),
        root_means=root_means,
    )


def _node_kind(node: str, observed: np.ndarray) -> str:
    values = set(np.unique(observed))
    if values <= {0.0, 1.0}:
        return "binary"
    if len(values) == 2:
        raise ValueError(
            f"{node}: two-valued column is not coded 0/1 ({sorted(values)}); "
            "recode it before calibrating, or replay will be silently wrong"
        )
    # One non-{0,1} value is a degenerate continuous column, floored later.
    return "continuous"


def _fit_ols(
    data: pd.DataFrame, node: str, parents: tuple[str, ...]
) -> tuple[float, tuple[float, ...], float, float]:
    design = np.column_stack(
        [np.ones(len(data))] + [data[parent].to_numpy(dtype=float) for parent in parents]
    )
    target = data[node].to_numpy(dtype=float)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ solution
    total_variance = float(target.var())
    r2 = 1.0 - float(residuals.var()) / total_variance if total_variance > 0 else 0.0
    return float(solution[0]), tuple(float(c) for c in solution[1:]), r2, float(residuals.std())


def _fit_logistic(
    data: pd.DataFrame, node: str, parents: tuple[str, ...], seed: int
) -> tuple[float, tuple[float, ...], float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError as error:  # calibration of binary nodes needs sklearn
        raise ImportError(
            "Calibrating binary nodes requires scikit-learn (workbench extra)"
        ) from error

    design = data[list(parents)].to_numpy(dtype=float)
    target = data[node].to_numpy(dtype=float)
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(design, target)
    predicted = model.predict_proba(design)[:, 1]
    auc = float(roc_auc_score(target, predicted)) if len(set(target)) == 2 else float("nan")
    return float(model.intercept_[0]), tuple(float(c) for c in model.coef_[0]), auc


def _row(node: str, kind: str, n_parents: int, stat: str, value: float) -> dict[str, object]:
    return {
        "node": node,
        "kind": kind,
        "n_parents": n_parents,
        "fit_stat": stat,
        "fit_value": value,
    }
