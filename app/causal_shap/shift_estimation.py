"""Targeted estimation of one shift effect per surviving lever.

The pricing engine's SCM arm trusts every calibrated equation at once; this
module is the other pole of Marschak's Maxim: survey with the structural model,
then estimate one functional per lever that survives the screen. Each ±shift
is read as a modified treatment policy d(a) = a + δ (Haneuse & Rotnitzky 2013),
and its population effect E[Y(A+δ)] − E[Y] is estimated double-robustly with
the AIPW / one-step estimator for stochastic shift interventions (Díaz Muñoz &
van der Laan 2012; Díaz et al. 2023), with cross-fitted nuisances so flexible
learners do not leak overfit into the estimate (Chernozhukov et al. 2018).

Two design commitments keep the arm honest. First, the adjustment set is
DAG-derived and minimal in ambition: the lever's parents under the current
graph, which block every backdoor path and — because a parent cannot be a
descendant — exclude every post-lever variable by construction, the guard a
non-root lever needs against post-lever confounding. Second, feasibility is
reported, never assumed: the Haneuse–Rotnitzky framework requires the shifted
treatment to be an option units could actually have received, so every
estimate carries a support check on A+δ and density-ratio weight diagnostics,
with the verdict on screen rather than buried.

Both poles still trust the DAG; what this arm stops trusting is every other
equation in the system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .seeds import SEED_SHIFT_ESTIMATION

LEARNERS = ("gbm", "linear")

FEASIBILITY_OK = "ok"
FEASIBILITY_CAUTION = "caution"

_PROBABILITY_CLIP = 1e-3
_SUPPORT_SHARE_LIMIT = 0.05
# A shift of one conditional SD costs ESS share exp(-1) ~ 0.37 by construction,
# so the floor sits well below that: it flags shifts of ~2+ conditional SDs,
# where the ratio arm of the estimator is running on a handful of rows.
_ESS_SHARE_FLOOR = 0.1
# The 99th-percentile weight, not the maximum: one extreme row in thousands is
# a tail event, while a heavy p99 means the whole tail is carrying the estimate.
_P99_WEIGHT_LIMIT = 10.0


def adjustment_for_lever(graph: nx.DiGraph, lever: str, outcome: str) -> tuple[str, ...]:
    """The lever's parents under the current graph, sorted.

    Parents block every backdoor path from lever to outcome, and no parent can
    be a descendant of the lever, so post-lever variables (mediators, proxies
    of the outcome) are excluded by construction rather than by vigilance.
    Validity is conditional on the graph being right — the same trust both
    estimation poles place in the DAG.
    """
    if lever not in graph:
        raise ValueError(f"Lever {lever} is not a node of the graph")
    if outcome not in graph:
        raise ValueError(f"Outcome {outcome} is not a node of the graph")
    return tuple(sorted(graph.predecessors(lever)))


@dataclass(frozen=True)
class FeasibilityCheck:
    """Haneuse–Rotnitzky-style diagnostics for one modified treatment policy."""

    n: int
    share_shifted_outside_support: float
    ess: float
    ess_share: float
    max_weight: float
    p99_weight: float
    verdict: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ShiftEstimate:
    """One lever's shift effect, estimated from data rather than simulated."""

    lever: str
    delta: float
    outcome: str
    adjustment: tuple[str, ...]
    benefit: float
    benefit_se: float
    ci_low: float
    ci_high: float
    theta_shift: float
    theta_natural: float
    direction: str
    n_folds: int
    learner: str
    seed: int
    feasibility: FeasibilityCheck

    @property
    def label(self) -> str:
        return f"{self.lever}{self.delta:+g}"


def estimate_shift_effect(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    lever: str,
    delta: float,
    outcome: str,
    *,
    direction: str = "increase",
    learner: str = "gbm",
    n_folds: int = 5,
    seed: int = SEED_SHIFT_ESTIMATION,
) -> ShiftEstimate:
    """Cross-fitted AIPW for the modified treatment policy d(a) = a + δ.

    The one-step estimator of E[Y(A+δ)] plugs cross-fitted nuisances into the
    efficient influence function for a shift intervention:

        θ̂_δ = mean[ r̂(A,W) · (Y − m̂(A,W)) + m̂(A+δ, W) ]

    where m̂ is the outcome regression E[Y|A,W] and r̂(a,w) estimates the
    density ratio g(a−δ|w)/g(a|w), learned here by the classification trick:
    a classifier separating (A+δ, W) from (A, W) recovers the ratio as its
    odds. The reported benefit is θ̂_δ − Ȳ, negated when the beneficial
    direction is "decrease", with a standard error from the empirical variance
    of the combined influence function.
    """
    if learner not in LEARNERS:
        raise ValueError(f"Unknown learner: {learner!r}")
    if delta == 0.0:
        raise ValueError("A zero shift has nothing to estimate")

    adjustment = adjustment_for_lever(graph, lever, outcome)
    columns = list(dict.fromkeys([lever, *adjustment, outcome]))
    frame = data[columns].dropna()
    n = len(frame)
    if n < 20:
        raise ValueError(f"Only {n} complete rows; too few for cross-fitted estimation")

    treatment = frame[lever].to_numpy(dtype=float)
    if set(np.unique(treatment)) <= {0.0, 1.0}:
        raise ValueError(
            f"{lever} is binary: an additive shift leaves its support, so the "
            "modified-treatment-policy estimand is not defined for it"
        )

    y = frame[outcome].to_numpy(dtype=float)
    binary_outcome = set(np.unique(y)) <= {0.0, 1.0}
    covariates = frame[list(adjustment)].to_numpy(dtype=float)  # (n, 0) for a root lever
    design_natural = np.column_stack([treatment, covariates])
    design_shifted = np.column_stack([treatment + delta, covariates])

    n_folds = max(2, min(n_folds, n // 10))
    rng = np.random.default_rng(seed)
    fold_of = rng.permuted(np.arange(n) % n_folds)

    m_natural = np.empty(n)
    m_shifted = np.empty(n)
    ratio = np.empty(n)
    for fold in range(n_folds):
        train, test = fold_of != fold, fold_of == fold
        regressor = _outcome_learner(learner, binary_outcome, seed)
        regressor.fit(design_natural[train], y[train])
        m_natural[test] = _predict_mean(regressor, design_natural[test], binary_outcome)
        m_shifted[test] = _predict_mean(regressor, design_shifted[test], binary_outcome)

        # Density-ratio classification trick: label the shifted copies 1, the
        # natural rows 0; with equal class sizes the fitted odds at (a, w)
        # estimate g(a−δ|w)/g(a|w), exactly the AIPW weight.
        classifier = _ratio_learner(learner, seed)
        stacked = np.vstack([design_natural[train], design_shifted[train]])
        labels = np.concatenate([np.zeros(train.sum()), np.ones(train.sum())])
        classifier.fit(stacked, labels)
        p_shifted = np.clip(
            classifier.predict_proba(design_natural[test])[:, 1],
            _PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP,
        )
        ratio[test] = p_shifted / (1.0 - p_shifted)

    theta_shift = float(np.mean(ratio * (y - m_natural) + m_shifted))
    theta_natural = float(np.mean(y))
    influence = (ratio * (y - m_natural) + m_shifted - theta_shift) - (y - theta_natural)
    contrast = theta_shift - theta_natural
    se = float(np.std(influence, ddof=1) / math.sqrt(n))

    sign = -1.0 if direction == "decrease" else 1.0
    benefit = sign * contrast
    half_width = 1.96 * se

    return ShiftEstimate(
        lever=lever,
        delta=delta,
        outcome=outcome,
        adjustment=adjustment,
        benefit=benefit,
        benefit_se=se,
        ci_low=benefit - half_width,
        ci_high=benefit + half_width,
        theta_shift=theta_shift,
        theta_natural=theta_natural,
        direction=direction,
        n_folds=n_folds,
        learner=learner,
        seed=seed,
        feasibility=_feasibility(lever, delta, treatment, ratio),
    )


def estimates_frame(estimates: Sequence[ShiftEstimate]) -> pd.DataFrame:
    """The targeted estimates as a table, feasibility verdict included."""
    return pd.DataFrame(
        [
            {
                "action": item.label,
                "adjustment": ", ".join(item.adjustment) or "(none: root lever)",
                "dr_benefit": item.benefit,
                "dr_se": item.benefit_se,
                "ci95": f"[{item.ci_low:.3g}, {item.ci_high:.3g}]",
                "feasibility": item.feasibility.verdict,
                "notes": "; ".join(item.feasibility.notes),
            }
            for item in estimates
        ]
    )


def _feasibility(
    lever: str, delta: float, treatment: np.ndarray, ratio: np.ndarray
) -> FeasibilityCheck:
    """Support and weight diagnostics; every triggered rule becomes a note."""
    shifted = treatment + delta
    outside = (shifted < treatment.min()) | (shifted > treatment.max())
    share_outside = float(np.mean(outside))
    ess = float(np.sum(ratio) ** 2 / np.sum(ratio**2))
    ess_share = ess / len(ratio)
    max_weight = float(np.max(ratio))
    p99_weight = float(np.quantile(ratio, 0.99))

    notes: list[str] = []
    if share_outside > _SUPPORT_SHARE_LIMIT:
        notes.append(
            f"{share_outside:.0%} of units would be shifted outside the observed "
            f"range of {lever}: the modified policy assigns treatment values no "
            "unit was seen to receive (Haneuse & Rotnitzky feasibility)"
        )
    if ess_share < _ESS_SHARE_FLOOR:
        notes.append(
            f"density-ratio weights concentrate: effective sample size {ess:.0f} "
            f"of {len(ratio)} rows"
        )
    if p99_weight > _P99_WEIGHT_LIMIT:
        notes.append(f"the top 1% of density-ratio weights exceed {p99_weight:.1f}")
    return FeasibilityCheck(
        n=len(ratio),
        share_shifted_outside_support=share_outside,
        ess=ess,
        ess_share=ess_share,
        max_weight=max_weight,
        p99_weight=p99_weight,
        verdict=FEASIBILITY_CAUTION if notes else FEASIBILITY_OK,
        notes=tuple(notes),
    )


def _outcome_learner(learner: str, binary_outcome: bool, seed: int):
    if learner == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        cls = GradientBoostingClassifier if binary_outcome else GradientBoostingRegressor
        return cls(n_estimators=100, max_depth=3, random_state=seed)
    from sklearn.linear_model import LinearRegression, LogisticRegression

    return LogisticRegression(max_iter=1000) if binary_outcome else LinearRegression()


def _ratio_learner(learner: str, seed: int):
    if learner == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000)


def _predict_mean(model, design: np.ndarray, binary_outcome: bool) -> np.ndarray:
    return model.predict_proba(design)[:, 1] if binary_outcome else model.predict(design)
