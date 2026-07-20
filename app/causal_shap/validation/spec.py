"""Layered simulation specification for Credence-style validation.

Adapted from the author's Instats workshop (Module 5, "Causally Credible"),
which itself follows Parikh et al. (ICML 2022). Where the workshop app exposed
only a single scalar ATE and a single confounding knob, ``SimulationSpec``
layers several parameters at once: a fitted baseline h(X), a heterogeneous
treatment effect tau(X), a vector of confounders each with its own outcome and
treatment strength, and an explicit bias function g(X, Z) that perturbs the
observed outcome while leaving the potential-outcome truth clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from ..seeds import SEED_VALIDATION_TRUTH

# A baseline maps covariates to E[Y(0) | X]; tau maps covariates to the
# individual treatment effect; bias maps (covariates, treatment) to an additive
# perturbation of the *observed* outcome only.
BaselineFn = Callable[[pd.DataFrame], np.ndarray]
TauFn = Callable[[pd.DataFrame], np.ndarray]
BiasFn = Callable[[pd.DataFrame, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ConfounderSpec:
    """A latent driver that shifts both treatment assignment and the outcome."""

    name: str
    outcome_strength: float  # gamma_j: effect on Y(0)
    treatment_strength: float  # rho_j: effect on the treatment logit
    observed: bool = False  # if True, exposed to estimators as a covariate

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ConfounderSpec requires a name")


@dataclass(frozen=True)
class SimulationSpec:
    baseline: BaselineFn
    tau: TauFn
    confounders: tuple[ConfounderSpec, ...] = ()
    bias: BiasFn | None = None
    noise_sd: float = 1.0
    propensity_clip: float = 0.02  # positivity floor/ceiling on the propensity
    base_treatment_rate: float = 0.5

    def __post_init__(self) -> None:
        if self.noise_sd <= 0:
            raise ValueError("noise_sd must be positive")
        if not 0.0 <= self.propensity_clip < 0.5:
            raise ValueError("propensity_clip must be in [0, 0.5)")
        names = [c.name for c in self.confounders]
        if len(names) != len(set(names)):
            raise ValueError("Confounder names must be unique")


# ---------------------------------------------------------------------------
# Baseline / tau / bias factories (the smell fix: baseline is a real model).
# ---------------------------------------------------------------------------
def fit_baseline(
    real_df: pd.DataFrame,
    outcome: str,
    feature_cols: Sequence[str],
    *,
    seed: int = SEED_VALIDATION_TRUTH,
) -> BaselineFn:
    """Fit E[Y | X] on real data so generated X actually drives the outcome."""
    features = list(feature_cols)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
    model.fit(real_df[features], real_df[outcome])

    def baseline(frame: pd.DataFrame) -> np.ndarray:
        return model.predict(frame[features])

    return baseline


def constant_tau(value: float) -> TauFn:
    """Homogeneous effect: tau(X) = value (recovers a single-ATE design)."""

    def tau(frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), float(value))

    return tau


def linear_tau(intercept: float, coefficients: dict[str, float]) -> TauFn:
    """Heterogeneous effect linear in named covariates."""

    def tau(frame: pd.DataFrame) -> np.ndarray:
        effect = np.full(len(frame), float(intercept))
        for column, weight in coefficients.items():
            effect += weight * frame[column].to_numpy(dtype=float)
        return effect

    return tau


def interaction_tau(intercept: float, left: str, right: str, weight: float) -> TauFn:
    """Effect modified by a two-covariate interaction."""

    def tau(frame: pd.DataFrame) -> np.ndarray:
        product = frame[left].to_numpy(dtype=float) * frame[right].to_numpy(dtype=float)
        return np.full(len(frame), float(intercept)) + weight * product

    return tau


def additive_bias(column: str, weight: float) -> BiasFn:
    """Observed-only measurement bias proportional to a covariate under treatment."""

    def bias(frame: pd.DataFrame, treatment: np.ndarray) -> np.ndarray:
        return weight * treatment * frame[column].to_numpy(dtype=float)

    return bias
