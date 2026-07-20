"""Forward simulation and scenario suites for Credence-style validation.

The layered spec is simulated forward (no constrained optimization — that is out
of scope) so every estimand is known. A scenario suite runs the whole ladder on
data where the answers are known and scorecards how far a naive analyst drifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..structural_value import sigmoid
from .estimands import naive_difference_in_means, true_estimands
from .generators import CovariateGenerator
from .spec import SimulationSpec


@dataclass(frozen=True)
class SimulatedDataset:
    observed: pd.DataFrame  # covariates + treatment + observed outcome (no latents)
    potential_y0: np.ndarray
    potential_y1: np.ndarray
    treatment: np.ndarray
    propensity: np.ndarray
    estimands: pd.DataFrame
    naive_ate: float


@dataclass(frozen=True)
class Scenario:
    label: str
    spec: SimulationSpec


@dataclass(frozen=True)
class ValidationSuite:
    datasets: dict[str, SimulatedDataset]
    scorecard: pd.DataFrame


def reference_covariates(seed: int) -> pd.DataFrame:
    """A compact real-data proxy: realistic covariates, unconfounded treatment.

    Shared by the app's live validation explorer and the validation build script
    so both learn p(X|Z) from the same synthetic reference. Treatment is
    randomized, keeping realism separate from the injected truth and bias.
    """
    rng = np.random.default_rng(seed)
    n = 3000
    age = rng.normal(size=n)
    comorbidity = 0.4 * age + rng.normal(size=n)
    hydration = rng.normal(size=n)
    z = rng.binomial(1, 0.5, size=n)
    y = age + 0.5 * comorbidity - 0.4 * hydration + rng.normal(size=n)
    return pd.DataFrame({"age": age, "comorbidity": comorbidity, "hydration": hydration, "Z": z, "Y": y})


def fit_generator(
    generator: CovariateGenerator,
    real_df: pd.DataFrame,
    feature_cols: Sequence[str],
    treatment_col: str,
) -> CovariateGenerator:
    """Learn p(X | Z) from real data before it is used to simulate."""
    return generator.fit(real_df[list(feature_cols)], real_df[treatment_col].to_numpy())


def simulate(
    spec: SimulationSpec,
    generator: CovariateGenerator,
    n: int,
    *,
    seed: int,
    subgroup_col: str | None = None,
    treatment_name: str = "Z",
    outcome_name: str = "Y",
) -> SimulatedDataset:
    """Draw one synthetic dataset with clean potential outcomes from the spec."""
    rng = np.random.default_rng(seed)
    latent = {c.name: rng.standard_normal(n) for c in spec.confounders}

    logits = np.full(n, np.log(spec.base_treatment_rate / (1 - spec.base_treatment_rate)))
    for confounder in spec.confounders:
        logits += confounder.treatment_strength * latent[confounder.name]
    propensity = np.clip(sigmoid(logits), spec.propensity_clip, 1 - spec.propensity_clip)
    treatment = rng.binomial(1, propensity).astype(int)

    covariates = generator.sample(treatment, seed=seed + 1)
    for confounder in spec.confounders:
        if confounder.observed:
            covariates[confounder.name] = latent[confounder.name]

    baseline = spec.baseline(covariates)
    confounding_shift = np.zeros(n)
    for confounder in spec.confounders:
        confounding_shift += confounder.outcome_strength * latent[confounder.name]
    noise = rng.normal(0.0, spec.noise_sd, n)

    y0 = baseline + confounding_shift + noise
    y1 = y0 + spec.tau(covariates)
    y_obs = np.where(treatment == 1, y1, y0)
    if spec.bias is not None:
        y_obs = y_obs + spec.bias(covariates, treatment)

    subgroups = None
    if subgroup_col is not None:
        median = covariates[subgroup_col].median()
        subgroups = {f"{subgroup_col}>median": covariates[subgroup_col].to_numpy() > median}

    observed = covariates.copy()
    observed[treatment_name] = treatment
    observed[outcome_name] = y_obs
    return SimulatedDataset(
        observed=observed,
        potential_y0=y0,
        potential_y1=y1,
        treatment=treatment,
        propensity=propensity,
        estimands=true_estimands(y0, y1, treatment, subgroups),
        naive_ate=naive_difference_in_means(y_obs, treatment),
    )


def generate_validation_suite(
    generator: CovariateGenerator,
    scenarios: Sequence[Scenario],
    n: int,
    *,
    seed: int,
    subgroup_col: str | None = None,
) -> ValidationSuite:
    """Run each scenario and scorecard true ATE vs the naive difference in means."""
    datasets: dict[str, SimulatedDataset] = {}
    rows: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios):
        dataset = simulate(scenario.spec, generator, n, seed=seed + index, subgroup_col=subgroup_col)
        datasets[scenario.label] = dataset
        true_ate = float(dataset.estimands.loc[dataset.estimands["estimand"] == "ATE", "value"].iloc[0])
        rows.append(
            {
                "scenario": scenario.label,
                "true_ate": true_ate,
                "naive_ate": dataset.naive_ate,
                "naive_drift": dataset.naive_ate - true_ate,
            }
        )
    return ValidationSuite(datasets=datasets, scorecard=pd.DataFrame(rows))
