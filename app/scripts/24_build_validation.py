"""Credence-style validation suite on NASA-like covariates.

Trains the CVAE (torch, local only) and also runs the torch-free MVN generator,
emitting layered-parameter estimand tables and a naive-drift scorecard. The
committed CVAE decoder weights let the app show the deep-generative path without
shipping torch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from causal_shap.seeds import SEED_VALIDATION_SCENARIOS  # noqa: E402
from causal_shap.validation import (  # noqa: E402
    ConfounderSpec,
    MVNGenerator,
    Scenario,
    SimulationSpec,
    additive_bias,
    constant_tau,
    fit_baseline,
    fit_generator,
    generate_validation_suite,
    linear_tau,
)

OUTPUT_DIR = APP_DIR / "bundles" / "nasa_renal_clean_v3" / "stages"
N_ROWS = 4000


def _reference_data(seed: int = SEED_VALIDATION_SCENARIOS) -> pd.DataFrame:
    """A compact real-data proxy: realistic covariates, unconfounded treatment.

    Treatment is randomized so the learned p(X|Z) carries no arm imbalance —
    realism (the covariate distribution) stays separate from the truth and bias
    injected later, as in the Credence design.
    """
    rng = np.random.default_rng(seed)
    n = 3000
    age = rng.normal(size=n)
    comorbidity = 0.4 * age + rng.normal(size=n)
    hydration = rng.normal(size=n)
    z = rng.binomial(1, 0.5, size=n)
    y = age + 0.5 * comorbidity - 0.4 * hydration + rng.normal(size=n)
    return pd.DataFrame({"age": age, "comorbidity": comorbidity, "hydration": hydration, "Z": z, "Y": y})


def main() -> None:
    real = _reference_data()
    features = ["age", "comorbidity", "hydration"]
    baseline = fit_baseline(real, "Y", features)
    generator = fit_generator(MVNGenerator(), real, features, "Z")

    scenarios = [
        Scenario("homogeneous_no_confounding", SimulationSpec(baseline=baseline, tau=constant_tau(1.0))),
        Scenario(
            "heterogeneous_effect",
            SimulationSpec(baseline=baseline, tau=linear_tau(1.0, {"age": 0.8})),
        ),
        Scenario(
            "layered_confounding",
            SimulationSpec(
                baseline=baseline,
                tau=linear_tau(1.0, {"age": 0.8}),
                confounders=(
                    ConfounderSpec("frailty", outcome_strength=1.5, treatment_strength=1.2),
                    ConfounderSpec("access", outcome_strength=-0.8, treatment_strength=0.9),
                ),
            ),
        ),
        Scenario(
            "confounding_plus_measurement_bias",
            SimulationSpec(
                baseline=baseline,
                tau=linear_tau(1.0, {"age": 0.8}),
                confounders=(ConfounderSpec("frailty", 1.5, 1.2),),
                bias=additive_bias("comorbidity", 0.6),
            ),
        ),
    ]

    suite = generate_validation_suite(generator, scenarios, n=N_ROWS, seed=SEED_VALIDATION_SCENARIOS, subgroup_col="age")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suite.scorecard.to_csv(OUTPUT_DIR / "validation_scorecard.csv", index=False)

    estimand_frames = []
    for label, dataset in suite.datasets.items():
        table = dataset.estimands.copy()
        table.insert(0, "scenario", label)
        estimand_frames.append(table)
    pd.concat(estimand_frames, ignore_index=True).to_csv(OUTPUT_DIR / "validation_estimands.csv", index=False)

    _train_cvae(real, features)

    summary = {
        "status": "validation",
        "scenarios": list(suite.datasets),
        "generator": "MVN (live) + CVAE weights (committed)",
        "max_naive_drift": float(suite.scorecard["naive_drift"].abs().max()),
    }
    print(json.dumps(summary, indent=2))


def _train_cvae(real: pd.DataFrame, features: list[str]) -> None:
    """Train and persist CVAE decoder weights (torch, local build only)."""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        print("  CVAE weights skipped: torch not installed (heavy extra).")
        return

    from causal_shap.validation import CVAEGenerator

    generator = CVAEGenerator(epochs=200)
    generator.fit(real[features], real["Z"].to_numpy())
    weights_path = OUTPUT_DIR / "cvae_decoder.pt"
    generator.save_decoder(weights_path)
    print(f"  CVAE decoder weights written to {weights_path.name}")


if __name__ == "__main__":
    main()
