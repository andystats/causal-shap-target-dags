from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import pandas as pd

from causal_shap.validation import (
    ConfounderSpec,
    MVNGenerator,
    Scenario,
    SimulationSpec,
    additive_bias,
    constant_tau,
    fit_generator,
    generate_validation_suite,
    linear_tau,
    simulate,
    true_estimands,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _real_data(seed: int = 0, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z = rng.binomial(1, 0.5, size=n)
    y = x1 + 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "Z": z, "Y": y})


def _baseline_from(df: pd.DataFrame):
    def baseline(frame: pd.DataFrame) -> np.ndarray:
        return frame["x1"].to_numpy() + 0.5 * frame["x2"].to_numpy()

    return baseline


class EstimandTests(unittest.TestCase):
    def test_true_estimands_recover_known_effects(self) -> None:
        n = 5000
        rng = np.random.default_rng(1)
        y0 = rng.normal(size=n)
        tau = 2.0 + 0.0 * y0
        y1 = y0 + tau
        z = rng.binomial(1, 0.5, size=n)
        table = true_estimands(y0, y1, z).set_index("estimand")
        self.assertAlmostEqual(table.loc["ATE", "value"], 2.0, places=6)
        self.assertAlmostEqual(table.loc["ATT", "value"], 2.0, places=6)

    def test_att_differs_from_ate_under_effect_heterogeneity(self) -> None:
        n = 20000
        rng = np.random.default_rng(2)
        x = rng.normal(size=n)
        y0 = rng.normal(size=n)
        y1 = y0 + (1.0 + x)  # tau correlates with x
        # treated units have higher x -> larger effect -> ATT > ATE
        z = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(int)
        table = true_estimands(y0, y1, z).set_index("estimand")
        self.assertGreater(table.loc["ATT", "value"], table.loc["ATE", "value"])


class SimulationTests(unittest.TestCase):
    def test_zero_confounding_naive_matches_true_ate(self) -> None:
        real = _real_data()
        spec = SimulationSpec(baseline=_baseline_from(real), tau=constant_tau(1.5), confounders=())
        generator = fit_generator(MVNGenerator(), real, ["x1", "x2"], "Z")
        dataset = simulate(spec, generator, n=8000, seed=20260730)
        true_ate = dataset.estimands.set_index("estimand").loc["ATE", "value"]
        self.assertAlmostEqual(true_ate, 1.5, places=2)
        # No confounding -> naive difference in means is close to the truth.
        self.assertLess(abs(dataset.naive_ate - true_ate), 0.15)

    def test_confounding_biases_naive_but_not_truth(self) -> None:
        real = _real_data()
        spec = SimulationSpec(
            baseline=_baseline_from(real),
            tau=constant_tau(1.0),
            confounders=(ConfounderSpec("U", outcome_strength=2.0, treatment_strength=2.0),),
        )
        generator = fit_generator(MVNGenerator(), real, ["x1", "x2"], "Z")
        dataset = simulate(spec, generator, n=8000, seed=20260730)
        true_ate = dataset.estimands.set_index("estimand").loc["ATE", "value"]
        self.assertAlmostEqual(true_ate, 1.0, places=2)
        # Positive confounding inflates the naive estimate well above the truth.
        self.assertGreater(dataset.naive_ate - true_ate, 0.3)

    def test_bias_shifts_observed_only(self) -> None:
        real = _real_data()
        common = dict(baseline=_baseline_from(real), tau=constant_tau(1.0), confounders=())
        generator = fit_generator(MVNGenerator(), real, ["x1", "x2"], "Z")
        clean = simulate(SimulationSpec(**common), generator, n=8000, seed=5)
        biased = simulate(
            SimulationSpec(**common, bias=additive_bias("x1", 3.0)), generator, n=8000, seed=5
        )
        # Truth is identical; only the observed-data naive estimate moves.
        self.assertAlmostEqual(
            clean.estimands.set_index("estimand").loc["ATE", "value"],
            biased.estimands.set_index("estimand").loc["ATE", "value"],
        )
        self.assertNotAlmostEqual(clean.naive_ate, biased.naive_ate, places=2)

    def test_suite_scorecard_has_one_row_per_scenario(self) -> None:
        real = _real_data()
        generator = fit_generator(MVNGenerator(), real, ["x1", "x2"], "Z")
        scenarios = [
            Scenario("no_confounding", SimulationSpec(baseline=_baseline_from(real), tau=constant_tau(1.0))),
            Scenario(
                "strong_confounding",
                SimulationSpec(
                    baseline=_baseline_from(real),
                    tau=linear_tau(1.0, {"x1": 0.5}),
                    confounders=(ConfounderSpec("U", 2.0, 2.0),),
                ),
            ),
        ]
        suite = generate_validation_suite(generator, scenarios, n=4000, seed=20260801, subgroup_col="x1")
        self.assertEqual(list(suite.scorecard["scenario"]), ["no_confounding", "strong_confounding"])
        self.assertIn("naive_drift", suite.scorecard.columns)
        # The confounded scenario drifts further than the clean one.
        drift = suite.scorecard.set_index("scenario")["naive_drift"].abs()
        self.assertGreater(drift["strong_confounding"], drift["no_confounding"])


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed (build-only dependency)")
class CVAEGeneratorTests(unittest.TestCase):
    def test_cvae_samples_have_expected_shape(self) -> None:
        from causal_shap.validation import CVAEGenerator

        real = _real_data(n=1500)
        generator = fit_generator(CVAEGenerator(epochs=20), real, ["x1", "x2"], "Z")
        samples = generator.sample(np.array([0, 1, 0, 1]), seed=20260731)
        self.assertEqual(samples.shape, (4, 2))
        self.assertEqual(list(samples.columns), ["x1", "x2"])


if __name__ == "__main__":
    unittest.main()
