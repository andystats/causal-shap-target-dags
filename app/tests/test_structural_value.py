from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from causal_shap.structural_value import (
    ExogenousDraws,
    LinearLogisticSCM,
    NodeSpec,
    compute_structural_asymmetric_shap,
)


class StructuralValueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scm = LinearLogisticSCM(
            [
                NodeSpec("a", "continuous", noise_sd=1.0),
                NodeSpec("m", "continuous", parents=("a",), coefficients=(2.0,), noise_sd=1.0),
                NodeSpec("y", "continuous", parents=("m",), coefficients=(3.0,), noise_sd=1.0),
            ]
        )
        self.exogenous = ExogenousDraws(
            {
                "a": np.array([0.0, 1.0]),
                "m": np.array([0.0, 0.0]),
                "y": np.array([0.0, 0.0]),
            }
        )

    def test_intervention_propagates_to_descendants(self) -> None:
        simulated = self.scm.simulate(self.exogenous, {"a": 2.0})
        np.testing.assert_allclose(simulated["m"], [4.0, 4.0])
        np.testing.assert_allclose(simulated["y"], [12.0, 12.0])

    def test_downstream_intervention_blocks_upstream_propagation(self) -> None:
        simulated = self.scm.simulate(self.exogenous, {"a": 2.0, "m": 5.0})
        np.testing.assert_allclose(simulated["m"], [5.0, 5.0])
        np.testing.assert_allclose(simulated["y"], [15.0, 15.0])

    def test_structural_shap_is_efficient(self) -> None:
        evaluation = pd.DataFrame({"a": [2.0], "m": [4.0]})

        def predict_margin(matrix: np.ndarray) -> np.ndarray:
            return matrix[:, 0] + matrix[:, 1]

        result = compute_structural_asymmetric_shap(
            predict_margin=predict_margin,
            scm=self.scm,
            evaluation=evaluation,
            background_exogenous=self.exogenous,
            feature_names=["a", "m"],
            feature_edges=[("a", "m")],
            n_permutations=8,
            seed=7,
        )
        self.assertLess(float(np.max(np.abs(result.efficiency_error))), 1e-12)
        self.assertEqual(result.values.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
