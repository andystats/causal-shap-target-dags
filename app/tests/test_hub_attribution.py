from __future__ import annotations

import unittest

from hub import stages, theater
from hub.datasets import DATASETS

QUICK = dict(arm="nonparametric", model_type="gbm",
             n_perms=8, n_background=8, n_instances=8)


class GraphGovernsAttributionTests(unittest.TestCase):
    """The current graph decides WHO gets attributed - the canary contract.

    A feature with no directed path to the outcome has causal share exactly
    zero by construction (matching the frozen record, which attributes over
    the outcome's ancestors), and surgery that makes it an ancestor restores
    its eligibility. If the collider proxy ever carries causal credit under
    the correct graph again, this test is the alarm.
    """

    @classmethod
    def setUpClass(cls) -> None:
        toy = DATASETS["toy_trap"]
        cls.data = toy.load_data()
        cls.graph = toy.truth_graph()
        cls.features = ("Diet", "Climate", "Hydration", "ClinicVisit")

    def test_non_ancestor_is_structurally_zero_under_the_correct_graph(self) -> None:
        result = stages.run_causal_shap(
            data=self.data, features=self.features, outcome="Y",
            graph=self.graph, truth_effects=None, **QUICK,
        )
        self.assertEqual(result["excluded"], ("ClinicVisit",))
        self.assertEqual(result["causal_importance"]["ClinicVisit"], 0.0)
        self.assertIn("ancestors", result["arm_note"])

    def test_surgery_that_makes_it_an_ancestor_restores_eligibility(self) -> None:
        revised = theater.apply_surgery(self.graph, "flip", ("Y", "ClinicVisit"), "")
        result = stages.run_causal_shap(
            data=self.data, features=self.features, outcome="Y",
            graph=revised, truth_effects=None, **QUICK,
        )
        self.assertEqual(result["excluded"], ())
        self.assertGreater(result["causal_importance"]["ClinicVisit"], 0.0)

    def test_no_ancestral_feature_at_all_raises_rather_than_attributing(self) -> None:
        with self.assertRaisesRegex(ValueError, "nothing to causally attribute"):
            stages.run_causal_shap(
                data=self.data, features=("ClinicVisit",), outcome="Y",
                graph=self.graph, truth_effects=None, **QUICK,
            )


if __name__ == "__main__":
    unittest.main()
