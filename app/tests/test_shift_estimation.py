from __future__ import annotations

import unittest

import networkx as nx
import numpy as np
import pandas as pd

from causal_shap.seeds import SEED_ACTION_ABDUCTION
from causal_shap.shift_estimation import (
    FEASIBILITY_CAUTION,
    FEASIBILITY_OK,
    adjustment_for_lever,
    estimate_shift_effect,
    estimates_frame,
)
from causal_shap.teaching_dags import simulate_dataframe, toy_chain_fork_collider

# Toy DAG total effects, exact for a linear SCM (see test_policy.py):
#   te(Hydration) = 1.0,  te(Diet) = 0.8,  te(Climate) = 1.1
TRUTH = {"Hydration": 1.0, "Diet": 0.8, "Climate": 1.1}


class AdjustmentTests(unittest.TestCase):
    """The adjustment set is the lever's parents under the current graph."""

    def setUp(self) -> None:
        self.graph = toy_chain_fork_collider().graph

    def test_non_root_lever_adjusts_for_its_parents_only(self) -> None:
        # Climate confounds Hydration -> Y; Diet is a harmless extra parent.
        # No descendant of Hydration can appear: parents cannot be descendants,
        # which is the structural guard against post-lever confounding.
        self.assertEqual(
            adjustment_for_lever(self.graph, "Hydration", "Y"), ("Climate", "Diet")
        )

    def test_root_lever_needs_no_adjustment(self) -> None:
        self.assertEqual(adjustment_for_lever(self.graph, "Diet", "Y"), ())
        self.assertEqual(adjustment_for_lever(self.graph, "Climate", "Y"), ())

    def test_unknown_lever_and_outcome_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a node"):
            adjustment_for_lever(self.graph, "Nonesuch", "Y")
        with self.assertRaisesRegex(ValueError, "not a node"):
            adjustment_for_lever(self.graph, "Diet", "Nonesuch")


class ShiftEffectTests(unittest.TestCase):
    """Cross-fitted AIPW recovers the known linear shift effects from data."""

    @classmethod
    def setUpClass(cls) -> None:
        dag = toy_chain_fork_collider()
        cls.graph = dag.graph
        cls.data = simulate_dataframe(dag, n=2000, seed=SEED_ACTION_ABDUCTION)

    def estimate(self, lever: str, delta: float = 1.0, **kwargs: object):
        return estimate_shift_effect(
            self.data, self.graph, lever, delta, "Y", learner="linear", **kwargs
        )

    def test_estimates_recover_the_exact_linear_total_effects(self) -> None:
        for lever, truth in TRUTH.items():
            with self.subTest(lever=lever):
                estimate = self.estimate(lever)
                self.assertLess(abs(estimate.benefit - truth), 0.1)
                self.assertLess(estimate.ci_low, truth)
                self.assertGreater(estimate.ci_high, truth)

    def test_decrease_direction_negates_the_benefit(self) -> None:
        increase = self.estimate("Hydration")
        decrease = self.estimate("Hydration", direction="decrease")
        self.assertEqual(decrease.benefit, -increase.benefit)

    def test_estimation_is_deterministic(self) -> None:
        first, second = self.estimate("Diet"), self.estimate("Diet")
        self.assertEqual(first, second)

    def test_moderate_shift_on_a_wide_lever_is_feasible(self) -> None:
        # Climate has SD ~1, so a +1 shift stays on support with tame weights;
        # this pins the diagnostic thresholds against silent drift.
        estimate = self.estimate("Climate")
        self.assertEqual(estimate.feasibility.verdict, FEASIBILITY_OK)
        self.assertEqual(estimate.feasibility.notes, ())

    def test_tight_conditional_spread_draws_a_weight_caution(self) -> None:
        # Hydration's conditional SD is 0.5, so a +1 shift is a 2-sigma ask:
        # the density-ratio weights concentrate and the verdict says so.
        estimate = self.estimate("Hydration")
        self.assertEqual(estimate.feasibility.verdict, FEASIBILITY_CAUTION)
        self.assertTrue(
            any("effective sample size" in note for note in estimate.feasibility.notes)
        )

    def test_off_support_shift_fails_the_haneuse_rotnitzky_check(self) -> None:
        estimate = self.estimate("Hydration", delta=5.0)
        check = estimate.feasibility
        self.assertEqual(check.verdict, FEASIBILITY_CAUTION)
        self.assertGreater(check.share_shifted_outside_support, 0.5)
        self.assertTrue(
            any("outside the observed range" in note for note in check.notes)
        )

    def test_binary_lever_is_refused_with_the_reason(self) -> None:
        rng = np.random.default_rng(0)
        lever = rng.integers(0, 2, size=200).astype(float)
        frame = pd.DataFrame({"A": lever, "Y": lever + rng.normal(size=200)})
        graph = nx.DiGraph([("A", "Y")])
        with self.assertRaisesRegex(ValueError, "binary"):
            estimate_shift_effect(frame, graph, "A", 1.0, "Y")

    def test_degenerate_requests_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero shift"):
            self.estimate("Diet", delta=0.0)
        with self.assertRaisesRegex(ValueError, "Unknown learner"):
            estimate_shift_effect(self.data, self.graph, "Diet", 1.0, "Y", learner="rf")
        with self.assertRaisesRegex(ValueError, "too few"):
            estimate_shift_effect(self.data.head(10), self.graph, "Diet", 1.0, "Y")

    def test_estimates_frame_names_the_empty_adjustment_honestly(self) -> None:
        frame = estimates_frame([self.estimate("Diet"), self.estimate("Hydration")])
        self.assertEqual(
            list(frame.columns),
            ["action", "adjustment", "dr_benefit", "dr_se", "ci95",
             "feasibility", "notes"],
        )
        self.assertEqual(frame.loc[0, "adjustment"], "(none: root lever)")
        self.assertEqual(frame.loc[1, "adjustment"], "Climate, Diet")

    def test_gbm_nuisances_stay_near_the_truth(self) -> None:
        estimate = estimate_shift_effect(
            self.data, self.graph, "Hydration", 1.0, "Y", learner="gbm"
        )
        self.assertLess(abs(estimate.benefit - TRUTH["Hydration"]), 0.2)


if __name__ == "__main__":
    unittest.main()
