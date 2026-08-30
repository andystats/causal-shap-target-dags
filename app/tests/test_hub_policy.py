from __future__ import annotations

import unittest

from hub import stages
from hub.datasets import DATASETS


class PolicyEstimationArmTests(unittest.TestCase):
    """The pricing stage's two estimation arms, stamped like the attribution arms.

    The scm arm trusts every calibrated equation at once; the semiparametric
    arm keeps the SCM as the survey and re-estimates each shortlisted lever's
    shift double-robustly from the data, one functional per survivor
    (Marschak's Maxim). Both stamp arm and arm_note on the payload so no
    number travels without its method.
    """

    @classmethod
    def setUpClass(cls) -> None:
        toy = DATASETS["toy_trap"]
        cls.data = toy.load_data()
        cls.graph = toy.truth_graph()
        cls.specs = toy.cost_specs()

    def run_policy(self, **kwargs: object) -> dict:
        base = dict(
            data=self.data, graph=self.graph, outcome="Y", specs=self.specs,
            budget=1.5, direction="increase", alpha=0.05,
        )
        base.update(kwargs)
        return stages.run_policy(**base)

    def test_scm_arm_is_the_default_and_is_stamped(self) -> None:
        payload = self.run_policy()
        self.assertEqual(payload["arm"], "scm")
        self.assertIn("do() contrast", payload["arm_note"])
        self.assertEqual(payload["estimates"], ())
        self.assertTrue(payload["estimates_table"].empty)

    def test_semiparametric_arm_estimates_every_surviving_lever(self) -> None:
        payload = self.run_policy(
            estimation_arm="semiparametric", estimation_learner="linear"
        )
        self.assertEqual(payload["arm"], "semiparametric")
        self.assertIn("modified treatment policy", payload["arm_note"])
        shortlisted = [item.label for item in payload["ranking"].feasible()]
        self.assertEqual([e.label for e in payload["estimates"]], shortlisted)
        self.assertEqual(len(shortlisted), 3)  # Hydration+1, Diet+1, Climate+1

        # The targeted number sits beside the SCM's for the same action, and
        # on this linear toy the two poles should roughly agree.
        table = payload["estimates_table"]
        self.assertIn("scm_benefit", table.columns)
        for _, row in table.iterrows():
            self.assertLess(abs(row["dr_benefit"] - row["scm_benefit"]), 0.25)

    def test_estimates_use_the_graph_derived_adjustment_set(self) -> None:
        payload = self.run_policy(
            estimation_arm="semiparametric", estimation_learner="linear"
        )
        by_lever = {e.lever: e for e in payload["estimates"]}
        self.assertEqual(by_lever["Hydration"].adjustment, ("Climate", "Diet"))
        self.assertEqual(by_lever["Diet"].adjustment, ())

    def test_unknown_arm_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown estimation arm"):
            self.run_policy(estimation_arm="bayesian")


if __name__ == "__main__":
    unittest.main()
