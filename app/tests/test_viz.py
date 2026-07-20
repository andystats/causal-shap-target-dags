from __future__ import annotations

import unittest

import matplotlib
import networkx as nx

matplotlib.use("Agg")

from causal_shap.viz import distortion_profile, homunculus_figure, homunculus_pair, ladder_svg
from causal_shap.teaching_dags import toy_chain_fork_collider


class VizSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dag = toy_chain_fork_collider()
        self.graph = self.dag.graph
        # Vanilla over-credits the collider proxy; truth gives it zero.
        self.truth = dict(self.dag.true_total_effects)
        self.vanilla = {name: 0.1 for name in self.dag.features}
        self.vanilla["ClinicVisit"] = 1.0
        self.structural = dict(self.dag.true_total_effects)

    def test_homunculus_figure_renders(self) -> None:
        fig = homunculus_figure(self.graph, "Y", self.vanilla, self.truth)
        self.assertEqual(len(fig.axes), 1)
        fig.clf()

    def test_homunculus_pair_has_three_panels(self) -> None:
        fig = homunculus_pair(self.graph, "Y", self.truth, self.vanilla, self.structural)
        self.assertEqual(len(fig.axes), 3)
        fig.clf()

    def test_distortion_profile_renders(self) -> None:
        depths = {"Diet": 2, "Climate": 1, "Hydration": 1, "ClinicVisit": 1}
        shares = {
            "Interventional truth": self.truth,
            "Ordinary SHAP": self.vanilla,
            "Structural Causal SHAP": self.structural,
        }
        fig = distortion_profile(depths, shares)
        self.assertTrue(fig.axes)
        fig.clf()

    def test_ladder_svg_marks_active_rung(self) -> None:
        svg = ladder_svg(active=3)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("Causal SHAP", svg)
        self.assertIn("#eff6ff", svg)  # active highlight present


if __name__ == "__main__":
    unittest.main()
