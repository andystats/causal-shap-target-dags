from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from causal_shap.teaching_dags import (
    layered_ladder,
    simulate_dataframe,
    toy_chain_fork_collider,
)


class TeachingDAGTests(unittest.TestCase):
    def test_toy_graph_is_acyclic_with_expected_shape(self) -> None:
        dag = toy_chain_fork_collider()
        self.assertTrue(nx.is_directed_acyclic_graph(dag.graph))
        self.assertEqual(dag.outcome, "Y")
        self.assertEqual(set(dag.features), {"Diet", "Climate", "Hydration", "ClinicVisit"})

    def test_toy_total_effects_match_path_products(self) -> None:
        effects = toy_chain_fork_collider().true_total_effects
        self.assertAlmostEqual(effects["Hydration"], 1.0)
        self.assertAlmostEqual(effects["Diet"], 0.8)  # 0.8 * 1.0
        self.assertAlmostEqual(effects["Climate"], 1.1)  # 0.6 * 1.0 + 0.5
        self.assertAlmostEqual(effects["ClinicVisit"], 0.0)  # collider proxy of Y

    def test_layered_ladder_has_two_zero_effect_proxies(self) -> None:
        dag = layered_ladder()
        self.assertTrue(nx.is_directed_acyclic_graph(dag.graph))
        self.assertEqual(len(dag.features), 8)
        self.assertAlmostEqual(dag.true_total_effects["LabProxy"], 0.0)
        self.assertAlmostEqual(dag.true_total_effects["MonitorProxy"], 0.0)
        self.assertAlmostEqual(dag.true_total_effects["Metabolism"], 1.0)
        self.assertAlmostEqual(dag.true_total_effects["Genetics"], 0.94)  # 0.7 + 0.4*0.6

    def test_simulated_moments_track_true_effects(self) -> None:
        dag = toy_chain_fork_collider()
        data = simulate_dataframe(dag, n=40000, seed=20260727)
        # The collider is strongly correlated with Y despite zero causal effect.
        corr = data["ClinicVisit"].corr(data["Y"])
        self.assertGreater(corr, 0.5)
        # Diet reaches Y only through Hydration; its regression-free covariance is positive.
        self.assertGreater(data["Diet"].corr(data["Y"]), 0.2)

    def test_simulation_is_seed_reproducible(self) -> None:
        dag = layered_ladder()
        first = simulate_dataframe(dag, n=500, seed=20260728)
        second = simulate_dataframe(dag, n=500, seed=20260728)
        np.testing.assert_allclose(first.to_numpy(), second.to_numpy())


if __name__ == "__main__":
    unittest.main()
