from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from causal_shap.structural_value import ExogenousDraws, NodeSpec
from causal_shap.teaching_dags import (
    TeachingDAG,
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


class ExogenousNoiseByKindTests(unittest.TestCase):
    """Continuous nodes add scaled noise; binary nodes threshold against it."""

    def test_continuous_dags_draw_exactly_as_before(self) -> None:
        # Regression guard for the frozen teaching data: every shipped teaching
        # DAG is continuous, so the draw sequence must be unchanged.
        for dag in (toy_chain_fork_collider(), layered_ladder()):
            rng = np.random.default_rng(20260727)
            legacy = ExogenousDraws(
                {spec.name: rng.standard_normal(400) for spec in dag.specs}
            )
            expected = dag.scm().simulate(legacy)
            produced = simulate_dataframe(dag, n=400, seed=20260727)
            for name in produced.columns:
                np.testing.assert_allclose(produced[name].to_numpy(), expected[name])

    def test_binary_root_realizes_its_stated_probability(self) -> None:
        # Standard-normal draws compared against a probability would realize
        # P(Z < 0.35) = 0.637 rather than 0.35.
        dag = TeachingDAG(
            name="binary_root",
            specs=(
                NodeSpec("Exposure", "binary", root_probability=0.35),
                NodeSpec(
                    "Y", "continuous", parents=("Exposure",), coefficients=(1.0,), noise_sd=0.5
                ),
            ),
            outcome="Y",
            features=("Exposure",),
            true_total_effects={"Exposure": 1.0},
        )
        data = simulate_dataframe(dag, n=200000, seed=20260727)
        self.assertEqual(set(np.unique(data["Exposure"])), {0.0, 1.0})
        self.assertAlmostEqual(data["Exposure"].mean(), 0.35, places=2)


if __name__ == "__main__":
    unittest.main()
