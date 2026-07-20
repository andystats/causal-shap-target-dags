from __future__ import annotations

import unittest

import networkx as nx

from causal_shap.discovery import (
    compare_graphs,
    enforce_dag,
    identify_adjustment_sets,
    pairwise_skeleton_disagreement,
    run_ges,
    run_pc,
    shrier_platt_check,
    skeleton_f1,
    to_dagitty,
)
from causal_shap.teaching_dags import simulate_dataframe, toy_chain_fork_collider


class DiscoveryMetricsTests(unittest.TestCase):
    def test_compare_graphs_classifies_edges(self) -> None:
        truth = [("A", "B"), ("B", "C"), ("C", "D")]
        discovered = [("A", "B"), ("C", "B"), ("A", "D")]  # correct, reversed, spurious
        result = compare_graphs(discovered, truth)
        self.assertEqual(result.correct, frozenset({("A", "B")}))
        self.assertEqual(result.reversed, frozenset({("C", "B")}))
        self.assertEqual(result.spurious, frozenset({("A", "D")}))
        self.assertEqual(result.missed, frozenset({("C", "D")}))
        self.assertAlmostEqual(result.precision, 1 / 3)
        self.assertAlmostEqual(result.recall, 1 / 3)
        # skeleton counts the reversed C-B edge as correct
        self.assertAlmostEqual(result.skeleton_precision, 2 / 3)

    def test_skeleton_f1_ignores_direction(self) -> None:
        self.assertAlmostEqual(skeleton_f1([("A", "B")], [("B", "A")]), 1.0)
        self.assertAlmostEqual(skeleton_f1([], []), 1.0)

    def test_enforce_dag_breaks_cycles(self) -> None:
        edges, removed = enforce_dag([("A", "B"), ("B", "C"), ("C", "A")], ["A", "B", "C"])
        graph = nx.DiGraph(edges)
        self.assertTrue(nx.is_directed_acyclic_graph(graph))
        self.assertEqual(len(removed), 1)

    def test_shrier_platt_detects_open_backdoor(self) -> None:
        graph = nx.DiGraph([("L", "A"), ("L", "Y"), ("A", "Y")])
        self.assertFalse(shrier_platt_check(graph, "A", "Y", [])["valid"])
        self.assertTrue(shrier_platt_check(graph, "A", "Y", ["L"])["valid"])

    def test_identify_adjustment_sets_returns_confounder(self) -> None:
        graph = nx.DiGraph([("L", "A"), ("L", "Y"), ("A", "Y")])
        sets = identify_adjustment_sets(graph, "A", "Y")
        self.assertEqual(sets["traditional"]["variables"], ["L"])
        self.assertTrue(sets["traditional"]["valid"])

    def test_to_dagitty_is_sorted_block(self) -> None:
        code = to_dagitty([("B", "C"), ("A", "B")])
        self.assertEqual(code, "dag {\n  A -> B\n  B -> C\n}")

    def test_pairwise_disagreement_single_algorithm_is_zero(self) -> None:
        data = simulate_dataframe(toy_chain_fork_collider(), n=800, seed=1)
        self.assertEqual(pairwise_skeleton_disagreement([run_pc(data)]), 0.0)


class DiscoveryLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dag = toy_chain_fork_collider()
        self.data = simulate_dataframe(self.dag, n=3000, seed=20260729)
        self.truth_skeleton = {tuple(sorted(edge)) for edge in self.dag.graph.edges}

    def test_pc_recovers_true_skeleton(self) -> None:
        result = run_pc(self.data, alpha=0.05)
        self.assertEqual(set(result.skeleton), self.truth_skeleton)

    def test_ges_recovers_true_skeleton(self) -> None:
        result = run_ges(self.data)
        self.assertEqual(set(result.skeleton), self.truth_skeleton)

    def test_forbidden_edge_constraint_is_respected(self) -> None:
        result = run_pc(self.data, alpha=0.05, forbidden_edges=[("ClinicVisit", "Y"), ("Y", "ClinicVisit")])
        self.assertNotIn(("ClinicVisit", "Y"), result.skeleton)
        self.assertNotIn(("Y", "ClinicVisit"), result.directed_edges)

    def test_required_edge_constraint_forces_direction(self) -> None:
        result = run_pc(self.data, alpha=0.05, required_edges=[("Hydration", "Y")])
        self.assertIn(("Hydration", "Y"), result.directed_edges)


if __name__ == "__main__":
    unittest.main()
