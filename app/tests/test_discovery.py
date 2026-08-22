from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from causal_shap.discovery import (
    compare_graphs,
    deterministic_consistent_extension,
    direct_lingam_adjacency_to_pdag,
    enforce_dag,
    identify_adjustment_sets,
    pairwise_skeleton_disagreement,
    run_ges,
    run_pc,
    shrier_platt_check,
    skeleton_f1,
    to_dagitty,
)
from causal_shap.graphs import PDAG
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

    def test_consistent_extension_preserves_equivalence_class(self) -> None:
        pdag = PDAG(
            nodes=("A", "B", "C", "D"),
            directed_edges=frozenset({("A", "B"), ("C", "B")}),
            undirected_edges=frozenset({("B", "D")}),
        )
        first = deterministic_consistent_extension(pdag)
        second = deterministic_consistent_extension(pdag)

        self.assertEqual(set(first.edges), set(second.edges))
        self.assertEqual(set(pdag.directed_edges) - set(first.edges), set())
        self.assertEqual(
            {tuple(sorted(edge)) for edge in first.edges}, set(pdag.skeleton)
        )
        self.assertIn(("B", "D"), first.edges)
        self.assertTrue(nx.is_directed_acyclic_graph(first))

    def test_consistent_extension_rejects_cyclic_compelled_edges(self) -> None:
        pdag = PDAG(
            nodes=("A", "B", "C"),
            directed_edges=frozenset({("A", "B"), ("B", "C"), ("C", "A")}),
            undirected_edges=frozenset(),
        )
        with self.assertRaisesRegex(ValueError, "directed component is cyclic"):
            deterministic_consistent_extension(pdag)

    def test_direct_lingam_matrix_uses_child_parent_indexing(self) -> None:
        matrix = np.zeros((3, 3))
        matrix[1, 0] = 0.8  # A -> B
        matrix[2, 1] = -0.4  # B -> C
        pdag = direct_lingam_adjacency_to_pdag(
            matrix, ("A", "B", "C"), threshold=0.1
        )
        self.assertEqual(
            pdag.directed_edges, frozenset({("A", "B"), ("B", "C")})
        )

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
