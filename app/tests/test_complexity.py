from __future__ import annotations

import unittest

import networkx as nx

from causal_shap.complexity import (
    REGISTRY,
    ComplexityInputs,
    ProvisionalStructuralComplexityIndex,
    get_score,
)
from causal_shap.graphs import PDAG
from causal_shap.teaching_dags import toy_chain_fork_collider


def _score(graph, outcome, **kwargs):
    return ProvisionalStructuralComplexityIndex().compute(
        ComplexityInputs(graph=graph, outcome=outcome, **kwargs)
    )


class ComplexityTests(unittest.TestCase):
    def test_report_is_marked_provisional(self) -> None:
        report = _score(toy_chain_fork_collider().graph, "Y")
        self.assertTrue(report.provisional)
        self.assertEqual(report.score_name, "PSCI")
        self.assertIn(report.band, {"low", "moderate", "high"})

    def test_monotonic_within_a_growing_chain_family(self) -> None:
        # Complexity is not purely size (a small dense graph can be complex), so
        # monotonicity is asserted on a controlled family where depth and size
        # grow together: longer chains are strictly more complex.
        def chain(k: int) -> nx.DiGraph:
            graph = nx.DiGraph()
            graph.add_edges_from((f"X{i}", f"X{i + 1}") for i in range(k))
            return graph

        totals = [_score(chain(k), f"X{k}").total for k in (3, 6, 10)]
        self.assertLess(totals[0], totals[1])
        self.assertLess(totals[1], totals[2])

    def test_large_graph_outscores_small_one(self) -> None:
        small = _score(toy_chain_fork_collider().graph, "Y").total
        big = nx.gnr_graph(45, 0.2, seed=3).reverse()
        big = nx.relabel_nodes(big, {n: f"n{n}" for n in big.nodes})
        self.assertGreater(_score(big, "n0").total, small)

    def test_degraded_mode_renormalizes_without_disagreement(self) -> None:
        report = _score(toy_chain_fork_collider().graph, "Y")
        self.assertFalse(report.subscores["disagreement"].available)
        with_disagreement = _score(toy_chain_fork_collider().graph, "Y", disagreement=1.0)
        self.assertTrue(with_disagreement.subscores["disagreement"].available)
        # Adding a high disagreement subscore must raise the total.
        self.assertGreater(with_disagreement.total, report.total)

    def test_pdag_ambiguity_raises_score(self) -> None:
        dag = PDAG(
            nodes=("A", "B", "Y"),
            directed_edges=frozenset({("A", "Y"), ("B", "Y")}),
            undirected_edges=frozenset(),
        )
        cpdag = PDAG(
            nodes=("A", "B", "Y"),
            directed_edges=frozenset({("A", "Y")}),
            undirected_edges=frozenset({tuple(sorted(("B", "Y")))}),
        )
        self.assertGreater(_score(cpdag, "Y").total, _score(dag, "Y").total)

    def test_registry_round_trip(self) -> None:
        self.assertIn("PSCI", REGISTRY)
        self.assertIs(get_score("PSCI").compute.__self__.__class__, ProvisionalStructuralComplexityIndex)
        with self.assertRaises(KeyError):
            get_score("LumaWarp")


if __name__ == "__main__":
    unittest.main()
