from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx

from causal_shap.graphs import (
    PDAG,
    ancestors_of,
    directed_distance_to_outcome,
    load_edges_csv,
    undirected_distance_to_outcome,
)


def toy_graph() -> nx.DiGraph:
    """Chain Diet->Hydration->Y, fork Climate->{Hydration,Y}, collider Diet->ClinicVisit<-Y."""
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("Diet", "Hydration"),
            ("Hydration", "Y"),
            ("Climate", "Hydration"),
            ("Climate", "Y"),
            ("Diet", "ClinicVisit"),
            ("Y", "ClinicVisit"),
        ]
    )
    return graph


class GraphsTests(unittest.TestCase):
    def test_load_edges_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edges.csv"
            path.write_text('from,to\nA,B\nB,C\n', encoding="utf-8")
            graph = load_edges_csv(path)
        self.assertEqual(set(graph.edges), {("A", "B"), ("B", "C")})

    def test_load_edges_csv_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edges.csv"
            path.write_text('source,target\nA,B\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_edges_csv(path)

    def test_ancestors_of_collider_includes_both_parents_lineages(self) -> None:
        self.assertEqual(
            ancestors_of(toy_graph(), "ClinicVisit"),
            frozenset({"Diet", "Y", "Hydration", "Climate"}),
        )

    def test_directed_distance_omits_co_descendant_proxy(self) -> None:
        distances = directed_distance_to_outcome(toy_graph(), "Y")
        self.assertEqual(distances, {"Hydration": 1, "Climate": 1, "Diet": 2})
        self.assertNotIn("ClinicVisit", distances)

    def test_undirected_distance_is_finite_for_proxy(self) -> None:
        distances = undirected_distance_to_outcome(toy_graph(), "Y")
        self.assertEqual(distances["ClinicVisit"], 1)
        self.assertEqual(distances["Diet"], 2)

    def test_pdag_skeleton_and_fraction_undirected(self) -> None:
        pdag = PDAG(
            nodes=("A", "B", "C"),
            directed_edges=frozenset({("A", "B")}),
            undirected_edges=frozenset({("B", "C")}),
        )
        self.assertEqual(pdag.skeleton, frozenset({("A", "B"), ("B", "C")}))
        self.assertAlmostEqual(pdag.fraction_undirected, 0.5)
        self.assertEqual(set(pdag.to_digraph().edges), {("A", "B")})

    def test_pdag_rejects_unknown_node_and_unsorted_undirected_edge(self) -> None:
        with self.assertRaises(ValueError):
            PDAG(nodes=("A",), directed_edges=frozenset({("A", "B")}), undirected_edges=frozenset())
        with self.assertRaises(ValueError):
            PDAG(nodes=("A", "B"), directed_edges=frozenset(), undirected_edges=frozenset({("B", "A")}))


if __name__ == "__main__":
    unittest.main()
