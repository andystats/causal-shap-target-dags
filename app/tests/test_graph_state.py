from __future__ import annotations

import unittest

import networkx as nx

from causal_shap.discovery import deterministic_consistent_extension
from causal_shap.graph_state import (
    ConstraintEntry,
    GraphProvenance,
    GraphState,
)
from causal_shap.graphs import PDAG


def toy_pdag() -> PDAG:
    """Compelled Diet->Hydration, with Climate--Hydration and Climate--Y open."""
    return PDAG(
        nodes=("Climate", "Diet", "Hydration", "Y"),
        directed_edges=frozenset({("Diet", "Hydration"), ("Hydration", "Y")}),
        undirected_edges=frozenset({("Climate", "Hydration"), ("Climate", "Y")}),
    )


def discovered(**kwargs: object) -> GraphProvenance:
    return GraphProvenance(source="discovered", algorithm="pc", **kwargs)


class GraphStateConstructionTests(unittest.TestCase):
    def test_from_pdag_matches_the_shared_consistent_extension(self) -> None:
        pdag = toy_pdag()
        state = GraphState.from_pdag(pdag, discovered())
        self.assertEqual(
            state.directed_edges,
            frozenset(deterministic_consistent_extension(pdag).edges),
        )

    def test_from_pdag_retains_undirected_pairs_sorted_for_m5(self) -> None:
        state = GraphState.from_pdag(toy_pdag(), discovered())
        self.assertEqual(
            state.undirected_pairs,
            (("Climate", "Hydration"), ("Climate", "Y")),
        )
        self.assertEqual(state.n_undirected_pairs, 2)
        self.assertFalse(state.is_fully_directed)

    def test_from_digraph_claims_no_ambiguity(self) -> None:
        graph = nx.DiGraph([("A", "B"), ("B", "C")])
        state = GraphState.from_digraph(graph, GraphProvenance(source="bundle"))
        self.assertEqual(state.directed_edges, frozenset({("A", "B"), ("B", "C")}))
        self.assertEqual(state.undirected_pairs, ())
        self.assertTrue(state.is_fully_directed)

    def test_digraph_is_the_single_inflation_point_and_is_frozen(self) -> None:
        state = GraphState.from_pdag(toy_pdag(), discovered())
        graph = state.digraph()
        self.assertIs(graph, state.digraph())  # cached, not rebuilt per caller
        self.assertEqual(set(graph.nodes), set(state.nodes))
        with self.assertRaises(nx.NetworkXError):
            graph.add_edge("Diet", "Y")

    def test_digraph_copy_is_mutable(self) -> None:
        state = GraphState.from_pdag(toy_pdag(), discovered())
        mutable = state.digraph().copy()
        mutable.add_edge("Diet", "Y")
        self.assertIn(("Diet", "Y"), set(mutable.edges))
        self.assertNotIn(("Diet", "Y"), state.directed_edges)


class GraphStateValidationTests(unittest.TestCase):
    def test_rejects_edge_to_unknown_node(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown node"):
            GraphState(nodes=("A",), directed_edges=frozenset({("A", "B")}))

    def test_rejects_unsorted_undirected_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be sorted"):
            GraphState(
                nodes=("A", "B"),
                directed_edges=frozenset(),
                undirected_pairs=(("B", "A"),),
            )

    def test_rejects_undirected_pair_with_no_directed_edge(self) -> None:
        # The extension orients every undirected edge, so a pair with no
        # corresponding directed edge means the two fields disagree.
        with self.assertRaisesRegex(ValueError, "must also appear as directed"):
            GraphState(
                nodes=("A", "B"),
                directed_edges=frozenset(),
                undirected_pairs=(("A", "B"),),
            )

    def test_accepts_undirected_pair_oriented_either_way(self) -> None:
        for edge in (("A", "B"), ("B", "A")):
            state = GraphState(
                nodes=("A", "B"),
                directed_edges=frozenset({edge}),
                undirected_pairs=(("A", "B"),),
            )
            self.assertEqual(state.n_undirected_pairs, 1)

    def test_rejects_duplicate_node_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            GraphState(nodes=("A", "A"), directed_edges=frozenset())

    def test_rejects_a_cycle(self) -> None:
        # Ancestor screens, topological simulation, and SCM calibration all
        # misbehave silently on a cycle rather than failing.
        with self.assertRaisesRegex(ValueError, "cyclic"):
            GraphState(
                nodes=("A", "B"),
                directed_edges=frozenset({("A", "B"), ("B", "A")}),
            )

    def test_from_digraph_rejects_a_cycle_too(self) -> None:
        with self.assertRaisesRegex(ValueError, "cyclic"):
            GraphState.from_digraph(
                nx.DiGraph([("A", "B"), ("B", "C"), ("C", "A")]),
                GraphProvenance(source="uploaded"),
            )

    def test_recovered_graph_must_name_its_parent(self) -> None:
        with self.assertRaisesRegex(ValueError, "must name the discovered graph"):
            GraphProvenance(source="recovered")

    def test_rejects_unknown_source_kind_and_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown graph source"):
            GraphProvenance(source="guessed")
        with self.assertRaisesRegex(ValueError, "Unknown constraint kind"):
            ConstraintEntry(edge=("A", "B"), kind="banned", applied="post_hoc")
        with self.assertRaisesRegex(ValueError, "Unknown constraint stage"):
            ConstraintEntry(edge=("A", "B"), kind="forbidden", applied="later")


class GraphStateFingerprintTests(unittest.TestCase):
    def test_fingerprint_is_stable_across_two_constructions(self) -> None:
        left = GraphState.from_pdag(toy_pdag(), discovered())
        right = GraphState.from_pdag(toy_pdag(), discovered())
        self.assertEqual(left.fingerprint(), right.fingerprint())

    def test_fingerprint_changes_when_one_edge_flips(self) -> None:
        base = GraphState(
            nodes=("A", "B"), directed_edges=frozenset({("A", "B")})
        )
        flipped = GraphState(
            nodes=("A", "B"), directed_edges=frozenset({("B", "A")})
        )
        self.assertNotEqual(base.fingerprint(), flipped.fingerprint())

    def test_fingerprint_ignores_provenance_so_reruns_do_not_invalidate(self) -> None:
        edges = frozenset({("A", "B")})
        pc_run = GraphState(
            nodes=("A", "B"),
            directed_edges=edges,
            provenance=discovered(params={"alpha": 0.05}),
        )
        ges_run = GraphState(
            nodes=("A", "B"),
            directed_edges=edges,
            provenance=GraphProvenance(source="uploaded"),
        )
        self.assertEqual(pc_run.fingerprint(), ges_run.fingerprint())

    def test_fingerprint_distinguishes_identified_from_chosen_orientation(self) -> None:
        # Same directed edge; one is compelled, the other merely a representative
        # extension of an undirected pair. Downstream honesty differs, so the
        # fingerprints must too.
        identified = GraphState(
            nodes=("A", "B"), directed_edges=frozenset({("A", "B")})
        )
        chosen = GraphState(
            nodes=("A", "B"),
            directed_edges=frozenset({("A", "B")}),
            undirected_pairs=(("A", "B"),),
        )
        self.assertNotEqual(identified.fingerprint(), chosen.fingerprint())


class GraphStateRecoveryTests(unittest.TestCase):
    def test_with_constraints_records_the_discovered_parent(self) -> None:
        found = GraphState.from_pdag(toy_pdag(), discovered())
        ledger = (
            ConstraintEntry(
                edge=("Climate", "Y"),
                kind="required",
                applied="post_hoc",
                rationale="Reynolds review",
            ),
        )
        # Requiring Climate->Y means dropping whichever orientation the
        # extension chose, not adding a second arrow between the same pair.
        resolved = (found.directed_edges - {("Y", "Climate")}) | {("Climate", "Y")}
        recovered = found.with_constraints(
            directed_edges=resolved,
            undirected_pairs=(("Climate", "Hydration"),),
            ledger=ledger,
        )
        self.assertEqual(recovered.provenance.source, "recovered")
        self.assertEqual(recovered.provenance.parent_fingerprint, found.fingerprint())
        self.assertEqual(recovered.provenance.constraint_ledger, ledger)
        self.assertEqual(recovered.n_undirected_pairs, 1)


if __name__ == "__main__":
    unittest.main()
