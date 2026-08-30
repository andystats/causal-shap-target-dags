from __future__ import annotations

import unittest

from causal_shap.graph_state import GraphProvenance, GraphState
from hub.theater import apply_surgery, render_theater


def discovered_graph() -> GraphState:
    """Compelled A -> B -> D, with the (A, C) pair unresolved (shown A -> C).

    A real CPDAG's unresolved pair is flippable without creating a cycle, so
    the fixture keeps C off every path back to A.
    """
    return GraphState(
        nodes=("A", "B", "C", "D"),
        directed_edges=frozenset({("A", "B"), ("B", "D"), ("A", "C")}),
        undirected_pairs=(("A", "C"),),
        provenance=GraphProvenance(source="discovered", algorithm="pc"),
    )


def chain_graph() -> GraphState:
    """A -> B -> C, fully compelled; adding C -> A closes the cycle."""
    return GraphState(
        nodes=("A", "B", "C"),
        directed_edges=frozenset({("A", "B"), ("B", "C")}),
        provenance=GraphProvenance(source="discovered", algorithm="pc"),
    )


class ApplySurgeryTests(unittest.TestCase):
    def test_add_asserts_a_missing_edge_into_the_ledger(self) -> None:
        state = GraphState(
            nodes=("A", "B", "C"),
            directed_edges=frozenset({("A", "B")}),
            provenance=GraphProvenance(source="discovered", algorithm="pc"),
        )
        revised = apply_surgery(state, "add", ("A", "C"), "domain knowledge")
        self.assertIn(("A", "C"), revised.directed_edges)
        self.assertEqual(revised.provenance.source, "recovered")
        entry = revised.provenance.constraint_ledger[-1]
        self.assertEqual((entry.edge, entry.kind, entry.applied),
                         (("A", "C"), "required", "post_hoc"))

    def test_add_refuses_duplicates_reverses_and_self_loops(self) -> None:
        state = discovered_graph()
        with self.assertRaisesRegex(ValueError, "already exists"):
            apply_surgery(state, "add", ("A", "B"), "")
        with self.assertRaisesRegex(ValueError, "Flip instead"):
            apply_surgery(state, "add", ("B", "A"), "")
        with self.assertRaisesRegex(ValueError, "two different nodes"):
            apply_surgery(state, "add", ("A", "A"), "")

    def test_add_that_closes_a_cycle_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "cyclic"):
            apply_surgery(chain_graph(), "add", ("C", "A"), "")

    def test_flip_of_an_unresolved_pair_adjudicates_it(self) -> None:
        revised = apply_surgery(discovered_graph(), "flip", ("A", "C"), "review")
        self.assertIn(("C", "A"), revised.directed_edges)
        self.assertNotIn(("A", "C"), revised.directed_edges)
        self.assertEqual(revised.undirected_pairs, ())  # the pair is now decided

    def test_remove_drops_the_edge_and_forbids_both_directions(self) -> None:
        revised = apply_surgery(discovered_graph(), "remove", ("A", "C"), "")
        self.assertNotIn(("A", "C"), revised.directed_edges)
        kinds = {(e.edge, e.kind) for e in revised.provenance.constraint_ledger}
        self.assertIn((("A", "C"), "forbidden"), kinds)
        self.assertIn((("C", "A"), "forbidden"), kinds)

    def test_forbid_on_an_unresolved_pair_resolves_it_the_other_way(self) -> None:
        revised = apply_surgery(discovered_graph(), "forbid", ("A", "C"), "")
        self.assertIn(("C", "A"), revised.directed_edges)
        self.assertEqual(revised.undirected_pairs, ())


class RenderTheaterTests(unittest.TestCase):
    def test_each_render_gets_unique_marker_ids(self) -> None:
        # Two graphs on one page sharing marker ids resolve to whichever defs
        # comes first in the document; inside a hidden tab pane that marker is
        # not painted and every arrowhead disappears. The regression that
        # motivated this test shipped with the answer-key card.
        state = discovered_graph()
        first = render_theater(state, None, "C", interactive=False)
        second = render_theater(state, None, "C", interactive=False)
        marker_of = lambda svg: svg.split('marker id="')[1].split('"')[0]
        self.assertNotEqual(marker_of(first), marker_of(second))
        # Every arrow reference points at the render's own marker.
        own = marker_of(second)
        self.assertIn(f'marker-end="url(#{own})"', second)

    def test_static_render_has_no_click_bridge_or_shared_dom_id(self) -> None:
        svg = render_theater(discovered_graph(), None, "C", interactive=False)
        self.assertNotIn("<script>", svg)
        self.assertNotIn('id="theater-svg"', svg)


if __name__ == "__main__":
    unittest.main()
