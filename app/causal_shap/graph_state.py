"""One carrier for a graph plus the provenance that says how much to trust it.

Discovery returns a CPDAG: an equivalence class in which some edges stay
undirected. DAG-only tools need a single directed graph, so callers take one
deterministic representative consistent extension -- but the pairs that were
undirected must travel alongside it, or M5 identification honesty has nothing to
measure and the app silently presents an equivalence class as a settled DAG.

``GraphState`` bundles the three things that must never be separated: the
directed edges actually used, the undirected pairs still unresolved, and the
provenance recording where the graph came from. The ``source`` field carries the
distinction the whole program turns on -- ``"discovered"`` is the raw
algorithmic output, ``"recovered"`` is what a human adjudicated afterwards -- so
the app and the manuscript share one vocabulary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property

import networkx as nx

from .discovery import deterministic_consistent_extension
from .graphs import PDAG

SOURCES = ("discovered", "recovered", "uploaded", "bundle")
CONSTRAINT_KINDS = ("forbidden", "required")
CONSTRAINT_STAGES = ("search_time", "post_hoc")


@dataclass(frozen=True)
class ConstraintEntry:
    """One expert edge judgement, with the stage at which it took effect.

    ``applied`` is not decoration. ``discovery.apply_constraints`` edits the
    discovered PDAG *after* the search, so a ledger that does not separate
    search-time background knowledge from post-hoc reorientation overstates what
    the algorithm was actually told.
    """

    edge: tuple[str, str]
    kind: str
    applied: str
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.kind not in CONSTRAINT_KINDS:
            raise ValueError(f"Unknown constraint kind: {self.kind!r}")
        if self.applied not in CONSTRAINT_STAGES:
            raise ValueError(f"Unknown constraint stage: {self.applied!r}")


@dataclass(frozen=True)
class GraphProvenance:
    """Where a graph came from and what was done to it."""

    source: str
    algorithm: str = ""
    params: dict[str, object] = field(default_factory=dict)
    n_rows: int = 0
    constraint_ledger: tuple[ConstraintEntry, ...] = ()
    parent_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"Unknown graph source: {self.source!r}")
        if self.source == "recovered" and not self.parent_fingerprint:
            raise ValueError(
                "A recovered graph must name the discovered graph it came from"
            )


@dataclass(frozen=True)
class GraphState:
    """A directed graph, its unresolved pairs, and its provenance.

    ``undirected_pairs`` does not hold a second set of edges. The consistent
    extension orients every undirected edge, so each pair listed here is also
    present in ``directed_edges`` in one orientation; the pair records that the
    orientation was *chosen*, not identified. That is the form
    ``evaluation.m5_identification_honesty`` expects: it drops these pairs from
    the graph and re-enumerates their orientations.
    """

    nodes: tuple[str, ...]
    directed_edges: frozenset[tuple[str, str]]
    undirected_pairs: tuple[tuple[str, str], ...] = ()
    provenance: GraphProvenance = field(
        default_factory=lambda: GraphProvenance(source="uploaded")
    )

    def __post_init__(self) -> None:
        known = set(self.nodes)
        if len(known) != len(self.nodes):
            raise ValueError("Node names must be unique")
        for source, target in set(self.directed_edges) | set(self.undirected_pairs):
            if source not in known or target not in known:
                raise ValueError(f"Edge references unknown node: {(source, target)}")
        for pair in self.undirected_pairs:
            if pair != tuple(sorted(pair)):
                raise ValueError(f"Undirected pair must be sorted: {pair}")
        if len(set(self.undirected_pairs)) != len(self.undirected_pairs):
            raise ValueError("Undirected pairs must be unique")
        skeleton = {tuple(sorted(edge)) for edge in self.directed_edges}
        unrepresented = set(self.undirected_pairs) - skeleton
        if unrepresented:
            raise ValueError(
                "Undirected pairs must also appear as directed edges, since the "
                "consistent extension orients them: "
                f"{sorted(unrepresented)}"
            )
        # Every consumer assumes acyclicity: ancestor screens, topological
        # simulation, and SCM calibration all silently misbehave on a cycle
        # rather than failing, so refuse one at construction.
        probe = nx.DiGraph()
        probe.add_nodes_from(self.nodes)
        probe.add_edges_from(self.directed_edges)
        if not nx.is_directed_acyclic_graph(probe):
            cycle = nx.find_cycle(probe)
            raise ValueError(f"Graph is cyclic: {[edge[:2] for edge in cycle]}")

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------
    @classmethod
    def from_pdag(cls, pdag: PDAG, provenance: GraphProvenance) -> "GraphState":
        """Extend a CPDAG to one deterministic DAG, keeping the undirected pairs.

        The extension is reproducible, not identified: it is one representative
        of the equivalence class. ``undirected_pairs`` records exactly how much
        ambiguity that hides.
        """
        extension = deterministic_consistent_extension(pdag)
        return cls(
            nodes=tuple(pdag.nodes),
            directed_edges=frozenset(extension.edges),
            undirected_pairs=tuple(sorted(pdag.undirected_edges)),
            provenance=provenance,
        )

    @classmethod
    def from_digraph(
        cls,
        graph: nx.DiGraph,
        provenance: GraphProvenance,
        *,
        nodes: tuple[str, ...] | None = None,
    ) -> "GraphState":
        """Wrap an already-directed graph; no ambiguity is claimed or implied."""
        return cls(
            nodes=tuple(nodes) if nodes is not None else tuple(graph.nodes),
            directed_edges=frozenset(graph.edges),
            undirected_pairs=(),
            provenance=provenance,
        )

    # -----------------------------------------------------------------------
    # Derived views
    # -----------------------------------------------------------------------
    @cached_property
    def _frozen_digraph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.directed_edges)
        return nx.freeze(graph)

    def digraph(self) -> nx.DiGraph:
        """The single inflation point; frozen, so call ``.copy()`` to mutate."""
        return self._frozen_digraph

    @property
    def n_undirected_pairs(self) -> int:
        return len(self.undirected_pairs)

    @property
    def is_fully_directed(self) -> bool:
        return not self.undirected_pairs

    def fingerprint(self) -> str:
        """Content hash over nodes and edges, deliberately excluding provenance.

        Re-running discovery with different settings that land on the same graph
        must not invalidate an expensive downstream stage, so two states with
        identical structure share a fingerprint even when their provenance differs.
        """
        payload = json.dumps(
            {
                "nodes": sorted(self.nodes),
                "directed": sorted(self.directed_edges),
                "undirected": sorted(self.undirected_pairs),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_constraints(
        self,
        directed_edges: frozenset[tuple[str, str]],
        undirected_pairs: tuple[tuple[str, str], ...],
        ledger: tuple[ConstraintEntry, ...],
    ) -> "GraphState":
        """Return the recovered graph produced by adjudicating this one."""
        return GraphState(
            nodes=self.nodes,
            directed_edges=directed_edges,
            undirected_pairs=undirected_pairs,
            provenance=GraphProvenance(
                source="recovered",
                algorithm=self.provenance.algorithm,
                params=dict(self.provenance.params),
                n_rows=self.provenance.n_rows,
                constraint_ledger=self.provenance.constraint_ledger + ledger,
                parent_fingerprint=self.fingerprint(),
            ),
        )
