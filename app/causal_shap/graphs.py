"""Shared graph utilities for discovery, complexity scoring, and visualization.

Distances to the outcome come in two flavors: directed hop counts for causal
depth, and undirected (skeleton) hop counts for figure layout, where co-descendant
proxies with no directed path to the outcome still need a finite position.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class PDAG:
    """Partially directed graph as returned by discovery algorithms."""

    nodes: tuple[str, ...]
    directed_edges: frozenset[tuple[str, str]]
    undirected_edges: frozenset[tuple[str, str]]

    def __post_init__(self) -> None:
        known = set(self.nodes)
        for source, target in self.directed_edges | self.undirected_edges:
            if source not in known or target not in known:
                raise ValueError(f"Edge references unknown node: {(source, target)}")
        for source, target in self.undirected_edges:
            if (source, target) != tuple(sorted((source, target))):
                raise ValueError(f"Undirected edge must be sorted: {(source, target)}")

    @property
    def skeleton(self) -> frozenset[tuple[str, str]]:
        directed = {tuple(sorted(edge)) for edge in self.directed_edges}
        return frozenset(directed | set(self.undirected_edges))

    @property
    def fraction_undirected(self) -> float:
        total = len(self.skeleton)
        return len(self.undirected_edges) / total if total else 0.0

    def to_digraph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.directed_edges)
        return graph


def load_edges_csv(path: Path) -> nx.DiGraph:
    """Load a directed graph from a CSV with `from`/`to` columns."""
    edges = pd.read_csv(path)
    missing = {"from", "to"} - set(edges.columns)
    if missing:
        raise ValueError(f"Edge file {path} is missing columns: {sorted(missing)}")
    graph = nx.DiGraph()
    graph.add_edges_from(zip(edges["from"], edges["to"]))
    return graph


def ancestors_of(graph: nx.DiGraph, node: str) -> frozenset[str]:
    if node not in graph:
        raise ValueError(f"Unknown node: {node}")
    return frozenset(nx.ancestors(graph, node))


def directed_distance_to_outcome(graph: nx.DiGraph, outcome: str) -> dict[str, int]:
    """Shortest directed hop count node -> outcome; unreachable nodes are omitted."""
    if outcome not in graph:
        raise ValueError(f"Unknown outcome: {outcome}")
    lengths = nx.shortest_path_length(graph.reverse(copy=False), source=outcome)
    return {node: distance for node, distance in lengths.items() if node != outcome}


def undirected_distance_to_outcome(graph: nx.DiGraph, outcome: str) -> dict[str, int]:
    """Skeleton hop count node -> outcome; finite for co-descendant proxies."""
    if outcome not in graph:
        raise ValueError(f"Unknown outcome: {outcome}")
    lengths = nx.shortest_path_length(graph.to_undirected(as_view=True), source=outcome)
    return {node: distance for node, distance in lengths.items() if node != outcome}
