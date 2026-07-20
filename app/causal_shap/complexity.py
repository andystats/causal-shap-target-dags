"""Problem-complexity scoring for the workflow's Rung 2.

A complexity score summarizes how much causal care a problem demands before its
attributions can be trusted. This module ships one PROVISIONAL score (PSCI v0)
behind a small registry seam; the authors' final score (private) is dropped in
later by registering another ``ComplexityScore`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import networkx as nx
import numpy as np

from .discovery import identify_adjustment_sets
from .graphs import PDAG, undirected_distance_to_outcome

Band = str  # "low" | "moderate" | "high"


@dataclass(frozen=True)
class Subscore:
    value: float
    rationale: str
    available: bool = True


@dataclass(frozen=True)
class ComplexityInputs:
    graph: nx.DiGraph | PDAG
    outcome: str
    disagreement: float | None = None

    def directed_graph(self) -> nx.DiGraph:
        graph = self.graph.to_digraph() if isinstance(self.graph, PDAG) else self.graph
        if self.outcome not in graph:
            raise ValueError(f"Outcome {self.outcome} is not a node of the graph")
        return graph

    def fraction_undirected(self) -> float:
        return self.graph.fraction_undirected if isinstance(self.graph, PDAG) else 0.0


@dataclass(frozen=True)
class ComplexityReport:
    score_name: str
    score_version: str
    subscores: Mapping[str, Subscore]
    total: float
    band: Band
    recommendations: tuple[str, ...]
    provisional: bool = False


@runtime_checkable
class ComplexityScore(Protocol):
    name: str
    version: str

    def compute(self, inputs: ComplexityInputs) -> ComplexityReport: ...


# Normalization references (documented, deliberately simple).
_SIZE_REFERENCE_NODES = 50.0  # ~NASA scale maps to 1.0
_DEPTH_REFERENCE_HOPS = 6.0  # NASA's deepest directed chain
_ADJUSTMENT_REFERENCE = 5.0

_WEIGHTS = {
    "size": 0.15,
    "density": 0.15,
    "depth": 0.15,
    "ambiguity": 0.20,
    "disagreement": 0.20,
    "adjustment_burden": 0.15,
}

_BAND_RECOMMENDATIONS = {
    "low": (
        "Structure is simple; ordinary and structural attributions likely agree.",
        "A single discovery run plus a quick DAG sketch is enough.",
    ),
    "moderate": (
        "Discovery disagreement and depth matter; compare algorithms before trusting edges.",
        "Prefer structural Causal SHAP over ordering-only variants here.",
    ),
    "high": (
        "Attribution is fragile; small graph changes move the answer.",
        "Iterate: refine the DAG, validate with simulation, and report uncertainty.",
    ),
}


def _band(total: float) -> Band:
    if total < 33.0:
        return "low"
    if total < 66.0:
        return "moderate"
    return "high"


class ProvisionalStructuralComplexityIndex:
    """PSCI v0 — a provisional structural complexity index.

    PROVISIONAL: a transparent placeholder combining graph size, density, depth
    to outcome, CPDAG ambiguity, cross-algorithm disagreement, and adjustment
    burden. To be replaced by the authors' final score before submission.
    """

    name = "PSCI"
    version = "0"

    def compute(self, inputs: ComplexityInputs) -> ComplexityReport:
        graph = inputs.directed_graph()
        outcome = inputs.outcome
        nodes = list(graph.nodes())
        n_nodes = len(nodes)

        subscores: dict[str, Subscore] = {}

        size = min(np.log(max(n_nodes, 1)) / np.log(_SIZE_REFERENCE_NODES), 1.0)
        subscores["size"] = Subscore(size, f"{n_nodes} nodes")

        max_skeleton = n_nodes * (n_nodes - 1) / 2
        skeleton_edges = len({tuple(sorted(edge)) for edge in graph.edges()})
        density = skeleton_edges / max_skeleton if max_skeleton else 0.0
        subscores["density"] = Subscore(min(density, 1.0), f"{skeleton_edges} skeleton edges")

        distances = undirected_distance_to_outcome(graph, outcome)
        mean_depth = float(np.mean(list(distances.values()))) if distances else 0.0
        subscores["depth"] = Subscore(
            min(mean_depth / _DEPTH_REFERENCE_HOPS, 1.0),
            f"mean {mean_depth:.1f} hops to {outcome}",
        )

        ambiguity = inputs.fraction_undirected()
        subscores["ambiguity"] = Subscore(ambiguity, f"{100 * ambiguity:.0f}% of edges undirected")

        if inputs.disagreement is None:
            subscores["disagreement"] = Subscore(
                0.0, "n/a (single-graph scoring)", available=False
            )
        else:
            subscores["disagreement"] = Subscore(
                min(max(inputs.disagreement, 0.0), 1.0),
                f"{100 * inputs.disagreement:.0f}% cross-algorithm disagreement",
            )

        burden = _adjustment_burden(graph, outcome)
        subscores["adjustment_burden"] = Subscore(
            min(burden / _ADJUSTMENT_REFERENCE, 1.0),
            f"mean adjustment set of {burden:.1f} variables",
        )

        total = _weighted_total(subscores)
        band = _band(total)
        return ComplexityReport(
            score_name=self.name,
            score_version=self.version,
            subscores=subscores,
            total=total,
            band=band,
            recommendations=_BAND_RECOMMENDATIONS[band],
            provisional=True,
        )


def _adjustment_burden(graph: nx.DiGraph, outcome: str) -> float:
    """Mean minimal-adjustment-set size across ancestor exposures of the outcome."""
    exposures = [node for node in nx.ancestors(graph, outcome) if node != outcome]
    if not exposures:
        return 0.0
    sizes: list[int] = []
    for exposure in exposures:
        sets = identify_adjustment_sets(graph, exposure, outcome)
        minimal = sets.get("minimal", {}).get("variables", [])
        sizes.append(len(minimal))
    return float(np.mean(sizes)) if sizes else 0.0


def _weighted_total(subscores: Mapping[str, Subscore]) -> float:
    """Weighted average over available subscores, renormalized for degraded mode."""
    active = {name: sub for name, sub in subscores.items() if sub.available}
    weight_sum = sum(_WEIGHTS[name] for name in active)
    if weight_sum == 0:
        return 0.0
    weighted = sum(_WEIGHTS[name] * active[name].value for name in active)
    return 100.0 * weighted / weight_sum


REGISTRY: dict[str, ComplexityScore] = {}


def register_score(score: ComplexityScore) -> None:
    REGISTRY[score.name] = score


def get_score(name: str) -> ComplexityScore:
    if name not in REGISTRY:
        raise KeyError(f"Unknown complexity score: {name}")
    return REGISTRY[name]


register_score(ProvisionalStructuralComplexityIndex())
