"""Causal-discovery wrappers and structure-quality metrics.

Adapted from the author's Instats workshop (Module 3, "Causal Discovery
Playground"). The live path wraps causal-learn's PC and GES and returns a
named-edge ``PDAG``; DirectLiNGAM (causal-learn) and NOTEARS (gcastle) are
provided for precomputed appendix artifacts only. Every causal-learn call is
kept minimal and funnelled through one adapter (``_general_graph_to_pdag``) so
that 0.x API drift is isolated to a single tested function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .graphs import PDAG


@dataclass(frozen=True)
class DiscoveryResult:
    algorithm: str
    pdag: PDAG
    params: dict[str, object]
    n_rows: int

    @property
    def directed_edges(self) -> frozenset[tuple[str, str]]:
        return self.pdag.directed_edges

    @property
    def skeleton(self) -> frozenset[tuple[str, str]]:
        return self.pdag.skeleton


@dataclass(frozen=True)
class GraphComparison:
    precision: float
    recall: float
    f1: float
    skeleton_precision: float
    skeleton_recall: float
    skeleton_f1: float
    correct: frozenset[tuple[str, str]]
    reversed: frozenset[tuple[str, str]]
    spurious: frozenset[tuple[str, str]]
    missed: frozenset[tuple[str, str]]


# ---------------------------------------------------------------------------
# causal-learn adapter (the API-instability firewall)
# ---------------------------------------------------------------------------
def _general_graph_to_pdag(graph, columns: Sequence[str]) -> PDAG:
    """Convert a causal-learn GeneralGraph to a named-edge PDAG.

    Endpoint encoding: TAIL(-1)-ARROW(1) is a directed edge; TAIL-TAIL is
    undirected; ARROW-ARROW/CIRCLE endpoints (rare for CPDAGs) fall back to the
    undirected skeleton.
    """
    from causallearn.graph.Endpoint import Endpoint

    directed: set[tuple[str, str]] = set()
    undirected: set[tuple[str, str]] = set()
    for edge in graph.get_graph_edges():
        source = edge.get_node1().get_name()
        target = edge.get_node2().get_name()
        end1, end2 = edge.get_endpoint1(), edge.get_endpoint2()
        if end1 == Endpoint.TAIL and end2 == Endpoint.ARROW:
            directed.add((source, target))
        elif end1 == Endpoint.ARROW and end2 == Endpoint.TAIL:
            directed.add((target, source))
        else:
            undirected.add(tuple(sorted((source, target))))
    return PDAG(
        nodes=tuple(columns),
        directed_edges=frozenset(directed),
        undirected_edges=frozenset(undirected),
    )


def _apply_constraints(
    pdag: PDAG,
    forbidden_edges: Sequence[tuple[str, str]],
    required_edges: Sequence[tuple[str, str]],
) -> PDAG:
    """Enforce forbidden/required directed edges on a discovered PDAG.

    Constraints act on the output graph, not the search: a forbidden (a, b)
    removes that directed edge and orients or drops the matching undirected
    edge; a required (a, b) forces the directed edge and clears any conflict.
    """
    forbidden = set(forbidden_edges)
    required = set(required_edges)
    directed = set(pdag.directed_edges)
    undirected = set(pdag.undirected_edges)

    for a, b in forbidden:
        directed.discard((a, b))
        key = tuple(sorted((a, b)))
        if key in undirected:
            undirected.discard(key)
            if (b, a) not in forbidden:  # only one direction forbidden: orient the other way
                directed.add((b, a))

    for a, b in required:
        undirected.discard(tuple(sorted((a, b))))
        directed.discard((b, a))
        directed.add((a, b))

    return PDAG(
        nodes=pdag.nodes,
        directed_edges=frozenset(directed),
        undirected_edges=frozenset(undirected),
    )


def _run_causal_learn(
    algorithm: str,
    df: pd.DataFrame,
    params: dict[str, object],
    forbidden_edges: Sequence[tuple[str, str]],
    required_edges: Sequence[tuple[str, str]],
) -> DiscoveryResult:
    columns = list(df.columns)
    data = df.to_numpy(dtype=float)
    if algorithm == "pc":
        from causallearn.search.ConstraintBased.PC import pc

        cg = pc(
            data,
            alpha=float(params["alpha"]),
            indep_test=str(params.get("indep_test", "fisherz")),
            node_names=columns,
            show_progress=False,
        )
        graph = cg.G
    elif algorithm == "ges":
        from causallearn.search.ScoreBased.GES import ges

        record = ges(data, score_func=str(params.get("score_func", "local_score_BIC")), node_names=columns)
        graph = record["G"]
    elif algorithm == "direct_lingam":
        from causallearn.search.FCMBased import lingam

        model = lingam.DirectLiNGAM()
        model.fit(data)
        directed = {
            (columns[parent], columns[child])
            for child in range(len(columns))
            for parent in range(len(columns))
            if model.adjacency_matrix_[child, parent] != 0.0
        }
        pdag = PDAG(nodes=tuple(columns), directed_edges=frozenset(directed), undirected_edges=frozenset())
        pdag = _apply_constraints(pdag, forbidden_edges, required_edges)
        return DiscoveryResult(algorithm=algorithm, pdag=pdag, params=params, n_rows=len(df))
    else:
        raise ValueError(f"Unknown causal-learn algorithm: {algorithm}")

    pdag = _general_graph_to_pdag(graph, columns)
    pdag = _apply_constraints(pdag, forbidden_edges, required_edges)
    return DiscoveryResult(algorithm=algorithm, pdag=pdag, params=params, n_rows=len(df))


def run_pc(
    df: pd.DataFrame,
    alpha: float = 0.05,
    forbidden_edges: Sequence[tuple[str, str]] = (),
    required_edges: Sequence[tuple[str, str]] = (),
) -> DiscoveryResult:
    """PC (constraint-based) discovery via causal-learn's Fisher-Z test."""
    return _run_causal_learn("pc", df, {"alpha": alpha, "indep_test": "fisherz"}, forbidden_edges, required_edges)


def run_ges(
    df: pd.DataFrame,
    forbidden_edges: Sequence[tuple[str, str]] = (),
    required_edges: Sequence[tuple[str, str]] = (),
) -> DiscoveryResult:
    """GES (score-based, BIC) discovery via causal-learn."""
    return _run_causal_learn("ges", df, {"score_func": "local_score_BIC"}, forbidden_edges, required_edges)


def run_direct_lingam(
    df: pd.DataFrame,
    forbidden_edges: Sequence[tuple[str, str]] = (),
    required_edges: Sequence[tuple[str, str]] = (),
) -> DiscoveryResult:
    """DirectLiNGAM (linear non-Gaussian) discovery. Precompute-only appendix."""
    return _run_causal_learn("direct_lingam", df, {"method": "direct_lingam"}, forbidden_edges, required_edges)


def run_notears(
    df: pd.DataFrame,
    threshold: float = 0.3,
    forbidden_edges: Sequence[tuple[str, str]] = (),
    required_edges: Sequence[tuple[str, str]] = (),
) -> DiscoveryResult:
    """NOTEARS (continuous optimization) via gcastle. Build-scripts only.

    gcastle pulls in torch and is not shipped to the app; this raises a clear
    error when the ``heavy`` extra is absent.
    """
    try:
        from castle.algorithms import Notears
    except ImportError as error:  # pragma: no cover - exercised only without the heavy extra
        raise ImportError(
            "run_notears requires the 'heavy' extra (gcastle + torch); "
            "install with pip install -e '.[heavy]'."
        ) from error

    columns = list(df.columns)
    model = Notears()
    model.learn(df.to_numpy(dtype=float))
    matrix = np.asarray(model.causal_matrix, dtype=float)
    directed = {
        (columns[i], columns[j])
        for i in range(len(columns))
        for j in range(len(columns))
        if abs(matrix[i, j]) >= threshold
    }
    pdag = PDAG(nodes=tuple(columns), directed_edges=frozenset(directed), undirected_edges=frozenset())
    pdag = _apply_constraints(pdag, forbidden_edges, required_edges)
    return DiscoveryResult(algorithm="notears", pdag=pdag, params={"threshold": threshold}, n_rows=len(df))


# ---------------------------------------------------------------------------
# Graph post-processing and quality metrics (ported from Module 3)
# ---------------------------------------------------------------------------
def enforce_dag(
    directed_edges: Sequence[tuple[str, str]],
    nodes: Sequence[str],
) -> tuple[frozenset[tuple[str, str]], list[tuple[str, str]]]:
    """Remove cycle-closing edges until the directed graph is acyclic."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(directed_edges)
    removed: list[tuple[str, str]] = []
    while not nx.is_directed_acyclic_graph(graph):
        try:
            cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:  # pragma: no cover - loop guard
            break
        source, target = cycle[-1][0], cycle[-1][1]
        graph.remove_edge(source, target)
        removed.append((source, target))
    return frozenset(graph.edges), removed


def shrier_platt_check(
    graph: nx.DiGraph,
    exposure: str,
    outcome: str,
    adjustment_set: Sequence[str],
) -> dict[str, object]:
    """Shrier-Platt six-step backdoor-path check (Shrier & Platt, 2008)."""
    nodes = set(graph.nodes())
    if exposure not in nodes or outcome not in nodes:
        return {"valid": False, "message": "Exposure or outcome not in graph", "steps": []}

    adjustment = [z for z in adjustment_set if z in nodes]
    keep = {exposure, outcome, *adjustment}
    keep |= nx.ancestors(graph, exposure) | nx.ancestors(graph, outcome)
    for z in adjustment:
        keep |= nx.ancestors(graph, z)

    kept_edges = [(u, v) for u, v in graph.edges() if u in keep and v in keep]
    backdoor_edges = [(u, v) for u, v in kept_edges if u != exposure]  # step 2: drop arrows out of A

    parents: dict[str, list[str]] = {}
    for u, v in backdoor_edges:
        parents.setdefault(v, []).append(u)
    moral_edges = list(backdoor_edges)
    for co_parents in parents.values():  # step 3: moralize colliders
        for i in range(len(co_parents)):
            for j in range(i + 1, len(co_parents)):
                moral_edges.append((co_parents[i], co_parents[j]))

    undirected = nx.Graph()  # step 4: undirect
    undirected.add_nodes_from(keep - set(adjustment))  # step 5: drop adjustment nodes
    for u, v in moral_edges:
        if u not in adjustment and v not in adjustment:
            undirected.add_edge(u, v)

    connected = (
        exposure in undirected
        and outcome in undirected
        and nx.has_path(undirected, exposure, outcome)
    )
    return {
        "valid": not connected,
        "message": "All backdoor paths blocked" if not connected else "Unblocked backdoor path exists",
        "steps": [
            f"1. Kept {len(keep)} ancestor nodes",
            f"2. Removed arrows out of {exposure}",
            f"3. Moralized {len(moral_edges) - len(backdoor_edges)} co-parent links",
            "4. Made edges undirected",
            f"5. Removed adjustment nodes: {', '.join(adjustment) if adjustment else 'none'}",
            f"6. {exposure}-{outcome}: {'BLOCKED' if not connected else 'CONNECTED (open backdoor)'}",
        ],
    }


def identify_adjustment_sets(
    graph: nx.DiGraph,
    exposure: str,
    outcome: str,
) -> dict[str, dict[str, object]]:
    """Traditional, disjunctive-cause, and minimal adjustment sets, each checked."""
    if exposure not in graph or outcome not in graph:
        return {}

    parents_exp = set(graph.predecessors(exposure))
    parents_out = set(graph.predecessors(outcome))
    descendants_exp = nx.descendants(graph, exposure)

    traditional = (parents_exp & parents_out) - descendants_exp - {exposure, outcome}
    disjunctive = (parents_exp | parents_out) - descendants_exp - {exposure, outcome}
    # The common-parent confounder set is the minimal candidate this heuristic offers;
    # the Shrier-Platt check below reports whether it actually blocks every backdoor path.
    minimal = traditional

    def described(name: str, description: str, variables: set[str]) -> dict[str, object]:
        variables_sorted = sorted(variables)
        check = shrier_platt_check(graph, exposure, outcome, variables_sorted)
        return {
            "name": name,
            "description": description,
            "variables": variables_sorted,
            "valid": check["valid"],
            "steps": check["steps"],
        }

    return {
        "traditional": described(
            "Traditional Confounders",
            "Variables that cause both exposure and outcome",
            traditional,
        ),
        "disjunctive": described(
            "Disjunctive Cause Criterion",
            "Causes of exposure OR outcome (VanderWeele & Shpitser, 2011)",
            disjunctive,
        ),
        "minimal": described(
            "Minimal Candidate Set",
            "Common-parent confounders; the validity flag says whether it blocks all backdoor paths",
            minimal,
        ),
    }


def compare_graphs(
    discovered_edges: Sequence[tuple[str, str]],
    truth_edges: Sequence[tuple[str, str]],
) -> GraphComparison:
    """Directed and skeleton precision/recall/F1 plus edge classification."""
    discovered = set(discovered_edges)
    truth = set(truth_edges)
    truth_reversed = {(t, s) for s, t in truth}
    discovered_reversed = {(t, s) for s, t in discovered}

    correct = discovered & truth
    reversed_edges = discovered & truth_reversed
    spurious = discovered - truth - truth_reversed
    missed = truth - discovered - discovered_reversed

    precision = len(correct) / len(discovered) if discovered else 0.0
    recall = len(correct) / len(truth) if truth else 0.0
    f1 = _harmonic(precision, recall)

    skeleton_discovered = {tuple(sorted(edge)) for edge in discovered}
    skeleton_truth = {tuple(sorted(edge)) for edge in truth}
    skeleton_correct = skeleton_discovered & skeleton_truth
    skeleton_precision = len(skeleton_correct) / len(skeleton_discovered) if skeleton_discovered else 0.0
    skeleton_recall = len(skeleton_correct) / len(skeleton_truth) if skeleton_truth else 0.0

    return GraphComparison(
        precision=precision,
        recall=recall,
        f1=f1,
        skeleton_precision=skeleton_precision,
        skeleton_recall=skeleton_recall,
        skeleton_f1=_harmonic(skeleton_precision, skeleton_recall),
        correct=frozenset(correct),
        reversed=frozenset(reversed_edges),
        spurious=frozenset(spurious),
        missed=frozenset(missed),
    )


def skeleton_f1(
    edges_a: Sequence[tuple[str, str]],
    edges_b: Sequence[tuple[str, str]],
) -> float:
    """F1 between two skeletons (edge direction ignored)."""
    skeleton_a = {tuple(sorted(edge)) for edge in edges_a}
    skeleton_b = {tuple(sorted(edge)) for edge in edges_b}
    if not skeleton_a and not skeleton_b:
        return 1.0
    shared = skeleton_a & skeleton_b
    precision = len(shared) / len(skeleton_a) if skeleton_a else 0.0
    recall = len(shared) / len(skeleton_b) if skeleton_b else 0.0
    return _harmonic(precision, recall)


def pairwise_skeleton_disagreement(results: Sequence[DiscoveryResult]) -> float:
    """Mean 1 - skeleton-F1 across algorithm pairs; 0.0 for a single algorithm."""
    if len(results) < 2:
        return 0.0
    scores: list[float] = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            scores.append(1.0 - skeleton_f1(results[i].skeleton, results[j].skeleton))
    return float(np.mean(scores))


def to_dagitty(directed_edges: Sequence[tuple[str, str]]) -> str:
    """Render directed edges as a dagitty.net `dag { ... }` block."""
    lines = [f"{source} -> {target}" for source, target in sorted(directed_edges)]
    body = "\n".join(f"  {line}" for line in lines)
    return f"dag {{\n{body}\n}}"


def _harmonic(precision: float, recall: float) -> float:
    total = precision + recall
    return 2 * precision * recall / total if total > 0 else 0.0
