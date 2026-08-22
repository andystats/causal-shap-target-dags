"""Evaluation metrics for learned target DAGs.

The battery separates graph concordance (M1-M2) from whether the learned
graph supports the target causal analysis (M3-M5).  All functions operate on
named :class:`networkx.DiGraph` objects; node names must match data columns.

M5 accepts a directed graph plus the node pairs that remain undirected in a
CPDAG.  It only counts unique, acyclic orientations that preserve the fixed
directed edges and introduce no new unshielded colliders.  When the orientation
space is larger than ``cap``, the result is explicitly labelled as capped
Monte Carlo rather than exhaustive enumeration.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd


def get_ancestors(graph: nx.DiGraph, node: str) -> set[str]:
    """Return ancestors of ``node``, or an empty set when it is absent."""

    return set(nx.ancestors(graph, node)) if node in graph else set()


def get_descendants(graph: nx.DiGraph, node: str) -> set[str]:
    """Return descendants of ``node``, or an empty set when it is absent."""

    return set(nx.descendants(graph, node)) if node in graph else set()


def shrier_platt_check(
    graph: nx.DiGraph,
    exposure: str,
    outcome: str,
    adjustment_set: Sequence[str],
) -> dict:
    """Run the six-step graphical backdoor-path check.

    This helper tests path blocking only.  Callers that claim a fully valid
    adjustment set must separately exclude descendants of the exposure; M3
    and M5 do so.
    """

    nodes = set(graph.nodes())
    if exposure not in nodes or outcome not in nodes:
        return {
            "valid": False,
            "message": "Exposure or outcome not in graph",
            "steps": [],
        }

    adjustment = [node for node in adjustment_set if node in nodes]
    keep = {exposure, outcome, *adjustment}
    keep.update(get_ancestors(graph, exposure))
    keep.update(get_ancestors(graph, outcome))
    for node in adjustment:
        keep.update(get_ancestors(graph, node))

    ancestor_edges = [
        (source, target)
        for source, target in graph.edges()
        if source in keep and target in keep
    ]
    without_exposure_arrows = [
        (source, target)
        for source, target in ancestor_edges
        if source != exposure
    ]

    parent_lists: dict[str, list[str]] = {}
    for source, target in without_exposure_arrows:
        parent_lists.setdefault(target, []).append(source)

    moralized = list(without_exposure_arrows)
    for parents in parent_lists.values():
        for first, second in itertools.combinations(parents, 2):
            moralized.extend([(first, second), (second, first)])

    undirected = {frozenset((source, target)) for source, target in moralized}
    adjustment_nodes = set(adjustment)
    filtered = [
        tuple(edge)
        for edge in undirected
        if edge.isdisjoint(adjustment_nodes) and len(edge) == 2
    ]

    remaining = keep - adjustment_nodes
    if exposure not in remaining or outcome not in remaining:
        blocked = True
    else:
        moral_graph = nx.Graph()
        moral_graph.add_nodes_from(remaining)
        moral_graph.add_edges_from(
            (source, target)
            for source, target in filtered
            if source in remaining and target in remaining
        )
        blocked = not nx.has_path(moral_graph, exposure, outcome)

    return {
        "valid": blocked,
        "message": (
            "All backdoor paths blocked"
            if blocked
            else "Unblocked backdoor path exists"
        ),
        "steps": [
            f"1. Kept {len(keep)} ancestor nodes",
            f"2. Removed arrows out of {exposure}",
            "3. Moralized co-parents",
            f"4. {len(undirected)} undirected edges",
            f"5. Removed: {', '.join(adjustment) if adjustment else 'none'}",
            f"6. {exposure}-{outcome}: {'BLOCKED' if blocked else 'CONNECTED'}",
        ],
    }


def _adjustment_valid(
    graph: nx.DiGraph,
    exposure: str,
    outcome: str,
    adjustment_set: Sequence[str],
) -> tuple[bool, list[str]]:
    """Return full adjustment validity and descendant offenders."""

    backdoor = shrier_platt_check(graph, exposure, outcome, adjustment_set)
    offenders = sorted(set(adjustment_set) & get_descendants(graph, exposure))
    return bool(backdoor["valid"] and not offenders), offenders


def minimal_adjustment_set(
    graph: nx.DiGraph, exposure: str, outcome: str
) -> dict:
    """Derive one deterministic minimal sufficient adjustment set."""

    if exposure not in graph or outcome not in graph:
        return {"set": [], "valid": False}

    pool = (
        (get_ancestors(graph, exposure) | get_ancestors(graph, outcome))
        - get_descendants(graph, exposure)
        - {exposure, outcome}
    )
    current = sorted(pool)
    if not shrier_platt_check(graph, exposure, outcome, current)["valid"]:
        if shrier_platt_check(graph, exposure, outcome, [])["valid"]:
            return {"set": [], "valid": True}
        return {"set": current, "valid": False}

    changed = True
    while changed:
        changed = False
        for node in list(current):
            trial = [candidate for candidate in current if candidate != node]
            if shrier_platt_check(graph, exposure, outcome, trial)["valid"]:
                current = trial
                changed = True
    return {"set": sorted(current), "valid": True}


def _pair_state(graph: nx.DiGraph, first: str, second: str) -> str:
    """Return the directed state of one unordered node pair."""

    forward = graph.has_edge(first, second)
    backward = graph.has_edge(second, first)
    if forward and backward:
        return "--"
    if forward:
        return "->"
    if backward:
        return "<-"
    return ""


def m1_concordance(graph_learned: nx.DiGraph, graph_true: nx.DiGraph) -> dict:
    """Compute directed/skeleton precision, recall, F1, and SHD."""

    learned = set(graph_learned.edges())
    truth = set(graph_true.edges())
    true_positives = len(learned & truth)
    precision = true_positives / len(learned) if learned else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    learned_skeleton = {frozenset(edge) for edge in learned}
    truth_skeleton = {frozenset(edge) for edge in truth}
    skeleton_true_positives = len(learned_skeleton & truth_skeleton)
    skeleton_precision = (
        skeleton_true_positives / len(learned_skeleton)
        if learned_skeleton
        else 0.0
    )
    skeleton_recall = (
        skeleton_true_positives / len(truth_skeleton)
        if truth_skeleton
        else 0.0
    )
    skeleton_f1 = (
        2
        * skeleton_precision
        * skeleton_recall
        / (skeleton_precision + skeleton_recall)
        if skeleton_precision + skeleton_recall
        else 0.0
    )

    nodes = sorted(set(graph_learned.nodes()) | set(graph_true.nodes()))
    shd = sum(
        _pair_state(graph_learned, first, second)
        != _pair_state(graph_true, first, second)
        for first, second in itertools.combinations(nodes, 2)
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "skeleton_precision": skeleton_precision,
        "skeleton_recall": skeleton_recall,
        "skeleton_f1": skeleton_f1,
        "shd": shd,
        "n_learned": len(learned),
        "n_true": len(truth),
        "tp": true_positives,
    }


def target_pathway_edges(
    graph: nx.DiGraph, exposure: str, outcome: str
) -> set[tuple[str, str]]:
    """Return edges lying on at least one directed exposure-to-outcome path."""

    if exposure not in graph or outcome not in graph:
        return set()
    downstream = get_descendants(graph, exposure) | {exposure}
    upstream = get_ancestors(graph, outcome) | {outcome}
    pathway_nodes = downstream & upstream
    return {
        (source, target)
        for source, target in graph.edges()
        if source in pathway_nodes and target in pathway_nodes
    }


def m2_target_pathway(
    graph_learned: nx.DiGraph,
    graph_true: nx.DiGraph,
    exposure: str,
    outcome: str,
) -> dict:
    """Classify learned edges on the true target pathway."""

    true_path = target_pathway_edges(graph_true, exposure, outcome)
    learned_path = target_pathway_edges(graph_learned, exposure, outcome)
    statuses: dict[tuple[str, str], str] = {}
    n_correct = n_reversed = n_missed = 0
    for source, target in sorted(true_path):
        if graph_learned.has_edge(source, target):
            statuses[(source, target)] = "correct"
            n_correct += 1
        elif graph_learned.has_edge(target, source):
            statuses[(source, target)] = "reversed"
            n_reversed += 1
        else:
            statuses[(source, target)] = "missed"
            n_missed += 1

    true_pairs = {frozenset(edge) for edge in true_path}
    spurious = sorted(
        edge for edge in learned_path if frozenset(edge) not in true_pairs
    )
    n_pathway = len(true_path)
    return {
        "edges": {
            f"{source}->{target}": status
            for (source, target), status in statuses.items()
        },
        "n_true_pathway": n_pathway,
        "n_correct": n_correct,
        "n_reversed": n_reversed,
        "n_missed": n_missed,
        "spurious": [f"{source}->{target}" for source, target in spurious],
        "score": n_correct / n_pathway if n_pathway else float("nan"),
    }


def m3_sufficiency_transfer(
    graph_learned: nx.DiGraph,
    graph_true: nx.DiGraph,
    exposure: str,
    outcome: str,
) -> dict:
    """Derive Z' in the learned graph and test full validity in the truth."""

    learned_set = minimal_adjustment_set(graph_learned, exposure, outcome)
    true_set = minimal_adjustment_set(graph_true, exposure, outcome)
    verdict = shrier_platt_check(
        graph_true, exposure, outcome, learned_set["set"]
    )
    valid, offenders = _adjustment_valid(
        graph_true, exposure, outcome, learned_set["set"]
    )
    learned_nodes = set(learned_set["set"])
    true_nodes = set(true_set["set"])
    union = learned_nodes | true_nodes
    jaccard = len(learned_nodes & true_nodes) / len(union) if union else 1.0
    return {
        "z_learned": learned_set["set"],
        "z_learned_valid_in_learned": learned_set["valid"],
        "z_true": true_set["set"],
        "valid_in_true": valid,
        "backdoor_blocked": verdict["valid"],
        "descendant_offenders": offenders,
        "verdict_steps": verdict["steps"]
        + [
            f"7. Z' descendants of {exposure} in true G: "
            f"{', '.join(offenders) if offenders else 'none'}"
        ],
        "jaccard": jaccard,
        "excess_size": len(learned_nodes) - len(true_nodes),
    }


def m4_parameter_fidelity(
    data: pd.DataFrame,
    exposure: str,
    outcome: str,
    z_learned: Sequence[str],
    true_effect: float | None = None,
    z_true: Sequence[str] | None = None,
) -> dict:
    """Estimate the exposure coefficient under learned and true adjustment."""

    def ols_beta(adjustment: Sequence[str]) -> tuple[float, float]:
        columns = [exposure] + [
            node
            for node in adjustment
            if node in data.columns and node not in (exposure, outcome)
        ]
        design = np.column_stack(
            [np.ones(len(data))]
            + [data[column].to_numpy(float) for column in columns]
        )
        response = data[outcome].to_numpy(float)
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        residuals = response - design @ coefficients
        degrees_freedom = max(len(response) - design.shape[1], 1)
        sigma_squared = float(residuals @ residuals) / degrees_freedom
        covariance = sigma_squared * np.linalg.pinv(design.T @ design)
        return float(coefficients[1]), float(np.sqrt(covariance[1, 1]))

    estimate, standard_error = ols_beta(z_learned)
    result = {
        "estimate": estimate,
        "se": standard_error,
        "adjustment_set": sorted(z_learned),
    }
    if z_true is not None:
        true_set_estimate, true_set_se = ols_beta(z_true)
        result["estimate_under_z_true"] = true_set_estimate
        result["se_under_z_true"] = true_set_se
    if true_effect is not None:
        result["true_effect"] = float(true_effect)
        result["bias"] = estimate - float(true_effect)
        result["relative_bias"] = (
            (estimate - float(true_effect)) / float(true_effect)
            if true_effect
            else float("nan")
        )
    return result


def _normalize_undirected_pairs(
    undirected_pairs: Iterable[Sequence[str]],
) -> list[tuple[str, str]]:
    """Deduplicate undirected pairs while retaining deterministic ordering."""

    unique: dict[frozenset[str], tuple[str, str]] = {}
    for pair in undirected_pairs:
        if len(pair) != 2:
            raise ValueError("Every undirected pair must contain two nodes")
        first, second = pair
        if first == second:
            raise ValueError("An undirected pair cannot be a self-loop")
        key = frozenset((first, second))
        unique.setdefault(key, (first, second))
    return sorted(unique.values(), key=lambda pair: (repr(pair[0]), repr(pair[1])))


def _skeleton_pairs(
    graph: nx.DiGraph, undirected_pairs: Sequence[tuple[str, str]]
) -> set[frozenset[str]]:
    skeleton = {frozenset(edge) for edge in graph.edges()}
    skeleton.update(frozenset(pair) for pair in undirected_pairs)
    return skeleton


def _unshielded_colliders(
    graph: nx.DiGraph, skeleton: set[frozenset[str]]
) -> set[tuple[str, str, str]]:
    """Return canonical parent-child-parent triples A->B<-C with A,C apart."""

    colliders: set[tuple[str, str, str]] = set()
    for child in graph.nodes():
        parents = sorted(graph.predecessors(child), key=repr)
        for first, second in itertools.combinations(parents, 2):
            if frozenset((first, second)) not in skeleton:
                outer = sorted((first, second), key=repr)
                colliders.add((outer[0], child, outer[1]))
    return colliders


def _orientation_draws(
    n_pairs: int, cap: int, seed: int
) -> tuple[list[tuple[int, ...]], str, int]:
    """Return unique orientation vectors, mode, and full-space size."""

    if cap < 1:
        raise ValueError("cap must be positive")
    n_possible = 1 << n_pairs
    if n_possible <= cap:
        return list(itertools.product((0, 1), repeat=n_pairs)), "exhaustive", n_possible

    rng = np.random.default_rng(seed)
    draws: set[tuple[int, ...]] = set()
    while len(draws) < cap:
        draws.add(tuple(int(bit) for bit in rng.integers(0, 2, n_pairs)))
    return sorted(draws), "capped_monte_carlo", n_possible


def _consistent_extensions(
    graph_learned: nx.DiGraph,
    undirected_pairs: Sequence[tuple[str, str]],
    orientation_draws: Iterable[tuple[int, ...]],
) -> list[nx.DiGraph]:
    """Construct unique DAG extensions compatible with the supplied CPDAG."""

    base = graph_learned.copy()
    for first, second in undirected_pairs:
        base.remove_edge(first, second) if base.has_edge(first, second) else None
        base.remove_edge(second, first) if base.has_edge(second, first) else None
        base.add_nodes_from((first, second))

    skeleton = _skeleton_pairs(base, undirected_pairs)
    allowed_colliders = _unshielded_colliders(base, skeleton)
    extensions: dict[frozenset[tuple[str, str]], nx.DiGraph] = {}

    for bits in orientation_draws:
        candidate = base.copy()
        for (first, second), bit in zip(undirected_pairs, bits, strict=True):
            candidate.add_edge(first, second) if bit else candidate.add_edge(second, first)
        if not nx.is_directed_acyclic_graph(candidate):
            continue
        candidate_colliders = _unshielded_colliders(candidate, skeleton)
        if candidate_colliders - allowed_colliders:
            continue
        key = frozenset(candidate.edges())
        extensions.setdefault(key, candidate)
    return list(extensions.values())


def m5_identification_honesty(
    graph_learned: nx.DiGraph,
    undirected_pairs: Iterable[Sequence[str]],
    exposure: str,
    outcome: str,
    z_learned: Sequence[str],
    cap: int = 256,
    seed: int = 0,
) -> dict:
    """Evaluate adjustment validity across consistent CPDAG extensions.

    Exhaustive enumeration is used when ``2**k <= cap``.  Otherwise ``cap``
    unique orientations are sampled without duplicate bit patterns; only
    consistent extensions are retained.  The reported fraction is therefore
    exact in exhaustive mode and a capped Monte Carlo diagnostic otherwise.
    """

    pairs = _normalize_undirected_pairs(undirected_pairs)
    draws, mode, n_possible = _orientation_draws(len(pairs), cap, seed)
    extensions = _consistent_extensions(graph_learned, pairs, draws)
    n_valid = sum(
        _adjustment_valid(extension, exposure, outcome, z_learned)[0]
        for extension in extensions
    )
    n_extensions = len(extensions)
    return {
        "fraction_valid": (
            n_valid / n_extensions if n_extensions else float("nan")
        ),
        "n_extensions": n_extensions,
        "n_valid": n_valid,
        "capped": mode != "exhaustive",
        "mode": mode,
        "n_undirected": len(pairs),
        "n_possible_orientations": n_possible,
        "n_orientations_evaluated": len(draws),
        "n_inconsistent_orientations": len(draws) - n_extensions,
    }


def evaluate_battery(
    graph_learned: nx.DiGraph,
    graph_true: nx.DiGraph,
    exposure: str,
    outcome: str,
    data: pd.DataFrame | None = None,
    true_effect: float | None = None,
    undirected_pairs: Iterable[Sequence[str]] | None = None,
) -> dict:
    """Run M1-M5, with M4 enabled when data are supplied."""

    m1 = m1_concordance(graph_learned, graph_true)
    m2 = m2_target_pathway(graph_learned, graph_true, exposure, outcome)
    m3 = m3_sufficiency_transfer(graph_learned, graph_true, exposure, outcome)
    result = {"m1": m1, "m2": m2, "m3": m3}
    if data is not None:
        result["m4"] = m4_parameter_fidelity(
            data,
            exposure,
            outcome,
            m3["z_learned"],
            true_effect=true_effect,
            z_true=m3["z_true"],
        )
    result["m5"] = m5_identification_honesty(
        graph_learned,
        undirected_pairs or [],
        exposure,
        outcome,
        m3["z_learned"],
    )
    return result
