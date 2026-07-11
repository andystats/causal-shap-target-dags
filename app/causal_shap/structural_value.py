"""Structural intervention propagation and DAG-asymmetric Shapley values.

The value function is E[f(X) | do(X_S=x_S)] under a known structural causal
model. Background observations are first abducted to exogenous draws, then each
coalition intervention is propagated through descendants before model scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NodeSpec:
    name: str
    kind: str
    parents: tuple[str, ...] = ()
    coefficients: tuple[float, ...] = ()
    intercept: float = 0.0
    noise_sd: float = 1.0
    root_probability: float = 0.35

    def __post_init__(self) -> None:
        if self.kind not in {"continuous", "binary"}:
            raise ValueError(f"Unsupported node kind for {self.name}: {self.kind}")
        if len(self.parents) != len(self.coefficients):
            raise ValueError(f"Parent/coefficient mismatch for {self.name}")
        if self.kind == "continuous" and self.noise_sd <= 0:
            raise ValueError(f"Continuous node {self.name} must have positive noise_sd")


@dataclass(frozen=True)
class ExogenousDraws:
    values: Mapping[str, np.ndarray]

    @property
    def n(self) -> int:
        lengths = {len(np.asarray(value)) for value in self.values.values()}
        if len(lengths) != 1:
            raise ValueError("Exogenous arrays do not share a common length")
        return next(iter(lengths))

    def tiled(self, repeats: int) -> "ExogenousDraws":
        return ExogenousDraws(
            {name: np.tile(np.asarray(value), repeats) for name, value in self.values.items()}
        )


@dataclass(frozen=True)
class StructuralShapResult:
    values: pd.DataFrame
    baseline_margin: float
    efficiency_error: np.ndarray
    permutations: int
    background_rows: int


class LinearLogisticSCM:
    def __init__(self, specs: Sequence[NodeSpec]):
        self.specs = {spec.name: spec for spec in specs}
        graph = nx.DiGraph()
        graph.add_nodes_from(self.specs)
        for spec in specs:
            for parent in spec.parents:
                if parent not in self.specs:
                    raise ValueError(f"Unknown parent {parent} for {spec.name}")
                graph.add_edge(parent, spec.name)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Structural model must be acyclic")
        self.graph = graph
        self.order = tuple(nx.topological_sort(graph))

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-value))

    def recover_exogenous(self, data: pd.DataFrame, seed: int) -> ExogenousDraws:
        """Abduct exogenous draws consistent with observed structural data."""
        missing = sorted(set(self.specs) - set(data.columns))
        if missing:
            raise ValueError(f"Structural data are missing nodes: {missing[:5]}")
        rng = np.random.default_rng(seed)
        draws: dict[str, np.ndarray] = {}

        for node in self.order:
            spec = self.specs[node]
            observed = data[node].to_numpy(dtype=float)
            if spec.parents:
                linear = np.full(len(data), spec.intercept, dtype=float)
                for parent, coefficient in zip(spec.parents, spec.coefficients):
                    linear += coefficient * data[parent].to_numpy(dtype=float)
            else:
                linear = np.full(len(data), spec.intercept, dtype=float)

            if spec.kind == "continuous":
                mean = linear if spec.parents else np.zeros(len(data), dtype=float)
                draws[node] = (observed - mean) / spec.noise_sd
            else:
                probability = (
                    self._sigmoid(linear)
                    if spec.parents
                    else np.full(len(data), spec.root_probability, dtype=float)
                )
                probability = np.clip(probability, 1e-9, 1 - 1e-9)
                low = np.where(observed == 1, 0.0, probability)
                high = np.where(observed == 1, probability, 1.0)
                draws[node] = rng.uniform(low=low, high=high)

        return ExogenousDraws(draws)

    def simulate(
        self,
        exogenous: ExogenousDraws,
        interventions: Mapping[str, float | np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        interventions = interventions or {}
        unknown = set(interventions) - set(self.specs)
        if unknown:
            raise ValueError(f"Interventions reference unknown nodes: {sorted(unknown)}")
        n = exogenous.n
        values: dict[str, np.ndarray] = {}

        for node in self.order:
            spec = self.specs[node]
            if node in interventions:
                intervention = np.asarray(interventions[node], dtype=float)
                if intervention.ndim == 0:
                    intervention = np.full(n, float(intervention))
                if len(intervention) != n:
                    raise ValueError(f"Intervention for {node} has length {len(intervention)}, expected {n}")
                values[node] = intervention
                continue

            noise = np.asarray(exogenous.values[node], dtype=float)
            if spec.parents:
                linear = np.full(n, spec.intercept, dtype=float)
                for parent, coefficient in zip(spec.parents, spec.coefficients):
                    linear += coefficient * values[parent]
            else:
                linear = np.full(n, spec.intercept, dtype=float)

            if spec.kind == "continuous":
                mean = linear if spec.parents else np.zeros(n, dtype=float)
                values[node] = mean + spec.noise_sd * noise
            else:
                probability = (
                    self._sigmoid(linear)
                    if spec.parents
                    else np.full(n, spec.root_probability, dtype=float)
                )
                values[node] = (noise < probability).astype(float)

        return values


def random_topological_order(
    nodes: Sequence[str],
    edges: Iterable[tuple[str, str]],
    rng: np.random.Generator,
) -> list[str]:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((source, target) for source, target in edges if source in graph and target in graph)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Feature graph must be acyclic")
    order: list[str] = []
    remaining = graph.copy()
    while remaining:
        available = sorted(node for node, degree in remaining.in_degree() if degree == 0)
        selected = available[int(rng.integers(0, len(available)))]
        order.append(selected)
        remaining.remove_node(selected)
    return order


def compute_structural_asymmetric_shap(
    predict_margin: Callable[[np.ndarray], np.ndarray],
    scm: LinearLogisticSCM,
    evaluation: pd.DataFrame,
    background_exogenous: ExogenousDraws,
    feature_names: Sequence[str],
    feature_edges: Iterable[tuple[str, str]],
    n_permutations: int,
    seed: int,
) -> StructuralShapResult:
    """Compute structural DAG-asymmetric values with batched coalition scoring."""
    feature_names = list(feature_names)
    if list(evaluation.columns) != feature_names:
        raise ValueError("Evaluation columns must exactly match feature_names")
    if not set(feature_names).issubset(scm.specs):
        raise ValueError("Every feature must be a node in the structural model")

    n_eval = len(evaluation)
    n_bg = background_exogenous.n
    batched_exogenous = background_exogenous.tiled(n_eval)
    base_values = scm.simulate(background_exogenous)
    base_matrix = np.column_stack([base_values[name] for name in feature_names])
    baseline_margin = float(np.mean(predict_margin(base_matrix)))

    result = np.zeros((n_eval, len(feature_names)), dtype=float)
    rng = np.random.default_rng(seed)
    evaluation_values = {
        name: evaluation[name].to_numpy(dtype=float) for name in feature_names
    }

    for _ in range(n_permutations):
        ordering = random_topological_order(feature_names, feature_edges, rng)
        interventions: dict[str, np.ndarray] = {}
        previous = np.full(n_eval, baseline_margin, dtype=float)
        for feature in ordering:
            interventions[feature] = np.repeat(evaluation_values[feature], n_bg)
            simulated = scm.simulate(batched_exogenous, interventions)
            matrix = np.column_stack([simulated[name] for name in feature_names])
            current = predict_margin(matrix).reshape(n_eval, n_bg).mean(axis=1)
            result[:, feature_names.index(feature)] += current - previous
            previous = current

    result /= n_permutations
    evaluation_margin = predict_margin(evaluation.to_numpy(dtype=float))
    efficiency_error = result.sum(axis=1) + baseline_margin - evaluation_margin
    return StructuralShapResult(
        values=pd.DataFrame(result, columns=feature_names, index=evaluation.index),
        baseline_margin=baseline_margin,
        efficiency_error=efficiency_error,
        permutations=n_permutations,
        background_rows=n_bg,
    )
