"""Small, deterministic helpers for the pedagogic mediator/proxy stress test.

This module is intentionally separate from the publication-facing structural
engine.  It reproduces the fast demonstration estimator used by the app; it is
not the estimand used for the NASA scientific analysis.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau
from sklearn.ensemble import GradientBoostingRegressor


TRUE_TOTAL_EFFECTS = {
    "BaselineSeverity": 3.52666965625,
    "Inflammation": 2.766340625,
    "ChronicBurden": 1.96463521875,
    "TreatmentIntensity": -1.81990875,
    "PerfusionDeficit": 1.7025625,
    "Lactate": 0.895,
    "RenalStress": 0.8425,
    "PracticeStyle": -0.8189589375,
    "OxygenDeficit": 0.77375,
    "Coagulation": 0.6775,
    "EndOrganStress": 0.65,
    "SocialRisk": 0.45997390625,
    "Age": 0.308365875,
    "CompositeScoreProxy": 0.0,
    "MonitoringProxy": 0.0,
    "RescueProxy": 0.0,
    "ShockIndexProxy": 0.0,
    "VasopressorProxy": 0.0,
}


def dag_from_edges_csv(path: Path, nodes: list[str]) -> nx.DiGraph:
    edges = pd.read_csv(path)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges[["from", "to"]].itertuples(index=False, name=None))
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Pedagogic edge list must be acyclic")
    return graph


def random_topological_sort(graph: nx.DiGraph) -> list[str]:
    remaining = graph.copy()
    order: list[str] = []
    while remaining:
        available = sorted(node for node, degree in remaining.in_degree() if degree == 0)
        chosen = available[np.random.randint(len(available))]
        order.append(chosen)
        remaining.remove_node(chosen)
    return order


def fit_conditionals(
    graph: nx.DiGraph, data: pd.DataFrame, outcome: str
) -> dict[str, tuple[GradientBoostingRegressor, list[str]]]:
    fitted: dict[str, tuple[GradientBoostingRegressor, list[str]]] = {}
    for node in graph.nodes:
        parents = [p for p in graph.predecessors(node) if p in data.columns and p != outcome]
        if node == outcome or node not in data.columns or not parents:
            continue
        model = GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=42)
        model.fit(data[parents], data[node])
        fitted[node] = (model, parents)
    return fitted


def interventional_prediction(
    model,
    instance: pd.Series,
    coalition: set[str],
    features: list[str],
    conditionals: dict[str, tuple[GradientBoostingRegressor, list[str]]],
    graph: nx.DiGraph,
    background: pd.DataFrame,
    n_background: int,
) -> float:
    predictions: list[float] = []
    topological_features = [node for node in nx.topological_sort(graph) if node in features]
    for _ in range(n_background):
        sample = background.iloc[np.random.randint(len(background))].copy()
        for node in topological_features:
            if node in coalition:
                sample[node] = instance[node]
            elif node in conditionals:
                conditional, parents = conditionals[node]
                parent_frame = pd.DataFrame([[sample[p] for p in parents]], columns=parents)
                sample[node] = conditional.predict(parent_frame)[0]
        predictions.append(float(model.predict(pd.DataFrame([sample[features]]))[0]))
    return float(np.mean(predictions))


def compute_causal_shap_fast(
    model,
    data: pd.DataFrame,
    graph: nx.DiGraph,
    features: list[str],
    outcome: str,
    *,
    n_perms: int,
    n_background: int,
    n_instances: int,
) -> pd.DataFrame:
    feature_graph = graph.subgraph(features).copy()
    conditionals = fit_conditionals(graph, data, outcome)
    background = data[features]
    sample_indices = np.random.choice(len(data), min(len(data), n_instances), replace=False)
    values = np.zeros((len(sample_indices), len(features)))
    feature_index = {feature: index for index, feature in enumerate(features)}

    for row_index, source_index in enumerate(sample_indices):
        instance = data.iloc[source_index]
        for _ in range(n_perms):
            permutation = random_topological_sort(feature_graph)
            permutation.extend(feature for feature in features if feature not in permutation)
            coalition: set[str] = set()
            previous = interventional_prediction(
                model, instance, coalition, features, conditionals, graph, background, n_background
            )
            for feature in permutation:
                coalition.add(feature)
                current = interventional_prediction(
                    model, instance, coalition, features, conditionals, graph, background, n_background
                )
                values[row_index, feature_index[feature]] += current - previous
                previous = current

    return pd.DataFrame(values / n_perms, columns=features)


def compute_standard_shap(
    model, data: pd.DataFrame, features: list[str], n_background: int
) -> pd.DataFrame:
    feature_data = data[features]
    background = feature_data.sample(min(n_background, len(feature_data)), random_state=42)
    evaluation = feature_data.sample(min(100, len(feature_data)), random_state=42)
    explanation = shap.Explainer(model, background)(evaluation)
    return pd.DataFrame(explanation.values, columns=features)


def tau_vs_truth(values: pd.DataFrame) -> float:
    importance = values.abs().mean()
    truth = pd.Series({feature: abs(TRUE_TOTAL_EFFECTS.get(feature, 0.0)) for feature in values})
    return float(kendalltau(importance.rank(ascending=False), truth.rank(ascending=False)).statistic)


def attribution_shift(
    standard: pd.DataFrame,
    causal: pd.DataFrame,
    mediators: list[str],
    roots: list[str],
) -> tuple[float, float]:
    standard_importance = standard.abs().mean()
    causal_importance = causal.abs().mean()
    proxy_inflation = float(
        standard_importance[mediators].sum() / causal_importance[mediators].sum()
    )
    root_boost = float(causal_importance[roots].sum() / standard_importance[roots].sum())
    return proxy_inflation, root_boost
