"""Pure compute behind every hub stage: no Shiny, safe on a worker thread.

Each ``run_*`` function takes plain values, returns plain values, and raises on
failure; the app wraps them in extended tasks and turns exceptions into error
cards. Two rules inherited from the program's review history are enforced here
rather than in the UI: the attribution model is always fit on exactly the
feature tuple being attributed (a stage-2 model fed a stage-6 subset raises a
sklearn feature mismatch), and the policy stage always abducts from the full
SCM frame, never from a feature subset.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from causal_shap.action_costs import ActionSpec, CostModel
from causal_shap.calibrate import CalibratedSCM, fit_linear_logistic_scm
from causal_shap.discovery import identify_adjustment_sets, run_ges, run_pc
from causal_shap.evaluation import m1_concordance, m3_sufficiency_transfer
from causal_shap.graph_state import GraphProvenance, GraphState
from causal_shap.policy import ActionRanking, InterventionProblem, abduct, rank_actions
from causal_shap.seeds import SEED_ACTION_ABDUCTION, SEED_HUB_DEMO
from causal_shap.structural_value import compute_structural_asymmetric_shap
from workbench.attribution import (
    _causal_shap_engine,
    compare_shap_rankings,
    compute_standard_shap,
    fit_conditional_models,
    mean_abs_shap,
    prediction_callable,
)

INK = "#111111"
AMBER = "#b45309"
MUTED = "#94a3b8"


# ---------------------------------------------------------------------------
# Outcome model: the ladder, factored once
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelFit:
    model: object
    task: str                    # "binary" | "continuous"
    features: tuple[str, ...]
    stat_name: str               # "holdout AUC" | "holdout R2"
    stat_value: float
    model_type: str
    seed: int


def fit_outcome_model(
    data: pd.DataFrame,
    features: Sequence[str],
    outcome: str,
    *,
    model_type: str = "gbm",
    seed: int = SEED_HUB_DEMO,
) -> ModelFit:
    """Fit the predictor and report an honest held-out statistic.

    AUC for a binary outcome, R2 for a continuous one — a continuous outcome
    has no AUC, and pretending otherwise was a documented plan defect.
    """
    from sklearn.metrics import r2_score, roc_auc_score

    frame = data[list(features) + [outcome]].dropna()
    X, y = frame[list(features)], frame[outcome]
    is_binary = len(y.unique()) <= 2
    model_class, kwargs = _model_class(model_type, is_binary)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    cut = int(len(frame) * 0.75)
    train, hold = order[:cut], order[cut:]

    probe = model_class(**kwargs).fit(X.iloc[train], y.iloc[train])
    if is_binary:
        held = y.iloc[hold]
        if len(held.unique()) == 2:
            stat = float(roc_auc_score(held, probe.predict_proba(X.iloc[hold])[:, 1]))
        else:
            stat = float("nan")
        stat_name = "holdout AUC"
    else:
        stat = float(r2_score(y.iloc[hold], probe.predict(X.iloc[hold])))
        stat_name = "holdout R2"

    model = model_class(**kwargs).fit(X, y)
    return ModelFit(
        model=model,
        task="binary" if is_binary else "continuous",
        features=tuple(features),
        stat_name=stat_name,
        stat_value=stat,
        model_type=model_type,
        seed=seed,
    )


def _model_class(model_type: str, is_binary: bool):
    if model_type == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        cls = GradientBoostingClassifier if is_binary else GradientBoostingRegressor
        return cls, dict(n_estimators=100, max_depth=4, random_state=42)
    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if is_binary else RandomForestRegressor
        return cls, dict(n_estimators=100, max_depth=6, random_state=42)
    if model_type == "linear":
        from sklearn.linear_model import LinearRegression, LogisticRegression

        if is_binary:
            return LogisticRegression, dict(max_iter=1000)
        return LinearRegression, {}
    raise ValueError(f"Unknown model type: {model_type!r}")


# ---------------------------------------------------------------------------
# Stage 2: naive SHAP benchmark
# ---------------------------------------------------------------------------
def run_naive_shap(
    data: pd.DataFrame,
    features: Sequence[str],
    outcome: str,
    *,
    model_type: str,
    n_background: int = 100,
    seed: int = SEED_HUB_DEMO,
) -> dict[str, object]:
    fit = fit_outcome_model(data, features, outcome, model_type=model_type, seed=seed)
    shap_df = compute_standard_shap(fit.model, data, list(features), n_background=n_background)
    importance = mean_abs_shap(shap_df)
    total = float(importance.sum())
    shares = {name: 100.0 * value / total for name, value in importance.items()} if total else {}
    return {
        "fit": fit,
        "shap_df": shap_df,
        "importance": importance.to_dict(),
        "shares": shares,
        "plot": bar_chart(importance.to_dict(), "Naive SHAP — what the model listened to", AMBER),
    }


# ---------------------------------------------------------------------------
# Stage 3: discovery
# ---------------------------------------------------------------------------
def run_discovery(
    data: pd.DataFrame,
    features: Sequence[str],
    outcome: str,
    *,
    algorithm: str,
    alpha: float,
    truth: GraphState | None,
) -> dict[str, object]:
    columns = list(dict.fromkeys(list(features) + [outcome]))
    frame = data[columns].dropna()
    if algorithm == "pc":
        result = run_pc(frame, alpha=alpha)
    elif algorithm == "ges":
        result = run_ges(frame)
    else:
        raise ValueError(f"Unknown discovery algorithm: {algorithm!r}")

    state = GraphState.from_pdag(
        result.pdag,
        GraphProvenance(
            source="discovered",
            algorithm=result.algorithm,
            params=dict(result.params),
            n_rows=result.n_rows,
        ),
    )
    payload: dict[str, object] = {"graph": state}
    if truth is not None:
        truth_view = _induced(truth, state.nodes)
        payload["m1"] = m1_concordance(state.digraph(), truth_view)
    return payload


def _induced(truth: GraphState, nodes: Sequence[str]) -> nx.DiGraph:
    keep = set(nodes)
    view = nx.DiGraph()
    view.add_nodes_from(name for name in truth.nodes if name in keep)
    view.add_edges_from(
        (a, b) for a, b in truth.directed_edges if a in keep and b in keep
    )
    return view


# ---------------------------------------------------------------------------
# Stage 4: depth-detector flags (module may be absent on a clean clone)
# ---------------------------------------------------------------------------
def run_flags(
    flags_id: str,
    outcome: str,
    feature_names: Sequence[str],
    *,
    block_root: Path | None,
    preference: str | None = None,
) -> dict[str, object]:
    import importlib.util

    if importlib.util.find_spec("causal_shap.node_flags") is None:
        return {
            "status": "module_missing",
            "message": "The depth-detector module is not present in this checkout.",
            "records": [],
            "halos": {},
            "provenance": "",
        }

    from causal_shap.node_flags import NodeFlagRequest, select_flag_provider

    provider = select_flag_provider(preference, block_root=block_root)
    result = provider.flags(
        NodeFlagRequest(
            dataset_id=flags_id, outcome=outcome, feature_names=tuple(feature_names)
        )
    )
    halos = {name: "h0" for name in result.flagged("h0")} if result.ran else {}
    for channel in ("h1", "eig"):
        if result.ran:
            for name in result.flagged(channel):
                halos.setdefault(name, channel)
    return {
        "status": result.status,
        "message": result.provenance,
        "records": result.per_feature.to_dict("records") if result.ran else [],
        "halos": halos,
        "provenance": result.provenance,
        "provider": f"{result.provider_name} v{result.provider_version}",
        "warnings": list(result.warnings),
        "cleared": result.cleared_for_circulation,
    }


# ---------------------------------------------------------------------------
# Surgery scorecard (cheap; recomputed per edit on the main thread)
# ---------------------------------------------------------------------------
def surgery_scorecard(
    graph: GraphState,
    exposure: str,
    outcome: str,
    truth: GraphState | None,
) -> dict[str, object]:
    digraph = graph.digraph()
    card: dict[str, object] = {
        "source": graph.provenance.source,
        "n_undirected": graph.n_undirected_pairs,
        "n_edges": len(graph.directed_edges),
        "ledger": [
            f"{entry.kind} {entry.edge[0]} → {entry.edge[1]}"
            + (f" — {entry.rationale}" if entry.rationale else "")
            for entry in graph.provenance.constraint_ledger
        ],
    }
    if exposure in digraph and outcome in digraph:
        candidates = identify_adjustment_sets(digraph, exposure, outcome)
        minimal = candidates.get("minimal", {})
        card["adjustment"] = sorted(minimal.get("variables", []))
        card["adjustment_valid"] = bool(minimal.get("valid", False))
    if truth is not None:
        truth_view = _induced(truth, graph.nodes)
        card["m1"] = m1_concordance(digraph, truth_view)
        if exposure in truth_view and outcome in truth_view:
            m3 = m3_sufficiency_transfer(digraph, truth_view, exposure, outcome)
            card["m3_valid_in_true"] = bool(m3.get("valid_in_true", False))
    return card


# ---------------------------------------------------------------------------
# Stage 6: causal SHAP, two arms
# ---------------------------------------------------------------------------
def run_causal_shap(
    data: pd.DataFrame,
    features: Sequence[str],
    outcome: str,
    graph: GraphState,
    *,
    arm: str,
    model_type: str,
    truth_effects: Mapping[str, float] | None,
    n_perms: int = 32,
    n_background: int = 16,
    n_instances: int = 32,
    seed: int = SEED_HUB_DEMO,
) -> dict[str, object]:
    """Attribute on the CURRENT graph, refitting the model on this feature tuple."""
    features = tuple(features)
    fit = fit_outcome_model(data, features, outcome, model_type=model_type, seed=seed)
    naive_df = compute_standard_shap(fit.model, data, list(features), n_background=100)

    digraph = graph.digraph()
    if arm == "structural":
        fitted = fit_linear_logistic_scm(
            data[list(graph.nodes)].dropna(), digraph, seed=seed,
            n_undirected_pairs=graph.n_undirected_pairs,
        )
        scm = fitted.scm
        rng = np.random.default_rng(seed)
        scm_frame = data[list(scm.order)].dropna()
        evaluation = data[list(features)].dropna().sample(
            min(n_instances, len(data)), random_state=seed
        )
        background_rows = scm_frame.sample(min(n_background, len(scm_frame)), random_state=seed + 1)
        background = scm.recover_exogenous(background_rows, seed=seed)
        feature_edges = [
            (a, b) for a, b in digraph.edges if a in set(features) and b in set(features)
        ]
        result = compute_structural_asymmetric_shap(
            prediction_callable(fit.model, list(features)),
            scm,
            evaluation,
            background,
            list(features),
            feature_edges,
            n_permutations=n_perms,
            seed=seed,
        )
        causal_df = result.values
        arm_note = (
            f"structural do()-propagation on a {fitted.grade}-grade SCM calibrated "
            f"to this data on the current graph ({graph.provenance.source})"
        )
    elif arm == "nonparametric":
        causal_df = _causal_shap_engine(
            fit.model, data, digraph, list(features), outcome,
            n_perms=n_perms, n_background=n_background, n_instances=n_instances,
        )
        arm_note = "nonparametric conditional-model propagation (GBM P(X|parents))"
    else:
        raise ValueError(f"Unknown attribution arm: {arm!r}")

    comparison = compare_shap_rankings(
        naive_df, causal_df, dict(truth_effects) if truth_effects else None
    )
    naive_importance = mean_abs_shap(naive_df).to_dict()
    causal_importance = mean_abs_shap(causal_df).to_dict()
    return {
        "arm": arm,
        "arm_note": arm_note,
        "fit": fit,
        "naive_importance": naive_importance,
        "causal_importance": causal_importance,
        "comparison": comparison,
        "plot": comparison_chart(naive_importance, causal_importance),
    }


# ---------------------------------------------------------------------------
# Stage 7: price and dice
# ---------------------------------------------------------------------------
def run_policy(
    data: pd.DataFrame,
    graph: GraphState,
    outcome: str,
    specs: Mapping[str, ActionSpec],
    *,
    budget: float | None,
    direction: str,
    alpha: float,
    seed: int = SEED_ACTION_ABDUCTION,
) -> dict[str, object]:
    """Calibrate on the current graph, then price every candidate lever.

    Calibrating on whatever graph the surgeon left behind is what keeps the
    screening graph and the propagation model the same object; simulating a
    frozen bundled SCM after the user removed one of its edges would be
    quietly incoherent.
    """
    digraph = graph.digraph()
    scm_frame = data[list(graph.nodes)].dropna()
    fitted = fit_linear_logistic_scm(
        scm_frame, digraph, seed=seed, n_undirected_pairs=graph.n_undirected_pairs
    )
    exogenous = abduct(fitted.scm, scm_frame, seed=seed)
    problem = InterventionProblem(
        scm=fitted.scm,
        outcome=outcome,
        cost_model=CostModel(specs=dict(specs), budget=budget),
        graph=digraph,
        direction=direction,
        alpha=alpha,
        n_undirected_pairs=graph.n_undirected_pairs,
    )
    ranking = rank_actions(problem, exogenous, seed=seed)
    return {
        "ranking": ranking,
        "table": ranking.to_frame(),
        "screened": ranking.screened_frame(),
        "calibration": fitted,
        "pareto_plot": pareto_chart(ranking, budget),
    }


# ---------------------------------------------------------------------------
# Charts (hub palette, base64 PNG; None when matplotlib is absent)
# ---------------------------------------------------------------------------
def _figure_to_base64(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode()


def bar_chart(importance: Mapping[str, float], title: str, color: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    items = sorted(importance.items(), key=lambda pair: abs(pair[1]))[-15:]
    names = [name for name, _ in items]
    values = [abs(value) for _, value in items]
    figure, axis = plt.subplots(figsize=(7.2, max(2.4, 0.34 * len(items))))
    axis.barh(names, values, color=color)
    axis.set_title(title, fontsize=11, fontfamily="serif", color=INK, loc="left")
    axis.tick_params(labelsize=9)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    return _figure_to_base64(figure)


def comparison_chart(
    naive: Mapping[str, float], causal: Mapping[str, float]
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    order = [name for name, _ in sorted(naive.items(), key=lambda p: abs(p[1]))][-12:]
    positions = np.arange(len(order))
    figure, axis = plt.subplots(figsize=(7.2, max(2.6, 0.42 * len(order))))
    axis.barh(positions + 0.2, [abs(naive[n]) for n in order], height=0.38,
              color=MUTED, label="naive")
    axis.barh(positions - 0.2, [abs(causal.get(n, 0.0)) for n in order], height=0.38,
              color=AMBER, label="causal")
    axis.set_yticks(positions, order, fontsize=9)
    axis.legend(frameon=False, fontsize=9)
    axis.set_title("Naive vs causal attribution", fontsize=11, fontfamily="serif",
                   color=INK, loc="left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    return _figure_to_base64(figure)


def pareto_chart(ranking: ActionRanking, budget: float | None) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for item in ranking.evaluations:
        color = AMBER if item.feasible else MUTED
        axis.scatter(item.cost, item.benefit, color=color, s=42, zorder=3)
        axis.annotate(item.label, (item.cost, item.benefit), fontsize=8,
                      xytext=(4, 4), textcoords="offset points", color=INK)
    if budget is not None:
        axis.axvline(budget, color=INK, linestyle="--", linewidth=1)
        axis.annotate(f"budget {budget:g}", (budget, axis.get_ylim()[0]), fontsize=8,
                      xytext=(4, 6), textcoords="offset points", color=INK)
    axis.set_xlabel("cost", fontsize=10)
    axis.set_ylabel("expected benefit", fontsize=10)
    axis.set_title("Benefit against cost", fontsize=11, fontfamily="serif",
                   color=INK, loc="left")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    return _figure_to_base64(figure)
