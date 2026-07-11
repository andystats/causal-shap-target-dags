"""Build deterministic, precomputed assets for the ACIC proxy stress test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from causal_shap.pedagogic import (  # noqa: E402
    TRUE_TOTAL_EFFECTS,
    attribution_shift,
    compute_causal_shap_fast,
    compute_standard_shap,
    dag_from_edges_csv,
    tau_vs_truth,
)


matplotlib.use("Agg")

OUTPUT_DIR = APP_DIR / "bundles" / "acic_proxy_stress_test"
DATA_PATH = OUTPUT_DIR / "data.csv"
EDGES_PATH = OUTPUT_DIR / "edges.csv"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    outcome = "AcuteRisk"
    features = [column for column in data.select_dtypes("number").columns if column != outcome]
    train, test = train_test_split(data, test_size=0.30, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(train[features], train[outcome])
    held_out_r2 = r2_score(test[outcome], model.predict(test[features]))

    np.random.seed(20260725)
    standard_values = compute_standard_shap(model, data, features, n_background=100)
    graph = dag_from_edges_csv(EDGES_PATH, features)
    np.random.seed(20260726)
    causal_values = compute_causal_shap_fast(
        model,
        data,
        graph,
        features,
        outcome,
        n_perms=8,
        n_background=4,
        n_instances=8,
    )

    standard = standard_values.abs().mean(axis=0)
    causal = causal_values.abs().mean(axis=0)
    truth = pd.Series(
        {feature: abs(TRUE_TOTAL_EFFECTS.get(feature, 0.0)) for feature in features}
    )
    rows = []
    for method, values in [
        ("Interventional truth", truth),
        ("Ordinary SHAP", standard),
        ("DAG-aware SHAP demo", causal),
    ]:
        normalized = values / values.sum()
        for feature in features:
            rows.append(
                {
                    "method": method,
                    "variable": feature,
                    "raw_importance": float(values[feature]),
                    "normalized_importance": float(normalized[feature]),
                }
            )
    importance = pd.DataFrame(rows)
    importance["rank"] = importance.groupby("method")["normalized_importance"].rank(
        ascending=False, method="average"
    )
    importance.to_csv(OUTPUT_DIR / "importance.csv", index=False)

    proxy_inflation, root_cause_boost = attribution_shift(
        standard_values,
        causal_values,
        mediators=[
            "ShockIndexProxy",
            "VasopressorProxy",
            "MonitoringProxy",
            "RescueProxy",
            "CompositeScoreProxy",
        ],
        roots=[
            "BaselineSeverity",
            "ChronicBurden",
            "SocialRisk",
            "PracticeStyle",
            "Age",
            "TreatmentIntensity",
        ],
    )
    metrics = {
        "status": "pedagogic_stress_test",
        "rows": len(data),
        "features": len(features),
        "held_out_r2": held_out_r2,
        "tau_vs_truth_standard": tau_vs_truth(standard_values),
        "tau_vs_truth_causal": tau_vs_truth(causal_values),
        "proxy_inflation": proxy_inflation,
        "root_cause_boost": root_cause_boost,
        "seed_standard": 20260725,
        "seed_causal": 20260726,
        "causal_permutations": 8,
        "causal_background": 4,
        "causal_instances": 8,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    position = nx.spring_layout(graph, seed=42, k=1.5)
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
    node_colors = [
        "#d97706" if node in {"ShockIndexProxy", "MonitoringProxy", "CompositeScoreProxy"}
        else "#2563eb" if node in {"BaselineSeverity", "ChronicBurden", "TreatmentIntensity"}
        else "#94a3b8"
        for node in graph.nodes
    ]
    nx.draw_networkx_edges(graph, position, ax=ax, edge_color="#cbd5e1", arrows=True, arrowsize=12)
    nx.draw_networkx_nodes(graph, position, ax=ax, node_color=node_colors, node_size=900)
    nx.draw_networkx_labels(graph, position, ax=ax, font_size=7)
    ax.set_axis_off()
    ax.set_title("Pedagogic mediator/proxy stress-test DAG", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dag.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
