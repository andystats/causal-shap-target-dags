"""Simulate the teaching DAGs and freeze their data, edges, truth, and figure.

Everything downstream (discovery, complexity, causal SHAP, figures) reads these
frozen artifacts, so the toy and ladder datasets are generated exactly once here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from _datasets import TEACHING_ROWS  # noqa: E402
from causal_shap.seeds import SEED_TEACHING_LADDER, SEED_TEACHING_TOY  # noqa: E402
from causal_shap.teaching_dags import (  # noqa: E402
    TeachingDAG,
    layered_ladder,
    simulate_dataframe,
    toy_chain_fork_collider,
)

matplotlib.use("Agg")

BUNDLES_DIR = APP_DIR / "bundles"


def _write_dag_figure(dag: TeachingDAG, path: Path) -> None:
    graph = dag.graph
    position = nx.spring_layout(graph, seed=42, k=1.6)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    colors = [
        "#111827" if node == dag.outcome
        else "#d97706" if abs(dag.true_total_effects.get(node, 0.0)) < 1e-9
        else "#2563eb"
        for node in graph.nodes
    ]
    nx.draw_networkx_edges(graph, position, ax=ax, edge_color="#cbd5e1", arrows=True, arrowsize=13)
    nx.draw_networkx_nodes(graph, position, ax=ax, node_color=colors, node_size=1100)
    nx.draw_networkx_labels(graph, position, ax=ax, font_size=8, font_color="#ffffff")
    ax.set_axis_off()
    ax.set_title(f"{dag.name} (amber = zero-effect proxy)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _build_one(dag: TeachingDAG, bundle_id: str, seed: int) -> dict[str, object]:
    bundle_dir = BUNDLES_DIR / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    data = simulate_dataframe(dag, TEACHING_ROWS, seed)
    data.to_csv(bundle_dir / "data.csv", index=False)

    edges = [{"from": source, "to": target} for source, target in dag.graph.edges()]
    pd.DataFrame(edges).to_csv(bundle_dir / "edges.csv", index=False)

    (bundle_dir / "true_effects.json").write_text(
        json.dumps({"outcome": dag.outcome, "true_total_effects": dict(dag.true_total_effects)}, indent=2),
        encoding="utf-8",
    )
    _write_dag_figure(dag, bundle_dir / "dag.png")

    if len(data) != TEACHING_ROWS:
        raise RuntimeError(f"{bundle_id}: expected {TEACHING_ROWS} rows, got {len(data)}")
    return {"bundle": bundle_id, "rows": len(data), "nodes": dag.graph.number_of_nodes(), "seed": seed}


def main() -> None:
    summary = {
        "status": "teaching_data",
        "datasets": [
            _build_one(toy_chain_fork_collider(), "toy_chain_fork_collider", SEED_TEACHING_TOY),
            _build_one(layered_ladder(), "layered_ladder", SEED_TEACHING_LADDER),
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
