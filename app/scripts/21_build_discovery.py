"""Run causal discovery on every live-discovery dataset and freeze the results.

PC and GES are the live in-app algorithms; DirectLiNGAM and NOTEARS are added
here as precomputed appendix columns when their dependencies are installed.
Each dataset gets one discovery.json with per-algorithm edges, comparison
metrics vs the known truth, and the cross-algorithm disagreement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from _datasets import discovery_datasets  # noqa: E402
from causal_shap.discovery import (  # noqa: E402
    DiscoveryResult,
    compare_graphs,
    pairwise_skeleton_disagreement,
    run_direct_lingam,
    run_ges,
    run_notears,
    run_pc,
)
from causal_shap.seeds import SEED_DISCOVERY  # noqa: E402


def _algorithm_results(data) -> dict[str, DiscoveryResult]:
    results: dict[str, DiscoveryResult] = {"PC": run_pc(data, alpha=0.05), "GES": run_ges(data)}
    try:
        results["DirectLiNGAM"] = run_direct_lingam(data)
    except Exception as error:  # pragma: no cover - appendix only
        print(f"  DirectLiNGAM skipped: {error}")
    try:
        results["NOTEARS"] = run_notears(data)
    except ImportError:
        pass  # gcastle/torch absent: NOTEARS is a build-only appendix column
    return results


def _serialize(result: DiscoveryResult, truth_edges) -> dict[str, object]:
    comparison = compare_graphs(sorted(result.directed_edges), truth_edges)
    return {
        "algorithm": result.algorithm,
        "directed_edges": sorted(list(edge) for edge in result.directed_edges),
        "undirected_edges": sorted(list(edge) for edge in result.pdag.undirected_edges),
        "precision": comparison.precision,
        "recall": comparison.recall,
        "f1": comparison.f1,
        "skeleton_f1": comparison.skeleton_f1,
        "n_reversed": len(comparison.reversed),
        "n_spurious": len(comparison.spurious),
        "n_missed": len(comparison.missed),
    }


def main() -> None:
    datasets = []
    for dataset in discovery_datasets():
        data = dataset.load_data()
        graph = dataset.load_graph()
        graph_nodes = set(graph.nodes())
        frame = data[[c for c in data.columns if c in graph_nodes]]
        truth_edges = sorted(graph.edges())
        results = _algorithm_results(frame)
        record = {
            "bundle": dataset.bundle_id,
            "seed": SEED_DISCOVERY,
            "n_rows": len(frame),
            "algorithms": {name: _serialize(result, truth_edges) for name, result in results.items()},
            "cross_algorithm_disagreement": pairwise_skeleton_disagreement(list(results.values())),
        }
        dataset.stage_dir.mkdir(parents=True, exist_ok=True)
        (dataset.stage_dir / "discovery.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        datasets.append(
            {
                "bundle": dataset.bundle_id,
                "algorithms": list(results),
                "disagreement": round(record["cross_algorithm_disagreement"], 3),
                "pc_skeleton_f1": round(record["algorithms"]["PC"]["skeleton_f1"], 3),
            }
        )
    print(json.dumps({"status": "discovery", "datasets": datasets}, indent=2))


if __name__ == "__main__":
    main()
