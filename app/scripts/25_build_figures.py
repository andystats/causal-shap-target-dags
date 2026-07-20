"""Render every site/app figure from frozen artifacts.

One script owns all generated imagery: homunculus pairs and distortion profiles
per dataset, plus the ladder SVG. Outputs land in site/assets/figures (with a
provenance MANIFEST) and are mirrored into each app bundle so the deployed app
is self-contained.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from _datasets import teaching_datasets  # noqa: E402
from causal_shap.graphs import load_edges_csv, undirected_distance_to_outcome  # noqa: E402
from causal_shap.viz import distortion_profile, homunculus_pair, ladder_svg  # noqa: E402

matplotlib.use("Agg")

SITE_FIGURES = PROJECT_DIR / "site" / "assets" / "figures"
BUNDLES_DIR = APP_DIR / "bundles"
MANIFEST: list[dict[str, str]] = []


def _shares(frame: pd.DataFrame, method: str) -> dict[str, float]:
    rows = frame[frame["method"] == method]
    return dict(zip(rows["variable"], rows["normalized_importance"]))


def _save(fig, name: str, bundle_id: str) -> None:
    SITE_FIGURES.mkdir(parents=True, exist_ok=True)
    site_path = SITE_FIGURES / name
    fig.savefig(site_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    bundle_copy = BUNDLES_DIR / bundle_id / "stages" / name
    bundle_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(site_path, bundle_copy)
    MANIFEST.append({"figure": name, "bundle": bundle_id, "sha256": _sha(site_path)})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _teaching_figures() -> None:
    for dataset in teaching_datasets():
        attribution = pd.read_csv(dataset.stage_dir / "attribution.csv")
        graph = dataset.load_graph()
        fig = homunculus_pair(
            graph,
            dataset.outcome,
            _shares(attribution, "Interventional truth"),
            _shares(attribution, "Ordinary SHAP"),
            _shares(attribution, "Structural Causal SHAP"),
        )
        _save(fig, f"{dataset.bundle_id}_homunculus.png", dataset.bundle_id)

        depths = undirected_distance_to_outcome(graph, dataset.outcome)
        shares = {method: _shares(attribution, method) for method in attribution["method"].unique()}
        _save(distortion_profile(depths, shares), f"{dataset.bundle_id}_distortion.png", dataset.bundle_id)


def _acic_figure() -> None:
    bundle_dir = BUNDLES_DIR / "acic_proxy_stress_test"
    importance = pd.read_csv(bundle_dir / "importance.csv")
    graph = load_edges_csv(bundle_dir / "edges.csv")
    fig = homunculus_pair(
        graph,
        "AcuteRisk",
        _shares(importance, "Interventional truth"),
        _shares(importance, "Ordinary SHAP"),
        _shares(importance, "DAG-aware SHAP demo"),
    )
    _save(fig, "acic_proxy_stress_test_homunculus.png", "acic_proxy_stress_test")


def _nasa_figure() -> None:
    bundle_dir = BUNDLES_DIR / "nasa_renal_clean_v3"
    comparison = pd.read_csv(bundle_dir / "importance_comparison.csv")
    structural = pd.read_csv(bundle_dir / "structural_causal_shap_importance.csv")
    combined = pd.concat([comparison, structural], ignore_index=True, sort=False)
    depths = dict(zip(comparison["variable"], comparison["distance"]))
    methods = [
        "Interventional truth",
        "Ordinary TreeSHAP",
        "DAG-constrained asymmetric SHAP",
        "Structural intervention-propagating SHAP prototype",
    ]
    shares = {m: _shares(combined, m) for m in methods if m in set(combined["method"])}
    _save(distortion_profile(depths, shares, truth_key="Interventional truth"), "nasa_distortion.png", "nasa_renal_clean_v3")


def main() -> None:
    _teaching_figures()
    _acic_figure()
    _nasa_figure()

    ladder_path = SITE_FIGURES / "ladder.svg"
    ladder_path.write_text(ladder_svg(), encoding="utf-8")
    (APP_DIR / "assets").mkdir(parents=True, exist_ok=True)
    (APP_DIR / "assets" / "ladder.svg").write_text(ladder_svg(), encoding="utf-8")
    MANIFEST.append({"figure": "ladder.svg", "bundle": "shared", "sha256": _sha(ladder_path)})

    (SITE_FIGURES / "MANIFEST.json").write_text(json.dumps({"figures": MANIFEST}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "figures", "count": len(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
