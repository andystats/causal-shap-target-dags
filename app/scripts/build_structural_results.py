"""Build a frozen structural Causal SHAP prototype for NASA clean v3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import kendalltau, spearmanr


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from causal_shap import build_nasa_renal_scm, compute_structural_asymmetric_shap  # noqa: E402


ANALYSIS_DIR = PROJECT_DIR / "analysis"
SHAP_DIR = ANALYSIS_DIR / "output" / "shap_nephrolithiasis_clean_v3"
DATA_PATH = (
    ANALYSIS_DIR
    / "output"
    / "source_aligned_clean"
    / "renal_stone_source_aligned_clean_v3.csv"
)
EDGES_PATH = ANALYSIS_DIR / "output" / "dag_validation" / "validated_clean_source_edges.csv"
MAP_PATH = (
    ANALYSIS_DIR
    / "output"
    / "source_aligned_clean"
    / "source_to_simulation_variable_map.csv"
)


def concentration(values: np.ndarray, distances: np.ndarray) -> np.ndarray:
    maximum = int(np.max(distances))
    return np.array([values[distances <= hop].sum() for hop in range(1, maximum + 1)])


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    ancestors = pd.read_csv(SHAP_DIR / "target_ancestor_table.csv")
    truth = pd.read_csv(SHAP_DIR / "interventional_truth.csv")
    evaluation_manifest = pd.read_csv(SHAP_DIR / "evaluation_manifest.csv")
    background_manifest = pd.read_csv(SHAP_DIR / "background_manifest.csv")
    source_edges = pd.read_csv(EDGES_PATH)
    variable_map = pd.read_csv(MAP_PATH)

    features = sorted(ancestors["variable"].tolist())
    feature_set = set(features)
    source_to_variable = dict(zip(variable_map["source_node"], variable_map["variable"]))
    feature_edges = [
        (source_to_variable[source], source_to_variable[target])
        for source, target in source_edges[["from", "to"]].itertuples(index=False, name=None)
        if source_to_variable[source] in feature_set and source_to_variable[target] in feature_set
    ]

    n_evaluation = min(32, len(evaluation_manifest))
    n_background = min(32, len(background_manifest))
    evaluation_rows = evaluation_manifest["source_row"].head(n_evaluation).to_numpy() - 1
    background_rows = background_manifest["source_row"].head(n_background).to_numpy() - 1
    evaluation = data.iloc[evaluation_rows][features].copy()
    evaluation.index = evaluation_manifest["source_row"].head(n_evaluation).astype(str)
    background_structural = data.iloc[background_rows][list(variable_map["variable"])]

    scm = build_nasa_renal_scm(EDGES_PATH, MAP_PATH)
    exogenous = scm.recover_exogenous(background_structural, seed=20260723)
    reconstructed = pd.DataFrame(scm.simulate(exogenous))
    reconstruction_error = float(
        np.max(
            np.abs(
                reconstructed[list(variable_map["variable"])].to_numpy()
                - background_structural[list(variable_map["variable"])].to_numpy()
            )
        )
    )

    booster = xgb.Booster()
    booster.load_model(SHAP_DIR / "nephrolithiasis_xgboost_clean_v3.ubj")

    def predict_margin(matrix: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(matrix, columns=features)
        return booster.predict(xgb.DMatrix(frame), output_margin=True)

    result = compute_structural_asymmetric_shap(
        predict_margin=predict_margin,
        scm=scm,
        evaluation=evaluation,
        background_exogenous=exogenous,
        feature_names=features,
        feature_edges=feature_edges,
        n_permutations=32,
        seed=20260724,
    )

    values_output = result.values.copy()
    values_output.insert(0, "source_row", values_output.index.astype(int))
    values_output.to_csv(SHAP_DIR / "structural_causal_shap_values.csv", index=False)

    raw_importance = result.values.abs().mean(axis=0)
    normalized_importance = raw_importance / raw_importance.sum()
    structural_importance = pd.DataFrame(
        {
            "method": "Structural intervention-propagating SHAP prototype",
            "variable": features,
            "raw_importance": [raw_importance[name] for name in features],
            "normalized_importance": [normalized_importance[name] for name in features],
        }
    )
    structural_importance = structural_importance.merge(
        ancestors[["variable", "source_node", "distance", "structural_role"]],
        on="variable",
        how="left",
    )
    structural_importance["rank"] = structural_importance["normalized_importance"].rank(
        ascending=False, method="average"
    )
    structural_importance.to_csv(SHAP_DIR / "structural_causal_shap_importance.csv", index=False)

    truth_aligned = truth.set_index("variable").loc[features]
    truth_normalized = truth_aligned["absolute_total_effect"].to_numpy(dtype=float)
    truth_normalized /= truth_normalized.sum()
    structural_normalized = structural_importance.set_index("variable").loc[features][
        "normalized_importance"
    ].to_numpy(dtype=float)
    distances = ancestors.set_index("variable").loc[features]["distance"].to_numpy(dtype=int)
    truth_mean_distance = float(np.sum(truth_normalized * distances))
    method_mean_distance = float(np.sum(structural_normalized * distances))
    truth_curve = concentration(truth_normalized, distances)
    method_curve = concentration(structural_normalized, distances)
    top_truth = set(np.argsort(-truth_normalized)[:5])
    top_method = set(np.argsort(-structural_normalized)[:5])

    summary = {
        "status": "prototype",
        "method": "Structural intervention-propagating DAG-asymmetric SHAP",
        "evaluation_rows": n_evaluation,
        "background_rows": n_background,
        "permutations": result.permutations,
        "background_reconstruction_max_abs_error": reconstruction_error,
        "max_absolute_efficiency_error": float(np.max(np.abs(result.efficiency_error))),
        "kendall_tau_vs_truth": float(kendalltau(structural_normalized, truth_normalized).statistic),
        "spearman_rho_vs_truth": float(spearmanr(structural_normalized, truth_normalized).statistic),
        "top5_recovery": len(top_truth & top_method) / 5,
        "mean_distance": method_mean_distance,
        "truth_mean_distance": truth_mean_distance,
        "pbi": truth_mean_distance - method_mean_distance,
        "poa": float(np.mean((method_curve - truth_curve)[:-1])),
        "proximal_mass_distance_le_2": float(structural_normalized[distances <= 2].sum()),
        "seed_exogenous_abduction": 20260723,
        "seed_permutations": 20260724,
    }
    (SHAP_DIR / "structural_causal_shap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if reconstruction_error > 1e-9:
        raise RuntimeError(f"SCM reconstruction error is too large: {reconstruction_error}")
    if summary["max_absolute_efficiency_error"] > 1e-5:
        raise RuntimeError("Structural SHAP failed the efficiency check")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
