"""Structural intervention-propagating Causal SHAP on the teaching DAGs.

Produces, for the toy and ladder DAGs, the three attribution families on a
shared scale (interventional truth, ordinary SHAP, structural Causal SHAP) so
the app and site can show the homunculus contrast on live-sized problems. NASA
and ACIC attribution already ship from the frozen analysis and are only read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from causal_shap.seeds import SEED_TEACHING_LADDER, SEED_TEACHING_TOY  # noqa: E402
from causal_shap.structural_value import compute_structural_asymmetric_shap  # noqa: E402
from causal_shap.teaching_dags import (  # noqa: E402
    TeachingDAG,
    layered_ladder,
    simulate_dataframe,
    toy_chain_fork_collider,
)

BUNDLES_DIR = APP_DIR / "bundles"
N_PERMUTATIONS = 64
N_BACKGROUND = 64
N_EVALUATION = 64


def _attribution_frame(dag: TeachingDAG, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    np.random.seed(seed)  # shap's masker draws from the global RNG; pin it for reproducibility
    features = list(dag.features)
    data = simulate_dataframe(dag, 6000, seed)
    model = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=seed)
    model.fit(data[features], data[dag.outcome])

    background_frame = simulate_dataframe(dag, N_BACKGROUND, seed + 1)
    scm = dag.scm()
    background_exogenous = scm.recover_exogenous(background_frame, seed=seed + 2)
    evaluation = data[features].head(N_EVALUATION).reset_index(drop=True)
    feature_edges = [(u, v) for u, v in dag.graph.edges() if u in features and v in features]

    def predict_margin(matrix: np.ndarray) -> np.ndarray:
        return model.predict(pd.DataFrame(matrix, columns=features))

    structural = compute_structural_asymmetric_shap(
        predict_margin=predict_margin,
        scm=scm,
        evaluation=evaluation,
        background_exogenous=background_exogenous,
        feature_names=features,
        feature_edges=feature_edges,
        n_permutations=N_PERMUTATIONS,
        seed=seed + 3,
    )
    structural_importance = structural.values.abs().mean(axis=0)

    explainer = shap.Explainer(model, data[features].sample(100, random_state=seed))
    vanilla = np.abs(explainer(evaluation).values).mean(axis=0)
    vanilla_importance = pd.Series(vanilla, index=features)

    truth = pd.Series({name: abs(dag.true_total_effects[name]) for name in features})

    rows = []
    for method, values in [
        ("Interventional truth", truth),
        ("Ordinary SHAP", vanilla_importance),
        ("Structural Causal SHAP", structural_importance),
    ]:
        total = values.sum() or 1.0
        for name in features:
            rows.append(
                {
                    "method": method,
                    "variable": name,
                    "raw_importance": float(values[name]),
                    "normalized_importance": float(values[name] / total),
                }
            )
    frame = pd.DataFrame(rows)
    efficiency = float(np.max(np.abs(structural.efficiency_error)))
    return frame, {"max_efficiency_error": efficiency, "permutations": N_PERMUTATIONS}


def _build(dag: TeachingDAG, bundle_id: str, seed: int) -> dict[str, object]:
    frame, meta = _attribution_frame(dag, seed)
    stage_dir = BUNDLES_DIR / bundle_id / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stage_dir / "attribution.csv", index=False)
    if meta["max_efficiency_error"] > 1e-6:
        raise RuntimeError(f"{bundle_id}: structural efficiency error {meta['max_efficiency_error']:.2e} too large")
    return {"bundle": bundle_id, **meta, "seed": seed}


def main() -> None:
    summary = [
        _build(toy_chain_fork_collider(), "toy_chain_fork_collider", SEED_TEACHING_TOY),
        _build(layered_ladder(), "layered_ladder", SEED_TEACHING_LADDER),
    ]
    print(json.dumps({"status": "causal_shap", "datasets": summary}, indent=2))


if __name__ == "__main__":
    main()
