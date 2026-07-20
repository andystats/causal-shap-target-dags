"""Build stages: the in-package home of the former numbered ``scripts/2x``.

Each function is one former script's ``main()`` body, carrying the same logic,
seeds, output paths, and JSON summary, so the frozen artifacts stay
byte-for-byte reproducible. Per-stage constants and helpers are kept local to
their stage; only the path block and the JSON status print are centralized in
``common``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml
from matplotlib import pyplot as plt
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .. import build_nasa_renal_scm
from ..complexity import ComplexityInputs, get_score
from ..datasets import TEACHING_ROWS, discovery_datasets, teaching_datasets
from ..discovery import (
    DiscoveryResult,
    compare_graphs,
    pairwise_skeleton_disagreement,
    run_direct_lingam,
    run_ges,
    run_notears,
    run_pc,
)
from ..graphs import load_edges_csv, undirected_distance_to_outcome
from ..pedagogic import (
    TRUE_TOTAL_EFFECTS,
    attribution_shift,
    compute_causal_shap_fast,
    compute_standard_shap,
    dag_from_edges_csv,
    tau_vs_truth,
)
from ..seeds import (
    SEED_DISCOVERY,
    SEED_TEACHING_LADDER,
    SEED_TEACHING_TOY,
    SEED_VALIDATION_SCENARIOS,
)
from ..structural_value import compute_structural_asymmetric_shap
from ..teaching_dags import (
    TeachingDAG,
    layered_ladder,
    simulate_dataframe,
    toy_chain_fork_collider,
)
from ..validation import (
    ConfounderSpec,
    MVNGenerator,
    Scenario,
    SimulationSpec,
    additive_bias,
    constant_tau,
    fit_baseline,
    fit_generator,
    generate_validation_suite,
    linear_tau,
)
from ..viz import distortion_profile, homunculus_pair, ladder_svg
from .common import APP_DIR, BUNDLES_DIR, PROJECT_DIR, print_status


def teaching_data() -> None:
    """Simulate the teaching DAGs and freeze data, edges, truth, and figures."""

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

    summary = {
        "status": "teaching_data",
        "datasets": [
            _build_one(toy_chain_fork_collider(), "toy_chain_fork_collider", SEED_TEACHING_TOY),
            _build_one(layered_ladder(), "layered_ladder", SEED_TEACHING_LADDER),
        ],
    }
    print_status(summary)


def discovery() -> None:
    """Run causal discovery on every live-discovery dataset and freeze results."""

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
    print_status({"status": "discovery", "datasets": datasets})


def complexity() -> None:
    """Score every dataset's problem complexity (PSCI v0) and freeze the reports."""
    ANALYSIS = PROJECT_DIR / "analysis" / "output"
    NASA_EDGES = ANALYSIS / "dag_validation" / "validated_clean_source_edges.csv"
    NASA_MAP = ANALYSIS / "source_aligned_clean" / "source_to_simulation_variable_map.csv"
    NASA_OUTCOME = "nephrolithiasis"

    def _report_dict(report) -> dict[str, object]:
        return {
            "score_name": report.score_name,
            "score_version": report.score_version,
            "provisional": report.provisional,
            "total": report.total,
            "band": report.band,
            "subscores": {
                name: {"value": sub.value, "rationale": sub.rationale, "available": sub.available}
                for name, sub in report.subscores.items()
            },
            "recommendations": list(report.recommendations),
        }

    def _write(bundle_dir: Path, report) -> None:
        stage_dir = bundle_dir / "stages"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "complexity.json").write_text(json.dumps(_report_dict(report), indent=2), encoding="utf-8")

    score = get_score("PSCI")
    summary = []

    for dataset in discovery_datasets():
        discovery_path = dataset.stage_dir / "discovery.json"
        disagreement = None
        if discovery_path.exists():
            disagreement = json.loads(discovery_path.read_text())["cross_algorithm_disagreement"]
        report = score.compute(
            ComplexityInputs(graph=dataset.load_graph(), outcome=dataset.outcome, disagreement=disagreement)
        )
        _write(dataset.bundle_dir, report)
        summary.append({"bundle": dataset.bundle_id, "total": round(report.total, 1), "band": report.band})

    nasa_scm = build_nasa_renal_scm(NASA_EDGES, NASA_MAP)
    nasa_report = score.compute(ComplexityInputs(graph=nasa_scm.graph, outcome=NASA_OUTCOME))
    _write(APP_DIR / "bundles" / "nasa_renal_clean_v3", nasa_report)
    summary.append({"bundle": "nasa_renal_clean_v3", "total": round(nasa_report.total, 1), "band": nasa_report.band})

    print_status({"status": "complexity", "datasets": summary})


def causal_shap() -> None:
    """Structural intervention-propagating Causal SHAP on the teaching DAGs."""
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

    summary = [
        _build(toy_chain_fork_collider(), "toy_chain_fork_collider", SEED_TEACHING_TOY),
        _build(layered_ladder(), "layered_ladder", SEED_TEACHING_LADDER),
    ]
    print_status({"status": "causal_shap", "datasets": summary})


def validation() -> None:
    """Credence-style validation suite on NASA-like covariates."""
    OUTPUT_DIR = APP_DIR / "bundles" / "nasa_renal_clean_v3" / "stages"
    N_ROWS = 4000

    def _reference_data(seed: int = SEED_VALIDATION_SCENARIOS) -> pd.DataFrame:
        """A compact real-data proxy: realistic covariates, unconfounded treatment.

        Treatment is randomized so the learned p(X|Z) carries no arm imbalance —
        realism (the covariate distribution) stays separate from the truth and bias
        injected later, as in the Credence design.
        """
        rng = np.random.default_rng(seed)
        n = 3000
        age = rng.normal(size=n)
        comorbidity = 0.4 * age + rng.normal(size=n)
        hydration = rng.normal(size=n)
        z = rng.binomial(1, 0.5, size=n)
        y = age + 0.5 * comorbidity - 0.4 * hydration + rng.normal(size=n)
        return pd.DataFrame({"age": age, "comorbidity": comorbidity, "hydration": hydration, "Z": z, "Y": y})

    def _train_cvae(real: pd.DataFrame, features: list[str]) -> None:
        """Train and persist CVAE decoder weights (torch, local build only)."""
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            print("  CVAE weights skipped: torch not installed (heavy extra).")
            return

        from ..validation import CVAEGenerator

        generator = CVAEGenerator(epochs=200)
        generator.fit(real[features], real["Z"].to_numpy())
        weights_path = OUTPUT_DIR / "cvae_decoder.pt"
        generator.save_decoder(weights_path)
        print(f"  CVAE decoder weights written to {weights_path.name}")

    real = _reference_data()
    features = ["age", "comorbidity", "hydration"]
    baseline = fit_baseline(real, "Y", features)
    generator = fit_generator(MVNGenerator(), real, features, "Z")

    scenarios = [
        Scenario("homogeneous_no_confounding", SimulationSpec(baseline=baseline, tau=constant_tau(1.0))),
        Scenario(
            "heterogeneous_effect",
            SimulationSpec(baseline=baseline, tau=linear_tau(1.0, {"age": 0.8})),
        ),
        Scenario(
            "layered_confounding",
            SimulationSpec(
                baseline=baseline,
                tau=linear_tau(1.0, {"age": 0.8}),
                confounders=(
                    ConfounderSpec("frailty", outcome_strength=1.5, treatment_strength=1.2),
                    ConfounderSpec("access", outcome_strength=-0.8, treatment_strength=0.9),
                ),
            ),
        ),
        Scenario(
            "confounding_plus_measurement_bias",
            SimulationSpec(
                baseline=baseline,
                tau=linear_tau(1.0, {"age": 0.8}),
                confounders=(ConfounderSpec("frailty", 1.5, 1.2),),
                bias=additive_bias("comorbidity", 0.6),
            ),
        ),
    ]

    suite = generate_validation_suite(generator, scenarios, n=N_ROWS, seed=SEED_VALIDATION_SCENARIOS, subgroup_col="age")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suite.scorecard.to_csv(OUTPUT_DIR / "validation_scorecard.csv", index=False)

    estimand_frames = []
    for label, dataset in suite.datasets.items():
        table = dataset.estimands.copy()
        table.insert(0, "scenario", label)
        estimand_frames.append(table)
    pd.concat(estimand_frames, ignore_index=True).to_csv(OUTPUT_DIR / "validation_estimands.csv", index=False)

    _train_cvae(real, features)

    summary = {
        "status": "validation",
        "scenarios": list(suite.datasets),
        "generator": "MVN (live) + CVAE weights (committed)",
        "max_naive_drift": float(suite.scorecard["naive_drift"].abs().max()),
    }
    print_status(summary)


def figures() -> None:
    """Render every site/app figure from frozen artifacts."""
    SITE_FIGURES = PROJECT_DIR / "site" / "assets" / "figures"
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

    _teaching_figures()
    _acic_figure()
    _nasa_figure()

    ladder_path = SITE_FIGURES / "ladder.svg"
    ladder_path.write_text(ladder_svg(), encoding="utf-8")
    (APP_DIR / "assets").mkdir(parents=True, exist_ok=True)
    (APP_DIR / "assets" / "ladder.svg").write_text(ladder_svg(), encoding="utf-8")
    MANIFEST.append({"figure": "ladder.svg", "bundle": "shared", "sha256": _sha(ladder_path)})

    (SITE_FIGURES / "MANIFEST.json").write_text(json.dumps({"figures": MANIFEST}, indent=2), encoding="utf-8")
    print_status({"status": "figures", "count": len(MANIFEST)})


def glossary() -> None:
    """Render the canonical glossary into an app-consumable JSON artifact."""
    GLOSSARY_YML = PROJECT_DIR / "site" / "data" / "glossary.yml"
    GLOSSARY_JSON = APP_DIR / "assets" / "glossary.json"
    GLOSSARY_INCLUDE = PROJECT_DIR / "site" / "_includes" / "glossary_body.md"

    def _static_markdown(terms: list[dict]) -> str:
        groups: dict[str, list[dict]] = {}
        for entry in terms:
            groups.setdefault(entry.get("group", "General"), []).append(entry)
        blocks = ["<!-- Generated by `python -m causal_shap.build glossary` from data/glossary.yml. Do not edit. -->"]
        for group in sorted(groups):
            blocks.append(f'\n::: {{.glossary-group}}\n### {group}\n\n<dl class="glossary">')
            for entry in sorted(groups[group], key=lambda e: e["term"].lower()):
                definition = " ".join(entry["definition"].split())
                blocks.append(f"<dt>{entry['term']}</dt><dd>{definition}</dd>")
            blocks.append("</dl>\n:::")
        return "\n".join(blocks) + "\n"

    terms = yaml.safe_load(GLOSSARY_YML.read_text(encoding="utf-8"))["terms"]

    flat = {entry["term"]: " ".join(entry["definition"].split()) for entry in terms}
    GLOSSARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    GLOSSARY_JSON.write_text(json.dumps(flat, indent=2, sort_keys=True), encoding="utf-8")

    # Static include so the site renders with zero code execution (no CI Python).
    GLOSSARY_INCLUDE.parent.mkdir(parents=True, exist_ok=True)
    GLOSSARY_INCLUDE.write_text(_static_markdown(terms), encoding="utf-8")

    print_status({"status": "glossary", "terms": len(flat)})


def acic() -> None:
    """Build deterministic, precomputed assets for the ACIC proxy stress test."""
    OUTPUT_DIR = APP_DIR / "bundles" / "acic_proxy_stress_test"
    DATA_PATH = OUTPUT_DIR / "data.csv"
    EDGES_PATH = OUTPUT_DIR / "edges.csv"

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

    print_status(metrics)


def nasa_structural() -> None:
    """Build a frozen structural Causal SHAP prototype for NASA clean v3."""
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

    print_status(summary)
