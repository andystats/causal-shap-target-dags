"""Causal SHAP tutorial app — a six-rung ladder companion to the paper.

The ladder is the product: each rung is a nav panel with a guided story read
from precomputed bundles, and rungs 1 and 4 add a live, torch-free explore path
(causal-learn discovery and an MVN validation generator).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib
import pandas as pd
from scipy.stats import kendalltau
from shiny import App, render, ui

from causal_shap.bundles import BundleRepository, read_json
from causal_shap.discovery import compare_graphs, run_ges, run_pc
from causal_shap.graphs import load_edges_csv
from causal_shap.seeds import SEED_VALIDATION_SCENARIOS
from causal_shap.validation import (
    ConfounderSpec,
    MVNGenerator,
    SimulationSpec,
    constant_tau,
    fit_baseline,
    fit_generator,
    reference_covariates,
    simulate,
)
from stages.common import callout, csv_data, image_html, metric_cards, table_html, term

VALIDATION_FEATURES = ["age", "comorbidity", "hydration"]

matplotlib.use("Agg")

APP_DIR = Path(__file__).resolve().parent
REPOSITORY = BundleRepository(APP_DIR, APP_DIR / "bundles" / "manifest.json")
BUNDLE_ERRORS = REPOSITORY.validate()  # the bundle set is fixed for the process; check once
LADDER_SVG = APP_DIR / "assets" / "ladder.svg"

DATASET_LABELS = {
    "toy_chain_fork_collider": "Toy: chain / fork / collider",
    "layered_ladder": "Ladder: 3 tiers",
    "acic_proxy_stress_test": "ACIC proxy stress test",
    "nasa_renal_clean_v3": "NASA renal-stone (flagship)",
}
LIVE_DISCOVERY = {"toy_chain_fork_collider", "layered_ladder", "acic_proxy_stress_test"}
RUNG_TITLES = [
    "Vanilla SHAP",
    "Causal discovery",
    "Complexity score",
    "Causal SHAP",
    "Validation",
    "Iteration",
]


# ---------------------------------------------------------------------------
# Guided content per rung
# ---------------------------------------------------------------------------
def _attribution(bundle_id: str) -> pd.DataFrame | None:
    bundle = REPOSITORY.get(bundle_id)
    if "causal_shap" in bundle.stages and "attribution" in bundle.stages["causal_shap"]:
        return csv_data(str(bundle.stages["causal_shap"]["attribution"]))
    if bundle_id == "acic_proxy_stress_test":
        return csv_data(str(bundle.paths["importance"]))
    return None


def rung_vanilla(bundle_id: str) -> str:
    bundle = REPOSITORY.get(bundle_id)
    homunculus = bundle.stages.get("causal_shap", {}).get("homunculus")
    body = [
        '<div class="eyebrow">RUNG 0 · VANILLA SHAP</div>',
        "<h2>The homunculus: attribution swells toward the outcome</h2>",
        f'<p class="lede">Ordinary SHAP rewards whatever predicts well. When a {term("proxy")} sits next to '
        f"the outcome, it gets credit it does not causally deserve — the graph bloats like a homunculus.</p>",
    ]
    attribution = _attribution(bundle_id)
    if attribution is not None and "Interventional truth" in set(attribution["method"]):
        body.append(_attribution_contrast(attribution))
    if homunculus:
        body.append(image_html(Path(homunculus), "Attribution homunculus: truth vs ordinary vs structural"))
    if bundle_id == "nasa_renal_clean_v3":
        body.append(
            callout(
                "On the flagship NASA problem the failure is honest, not theatrical: ordinary and DAG-ordering "
                "SHAP are statistically tied. The dramatic proxy inflation lives in the teaching datasets above.",
                "warning",
            )
        )
    else:
        body.append(
            callout(
                f'The {term("collider", "collider/proxy")} carries large ordinary-SHAP mass but zero total effect. '
                "That gap is the whole motivation for the rungs below."
            )
        )
    return "".join(body)


def _attribution_contrast(attribution: pd.DataFrame) -> str:
    pivot = attribution.pivot(index="variable", columns="method", values="normalized_importance")
    truth = pivot["Interventional truth"]
    cards: list[tuple[str, str, str | None]] = []
    for method in pivot.columns:
        if method == "Interventional truth":
            continue
        tau = kendalltau(pivot[method].rank(), truth.rank()).statistic
        cards.append(("τ vs truth", f"{tau:+.2f}", method))
    return metric_cards(cards)


def rung_discovery(bundle_id: str) -> str:
    bundle = REPOSITORY.get(bundle_id)
    if "discovery" not in bundle.stages:
        return (
            '<div class="eyebrow">RUNG 1 · CAUSAL DISCOVERY</div>'
            "<h2>Discovery is a teaching move here</h2>"
            + callout(
                "The NASA graph is source-exact and taken as given, so discovery is demonstrated on the "
                "teaching datasets. Switch datasets to run PC and GES against a known truth.",
                "warning",
            )
        )
    record = read_json(bundle.stages["discovery"]["result"])
    rows = []
    for name, result in record["algorithms"].items():
        rows.append(
            {
                "algorithm": name,
                "precision": result["precision"],
                "recall": result["recall"],
                "skeleton F1": result["skeleton_f1"],
                "spurious": result["n_spurious"],
                "missed": result["n_missed"],
            }
        )
    table = pd.DataFrame(rows)
    return (
        '<div class="eyebrow">RUNG 1 · CAUSAL DISCOVERY</div>'
        '<h2>Learn structure — but treat it as a hypothesis</h2>'
        f'<p class="lede">Algorithms recover a {term("CPDAG")}, not the truth. They disagree with each other, '
        "which is exactly why the next rung scores how much to trust them.</p>"
        + metric_cards([("Cross-algorithm disagreement", f"{record['cross_algorithm_disagreement']:.2f}", "1 − mean skeleton F1")])
        + table_html(table)
        + callout('"Causal discovery is a tool, not an oracle." Use the Explore tab to run it live.')
    )


def rung_complexity(bundle_id: str) -> str:
    bundle = REPOSITORY.get(bundle_id)
    report = read_json(bundle.stages["complexity"]["report"])
    subscore_rows = pd.DataFrame(
        [
            {"subscore": name, "value": f"{sub['value']:.2f}", "reading": sub["rationale"]}
            for name, sub in report["subscores"].items()
        ]
    )
    band_kind = {"low": "callout", "moderate": "callout", "high": "warning"}[report["band"]]
    return (
        '<div class="eyebrow">RUNG 2 · COMPLEXITY SCORE</div>'
        '<h2>PSCI v0: how much causal care does this problem need?</h2>'
        + callout("PSCI v0 is provisional — a transparent placeholder for the authors' final score.", "warning")
        + metric_cards(
            [
                ("PSCI total", f"{report['total']:.0f}", "0–100"),
                ("Band", report["band"].title(), report["score_name"] + " v" + report["score_version"]),
            ]
        )
        + table_html(subscore_rows)
        + callout(report["recommendations"][0], band_kind)
    )


def rung_causal_shap(bundle_id: str) -> str:
    attribution = _attribution(bundle_id)
    body = [
        '<div class="eyebrow">RUNG 3 · CAUSAL SHAP</div>',
        "<h2>Propagate the intervention through the DAG</h2>",
        '<p class="lede">Structural Causal SHAP asks what changes downstream under do(X=x). Credit flows to '
        "upstream causes; the proxy deflates.</p>",
    ]
    if bundle_id == "nasa_renal_clean_v3":
        summary = read_json(REPOSITORY.get(bundle_id).paths["structural_summary"])
        body.append(
            metric_cards(
                [
                    ("Kendall τ", f"{summary['kendall_tau_vs_truth']:.3f}", "structural vs truth"),
                    ("Top-5 recovery", f"{100 * summary['top5_recovery']:.0f}%", "all five targets"),
                    ("PBI", f"{summary['pbi']:.3f}", "near zero"),
                ]
            )
        )
        body.append(image_html(REPOSITORY.get(bundle_id).stages["causal_shap"]["distortion"], "NASA distortion profile"))
    elif attribution is not None:
        pivot = attribution.pivot(index="variable", columns="method", values="normalized_importance")
        body.append(table_html(pivot.reset_index().rename(columns={"index": "variable"}), index=False))
    body.append(callout("Structural attribution corrects the direction, though it is a prototype, not a proof."))
    return "".join(body)


def rung_validation(bundle_id: str) -> str:
    bundle = REPOSITORY.get("nasa_renal_clean_v3")
    scorecard = csv_data(str(bundle.stages["validation"]["scorecard"]))
    estimands = csv_data(str(bundle.stages["validation"]["estimands"]))
    layered = estimands[estimands["scenario"] == "layered_confounding"][["estimand", "value", "mc_std_error"]]
    return (
        '<div class="eyebrow">RUNG 4 · SIMULATION VALIDATION</div>'
        "<h2>Credence-style: run the ladder where every answer is known</h2>"
        f'<p class="lede">Layer several parameters at once — heterogeneous τ(X), multiple confounders, measurement '
        f"bias — then watch a naive {term('estimand', 'estimate')} drift from the known truth.</p>"
        + table_html(scorecard.rename(columns={"true_ate": "true ATE", "naive_ate": "naive ATE", "naive_drift": "naive drift"}))
        + "<h3>Several estimands, one dataset (layered confounding)</h3>"
        + table_html(layered)
        + callout("Truth stays fixed near the true ATE; only the naive estimate drifts as confounding and bias stack up.")
    )


def rung_iteration(bundle_id: str) -> str:
    return (
        '<div class="eyebrow">RUNG 5 · THOUGHTFUL ITERATION</div>'
        "<h2>Climb down when the score says so</h2>"
        '<p class="lede">The ladder is a loop. A high complexity score or wide discovery disagreement sends you '
        "back to refine the DAG, add domain constraints, and re-validate before trusting any attribution.</p>"
        '<div class="flow">'
        "<div><strong>Discover</strong><span>Learn structure, note disagreement</span></div><b>→</b>"
        "<div><strong>Score</strong><span>PSCI flags fragility</span></div><b>→</b>"
        "<div class=\"active\"><strong>Attribute + validate</strong><span>Structural SHAP, known-truth checks</span></div>"
        "</div>"
        + callout(
            'The nine-step Causal Roadmap makes each assumption explicit. "A wrong DAG is better than no DAG, '
            'because at least we can critique and refine it."'
        )
    )


RUNG_RENDERERS = [rung_vanilla, rung_discovery, rung_complexity, rung_causal_shap, rung_validation, rung_iteration]


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
APP_CSS = """
<style>
body { background:#f8fafc!important; color:#0f172a!important; font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important; }
.page { max-width:1180px; margin:0 auto; padding:22px 18px 56px; }
.hero { display:grid; grid-template-columns:1fr 320px; gap:24px; align-items:center; margin-bottom:20px; padding-bottom:18px; border-bottom:1px solid #cbd5e1; }
.hero h1 { margin:0; font-size:2.05rem; letter-spacing:-0.02em; }
.hero p { color:#475569; margin:8px 0 0; max-width:640px; }
.hero .ladder-art { width:100%; }
.layout { display:grid; grid-template-columns:230px minmax(0,1fr); gap:20px; align-items:start; }
.controls { position:sticky; top:12px; }
.panel { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; min-width:0; }
.eyebrow { color:#1d4ed8; font-weight:750; letter-spacing:.09em; font-size:.74rem; }
h2 { margin:.35rem 0 .6rem; font-size:1.5rem; letter-spacing:-0.01em; }
h3 { margin:1.2rem 0 .5rem; font-size:1.08rem; }
.lede { color:#475569; font-size:1.02rem; max-width:760px; }
.metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:16px 0; }
.metric { border:1px solid #e2e8f0; border-radius:9px; padding:12px; }
.metric-label { color:#64748b; font-size:.75rem; }
.metric-value { font-size:1.4rem; font-weight:770; letter-spacing:-0.01em; }
.metric-detail { color:#64748b; font-size:.74rem; }
.figure { width:100%; height:auto; display:block; border-radius:9px; margin:14px 0; }
.callout,.warning { padding:12px 14px; border-radius:9px; margin-top:14px; font-size:.94rem; }
.callout { background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a8a; }
.warning { background:#fffbeb; border:1px solid #fde68a; color:#92400e; }
.term { border-bottom:1px dotted #2563eb; cursor:help; }
.table-wrap { overflow-x:auto; margin:12px 0; }
.data-table { width:100%; border-collapse:collapse; font-size:.9rem; }
.data-table th { text-align:left; color:#64748b; font-weight:650; border-bottom:2px solid #e2e8f0; padding:7px 10px; }
.data-table td { border-bottom:1px solid #f1f5f9; padding:7px 10px; }
.flow { display:grid; grid-template-columns:1fr auto 1fr auto 1fr; gap:10px; align-items:center; margin:20px 0; }
.flow div { border:1px solid #cbd5e1; border-radius:9px; padding:13px; }
.flow div.active { border-color:#2563eb; background:#eff6ff; }
.flow span { display:block; color:#64748b; font-size:.82rem; margin-top:4px; }
.explore { margin-top:18px; padding-top:16px; border-top:1px dashed #cbd5e1; }
.provenance { color:#64748b; font-size:.8rem; margin-top:10px; }
.footer { margin-top:26px; padding-top:16px; border-top:1px solid #cbd5e1; color:#64748b; font-size:.85rem; text-align:center; }
.footer a { color:#1d4ed8; }
@media (max-width:900px){ .hero{grid-template-columns:1fr}.layout{grid-template-columns:1fr}.controls{position:static}.flow{grid-template-columns:1fr}.flow>b{transform:rotate(90deg);text-align:center} }
</style>
"""


def _rung_panel(index: int, title: str) -> ui.NavPanel:
    controls: list = [ui.output_ui(f"rung_{index}")]
    if index == 1:
        controls += [
            ui.div(
                ui.h3("Explore: run discovery live"),
                ui.input_slider("pc_alpha", "PC significance α", min=0.01, max=0.2, value=0.05, step=0.01),
                ui.input_select("disc_algo", "Algorithm", {"PC": "PC", "GES": "GES"}, selected="PC"),
                ui.output_ui("discovery_live"),
                class_="explore",
            )
        ]
    if index == 4:
        controls += [
            ui.div(
                ui.h3("Explore: layer parameters live"),
                ui.input_slider("tau_value", "Baseline effect τ", min=0.0, max=3.0, value=1.0, step=0.25),
                ui.input_slider("conf_strength", "Latent confounding strength", min=0.0, max=2.5, value=0.0, step=0.25),
                ui.output_ui("validation_live"),
                class_="explore",
            )
        ]
    return ui.nav_panel(f"{index} · {title}", *controls, value=f"rung{index}")


app_ui = ui.page_fluid(
    ui.HTML(APP_CSS),
    ui.div(
        ui.div(
            ui.div(
                ui.h1("Causal SHAP for target DAGs"),
                ui.p("A tutorial ladder: watch ordinary SHAP fail, then climb through discovery, a complexity score, structural attribution, and simulation validation."),
            ),
            ui.HTML(f'<div class="ladder-art">{LADDER_SVG.read_text(encoding="utf-8") if LADDER_SVG.exists() else ""}</div>'),
            class_="hero",
        ),
        ui.div(
            ui.div(
                ui.input_select("dataset", "Dataset", choices=DATASET_LABELS, selected="toy_chain_fork_collider"),
                ui.output_ui("dataset_note"),
                class_="panel controls",
            ),
            ui.div(
                ui.navset_pill_list(
                    *[_rung_panel(index, title) for index, title in enumerate(RUNG_TITLES)],
                    widths=(3, 9),
                ),
                class_="panel",
            ),
            class_="layout",
        ),
        ui.HTML(
            '<div class="footer">Companion site — '
            '<a href="https://andystats.github.io/causal-shap-target-dags/">the ladder, cheatsheets &amp; glossary</a>'
            ' · <a href="https://github.com/andystats/causal-shap-target-dags">source on GitHub</a>. '
            "All datasets are synthetic; the complexity score is provisional.</div>"
        ),
        class_="page",
    ),
)


def server(input, output, session):
    def make_rung(index: int):
        @render.ui
        def _renderer():
            return ui.HTML(RUNG_RENDERERS[index](input.dataset()))

        return _renderer

    for index in range(len(RUNG_TITLES)):
        output(id=f"rung_{index}")(make_rung(index))

    @output
    @render.ui
    def dataset_note():
        if BUNDLE_ERRORS:
            return ui.HTML(callout(f"Bundle validation failed: {BUNDLE_ERRORS[0]}", "warning"))
        bundle = REPOSITORY.get(input.dataset())
        return ui.HTML(f'<div class="provenance"><strong>{bundle.label}</strong><br>{bundle.description}<br>{bundle.provenance}</div>')

    @output
    @render.ui
    def discovery_live():
        dataset_id = input.dataset()
        if dataset_id not in LIVE_DISCOVERY:
            return ui.HTML(callout("Live discovery runs on the teaching and ACIC datasets; NASA is taken as given.", "warning"))
        bundle = REPOSITORY.get(dataset_id)
        data = csv_data(str(bundle.paths["data"]))
        graph = load_edges_csv(str(bundle.paths["edges"]))
        frame = data[[c for c in data.columns if c in set(graph.nodes())]]
        result = run_pc(frame, alpha=input.pc_alpha()) if input.disc_algo() == "PC" else run_ges(frame)
        comparison = compare_graphs(sorted(result.directed_edges), sorted(graph.edges()))
        return ui.HTML(
            metric_cards(
                [
                    ("Skeleton F1", f"{comparison.skeleton_f1:.2f}", "vs known truth"),
                    ("Directed F1", f"{comparison.f1:.2f}", None),
                    ("Spurious / missed", f"{len(comparison.spurious)} / {len(comparison.missed)}", "edges"),
                ]
            )
        )

    @output
    @render.ui
    def validation_live():
        baseline, generator = _validation_baseline_and_generator()
        confounders = ()
        if input.conf_strength() > 0:
            confounders = (ConfounderSpec("frailty", outcome_strength=input.conf_strength(), treatment_strength=input.conf_strength()),)
        spec = SimulationSpec(baseline=baseline, tau=constant_tau(input.tau_value()), confounders=confounders)
        dataset = simulate(spec, generator, n=3000, seed=SEED_VALIDATION_SCENARIOS, subgroup_col="age")
        true_ate = dataset.estimands.set_index("estimand").loc["ATE", "value"]
        table = dataset.estimands[["estimand", "value", "mc_std_error"]]
        return ui.HTML(
            metric_cards(
                [
                    ("True ATE", f"{true_ate:.2f}", "known"),
                    ("Naive estimate", f"{dataset.naive_ate:.2f}", "difference in means"),
                    ("Naive drift", f"{dataset.naive_ate - true_ate:+.2f}", "bias"),
                ]
            )
            + table_html(table)
        )


@lru_cache(maxsize=1)
def _validation_baseline_and_generator():
    """Fit the baseline and generator once; neither depends on the live sliders."""
    real = reference_covariates(SEED_VALIDATION_SCENARIOS)
    baseline = fit_baseline(real, "Y", VALIDATION_FEATURES)
    generator = fit_generator(MVNGenerator(), real, VALIDATION_FEATURES, "Z")
    return baseline, generator


app = App(app_ui, server)
