from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from shiny import App, reactive, render, ui

from causal_shap.bundles import BundleRepository, read_json


matplotlib.use("Agg")

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
REPOSITORY = BundleRepository(PROJECT_DIR, APP_DIR / "bundles" / "manifest.json")
GUIDED_STEPS = 6


@lru_cache(maxsize=64)
def image_uri(path_text: str) -> str:
    path = Path(path_text)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@lru_cache(maxsize=16)
def csv_data(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text)


def image_html(path: Path, alt: str) -> str:
    return f'<img class="figure" src="{image_uri(str(path))}" alt="{alt}">'


def metric_cards(items: list[tuple[str, str, str | None]]) -> str:
    cards = []
    for label, value, detail in items:
        detail_html = f'<div class="metric-detail">{detail}</div>' if detail else ""
        cards.append(
            f'<div class="metric"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>{detail_html}</div>'
        )
    return f'<div class="metrics">{"".join(cards)}</div>'


def importance_plot_uri(data: pd.DataFrame, methods: tuple[str, ...], top_n: int = 10) -> str:
    subset = data[data["method"].isin(methods)].copy()
    top_features: list[str] = []
    for method in methods:
        rows = subset[subset["method"] == method].nlargest(top_n, "normalized_importance")
        top_features.extend(rows["variable"].tolist())
    top_features = list(dict.fromkeys(top_features))
    subset = subset[subset["variable"].isin(top_features)]
    pivot = subset.pivot(index="variable", columns="method", values="normalized_importance").fillna(0)
    order = pivot.max(axis=1).sort_values().index
    pivot = pivot.loc[order]

    colors = ["#111827", "#6b7280", "#d97706", "#2563eb", "#059669"]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(pivot))), facecolor="white")
    pivot.plot.barh(ax=ax, color=colors[: len(methods)], width=0.78)
    ax.set_xlabel("Normalized global importance")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda value, position: f"{100 * value:.0f}%")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def nasa_content(step: int) -> str:
    bundle = REPOSITORY.get("nasa_renal_clean_v3")
    data = csv_data(str(bundle.paths["data"]))
    model_metrics = csv_data(str(bundle.paths["model_metrics"])).iloc[0]
    attribution = csv_data(str(bundle.paths["attribution_metrics"]))
    structural = read_json(bundle.paths["structural_summary"])

    if step == 1:
        prevalence = data["nephrolithiasis"].mean()
        return (
            '<div class="eyebrow">SOURCE-ALIGNED PRIMARY ANALYSIS</div>'
            '<h2>Mission: find useful levers for renal-stone risk</h2>'
            '<p class="lede">Start with prediction, then reveal what the known NASA DAG changes—and what it does not.</p>'
            + metric_cards(
                [
                    ("Rows", f"{len(data):,}", "clean v3"),
                    ("Outcome rate", f"{100 * prevalence:.1f}%", "Nephrolithiasis"),
                    ("Eligible ancestors", "28", "1–6 directed hops"),
                    ("Graph", "51 / 75", "nodes / edges"),
                ]
            )
            + '<div class="callout">The graph topology is source-exact. Coefficients remain provisional simulation choices.</div>'
        )
    if step == 2:
        return (
            '<div class="eyebrow">STEP 2 · PREDICT</div>'
            '<h2>The first learner was XGBoost—and the ceiling is genuinely low</h2>'
            + metric_cards(
                [
                    ("XGBoost AUC", f'{model_metrics["auc"]:.3f}', "all 28 ancestors"),
                    ("Oracle AUC", "0.701", "true structural risk score"),
                    ("Brier score", f'{model_metrics["brier_score"]:.3f}', None),
                ]
            )
            + image_html(bundle.paths["auc_plot"], "AUC comparison of the oracle structural score and fitted learners")
        )
    if step == 3:
        return (
            '<div class="eyebrow">STEP 3 · REVEAL THE DAG</div>'
            '<h2>Prediction sees features; the Living DAG reveals positions</h2>'
            '<p class="lede">The target has 28 ancestors. Direct parents sit one hop away; prevention and exposure nodes can lie much farther upstream.</p>'
            + image_html(bundle.paths["dag"], "NASA renal-stone directed acyclic graph")
        )
    if step == 4:
        rows = attribution.set_index("method")
        return (
            '<div class="eyebrow">STEP 4 · COMPARE FAIRLY</div>'
            '<h2>Ordering alone does not earn the causal claim</h2>'
            + metric_cards(
                [
                    ("TreeSHAP PBI", f'{rows.loc["Ordinary TreeSHAP", "pbi"]:.3f}', None),
                    ("Matched ordinary PBI", f'{rows.loc["Ordinary interventional SHAP", "pbi"]:.3f}', None),
                    ("DAG-asymmetric PBI", f'{rows.loc["DAG-constrained asymmetric SHAP", "pbi"]:.3f}', None),
                ]
            )
            + image_html(bundle.paths["distance_plot"], "Distance-concentration curves for truth and three attribution methods")
            + '<div class="callout">With the background distribution matched, DAG ordering and ordinary permutations are essentially indistinguishable.</div>'
        )
    if step == 5:
        original = csv_data(str(PROJECT_DIR / "analysis" / "output" / "shap_nephrolithiasis_clean_v3" / "importance_comparison.csv"))
        structural_importance = csv_data(str(bundle.paths["structural_importance"]))
        combined = pd.concat([original, structural_importance], ignore_index=True, sort=False)
        uri = importance_plot_uri(
            combined,
            (
                "Interventional truth",
                "Ordinary TreeSHAP",
                "DAG-constrained asymmetric SHAP",
                "Structural intervention-propagating SHAP prototype",
            ),
            top_n=7,
        )
        return (
            '<div class="eyebrow">STEP 5 · PROPAGATE INTERVENTIONS</div>'
            '<h2>The structural prototype recovers the distributed truth</h2>'
            + metric_cards(
                [
                    ("Kendall τ", f'{structural["kendall_tau_vs_truth"]:.3f}', "prototype vs truth"),
                    ("Top-5 recovery", f'{100 * structural["top5_recovery"]:.0f}% ', "all five targets"),
                    ("PBI", f'{structural["pbi"]:.3f}', "near zero; slightly upstream"),
                    ("≤2-hop mass", f'{100 * structural["proximal_mass_distance_le_2"]:.1f}% ', "truth: 42.1%"),
                ]
            )
            + f'<img class="figure" src="{uri}" alt="Feature importance comparison including the structural Causal SHAP prototype">'
            + '<div class="warning">Prototype: 32 evaluation records × 32 backgrounds × 32 permutations. Scale and bootstrap before manuscript claims.</div>'
        )
    return (
        '<div class="eyebrow">STEP 6 · THE DECISION QUESTION</div>'
        '<h2>From feature order to intervention propagation</h2>'
        '<div class="flow"><div><strong>Ordinary SHAP</strong><span>Which observed features help the model predict?</span></div>'
        '<b>→</b><div><strong>DAG ordering</strong><span>In what sequence may information enter?</span></div>'
        '<b>→</b><div class="active"><strong>Structural Causal SHAP</strong><span>What changes downstream under do(X=x)?</span></div></div>'
        '<p class="lede">Next scientific gate: scale the structural estimator, repeat seeds and bootstraps, then move to Loss of Mission Objectives and NASA-like v4.</p>'
    )


def acic_content(step: int) -> str:
    bundle = REPOSITORY.get("acic_proxy_stress_test")
    importance = csv_data(str(bundle.paths["importance"]))
    metrics = read_json(bundle.paths["metrics"])
    standard_uri = importance_plot_uri(importance, ("Ordinary SHAP",), top_n=12)
    comparison_uri = importance_plot_uri(
        importance,
        ("Interventional truth", "Ordinary SHAP", "DAG-aware SHAP demo"),
        top_n=8,
    )

    if step == 1:
        return (
            '<div class="eyebrow stress">DESIGNED PEDAGOGIC STRESS TEST</div>'
            '<h2>Make the wrong-lever problem impossible to miss</h2>'
            '<p class="lede">Downstream proxies are intentionally predictive while having zero intervention effect by construction.</p>'
            + metric_cards(
                [
                    ("Rows", f'{metrics["rows"]:,}', None),
                    ("Features", str(metrics["features"]), None),
                    ("Held-out R²", f'{metrics["held_out_r2"]:.3f}', "strong predictive signal"),
                ]
            )
            + '<div class="warning">This dataset teaches the mechanism. It is not the primary NASA result.</div>'
        )
    if step == 2:
        return (
            '<div class="eyebrow stress">STEP 2 · ORDINARY SHAP</div>'
            '<h2>The predictive ranking rewards late mediators and proxies</h2>'
            f'<img class="figure" src="{standard_uri}" alt="Ordinary SHAP ranking in the ACIC proxy stress test">'
        )
    if step == 3:
        return (
            '<div class="eyebrow stress">STEP 3 · REVEAL THE KNOWN DAG</div>'
            '<h2>The graph separates predictive proximity from intervention leverage</h2>'
            + image_html(bundle.paths["dag"], "Pedagogic many-mediator and proxy DAG")
        )
    if step == 4:
        return (
            '<div class="eyebrow stress">STEP 4 · RERUN WITH CAUSAL KNOWLEDGE</div>'
            '<h2>Importance moves toward upstream causes</h2>'
            f'<img class="figure" src="{comparison_uri}" alt="Truth, ordinary SHAP, and DAG-aware SHAP comparison">'
        )
    if step == 5:
        return (
            '<div class="eyebrow stress">STEP 5 · SCORE THE REVEAL</div>'
            '<h2>The stress test creates a clear teaching contrast</h2>'
            + metric_cards(
                [
                    ("τ vs truth: ordinary", f'{metrics["tau_vs_truth_standard"]:.3f}', None),
                    ("τ vs truth: DAG-aware", f'{metrics["tau_vs_truth_causal"]:.3f}', None),
                    ("Proxy inflation", f'{metrics["proxy_inflation"]:.2f}×', None),
                    ("Upstream boost", f'{metrics["root_cause_boost"]:.2f}×', None),
                ]
            )
            + '<div class="warning">Fast precomputed demo estimate: 8 instances, 4 backgrounds, 8 permutations. Use it for intuition, not inference.</div>'
        )
    return (
        '<div class="eyebrow stress">STEP 6 · CARRY THE QUESTION FORWARD</div>'
        '<h2>Now test whether the lesson survives a source-aligned system</h2>'
        '<p class="lede">The app deliberately moves next to the NASA case, where the ordering-only advantage disappears and structural propagation has to earn the result.</p>'
    )


def lab_overview(bundle_id: str) -> str:
    bundle = REPOSITORY.get(bundle_id)
    is_primary = bundle.kind == "source_aligned_primary"
    eyebrow_class = "eyebrow" if is_primary else "eyebrow stress"
    eyebrow = "LAB OVERVIEW · SOURCE-ALIGNED ANALYSIS" if is_primary else "LAB OVERVIEW · DESIGNED STRESS TEST"
    result = (
        "Matched ordinary and ordering-only SHAP are effectively tied; the small structural prototype is promising but not yet publication-scale."
        if is_primary
        else "The teaching bundle makes proxy over-credit visible; its fast estimate is intuition, not primary evidence."
    )
    next_gate = (
        "Scale structural propagation, add paired uncertainty and seed replication, then run the longer-path endpoint and NASA-like regime."
        if is_primary
        else "Carry the mechanism into the source-aligned NASA analysis and require it to survive a fair matched-background control."
    )
    return (
        f'<div class="{eyebrow_class}">{eyebrow}</div>'
        '<h2>One workspace, two views of the same frozen evidence</h2>'
        '<p class="lede"><strong>Guided story</strong> controls the reveal for a reader or recording. '
        '<strong>Lab overview</strong> exposes the estimands, evidence flow, and tool architecture at once.</p>'
        '<section class="lab-section"><h3>Scientific concept</h3>'
        '<div class="lab-flow">'
        '<div><strong>Known DAG + equations</strong><span>Declare topology and structural mechanisms.</span></div><b>→</b>'
        '<div><strong>Synthetic data + frozen truth</strong><span>Generate records and total intervention effects independently of SHAP.</span></div><b>→</b>'
        '<div><strong>One fixed learner</strong><span>Hold the prediction model, records, and background contract constant.</span></div><b>→</b>'
        '<div><strong>Three attribution questions</strong><span>Prediction, DAG ordering, and structural intervention propagation.</span></div><b>→</b>'
        '<div><strong>Recovery + decisions</strong><span>Rank recovery, PBI/POA, uncertainty, then feasibility and cost.</span></div>'
        '</div></section>'
        '<section class="lab-section"><h3>Method contract</h3>'
        '<div class="method-grid">'
        '<div><em>PREDICTIVE</em><strong>Ordinary SHAP</strong><span>Which observed features help this fitted model predict?</span></div>'
        '<div><em>ORDERING-ONLY CONTROL</em><strong>DAG-asymmetric SHAP</strong><span>What changes if feature arrival order must respect the DAG?</span></div>'
        '<div class="active"><em>STRUCTURAL PROTOTYPE</em><strong>Intervention-propagating SHAP</strong><span>What changes after do(X=x) propagates through descendants?</span></div>'
        '</div>'
        f'<div class="callout"><strong>Current read:</strong> {result}</div></section>'
        '<section class="lab-section"><h3>Deterministic tool/demo</h3>'
        '<div class="tool-flow">'
        '<div><strong>Versioned local bundles</strong><span>Data, DAG, model, truth, manifests, metrics, and figures</span></div><b>→</b>'
        '<div class="split"><strong>Python Shiny presentation</strong><span>Guided story for narration</span><span>Lab overview for audit and comparison</span></div><b>→</b>'
        '<div><strong>Paper + video</strong><span>Every displayed number traces to a checked artifact.</span></div>'
        '</div>'
        f'<div class="warning"><strong>Next gate:</strong> {next_gate}</div></section>'
    )


APP_CSS = """
<style>
body { background:#f8fafc!important; color:#111827!important; font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important; }
.page { max-width:1120px; margin:0 auto; padding:24px 18px 48px; }
.hero { margin-bottom:18px; padding-bottom:16px; border-bottom:1px solid #cbd5e1; }
.hero h1 { margin:0; font-size:2rem; }
.hero p { color:#475569; max-width:850px; margin:8px 0 0; }
.layout { display:grid; grid-template-columns:275px minmax(0,1fr); gap:16px; align-items:start; }
.panel { background:#fff; border:1px solid #cbd5e1; border-radius:10px; padding:16px; min-width:0; overflow:hidden; }
.controls { position:sticky; top:12px; }
.nav { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; }
.step { color:#64748b; font-size:.85rem; margin-top:12px; }
.eyebrow { color:#1d4ed8; font-weight:750; letter-spacing:.08em; font-size:.76rem; }
.eyebrow.stress { color:#b45309; }
h2 { margin:.35rem 0 .65rem; font-size:1.55rem; }
.lede { color:#475569; font-size:1.02rem; max-width:850px; }
.metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:10px; margin:14px 0; }
.metric { border:1px solid #e2e8f0; border-radius:8px; padding:11px; background:#fff; }
.metric-label { color:#64748b; font-size:.76rem; }
.metric-value { font-size:1.35rem; font-weight:780; }
.metric-detail { color:#64748b; font-size:.75rem; }
.figure { width:100%; height:auto; display:block; background:#fff; border-radius:8px; }
.callout,.warning { padding:11px 13px; border-radius:8px; margin-top:12px; }
.callout { background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a8a; }
.warning { background:#fffbeb; border:1px solid #fde68a; color:#92400e; }
.flow { display:grid; grid-template-columns:1fr auto 1fr auto 1fr; gap:10px; align-items:center; margin:24px 0; }
.flow div { border:1px solid #cbd5e1; border-radius:8px; padding:14px; background:#fff; }
.flow div.active { border-color:#2563eb; background:#eff6ff; }
.flow span { display:block; color:#64748b; font-size:.85rem; margin-top:5px; }
.lab-section { margin-top:24px; padding-top:18px; border-top:1px solid #e2e8f0; }
.lab-section h3 { margin:0 0 12px; font-size:1.05rem; }
.lab-flow { display:grid; grid-template-columns:repeat(5,minmax(120px,1fr)); gap:8px; align-items:stretch; overflow-x:auto; }
.lab-flow div { grid-column:span 1; min-width:0; border:1px solid #cbd5e1; border-radius:8px; padding:10px; background:#f8fafc; }
.lab-flow b { display:none; }
.tool-flow b { text-align:center; color:#64748b; }
.lab-flow span,.method-grid span,.tool-flow span { display:block; color:#64748b; font-size:.78rem; margin-top:5px; }
.method-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
.method-grid div { border:1px solid #cbd5e1; border-radius:8px; padding:12px; background:#fff; }
.method-grid div.active { border-color:#2563eb; background:#eff6ff; }
.method-grid em { display:block; color:#64748b; font-size:.68rem; font-style:normal; font-weight:750; letter-spacing:.06em; margin-bottom:5px; }
.tool-flow { display:grid; grid-template-columns:1fr auto 1.2fr auto 1fr; gap:10px; align-items:center; }
.tool-flow div { border:1px solid #cbd5e1; border-radius:8px; padding:12px; background:#f8fafc; }
.tool-flow .split span { border-top:1px solid #e2e8f0; padding-top:5px; }
.nav-note { margin-top:12px; padding:9px 10px; border-radius:7px; background:#f1f5f9; color:#475569; font-size:.78rem; }
@media (max-width:850px) { .layout{grid-template-columns:1fr}.controls{position:static}.flow,.lab-flow,.method-grid,.tool-flow{grid-template-columns:1fr}.flow>b,.lab-flow>b,.tool-flow>b{transform:rotate(90deg);text-align:center} }
</style>
"""


app_ui = ui.page_fluid(
    ui.HTML(APP_CSS),
    ui.div(
        ui.div(
            ui.h1("Causal SHAP for Living DAGs"),
            ui.p("A deterministic paper companion: start with prediction, reveal causal structure, test a fair control, then propagate interventions."),
            class_="hero",
        ),
        ui.div(
            ui.div(
                ui.div(
                    ui.input_select("bundle", "Dataset", choices=REPOSITORY.choices(), selected="nasa_renal_clean_v3"),
                    ui.input_radio_buttons("mode", "Experience", choices={"guided":"Guided story","lab":"Lab overview"}, selected="guided"),
                    ui.output_ui("navigation"),
                    ui.output_ui("step_status"),
                    ui.output_ui("bundle_status"),
                    class_="panel controls",
                ),
                ui.div(ui.output_ui("content"), class_="panel"),
                class_="layout",
            ),
        ),
        class_="page",
    ),
)


def server(input, output, session):
    step = reactive.Value(1)

    @reactive.effect
    @reactive.event(input.bundle)
    def _reset_bundle_step():
        step.set(1)

    @reactive.effect
    @reactive.event(input.previous)
    def _previous():
        step.set(max(1, step.get() - 1))

    @reactive.effect
    @reactive.event(input.next)
    def _next():
        step.set(min(GUIDED_STEPS, step.get() + 1))

    @output
    @render.ui
    def navigation():
        if input.mode() == "lab":
            return ui.HTML('<div class="nav-note">Switch datasets to compare the primary analysis with the labeled teaching stress test.</div>')
        return ui.div(
            ui.input_action_button("previous", "Previous"),
            ui.input_action_button("next", "Next", class_="btn-primary"),
            class_="nav",
        )

    @output
    @render.ui
    def step_status():
        if input.mode() == "lab":
            return ui.HTML('<div class="step"><strong>Lab overview</strong><br>Concept, method contract, and deterministic tool architecture. Switch to Guided story for the six-step reveal.</div>')
        return ui.HTML(f'<div class="step">Step {step.get()} of {GUIDED_STEPS}</div>')

    @output
    @render.ui
    def bundle_status():
        errors = REPOSITORY.validate()
        if errors:
            return ui.HTML(f'<div class="warning">Bundle validation failed: {errors[0]}</div>')
        bundle = REPOSITORY.get(input.bundle())
        label = "Primary source-aligned analysis" if bundle.kind == "source_aligned_primary" else "Designed pedagogic stress test"
        return ui.HTML(f'<div class="step"><strong>{label}</strong><br>Target: {bundle.target}<br>All displayed results are local and precomputed.</div>')

    @output
    @render.ui
    def content():
        if input.mode() == "lab":
            return ui.HTML(lab_overview(input.bundle()))
        html = nasa_content(step.get()) if input.bundle() == "nasa_renal_clean_v3" else acic_content(step.get())
        return ui.HTML(html)


app = App(app_ui, server)
