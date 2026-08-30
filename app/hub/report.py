"""One self-contained HTML report of everything the session established.

Plain sentences with the actual numbers, the tables and charts already
computed, both graphs, and a code appendix showing the exact library calls.
Charts arrive as base64 PNGs and the graphs as static SVG, so the file has no
external dependencies and can be mailed as-is.

The honesty rules travel with the results: the arm that produced an
attribution, the grade of a calibrated SCM, illustrative-cost banners, and the
NASA-topology wording are part of the report, not optional decoration.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Mapping, Sequence

CSS = """
body { font-family: Georgia, serif; color: #1a1a1a; background: #fdfcfa;
  max-width: 900px; margin: 0 auto; padding: 32px 24px 80px; line-height: 1.55; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 34px 0 8px; border-bottom: 2px solid #111; padding-bottom: 4px; }
.meta { color: #6b6b6b; font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
.pill { display: inline-block; font-family: ui-monospace, Consolas, monospace; font-size: 11px;
  padding: 2px 9px; border-radius: 12px; margin: 0 4px 4px 0; background: #eaf0f9; color: #1e4d8c; }
.pill.warn { background: #fdf3e7; color: #b45309; }
.pill.bad { background: #fbeaea; color: #a32020; }
.pill.ok { background: #e8f5ee; color: #1a7f4b; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 10px 0; }
th { font-family: ui-monospace, Consolas, monospace; font-size: 10.5px; text-transform: uppercase;
  letter-spacing: 0.06em; color: #6b6b6b; text-align: left; padding: 5px 8px;
  border-bottom: 1px solid #111; }
td { padding: 5px 8px; border-bottom: 1px solid #e2ddd6; }
img { max-width: 100%; border: 1px solid #e2ddd6; border-radius: 4px; margin: 8px 0; }
.note { background: #fdf3e7; border-left: 2px solid #b45309; padding: 8px 12px;
  border-radius: 0 4px 4px 0; font-size: 13.5px; margin: 10px 0; }
pre { background: #1b1a16; color: #e8e4da; border-radius: 5px; padding: 10px 12px;
  font-family: ui-monospace, Consolas, monospace; font-size: 11px; line-height: 1.5;
  overflow-x: auto; }
svg { max-width: 100%; height: auto; border: 1px solid #e2ddd6; border-radius: 4px; }
.footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid #e2ddd6;
  color: #6b6b6b; font-size: 12px; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _table(rows: Sequence[Mapping[str, object]], columns: Sequence[str],
           limit: int = 30) -> str:
    if not rows:
        return "<p>Nothing to show.</p>"
    head = "".join(f"<th>{_esc(c)}</th>" for c in columns)
    body = []
    for row in list(rows)[:limit]:
        cells = []
        for c in columns:
            value = row.get(c, "")
            if isinstance(value, float):
                value = f"{value:.4g}"
            elif isinstance(value, bool):
                value = "yes" if value else "-"
            cells.append(f"<td>{_esc(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    more = f"<p class='meta'>... and {len(rows) - limit} more rows</p>" if len(rows) > limit else ""
    return f"<table><tr>{head}</tr>{''.join(body)}</table>{more}"


def _figure(b64: str | None, caption: str) -> str:
    if not b64:
        return ""
    return f'<img alt="{_esc(caption)}" src="data:image/png;base64,{b64}"/>'


def build_report(
    *,
    dataset_label: str,
    dataset_note: str,
    n_rows: int,
    outcome: str,
    features: Sequence[str],
    naive: Mapping[str, object] | None,
    discover_m1: Mapping[str, float] | None,
    graph_summary: Mapping[str, object] | None,
    truth_svg: str,
    current_svg: str,
    flags: Mapping[str, object] | None,
    shap: Mapping[str, object] | None,
    policy: Mapping[str, object] | None,
    code_appendix: Sequence[tuple[str, str]],
) -> str:
    """Assemble the report from whatever stages have actually run."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts: list[str] = []
    parts.append(f"<h1>Guided causal discovery — session report</h1>")
    parts.append(
        f'<p class="meta">{_esc(dataset_label)} · {n_rows:,} rows · generated {stamp} '
        f"· cross-sectional v1 (one row per unit, no time ordering)</p>"
    )
    if dataset_note:
        parts.append(f'<div class="note">{_esc(dataset_note)}</div>')
    parts.append(
        f"<p>Outcome under study: <b>{_esc(outcome)}</b>, over {len(features)} candidate "
        "features. No lever was specified in advance: which node is worth acting on is "
        "what the pipeline is for.</p>"
    )

    # ---- naive benchmark -------------------------------------------------
    if naive:
        fit = naive["fit"]
        shares = sorted(naive["shares"].items(), key=lambda p: -p[1])
        top_name, top_share = shares[0]
        parts.append("<h2>1 · What the model listened to (naive SHAP)</h2>")
        parts.append(
            f"<p>A {_esc(fit.model_type)} model of <b>{_esc(outcome)}</b> reached "
            f"{_esc(fit.stat_name)} {fit.stat_value:.3f}. Ordinary SHAP placed the "
            f"largest share of predictive credit on <b>{_esc(top_name)}</b> "
            f"({top_share:.1f}%). Predictive credit reports what the model used; it is "
            "not evidence that acting on a feature would move the outcome.</p>"
        )
        parts.append(_figure(naive.get("plot"), "naive SHAP importance"))

    # ---- structure -------------------------------------------------------
    parts.append("<h2>2 · Structure</h2>")
    if graph_summary:
        source = str(graph_summary.get("source", ""))
        n_undirected = int(graph_summary.get("n_undirected", 0))
        ledger = list(graph_summary.get("ledger", []))
        origin = (
            "after expert adjudication in the surgery theater"
            if source == "recovered" else
            "as produced by the discovery algorithm (no expert edits)"
            if source == "discovered" else "the bundled reference graph"
        )
        parts.append(
            f"<p>The working graph has {graph_summary.get('n_edges', '?')} directed edges, "
            f"{origin}. {n_undirected} edge pair(s) remain undirected in the underlying "
            "equivalence class; where a direction is shown for these, it is a labelled "
            "representative choice, not a data-identified finding.</p>"
        )
        if ledger:
            parts.append("<p><b>Constraint ledger</b> — every judgement recorded:</p><ol>")
            parts.extend(f"<li>{_esc(entry)}</li>" for entry in ledger)
            parts.append("</ol>")
        if discover_m1:
            caveat = " (directed metrics are representative-dependent)" if n_undirected else ""
            parts.append(
                f"<p>Against the sealed answer key{caveat}: skeleton F1 "
                f"{discover_m1['skeleton_f1']:.2f}, directed F1 {discover_m1['f1']:.2f}, "
                f"SHD {discover_m1['shd']}.</p>"
            )
    if current_svg:
        parts.append("<p><b>Working graph</b></p>")
        parts.append(current_svg)
    if truth_svg:
        parts.append("<p><b>Sealed answer key</b> (known truth for this bundled dataset)</p>")
        parts.append(truth_svg)

    # ---- flags -----------------------------------------------------------
    if flags and flags.get("status") == "ok":
        parts.append("<h2>3 · Depth-detector flags</h2>")
        parts.append(
            f'<p><span class="pill">{_esc(flags.get("provider", ""))}</span>'
            '<span class="pill warn">advisory diagnostic, never a causal claim</span>'
            + ('<span class="pill bad">not cleared for circulation</span>'
               if not flags.get("cleared") else "")
            + "</p><p>Per-node depth signals mark where to distrust the current "
            "attribution and look again. The three channels measure different things "
            "and are reported separately.</p>"
        )
        parts.append(_table(
            list(flags.get("records", [])),
            ["feature", "h0_z", "h0_flagged", "h1_z", "h1_flagged",
             "eig_z", "eig_flagged"],
        ))

    # ---- causal shap -----------------------------------------------------
    if shap:
        comparison = shap["comparison"]
        parts.append("<h2>4 · Attribution under the graph (causal SHAP)</h2>")
        arm_line = (
            f'<span class="pill">arm: {_esc(shap["arm"])}</span> '
            f"<span class='pill'>τ naive-vs-causal {comparison['kendall_tau']:.2f}</span>"
        )
        tau_naive = comparison.get("tau_vs_truth_standard")
        tau_causal = comparison.get("tau_vs_truth_causal")
        if tau_naive is not None:
            arm_line += (
                f' <span class="pill warn">τ vs truth: naive {tau_naive:.2f}</span>'
                f' <span class="pill ok">causal {tau_causal:.2f}</span>'
            )
        parts.append(f"<p>{arm_line}</p>")
        parts.append(f'<div class="note">{_esc(shap["arm_note"])}</div>')
        if tau_naive is not None:
            direction_word = "improves on" if tau_causal > tau_naive else "does not improve on"
            parts.append(
                f"<p>Against the known intervention truth, the causal attribution "
                f"({tau_causal:.2f}) {direction_word} the naive benchmark "
                f"({tau_naive:.2f}) in rank agreement.</p>"
            )
        changes = sorted(comparison["rank_changes"].items(),
                         key=lambda p: -abs(p[1]["change"]))
        truth = comparison.get("true_effects") or {}
        truth_total = sum(abs(v) for v in truth.values()) or None
        change_rows = []
        for name, d in changes:
            row = {"feature": name, "naive rank": d["standard_rank"],
                   "causal rank": d["causal_rank"], "change": d["change"]}
            if truth_total and name in truth:
                row["true effect %"] = 100.0 * abs(truth[name]) / truth_total
            change_rows.append(row)
        change_columns = ["feature", "naive rank", "causal rank", "change"]
        if truth_total:
            change_columns.append("true effect %")
            parts.append(
                '<div class="note">The graph governs eligibility: a feature with '
                "no directed path to the outcome under the working DAG has causal "
                "share zero by construction. Attribution is conditional on the "
                "hypothesized graph; the priced interventions then act on the "
                "outcome itself.</div>"
            )
        parts.append(_table(change_rows, change_columns))
        parts.append(_figure(shap.get("plot"), "naive vs causal attribution"))

    # ---- policy ----------------------------------------------------------
    if policy:
        ranking = policy["ranking"]
        fitted = policy["calibration"]
        best = ranking.best()
        parts.append("<h2>5 · Priced interventions (price and dice)</h2>")
        headline = (
            f"Under a budget of {ranking.budget:g} with beneficial direction "
            f"'{ranking.direction}', the best affordable action is "
            f"<b>{_esc(best.label)}</b> (expected benefit {best.benefit:.3g} at cost "
            f"{best.cost:.3g})." if best is not None else
            "No action was simultaneously affordable and confident under the current "
            "constraints."
        )
        parts.append(
            f"<p>{headline} Benefit is measured by simulating each action through the "
            f"structural model against shared exogenous draws (SCM grade: "
            f"{_esc(fitted.grade)}), never read off an attribution score.</p>"
        )
        if policy.get("arm_note"):
            parts.append(f'<div class="note">{_esc(policy["arm_note"])}</div>')
        parts.append(
            '<div class="note">Cost sheet is ILLUSTRATIVE — not domain-reviewed. '
            "Outputs are recommendation candidates worth testing, not recommendations."
            "</div>"
        )
        parts.append(_table(policy["table"].to_dict("records"),
                            ["action", "benefit", "cost", "ratio",
                             "p_unit_benefit", "feasible", "screened_out"]))
        estimates_table = policy.get("estimates_table")
        if estimates_table is not None and not estimates_table.empty:
            parts.append(
                "<p><b>Targeted estimates</b> — one double-robust functional per "
                "surviving lever (cross-fitted AIPW for the modified treatment "
                "policy, adjustment set = the lever's parents under the working "
                "graph), with Haneuse–Rotnitzky feasibility verdicts:</p>"
            )
            parts.append(_table(estimates_table.to_dict("records"),
                                ["action", "adjustment", "dr_benefit", "scm_benefit",
                                 "dr_se", "ci95", "feasibility", "notes"]))
        parts.append("<p><b>Screened before pricing</b> — nothing is dropped silently:</p>")
        parts.append(_table(policy["screened"].to_dict("records"),
                            ["node", "screened_out"], limit=60))
        parts.append(_figure(policy.get("pareto_plot"), "benefit against cost"))

    # ---- code appendix ---------------------------------------------------
    ran = [(label, code) for label, code in code_appendix if code]
    if ran:
        parts.append("<h2>Appendix · The code that ran</h2>")
        parts.append("<p>Generated at launch from the exact values each stage received.</p>")
        for label, code in ran:
            parts.append(f"<p><b>{_esc(label)}</b></p><pre>{_esc(code)}</pre>")

    parts.append(
        '<div class="footer">Produced by the Guided Causal Discovery Hub. '
        "Synthetic or source-aligned simulated data; topology-level claims only "
        "(say “NASA-topology simulation”, never “NASA effect”). Structural results "
        "are research prototypes.</div>"
    )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Hub session report</title><style>{CSS}</style></head>"
        f"<body>{''.join(parts)}</body></html>"
    )
