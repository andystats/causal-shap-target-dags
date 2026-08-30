"""Stage-side education: every control gets an explanation on demand.

Each topic is a short, honest paragraph in the study guide's voice, with the
occasional formula in plain HTML (no rendering dependency, so the hub works
offline). Tabs compose their own reader from topic keys; the drop-down reuses
the code-card visual so "learn" and "show the code" sit as siblings.
"""

from __future__ import annotations

import html

_PHI = (
    '<p style="font-family:var(--mono);font-size:.78rem;margin:6px 0">'
    "φ<sub>i</sub> = Σ<sub>S ⊆ F∖{i}</sub> "
    "[|S|!·(|F|−|S|−1)!/|F|!] · [v(S ∪ {i}) − v(S)]</p>"
)

TOPICS: dict[str, tuple[str, str]] = {
    "shap": (
        "SHAP (Shapley additive explanations)",
        "Splits a prediction among the features by averaging each feature's "
        "marginal contribution over orderings, the Shapley value from "
        "cooperative game theory:" + _PHI +
        "Everything turns on the value function v(S): what it means to 'know' "
        "a coalition S. Naive SHAP conditions by observation; the causal arm "
        "intervenes.",
    ),
    "naive": (
        "Why 'naive'?",
        "Nothing is wrong with the estimator: it faithfully reports what the "
        "fitted model listened to. It is naive only as an intervention guide, "
        "because a mediator or proxy near the outcome can absorb credit that "
        "causally belongs upstream (the trap this whole hub demonstrates).",
    ),
    "holdout": (
        "Holdout statistic",
        "A quarter of the rows are held out (stratified when the outcome is "
        "binary and both classes allow it); AUC is reported for binary "
        "outcomes and R² for continuous ones. It says the model is usable, "
        "not that it is causal.",
    ),
    "pc": (
        "PC algorithm",
        "Constraint-based discovery (Spirtes-Glymour-Scheines): start from a "
        "complete undirected graph, delete edges that fail conditional-"
        "independence tests (Fisher-Z here), then orient what logic compels "
        "(colliders first, then propagation). Fast and interpretable; "
        "directions are often left open, and that honesty is the CPDAG.",
    ),
    "ges": (
        "GES algorithm",
        "Score-based discovery: greedily add then prune edges to maximize a "
        "penalized fit score (BIC). Same output object as PC, an equivalence "
        "class, found by a different route; the two disagreeing on real data "
        "is a measurement, not a malfunction.",
    ),
    "alpha_pc": (
        "Significance α (PC)",
        "The p-value threshold for each conditional-independence test. Lower "
        "α deletes more edges (sparser graph, more missed edges); higher α "
        "keeps more (denser, more spurious). It tunes skepticism per test, "
        "not overall correctness.",
    ),
    "cpdag": (
        "CPDAG and the representative",
        "What discovery can actually identify from observational data is an "
        "equivalence class: some edges stay undirected because both "
        "directions fit the same independences. DAG-only tools receive ONE "
        "deterministic representative extension; the unresolved pairs travel "
        "along (dashed in the surgery view) so no one mistakes a choice for "
        "a finding.",
    ),
    "prep": (
        "What surgical prep is for",
        "An optional stage between discovery and surgery: gather node-level "
        "evidence from ANY source — an external depth detector, literature, "
        "expert priors — to decide where the surgeon's attention goes first. "
        "The evidence is advisory; only the surgeon's recorded judgements "
        "change the graph.",
    ),
    "channels": (
        "Reading detector channels",
        "A detector may report several per-node channels; z-scores are within-"
        "dataset, flagged above z > 1. Channels measure different things and "
        "are never averaged away; read them separately, and treat any "
        "composite as provisional. Flags mean 'look here again', never 'this "
        "node is causal'.",
    ),
    "surgery": (
        "Surgery operations",
        "Flip reverses an edge; Require asserts the shown direction; Forbid "
        "rejects it (resolving an unresolved pair the other way, or deleting "
        "a compelled edge); Remove deletes with both directions forbidden; "
        "Add asserts an edge the algorithm never proposed. Every operation "
        "lands in the ledger as a post-hoc expert judgement, and a cycle-"
        "creating edit is refused with the cycle named.",
    ),
    "ledger": (
        "The constraint ledger",
        "The versioned record of expert judgement: which edge, which verdict, "
        "when applied, and the stated rationale. It is what turns a DISCOVERED "
        "graph into a RECOVERED one, and it travels into the export so the "
        "adjudication is auditable.",
    ),
    "arms": (
        "Attribution arms",
        "Structural: calibrate linear/logistic equations on the current graph, "
        "then evaluate v(S) = E[f(X) | do(X<sub>S</sub> = x<sub>S</sub>)] by "
        "propagating each coalition through the equations (the frozen record's "
        "engine). Nonparametric: fit small conditional models P(X|parents) and "
        "propagate by sampling, for data where linearity is indefensible. Both "
        "are stamped on their results; they answer with different assumptions.",
    ),
    "ancestors": (
        "Why some features are 'structurally zero'",
        "The graph governs eligibility: only ancestors of the outcome under "
        "the CURRENT graph are attributed, exactly as in the frozen record. A "
        "feature with no directed path to the outcome cannot move it, so its "
        "causal share is zero by construction. Edit the graph and eligibility "
        "changes with it: attribution is conditional on the hypothesis.",
    ),
    "knobs": (
        "Permutations, explained rows, background draws",
        "Monte Carlo budget, not science settings. Permutations: how many "
        "feature orderings are averaged for the Shapley expectation. Explained "
        "rows: how many units get attributions. Background draws: samples "
        "standing in for the unknown exogenous state. More = smoother numbers, "
        "linearly slower; all seeded, so a rerun reproduces exactly.",
    ),
    "cost_sheet": (
        "The cost sheet",
        "Per node: whether it is manipulable at all, how far it may be moved "
        "(shift bounds), a fixed cost for touching it, and a per-unit cost for "
        "distance moved. Costs are charged on what YOU set, never on changes "
        "the model propagates downstream. The bundled sheets are illustrative "
        "until domain review; every output says so.",
    ),
    "grid": (
        "Why each lever appears twice",
        "Every manipulable lever is priced in BOTH allowed directions (its "
        "+shift and −shift). The direction that harms the outcome is kept in "
        "the table with its refusal reason rather than hidden, because "
        "'nothing is dropped silently' is the standing rule of this stage.",
    ),
    "alpha_floor": (
        "Confidence floor α",
        "An action is feasible only if the share of units it actually helps "
        "is at least 1 − α. With a rare binary outcome most units cannot "
        "change under any single lever, so a strict floor rules out "
        "everything; the renal default is therefore permissive, and the "
        "per-unit column is shown so you can judge.",
    ),
    "budget": (
        "Budget as the objective",
        "The optimum is the largest expected benefit that fits the budget. "
        "The benefit/cost ratio is shown as a diagnostic but never optimized: "
        "under per-unit costs the ratio is dose-invariant, so arbitrarily "
        "tiny cheap actions would tie for first. Slide the budget and watch "
        "the optimum change; that is the whole 'price' thesis.",
    ),
    "benefit": (
        "How benefit is measured",
        "Never from an attribution score. Each candidate action is simulated "
        "through the calibrated structural model against the SAME exogenous "
        "draws (a paired do() contrast), so the number is an intervention "
        "effect on the outcome, with the screening reasons reported for every "
        "node that never got priced.",
    ),
    "sandbox": (
        "The local sandbox",
        "Runs Python against the loaded data on THIS machine only (df, pd, "
        "np, plt are in scope; last expression or printed output is shown, "
        "figures are captured). It exists for honest data exploration before "
        "the pipeline starts; nothing it does feeds the stages.",
    ),
}


# Vendor-neutral explainer for the Surgical Prep stage: how ANY external
# depth detector plugs in, and how to bring results back by hand. A local
# module may replace this with runtime-specific documentation.
GENERIC_DETECTOR_DOCS = """
<details class="code-details"><summary>How an external detector plugs in</summary>
<div style="border:1px solid var(--ink);padding:10px 14px;margin-top:6px;background:var(--paper)">
<svg viewBox="0 0 940 100" style="width:100%">
  <defs><marker id="gdd-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
    markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#111"/></marker></defs>
  <g font-family="'Courier New',monospace" font-size="10.5">
    <rect x="8" y="24" width="160" height="52" fill="none" stroke="#111"/>
    <text x="88" y="46" text-anchor="middle">your data</text>
    <text x="88" y="60" text-anchor="middle" fill="#666">features + outcome</text>
    <rect x="238" y="24" width="180" height="52" fill="none" stroke="#111"/>
    <text x="328" y="46" text-anchor="middle">external detector</text>
    <text x="328" y="60" text-anchor="middle" fill="#666">runs anywhere, offline</text>
    <rect x="488" y="24" width="180" height="52" fill="none" stroke="#111"/>
    <text x="578" y="46" text-anchor="middle">per-node scores CSV</text>
    <text x="578" y="60" text-anchor="middle" fill="#666">one row per feature</text>
    <rect x="738" y="24" width="190" height="52" fill="none" stroke="#111"/>
    <text x="833" y="46" text-anchor="middle">z-scores + flags</text>
    <text x="833" y="60" text-anchor="middle" fill="#666">halos in Graph Surgery</text>
    <line x1="168" y1="50" x2="234" y2="50" stroke="#111" marker-end="url(#gdd-a)"/>
    <line x1="418" y1="50" x2="484" y2="50" stroke="#111" marker-end="url(#gdd-a)"/>
    <line x1="668" y1="50" x2="734" y2="50" stroke="#111" marker-end="url(#gdd-a)"/>
  </g>
</svg>
<p style="font-size:.84rem;margin:6px 0 0">The hub's seam is a small provider
protocol (<code>causal_shap.node_flags.NodeFlagProvider</code>): given a
dataset id, outcome, and feature names, return per-feature channel scores or
an honest status (<code>unavailable</code> / <code>not_run</code> /
<code>error</code>). Three ways to fill it:</p>
<ol style="font-size:.84rem;margin:6px 0 0 20px">
<li><b>Frozen tables</b>: run the detector offline, drop a
<code>&lt;dataset&gt;_blocks.csv</code> of per-node scores into the block
root, and the precomputed provider reads it here: no live dependency.</li>
<li><b>Live provider</b>: point <code>CAUSAL_SHAP_FLAG_PROVIDER</code> at a
<code>module:Class</code> that wraps your runtime; the hub trains and reduces
in a worker task.</li>
<li><b>No detector</b>: skip the stage, or bring evidence from literature and
expert priors straight to the surgeon. The stage is optional by design.</li>
</ol>
<p style="font-size:.84rem;margin:6px 0 0">Scores are z-scored within dataset
and flagged above z &gt; 1, per channel. A flag means "look here again",
never "this node is causal": only the surgeon's ledger changes the graph.</p>
</div></details>
"""


def learn(*keys: str, title: str = "Learn: what these controls mean") -> str:
    """A collapsible reader over the requested topics, code-card styled."""
    sections: list[str] = []
    for key in keys:
        heading, body = TOPICS[key]
        sections.append(
            f'<p style="margin:10px 0 2px"><b>{html.escape(heading)}</b></p>'
            f'<p style="font-size:.84rem;margin:0">{body}</p>'
        )
    return (
        '<details class="code-details"><summary>'
        + html.escape(title)
        + '</summary><div style="border:1px solid var(--ink);padding:10px 14px;'
        'margin-top:6px;background:var(--paper)">'
        + "".join(sections)
        + "</div></details>"
    )
