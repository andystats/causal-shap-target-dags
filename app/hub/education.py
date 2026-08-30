"""Stage-side education: every control gets an explanation on demand.

Each topic is a short, honest paragraph in the study guide's voice. Formulas
are written in TeX and typeset by MathJax (loaded from a CDN in app.py); with
no network they degrade to readable TeX source rather than breaking. Tabs
compose their own reader from topic keys; the drop-down reuses the code-card
visual so "learn" and "show the code" sit as siblings.
"""

from __future__ import annotations

import html

_PHI = (
    r"\[ \phi_i \;=\; \sum_{S \subseteq F \setminus \{i\}}"
    r" \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\,"
    r"\bigl[\,v(S \cup \{i\}) - v(S)\,\bigr] \]"
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
        "independence tests (Fisher-Z here, "
        r"\( z = \tfrac{1}{2}\sqrt{n-|S|-3}\;"
        r"\ln\tfrac{1+\hat\rho}{1-\hat\rho} \)"
        "), then orient what logic compels "
        "(colliders first, then propagation). Fast and interpretable; "
        "directions are often left open, and that honesty is the CPDAG.",
    ),
    "ges": (
        "GES algorithm",
        "Score-based discovery: greedily add then prune edges to maximize a "
        "penalized fit score, "
        r"\( \mathrm{BIC} = \log L - \tfrac{k}{2}\log n \). Same output object as PC, an equivalence "
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
        "then evaluate "
        r"\( v(S) = \mathbb{E}\bigl[f(X) \mid do(X_S = x_S)\bigr] \)"
        " by "
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
        "is at least "
        r"\( 1 - \alpha \). With a rare binary outcome most units cannot "
        "change under any single lever, so a strict floor rules out "
        "everything; the renal default is therefore permissive, and the "
        "per-unit column is shown so you can judge.",
    ),
    "budget": (
        "Budget as the objective",
        "The optimum is the largest expected benefit that fits the budget, "
        r"\( \max_a \; \mathbb{E}[\Delta Y(a)] \;\text{s.t.}\;"
        r" \mathrm{cost}(a) \le B \)"
        ". "
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
    "estimation_arms": (
        "Estimation arms: full-model versus targeted",
        "The SCM arm simulates every shift through the calibrated equations, "
        "trusting the whole system at once. The semiparametric arm follows "
        "Marschak's Maxim (Marschak 1953; Heckman 2010): let the SCM survey "
        "and shortlist, then estimate ONE functional per surviving lever. "
        "Each ±shift is a modified treatment policy "
        r"\( d(a) = a + \delta \) (Haneuse &amp; Rotnitzky 2013), and its "
        "effect "
        r"\( \mathbb{E}[Y(A+\delta)] - \mathbb{E}[Y] \) is estimated "
        "double-robustly by the cross-fitted one-step estimator "
        r"\[ \hat\theta_\delta = \frac{1}{n}\sum_i \Bigl[\, \hat r(A_i, W_i)"
        r"\,\bigl(Y_i - \hat m(A_i, W_i)\bigr) + \hat m(A_i + \delta, W_i)"
        r" \,\Bigr] - \bar Y \]"
        "(Díaz Muñoz &amp; van der Laan 2012; Díaz et al. "
        "2023; cross-fitting per Chernozhukov et al. 2018), where "
        r"\( \hat m \) is the outcome regression and \( \hat r \) the "
        "treatment density ratio. The adjustment set W is the lever's "
        "parents under the current graph: parents block every backdoor path "
        "and can never be descendants, so post-lever variables (the "
        "confounding trap for non-root levers) are excluded by construction. "
        "Feasibility is checked, not assumed: units the shift would push "
        "outside the observed treatment support, and concentrating density-"
        "ratio weights, both flag the estimate with a caution.",
    ),
    "doshapley": (
        "Decision-grade attribution: do-Shapley",
        "This stage's attributions explain the fitted model under the current "
        "graph: a survey instrument for triage. When the attribution itself "
        "must carry decision weight, do-Shapley values (Jung et al. 2022) "
        "define the shares on the real outcome, give graphical identification "
        "conditions, and come with double/debiased-ML estimators. Full "
        "citations sit in References below the tabs.",
    ),
    "routes": (
        "Alternate routes to the same question",
        "This stage simulates each shift through the calibrated SCM: the "
        "full-model pole, as DoWhy-GCM does (Bl\u00f6baum et al. 2024). The "
        "published alternatives are considerations the hub is meant to stage, "
        "not competitors. (1) Targeted estimation: each shortlisted "
        "\u00b1shift is a modified treatment policy, estimable double-"
        "robustly from its DAG-derived adjustment set alone (D\u00edaz "
        "Mu\u00f1oz &amp; van der Laan 2012; Haneuse &amp; Rotnitzky 2013; "
        "D\u00edaz et al. 2023; Kennedy 2019), in the spirit of Marschak's "
        "Maxim (Marschak 1953; Heckman 2010): trust one functional per "
        "lever, not every equation. This stage's semiparametric arm "
        "implements exactly that route for the levers that survive the "
        "screen. (2) Single-estimand "
        "identification tools: DoWhy (Sharma &amp; Kiciman 2020), Ananke "
        "(Lee et al. 2023). (3) Choosing where to intervene under "
        "constraints: policy learning (Athey &amp; Wager 2021), causal "
        "bandits (Lattimore et al. 2016; Lee &amp; Bareinboim 2018), and "
        "cost-aware causal Bayesian optimization (Aglietti et al. 2020). "
        "Full citations in References below the tabs.",
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


REFERENCE_LIST: tuple[str, ...] = (
    "Marschak J (1953). Economic measurements for policy and prediction. In "
    "Hood WC &amp; Koopmans TC (eds.), Studies in Econometric Method, Cowles "
    "Commission Monograph 14. Wiley, 1-26.",
    "Heckman JJ (2010). Building bridges between structural and program "
    "evaluation approaches to evaluating policy. Journal of Economic "
    "Literature 48(2), 356-398.",
    "Heckman JJ &amp; Vytlacil EJ (2007). Econometric evaluation of social "
    "programs. In Handbook of Econometrics, vol. 6B. Elsevier.",
    "Robins JM, Rotnitzky A &amp; Zhao LP (1994). Estimation of regression "
    "coefficients when some regressors are not always observed. JASA "
    "89(427), 846-866.",
    "van der Laan MJ &amp; Rubin D (2006). Targeted maximum likelihood "
    "learning. The International Journal of Biostatistics 2(1), Article 11.",
    "Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W "
    "&amp; Robins J (2018). Double/debiased machine learning for treatment "
    "and structural parameters. The Econometrics Journal 21(1), C1-C68.",
    "Pearl J (1995). Causal diagrams for empirical research (with "
    "discussion). Biometrika 82(4), 669-688.",
    "Shpitser I &amp; Pearl J (2006). Identification of joint interventional "
    "distributions in recursive semi-Markovian causal models. AAAI-06.",
    "Heskes T, Sijben E, Bucur IG &amp; Claassen T (2020). Causal Shapley "
    "values: exploiting causal knowledge to explain individual predictions "
    "of complex models. NeurIPS 33.",
    "Frye C, Rowat C &amp; Feige I (2020). Asymmetric Shapley values: "
    "incorporating causal knowledge into model-agnostic explainability. "
    "NeurIPS 33.",
    "Jung Y, Kasiviswanathan S, Tian J, Janzing D, Bl\u00f6baum P &amp; "
    "Bareinboim E (2022). On measuring causal contributions via "
    "do-interventions. ICML, PMLR 162, 10476-10501.",
    "Bl\u00f6baum P, G\u00f6tz P, Budhathoki K, Mastakouri AA &amp; "
    "Janzing D (2024). DoWhy-GCM: an extension of DoWhy for causal "
    "inference in graphical causal models. JMLR 25(147), 1-7.",
    "Sharma A &amp; Kiciman E (2020). DoWhy: an end-to-end library for "
    "causal inference. arXiv:2011.04216.",
    "Lee JJR, Bhattacharya R, Nabi R &amp; Shpitser I (2023). Ananke: a "
    "Python package for causal inference using graphical models. "
    "arXiv:2301.11477.",
    "D\u00edaz Mu\u00f1oz I &amp; van der Laan M (2012). Population "
    "intervention causal effects based on stochastic interventions. "
    "Biometrics 68(2), 541-549.",
    "Haneuse S &amp; Rotnitzky A (2013). Estimation of the effect of "
    "interventions that modify the received treatment. Statistics in "
    "Medicine 32(30), 5260-5277.",
    "D\u00edaz I, Williams N, Hoffman KL &amp; Schenck EJ (2023). "
    "Nonparametric causal effects based on longitudinal modified treatment "
    "policies. JASA 118(542), 846-857.",
    "Kennedy EH (2019). Nonparametric causal effects based on incremental "
    "propensity score interventions. JASA 114(526), 645-656.",
    "Athey S &amp; Wager S (2021). Policy learning with observational data. "
    "Econometrica 89(1), 133-161.",
    "Lattimore F, Lattimore T &amp; Reid MD (2016). Causal bandits: "
    "learning good interventions via causal inference. NeurIPS 29.",
    "Lee S &amp; Bareinboim E (2018). Structural causal bandits: where to "
    "intervene? NeurIPS 31.",
    "Aglietti V, Lu X, Paleyes A &amp; Gonz\u00e1lez J (2020). Causal "
    "Bayesian optimization. AISTATS, PMLR 108, 3155-3164.",
)


def references(*, collapsible: bool = True) -> str:
    """The methods bibliography, shared by the hub footer and the report."""
    items = "".join(f'<li style="margin:3px 0">{entry}</li>'
                    for entry in REFERENCE_LIST)
    body = (
        '<ol style="font-size:.8rem;margin:0 0 0 18px;line-height:1.5">'
        f"{items}</ol>"
    )
    if not collapsible:
        return f"<h2>References</h2>{body}"
    return (
        '<details class="code-details"><summary>'
        "References (methods cited in the Learn notes)</summary>"
        '<div style="border:1px solid #111;padding:10px 14px;margin-top:6px;'
        'background:#fff">' + body + "</div></details>"
    )


_FUTURE_DIRECTIONS = (
    "Targeted estimation arm for Price &amp; Dice: each shortlisted "
    "\u00b1shift re-estimated double-robustly (AIPW/TMLE, cross-fitting) "
    "from its DAG-derived adjustment set alone, with feasibility/positivity "
    "checks surfaced. Tabled; the survey-then-target logic is in Methods "
    "position above.",
    "A do-Shapley attribution arm (Jung et al. 2022): decision-grade shares "
    "on the real outcome where the graph class identifies them.",
    "Longitudinal support: the hub is cross-sectional v1, one row per unit; "
    "time-varying exposures and the longitudinal MTP literature (D\u00edaz "
    "et al. 2023) are the natural extension.",
    "Domain-reviewed cost sheets: every bundled sheet is illustrative until "
    "a domain expert prices the levers.",
    "Repeated-seed uncertainty over the whole pipeline, not only within "
    "stages.",
)


def methods_tab() -> str:
    """The Methods tab: position, alternate routes, future directions,
    references. Everything here is reading, not controls."""
    def section(title: str, body: str) -> str:
        return (
            f'<div class="hub-card"><h4>{html.escape(title)}</h4>'
            f'<div style="font-size:.86rem">{body}</div></div>'
        )

    position = (
        "<p>The working problem comes first: renal stone risk under altered "
        "gravity, a spaceflight epidemiology setting where randomized answers "
        "are impossible and decisions must be argued from mechanisms, "
        "models, and simulated evidence. The hub is one guided instrument "
        "for that argument, keeping every step's assumptions on screen: what "
        "the predictor listened to, what the data can identify about "
        "structure, where outside evidence says to look twice, what a domain "
        "expert decided and why, what attribution means under the current "
        "graph, and which affordable action survives a budget. Its bundled "
        "testbeds carry answer keys, so every claim in a session is "
        "checkable.</p>"
        "<p>The failure it guards against is proximity bias: a node caused "
        "by the outcome can dominate predictive attribution while carrying "
        "no intervention effect at all. The chain makes the repair visible: "
        "the graph, not the model, governs who can be attributed and who "
        "can be priced.</p>"
        "<p>On estimation the hub deliberately spans two poles. The "
        "causal-SHAP survey trusts the whole fitted system in order to map "
        "where credit can live (Marschak's Maxim would disapprove, which is "
        "why it is called a survey); Price &amp; Dice then asks one narrow "
        "question per surviving lever, and its targeted arm estimates that "
        "one functional with minimal trust. The routes below are staged "
        "here, not rivaled.</p>"
    )
    routes_body = f"<p>{TOPICS['routes'][1]}</p><p>{TOPICS['doshapley'][1]}</p>"
    future = (
        "<ul style='margin:0 0 0 18px'>"
        + "".join(f"<li style='margin:4px 0'>{item}</li>"
                  for item in _FUTURE_DIRECTIONS)
        + "</ul>"
    )
    refs = (
        '<ol style="font-size:.8rem;margin:0 0 0 18px;line-height:1.5">'
        + "".join(f'<li style="margin:3px 0">{entry}</li>'
                  for entry in REFERENCE_LIST)
        + "</ol>"
    )
    return (
        section("Position", position)
        + section("Alternate routes, as considerations", routes_body)
        + section("Future directions", future)
        + section("References", refs)
    )


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
