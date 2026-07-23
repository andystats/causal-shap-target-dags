# Documentation map

This folder is the durable scientific and editorial record for **Causal SHAP
for Target DAGs**. The website should be concise; the reasoning, references,
implementation notes, and scientific guardrails belong here.

## Start here

- [`PROJECT_NARRATIVE.md`](PROJECT_NARRATIVE.md) — the canonical elevator pitch,
  Markov explanation, evidence sequence, recommendation target, and claim
  boundaries.
- [`RESEARCH_REFERENCES.md`](RESEARCH_REFERENCES.md) — annotated primary sources
  and a claim-to-citation map.
- [`SITE_INTEGRATION_GUIDE.md`](SITE_INTEGRATION_GUIDE.md) — which web surface is
  canonical, where each piece of the narrative belongs, and how to replace the
  hand-drawn illustration placeholders.

## Scientific record

- [`METHODS.md`](METHODS.md) — data-generating systems, estimators, and metrics.
- [`RESULTS.md`](RESULTS.md) — current quantitative results and their status.
- [`LIMITATIONS.md`](LIMITATIONS.md) — constraints on interpretation.
- [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) — graph and data lineage.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — environments and build commands.
- [`ROADMAP.md`](ROADMAP.md) — publication-critical and exploratory next steps.
- [`ACIC_LINEAGE.md`](ACIC_LINEAGE.md) — continuity with the earlier Causal SHAP
  poster.
- [`ROBERT_REYNOLDS_DAGS_2026-07-13.md`](ROBERT_REYNOLDS_DAGS_2026-07-13.md) —
  the renal-stone and SANS DAG handoff, concordance, and modeling decisions.

## Public web surface

The repository has one public narrative: the single-page Quarto source at
`site/index.qmd`. The Pages workflow renders `site/_site` and deploys it. The
root `index.html` only redirects readers to that canonical page.

The public page is intentionally spare: ACIC motivation, Markov explanation,
five-rung hierarchy, three decisive results, recommendation target, and claim
boundary. Narrative or scientific claims should be settled in this
documentation first, then incorporated into that page. Methodological depth
stays here instead of spawning parallel public pages.

## Documentation rule

Every substantive result should state:

- the estimand and intervention semantics;
- the graph and data-generating regime;
- whether the result is teaching evidence, a matched comparison, or a
  provisional prototype;
- the learner, background population, evaluation records, and Monte Carlo
  budget; and
- the uncertainty procedure and remaining limitation.
