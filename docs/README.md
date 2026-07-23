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

## Two web surfaces

The repository currently contains two intentionally distinct front ends:

1. **Canonical GitHub Pages site:** Quarto source under `site/*.qmd`. The Pages
   workflow renders `site/_site` and deploys it.
2. **Exploratory static research companion:** root `index.html` with
   `site/styles.css` and `site/app.js`. It is a no-build prototype and is not the
   artifact deployed by the current Pages workflow.

Narrative or scientific claims should be settled in this documentation first,
then incorporated into the Quarto site. Port them to the static companion only
when that surface is being deliberately maintained; do not let two versions of
the claim evolve independently.

## Documentation rule

Every substantive result should state:

- the estimand and intervention semantics;
- the graph and data-generating regime;
- whether the result is teaching evidence, a matched comparison, or a
  provisional prototype;
- the learner, background population, evaluation records, and Monte Carlo
  budget; and
- the uncertainty procedure and remaining limitation.
