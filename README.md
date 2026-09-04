# Causal SHAP for Target DAGs

> **Development moved on 2026-09-03.** This repository is kept as provenance
> and for its published site. Its code, frozen results, and documentation were
> consolidated into
> [ahhatype/causal-shap-spaceflight-renal-stones](https://github.com/ahhatype/causal-shap-spaceflight-renal-stones)
> (site: <https://ahhatype.github.io/causal-shap-spaceflight-renal-stones/>),
> the single hub for the Space SHAP paper. Open issues and new work there.

[![Python tests](https://github.com/andystats/causal-shap-target-dags/actions/workflows/python-tests.yml/badge.svg)](https://github.com/andystats/causal-shap-target-dags/actions/workflows/python-tests.yml)
[![Companion site](https://img.shields.io/badge/GitHub%20Pages-open%20site-2563eb)](https://andystats.github.io/causal-shap-target-dags/)

**SHAP explains the model's ears. Target DAGs look for the system's levers.**

The research question is whether a predictive attribution ranking also
identifies useful upstream intervention targets. On a causal DAG, a mediator can
screen off its ancestors for prediction while still transmitting their
intervention effects. Ordinary SHAP can therefore concentrate importance on the
last measured nodes—even a downstream proxy with zero total effect.

This repository tests that mismatch entirely with synthetic data from known
DAGs. The result is deliberately nuanced: ordinary SHAP can reward proximity to
the outcome; constraining feature order to a DAG does not fix that on its own;
an intervention-propagating structural value function is promising, but still
needs larger runs and uncertainty estimates.

- **[Focused demonstration](https://andystats.github.io/causal-shap-target-dags/):**
  one application of the [archived ACIC Causal SHAP
  project](https://www.tao-rwd.com/acic-2026/causal-shap), moving from prediction
  through recommendation with the Markov explanation and three decisive
  results on one page.
- **[Research record](docs/README.md):** methods, evidence, references,
  provenance, limitations, reproducibility, and the ACIC lineage.
- **Interactive app:** climb every rung on four datasets; run discovery and
  validation live ([`app/app.py`](app/app.py)).
- **Experimental M1–M5 Workbench:** inspect how a learned graph performs for a
  target exposure–outcome question, using the frozen synthetic teaching fixture
  or a local upload ([`app/workbench/`](app/workbench/)). Its discovery battery
  is a single-seed pilot, not a validated benchmark; no clinical rows are
  distributed with the repository.

The analyzed worked example uses NASA's public SA-07566 renal-stone DAG. A
second source topology for spaceflight-associated neuro-ocular syndrome (SANS)
is now ingested and validated, but it does not yet have calibrated structural
equations or SHAP results.

## What the evidence says

On the teaching DAGs, ordinary SHAP's importance ranking is *negatively*
correlated with the causal truth — the collider/proxy wins the most credit
despite zero total effect — and structural Causal SHAP flips it positive.

| Dataset | Ordinary SHAP (Kendall τ vs truth) | Structural Causal SHAP |
| --- | ---: | ---: |
| Toy chain/fork/collider | −0.33 | +0.33 |
| Layered ladder | −0.33 | +0.26 |

On the source-exact **NASA** flagship the story is told straight: ordinary and
DAG-*ordering* SHAP are statistically tied (a deliberate null), and only the
structural prototype closes the gap to the frozen interventional truth.

| NASA method | Kendall τ vs truth | Top-5 recovery | PBI |
| --- | ---: | ---: | ---: |
| Exact TreeSHAP | 0.522 | 0.60 | 1.082 |
| DAG-asymmetric SHAP | 0.528 | 0.60 | 1.051 |
| Structural prototype | 0.794 | 1.00 | −0.113 |

## Repository map

- [`app/causal_shap/`](app/causal_shap/) — the tested library: teaching DAGs,
  discovery (`discovery.py`), complexity score (`complexity.py`), structural
  attribution (`structural_value.py`), Credence-style validation
  (`validation/`), and figures (`viz/`).
- [`app/`](app/) — the six-rung tutorial Shiny app and self-contained bundles.
- [`app/workbench/`](app/workbench/) — the experimental graph-discovery and
  M1–M5 structural-importance Workbench, with a synthetic teaching fixture.
- [`app/causal_shap/build/`](app/causal_shap/build/) — the consolidated build CLI
  for teaching data, discovery, attribution, validation, figures, and release
  checks.
- [`site/`](site/) — the single-page Quarto exposition, visually and
  rhetorically continuous with the archived Tao RWD ACIC project.
- [`index.html`](index.html) — a lightweight redirect to the canonical GitHub
  Pages site; it intentionally contains no second version of the narrative.
- [`analysis/`](analysis/) — the R analysis pipeline and frozen result artifacts.
- [`references/robert-reynolds-2026-07-13/`](references/robert-reynolds-2026-07-13/)
  — Robert Reynolds's renal-stone and SANS DAGitty source files.
- [`docs/PROVENANCE_AND_REFERENCES.md`](docs/PROVENANCE_AND_REFERENCES.md)
  — provenance, renal concordance result, and modeling notes from the handoff.
- [`docs/README.md`](docs/README.md) — documentation index for the canonical
  narrative, annotated references, site-integration guide, methods, results,
  provenance, reproducibility, limitations, and roadmap.

The checked CSVs, model, and figures under `analysis/output/` are the auditable
result record. Selected copies under `app/bundles/` support the deterministic
app and are validated against the frozen outputs.

## Quickstart

Run the guided causal-discovery hub (no torch, no server-side model):

```bash
git clone https://github.com/andystats/causal-shap-target-dags.git
cd causal-shap-target-dags
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[discovery]"
python -m shiny run --port 8002 --app-dir app hub.app:app  # open http://localhost:8002
```

The hub walks data upload, naive SHAP, structure discovery, an optional
evidence-gathering stage (bring your own detector or priors), interactive
graph surgery, graph-governed causal SHAP, and budget-constrained
intervention ranking. The older teaching app remains available:

```bash
cd app && shiny run app.py  # open http://127.0.0.1:8000
```

Rebuild the teaching pipeline and app figures:

```bash
cd ..
python -m pip install -e ".[discovery,site]"
python -m causal_shap.build all            # teaching data → discovery → … → validate
```

Add the `heavy` extra for the optional CVAE and NOTEARS build paths.

From the repository root, run the experimental Workbench separately:

```bash
python -m pip install -e ".[workbench]"
cd app/workbench
python -m shiny run --port 8001 app.py
```

## Validate

```bash
Rscript analysis/validate_outputs.R             # frozen R outputs
python -m unittest discover -s app/tests -v     # library + app tests
python -m causal_shap.build validate            # bundles + frozen-output hash gate
```

See [`docs/REPRODUCIBILITY_AND_SITE.md`](docs/REPRODUCIBILITY_AND_SITE.md) for the full
two-language pipeline.

## Scientific status

Research software under active development. **All datasets are synthetic.** Graph
topologies are source-aligned; coefficients are simulation parameters, not NASA
estimates. The structural NASA result is an explicitly labeled prototype, the
pedagogic proxy example is not evidence, and the **complexity score (PSCI v0) is
provisional** — it plugs into a registry seam for the authors' final score.

## Method background

- Heskes et al. [Causal Shapley Values](https://proceedings.neurips.cc/paper_files/paper/2020/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html), NeurIPS 2020.
- Frye, Rowat, Feige. [Asymmetric Shapley Values](https://papers.nips.cc/paper/2020/file/0d770c496aa3da6d2c3f2bd19e7b9d6b-Paper.pdf), NeurIPS 2020.
- Janzing, Minorics, Blöbaum. [Feature Relevance Quantification in Explainable AI](https://proceedings.mlr.press/v108/janzing20a), AISTATS 2020.
- Karimi, Schölkopf, Valera. [Algorithmic Recourse: from Counterfactual Explanations to Interventions](https://arxiv.org/abs/2002.06278), FAccT 2021.
- Parikh et al. [Validating Causal Inference Methods (Credence)](https://proceedings.mlr.press/v162/parikh22a.html), ICML 2022.

The simulation-validation layer is implemented from the author's own Instats
workshop code in the spirit of Credence; no Credence-repository code is
redistributed.

## License

Code and original documentation are available under the [MIT License](LICENSE).
External source material remains subject to its original terms.
