# Causal SHAP for Target DAGs

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
  the elevator pitch, the five-node failure, and the path from attribution to
  intervenable recommendations.
- **[Why it happens](https://andystats.github.io/causal-shap-target-dags/why-it-happens.html):**
  the three-node Markov proof and its assumptions.
- **[Evidence ledger](https://andystats.github.io/causal-shap-target-dags/evidence.html):**
  the teaching result, the ordering-only null, and the structural prototype.
- **Interactive app:** climb every rung on four datasets; run discovery and
  validation live ([`app/app.py`](app/app.py)).
- **Research companion (static):** the applied “valor-stealing mediator” example
  and the larger path from structural Causal SHAP through LumaWarp, DiCE, and
  cost-sensitive counterfactual recourse — a no-build static page at
  [`index.html`](index.html).

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
- [`app/causal_shap/build/`](app/causal_shap/build/) — the consolidated build CLI
  for teaching data, discovery, attribution, validation, figures, and release
  checks.
- [`site/`](site/) — the Quarto companion website.
- [`index.html`](index.html) — public-facing static research companion centered
  on mediator credit transfer, actionable intervention targets, an animated
  method-stack schematic, and annotated evidence.
- [`analysis/`](analysis/) — the R analysis pipeline and frozen result artifacts.
- [`references/robert-reynolds-2026-07-13/`](references/robert-reynolds-2026-07-13/)
  — Robert Reynolds's renal-stone and SANS DAGitty source files.
- [`docs/ROBERT_REYNOLDS_DAGS_2026-07-13.md`](docs/ROBERT_REYNOLDS_DAGS_2026-07-13.md)
  — provenance, renal concordance result, and modeling notes from the handoff.
- [`docs/`](docs/) — methods, results, reproducibility, limitations, roadmap.

The checked CSVs, model, and figures under `analysis/output/` are the auditable
result record. Selected copies under `app/bundles/` and `site/assets/` make the
app and site self-contained; Git stores identical files as one blob, so these
copies add much less repository weight than their checkout sizes suggest.

## Quickstart

Run the interactive app (no torch, no server-side model):

```bash
git clone https://github.com/andystats/causal-shap-target-dags.git
cd causal-shap-target-dags
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[discovery]"
cd app && shiny run app.py  # open http://127.0.0.1:8000
```

Rebuild the teaching pipeline and site figures:

```bash
cd ..
python -m pip install -e ".[discovery,site]"
python -m causal_shap.build all            # teaching data → discovery → … → validate
```

Add the `heavy` extra for the optional CVAE and NOTEARS build paths.

## Validate

```bash
Rscript analysis/validate_outputs.R             # frozen R outputs
python -m unittest discover -s app/tests -v     # library + app tests
python -m causal_shap.build validate            # bundles + frozen-output hash gate
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and the site's
[Reproduce](site/reproducibility.qmd) page for the full two-language pipeline.

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
