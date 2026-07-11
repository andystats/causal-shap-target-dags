# Causal SHAP Target DAGs

[![Python tests](https://github.com/andystats/causal-shap-target-dags/actions/workflows/python-tests.yml/badge.svg)](https://github.com/andystats/causal-shap-target-dags/actions/workflows/python-tests.yml)

An open research implementation for comparing predictive SHAP, DAG-constrained
asymmetric SHAP, and intervention-propagating structural attribution against
known causal truth in synthetic human-system-risk models.

The worked example uses NASA's public SA-07566 renal-stone DAG. The scientific
question is deliberately narrower than “does causal SHAP win?”:

> When a structural model and intervention truth are known, which attribution
> value function recovers useful upstream intervention targets—and which merely
> rewards features near the outcome?

## Current finding

DAG ordering alone does not provide a reliable advantage over a matched ordinary
interventional-SHAP estimator. A small structural intervention-propagating
prototype is much closer to the frozen total-effect truth, but it remains
provisional until scaled and bootstrapped.

| Method | Kendall tau vs truth | Top-5 recovery | PBI |
| --- | ---: | ---: | ---: |
| Exact TreeSHAP | 0.522 | 0.60 | 1.082 |
| Matched ordinary SHAP | 0.506 | 0.60 | 1.051 |
| DAG-asymmetric SHAP | 0.528 | 0.60 | 1.051 |
| Structural prototype | 0.794 | 1.00 | -0.113 |

The first learner was XGBoost. Its held-out AUC is 0.684, while the true
structural probability has AUC 0.701; the modest discrimination is largely a
property of the current weak-signal simulation.

## Repository map

- [`analysis/`](analysis/) — R structural simulation, frozen intervention truth,
  SHAP estimators, paired bootstrap, diagnostics, figures, and validation.
- [`app/`](app/) — deterministic Python Shiny app, self-contained bundles,
  structural value function, builders, and tests.
- [`docs/METHODS.md`](docs/METHODS.md) — estimands, simulation, learner, attribution
  methods, and evaluation metrics.
- [`docs/RESULTS.md`](docs/RESULTS.md) — current numerical results and their status.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — exact setup, rebuild, and
  validation commands.
- [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) — public DAG sources and
  synthetic-data boundaries.
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — scientific and implementation
  guardrails.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — publication-critical next steps.
- [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) — app walkthrough and video assets.

## Run the app

```powershell
cd app
py -3.13 -m pip install -r requirements.txt
py -3.13 -m shiny run --port 8010 app.py
```

Then open <http://127.0.0.1:8010>. Guided mode reads checked-in result bundles
and performs no live attribution computation.

## Validate

```powershell
Rscript analysis\validate_outputs.R
cd app
py -3.13 -m unittest discover -s tests -v
```

See [the full reproducibility guide](docs/REPRODUCIBILITY.md) before regenerating
all simulations and attribution outputs.

## Scientific status

This is research software under active development. All included datasets are
synthetic. The graph topology is source-aligned, but model coefficients are
simulation parameters rather than NASA estimates. The structural result is an
explicitly labeled prototype, and the pedagogic proxy example is not NASA
evidence.

## Method background

- Heskes et al. [Causal Shapley Values](https://proceedings.neurips.cc/paper_files/paper/2020/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html), NeurIPS 2020.
- Frye, Rowat, and Feige. [Asymmetric Shapley Values](https://papers.nips.cc/paper/2020/file/0d770c496aa3da6d2c3f2bd19e7b9d6b-Paper.pdf), 2020.

The method family is established; the causal-specific Python package ecosystem
is not standardized. This repository therefore keeps its structural value
function explicit, inspectable, and tested while relying on mature packages for
ordinary SHAP, graphs, learners, and numerical work.

## License

Code and original documentation are available under the [MIT License](LICENSE).
External source material remains subject to its original terms.
