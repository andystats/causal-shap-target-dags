# Causal SHAP for Target DAGs

[![Python tests](https://github.com/andystats/causal-shap-target-dags/actions/workflows/python-tests.yml/badge.svg)](https://github.com/andystats/causal-shap-target-dags/actions/workflows/python-tests.yml)

When feature importance points to the wrong lever. This project shows — entirely
with simulated data from known DAGs — that ordinary SHAP rebuilds a distorted
picture of causal structure (a homunculus with over-developed nodes near the
outcome), then climbs a workflow ladder that fixes it.

```
Rung 0  Vanilla SHAP        →  the homunculus: proximity bias
Rung 1  Causal discovery    →  learn structure (a tool, not an oracle)
Rung 2  Complexity score    →  how much causal care does this need?
Rung 3  Causal SHAP         →  propagate do(X=x) through descendants
Rung 4  Validation          →  Credence-style, layered known truth
Rung 5  Iteration           →  refine the DAG; report uncertainty
```

- **Companion site:** the ladder, cheatsheets, glossary, and reproducibility guide
  (Quarto → GitHub Pages, source in [`site/`](site/)).
- **Interactive app:** climb every rung on four datasets; run discovery and
  validation live ([`app/app.py`](app/app.py)).

## The result in one figure

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
- [`app/scripts/`](app/scripts/) — the numbered build pipeline (`20`–`29`).
- [`site/`](site/) — the Quarto companion website.
- [`analysis/`](analysis/) — the frozen R science (kept byte-for-byte stable).
- [`docs/`](docs/) — methods, results, reproducibility, limitations, roadmap.

## Quickstart

Run the interactive app (no torch, no server-side model):

```bash
cd app
pip install -r requirements.txt
shiny run app.py            # open http://127.0.0.1:8000
```

Rebuild the teaching pipeline and site figures:

```bash
pip install -e ".[discovery]"              # from repo root
python -m causal_shap.build all            # teaching data → discovery → … → validate
```

## Validate

```bash
Rscript analysis/validate_outputs.R          # frozen R science
cd app
python -m unittest discover -s tests -v      # library + app tests
python -m causal_shap.build validate         # bundles + frozen-output hash gate
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
- Parikh et al. [Validating Causal Inference Methods (Credence)](https://proceedings.mlr.press/v162/parikh22a.html), ICML 2022.

The simulation-validation layer is implemented from the author's own Instats
workshop code in the spirit of Credence; no Credence-repository code is
redistributed.

## License

Code and original documentation are available under the [MIT License](LICENSE).
External source material remains subject to its original terms.
