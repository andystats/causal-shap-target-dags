# Reproducibility

## Repository contents

- `analysis/`: R structural simulation, truth estimation, SHAP comparison,
  bootstrap, diagnostics, checked CSV outputs, and figures.
- `app/`: Python Shiny companion app, self-contained deterministic bundles,
  structural engine, builders, and unit tests.
- `site/`: Quarto source for the public GitHub Pages companion.
- `references/renal-stone-dag-code-SA-07566.txt`: public machine-readable NASA
  DAG input required by the simulator.
- `references/robert-reynolds-2026-07-13/`: the renal-stone and SANS DAGitty
  files supplied by Robert Reynolds on 2026-07-13.
- `docs/`: public methods, results, provenance, limitations, demo guide, and
  roadmap.

Private email records, third-party PDFs, and working whiteboards are not part of
the repository.

## Python environment

Python 3.13 was used for the current build.

```powershell
cd causal-shap-target-dags
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[discovery,site]"
python -m unittest discover -s app/tests -v
cd app
python -m shiny run --port 8010 app.py
```
The app reads checked-in bundles; rungs 1 and 4 add live, torch-free computation
(causal-learn discovery and a moment-matched MVN validation generator). The whole
Python pipeline is one CLI — `python -m causal_shap.build <stage>`. Add the
`heavy` extra to the install command for the optional torch/CVAE and NOTEARS
paths:

```bash
python -m causal_shap.build all             # teaching-data → discovery → complexity
                                            # → causal-shap → validation → figures
                                            # → glossary → validate (the whole flow)
python -m causal_shap.build validate        # just the release gate + frozen-output hashes
python -m causal_shap.build discovery       # or any single stage
python -m causal_shap.build acic            # the pre-frozen pedagogic bundle (run individually)
python -m causal_shap.build nasa-structural # the pre-frozen NASA structural prototype
```

Non-torch artifacts are bit-for-bit reproducible across runs. The `validate` stage
hashes `analysis/output/` against a committed baseline and fails on any change.
When a reviewed pipeline stage intentionally adds a new checked output—such as
the Robert Reynolds DAG source artifacts—the same commit must add its SHA-256 to
`app/bundles/analysis_output_baseline_hashes.json`. This preserves the gate
instead of exempting the new output directory.

## Companion site (Quarto)

The public site is
[andystats.github.io/causal-shap-target-dags](https://andystats.github.io/causal-shap-target-dags/).

```bash
quarto render site       # renders with zero code execution; output in site/_site
```

The current pages contain prose, pre-built figures, and a generated glossary, so
no Python or R runs at render time. `freeze: true` protects that boundary if an
executable chunk is added later. Deploy is handled by
`.github/workflows/publish-site.yml` (GitHub Pages). The interactive app is **not
hosted** — install from the repository root, then run it locally
(`pip install -e ".[discovery]" && cd app && shiny run app.py`). This is a
reproducibility choice: a hosted instance can't pin the environment behind the
published results, and the full workflow depends on packages that don't run in a
browser sandbox.

## R environment

The current outputs were generated under R 4.5.2. The whole pipeline runs from one
command:

```bash
Rscript analysis/install_dependencies.R
Rscript analysis/run_all.R     # generate all datasets → DAG validation → truth
                               # → SHAP comparison → bootstrap → diagnostics → gate
Rscript analysis/10_ingest_robert_dags.R   # Robert Reynolds DAG ingest (standalone)
```

Individual stages are still runnable on their own, and — unlike the old numbered
scripts — now run interactively too (`source("analysis/generate.R")`,
`Rscript analysis/06_compute_interventional_truth.R`, etc.); every entry point
self-locates the repo via `analysis/R/paths.R`. The four dataset generators are
consolidated into `analysis/generate.R` (`generate_dataset(variant)`).

All seeds and evaluation/background manifests are checked into the outputs. A
complete run is substantially slower than opening the deterministic app.

## Validation

At minimum, a release should pass:

```powershell
Rscript analysis/validate_outputs.R
python -m unittest discover -s app/tests -v
python -m compileall -q app/causal_shap app/stages app/tests app/app.py
python -m causal_shap.build validate
```
