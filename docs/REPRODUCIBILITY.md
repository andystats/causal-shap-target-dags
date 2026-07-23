# Reproducibility

## Repository contents

- `analysis/`: R structural simulation, truth estimation, SHAP comparison,
  bootstrap, diagnostics, checked CSV outputs, and figures.
- `app/`: Python Shiny companion app, self-contained deterministic bundles,
  structural engine, builders, and unit tests.
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
cd app
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m shiny run --port 8010 app.py
```
The app reads checked-in bundles; rungs 1 and 4 add live, torch-free computation
(causal-learn discovery and a moment-matched MVN validation generator). The whole
Python pipeline is one CLI — `python -m causal_shap.build <stage>` — after a root
editable install (`pip install -e ".[discovery]"`, plus `app/requirements-build.txt`
for the torch/CVAE step):

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

## Companion site (Quarto)

```bash
quarto render site       # renders with zero code execution; output in site/_site
```

The site embeds pre-built figures and includes the generated glossary, so no
Python runs at render time. Deploy is handled by `.github/workflows/publish-site.yml`
(GitHub Pages). The interactive app is **not hosted** — it runs locally from the
repo (`cd app && pip install -r requirements.txt && shiny run app.py`). This is a
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
Rscript analysis\validate_outputs.R
cd app
python -m unittest discover -s tests -v
python -m compileall -q causal_shap stages tests app.py
```
