# Reproducibility

## Repository contents

- `analysis/`: R structural simulation, truth estimation, SHAP comparison,
  bootstrap, diagnostics, checked CSV outputs, and figures.
- `app/`: Python Shiny companion app, self-contained deterministic bundles,
  structural engine, builders, and unit tests.
- `references/renal-stone-dag-code-SA-07566.txt`: public machine-readable NASA
  DAG input required by the simulator.
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
(causal-learn discovery and a moment-matched MVN validation generator). To rebuild
the ladder bundles and site figures, run the numbered pipeline in order (from a
root editable install, `pip install -e ".[discovery]"`, plus
`requirements-build.txt` for the torch/CVAE step):

```bash
cd app
python scripts/build_acic_bundle.py       # legacy pedagogic bundle
python scripts/build_structural_results.py # legacy NASA structural prototype
python scripts/20_build_teaching_data.py   # teaching DAGs
python scripts/21_build_discovery.py       # PC/GES/LiNGAM vs truth
python scripts/22_build_complexity.py      # PSCI v0 reports
python scripts/23_build_causal_shap.py     # structural attribution (teaching)
python scripts/24_build_validation.py      # CVAE (torch) + MVN validation suite
python scripts/25_build_figures.py         # homunculus + distortion + ladder
python scripts/26_build_glossary.py        # glossary.yml -> app JSON + site include
python scripts/29_validate_bundles.py      # release gate + frozen-output hashes
```

Non-torch artifacts are bit-for-bit reproducible across runs. `29_validate_bundles.py`
hashes `analysis/output/` against a committed baseline and fails on any change.

## Companion site (Quarto)

```bash
quarto render site       # renders with zero code execution; output in site/_site
```

The site embeds pre-built figures and includes the generated glossary, so no
Python runs at render time. Deploy is handled by `.github/workflows/publish-site.yml`
(GitHub Pages). The interactive app deploys manually with
`rsconnect deploy shiny app/ --name <account> --title causal-shap-ladder`.

## R environment

The current outputs were generated under R 4.5.2.

```powershell
Rscript analysis\install_dependencies.R
cd analysis
Rscript .\01_generate_clean.R
Rscript .\02_generate_nasa_like.R
Rscript .\03_validate_and_plot_dags.R
Rscript .\04_generate_source_aligned_clean.R
Rscript .\05_generate_source_aligned_nasa_like.R
Rscript .\06_compute_interventional_truth.R
Rscript .\07_run_shap_comparison.R
Rscript .\08_bootstrap_shap_comparison.R
Rscript .\09_diagnose_predictive_signal.R
Rscript .\validate_outputs.R
```

All seeds and evaluation/background manifests are checked into the outputs. A
complete run is substantially slower than opening the deterministic app.

## Validation

At minimum, a release should pass:

```powershell
Rscript analysis\validate_outputs.R
cd app
python -m unittest discover -s tests -v
python -m py_compile app.py causal_shap\*.py scripts\*.py tests\*.py
```
