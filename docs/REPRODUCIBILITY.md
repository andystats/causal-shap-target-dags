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
Guided mode reads checked-in bundles and performs no attribution recomputation.
To rebuild them:

```powershell
cd app
python scripts\build_acic_bundle.py
python scripts\build_structural_results.py
```

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
