# Causal SHAP Workbench

An experimental teaching app for moving from graph discovery to target-path
evaluation and attribution. The bundled example is fully synthetic. Users may
also upload a CSV and, separately, a `from,to` reference-edge list; an uploaded
reference graph is not assumed to be causal truth.

## Run

Install the repository's Workbench extra from the repository root, then launch
the app from this directory:

```powershell
py -3.13 -m pip install -e ".[workbench]"
Set-Location app\workbench
py -3.13 -m shiny run --port 8001 app.py
```

On Windows, `run_workbench.bat` performs the launch after the environment has
been installed. Open <http://127.0.0.1:8001>.

## Layout

| file | role |
| --- | --- |
| `app.py` | Shiny app: Map, six stations, and Guide |
| `attribution.py` | experimental Workbench attribution engine |
| `schematic.py` | clickable SVG map |
| `guide.py` | embedded study guide |
| `data/` | synthetic teaching data, known graph, and frozen effects |

The M1-M5 implementation lives in the tested package module
`causal_shap.evaluation`. Run its focused tests from the repository root:

```powershell
Set-Location app
py -3.13 -m unittest tests.test_workbench_evaluation -v
```

## Data handling and interpretation

- The repository contains no row-level clinical data and the Workbench has no
  built-in clinical-data loader.
- Uploaded files remain in the current local Shiny session; the app does not
  write them into the repository.
- The bundled simulation's graph is known truth. A user-uploaded graph is only
  a reference supplied by that user.
- The attribution and CPDAG diagnostics are research prototypes, not clinical
  or policy decision tools. Capped M5 runs are labelled Monte Carlo estimates.
- MathJax is loaded from a CDN when available and otherwise leaves readable TeX.
