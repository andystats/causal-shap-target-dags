# Interactive Research Companion

This Python Shiny app is the deterministic, guided version of the project's
six-rung workflow. Its self-contained bundles do not require private project
materials or network access.

## Run

```powershell
cd causal-shap-target-dags
py -3.13 -m pip install -e ".[discovery]"
cd app
py -3.13 -m shiny run --port 8010 app.py
```

Open `http://127.0.0.1:8010`.

## Rebuild frozen results

```powershell
cd causal-shap-target-dags
py -3.13 -m pip install -e ".[discovery,site]"
py -3.13 -m causal_shap.build all
py -3.13 -m causal_shap.build validate
py -3.13 -m unittest discover -s app/tests -v
```

The pre-frozen ACIC and NASA structural bundles are excluded from `all`; rebuild
them explicitly with the `acic` and `nasa-structural` stages. Install the
`heavy` extra first if you also want the optional CVAE or NOTEARS paths.

Guided mode never recomputes attribution. It reads checked-in local bundles so
the recorded story is fast and reproducible. The structural result is currently
marked **prototype** because it uses 32 evaluation records, 32 backgrounds, and
32 permutations. Scale and bootstrap it before manuscript claims.

## Current experiences

- NASA renal-stone clean v3: source-aligned primary analysis, fair
  matched-background control, and structural intervention-propagation prototype.
- ACIC mediator/proxy stress test: deliberately dramatic pedagogic example,
  visibly separated from the NASA analysis.
