# Paper Companion App

This project-level Python Shiny app is the deterministic guided experience for
the paper and video. Its self-contained bundles do not require private project
materials or network access.

## Run

```powershell
cd app
py -3.13 -m pip install -r requirements.txt
py -3.13 -m shiny run --port 8010 app.py
```

Open `http://127.0.0.1:8010`.

## Rebuild frozen results

```powershell
cd app
py -3.13 scripts\build_acic_bundle.py
py -3.13 scripts\build_structural_results.py
py -3.13 -m unittest discover -s tests -v
```

Guided mode never recomputes attribution. It reads checked-in local bundles so
the recorded story is fast and reproducible. The structural result is currently
marked **prototype** because it uses 32 evaluation records, 32 backgrounds, and
32 permutations. Scale and bootstrap it before manuscript claims.

## Current experiences

- NASA renal-stone clean v3: source-aligned primary analysis, fair
  matched-background control, and structural intervention-propagation prototype.
- ACIC mediator/proxy stress test: deliberately dramatic pedagogic example,
  visibly separated from the NASA analysis.
