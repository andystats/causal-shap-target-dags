# Python Causal SHAP App: Audit and Adaptation Plan

## Recommendation

Use the existing Python Shiny app as the interaction and presentation seed, but
do not use its current fast causal-SHAP calculation as the publication analysis
engine. The paper-facing app should load verified precomputed outputs by default
and offer live recomputation only as an explicitly labeled exploratory mode.

## Implementation status — 2026-07-11

The project-level app is implemented in `../app/`, with deterministic guided/lab
modes and checked-in NASA clean-v3 and pedagogic stress-test bundles. It has been
launched and exercised in a browser through both datasets and all NASA story
steps. The separate structural engine is unit-tested for descendant propagation,
intervention blocking, reconstruction, and Shapley efficiency. Its current NASA
result is a small 32-record/32-background/32-permutation prototype; scaling,
paired uncertainty, seed replication, and the longer-path endpoint remain the
publication gate.

## What is already reusable

- A runnable Python Shiny shell with a compact control/results layout.
- A working ACIC many-mediator/proxy dataset and edge list.
- Standard-SHAP and topological causal-SHAP functions.
- Rank-movement, proxy-inflation, and importance-comparison views.
- A simple local launch pattern and checked-in requirements.

The reusable source is now incorporated into the self-contained `../app/`
implementation and pedagogic bundle.

## Why the original seed app was not paper-ready

1. It is hard-coded to the continuous ACIC outcome `AcuteRisk`, uses
   `GradientBoostingRegressor`, and reports held-out R². The NASA target is
   binary and needs AUC, Brier score, log loss, and calibration.
2. Its causal run uses only 12 evaluation instances, eight background draws, and
   a small number of permutations. That is suitable for responsiveness, not for
   a definitive comparison.
3. Random sampling is not tied to a persisted evaluation/background manifest,
   so repeated app runs are not the frozen paper analysis.
4. Conditional node models are generic gradient-boosting regressors, including
   for binary nodes, and deterministic mean predictions omit realistic residual
   propagation.
5. Standard SHAP and causal SHAP do not yet share a fully documented value
   function/background contract.
6. The app has no NASA node-name mapping, target-specific ancestor extraction,
   intervention-truth layer, distance metrics, uncertainty display, or
   source-version register.
7. It had not yet been promoted into a validated project-level application.

## Target information architecture

### Guided story mode

Six locked steps, matching the video:

1. **Choose the mission** — dataset provenance and target.
2. **Predict** — fitted learner performance and ordinary attribution.
3. **Reveal the DAG** — target ancestor graph and node roles.
4. **Compare fairly** — TreeSHAP, matched ordinary SHAP, and DAG-asymmetric SHAP.
5. **Reveal truth** — total effects, PBI/POA, rank recovery, and uncertainty.
6. **Propagate interventions** — structural Causal SHAP once implemented.

The user advances explicitly; later results remain hidden until their step.

### Lab mode

- Dataset: ACIC stress test, NASA clean v3, NASA-like v4, later mission outcome.
- Target: only targets with a complete truth bundle.
- Learner: paper-locked model plus logistic and tree-ensemble benchmarks.
- Attribution: exact TreeSHAP, matched ordinary SHAP, DAG-asymmetric SHAP,
  structural Causal SHAP.
- Display: feature ranking, ancestor map, distance curves, rank recovery, PBI,
  POA, top-k recovery, and uncertainty.

## Architecture

```text
Verified R/Python analysis outputs
        |
        v
Versioned dataset/result manifest ----> Research audit tables
        |
        v
Python Shiny presentation layer
        |-- Guided story (precomputed, deterministic)
        |-- Lab mode (precomputed comparisons)
        `-- Optional live recomputation (clearly labeled)
```

The implemented project-level `app/` keeps the pedagogic bundle separate from
the structural publication engine and no longer depends on a provenance copy.

## Proposed project structure

```text
app/
  app.py
  causal_shap/
    attribution.py
    structural_value.py
    metrics.py
    schemas.py
  components/
    guided_story.py
    dag_view.py
    ranking_view.py
    distance_view.py
  bundles/
    manifest.json
    acic_proxy_stress_test/
    nasa_renal_clean_v3/
  assets/
  tests/
```

## Implementation sequence

1. Define and validate a bundle schema against current clean-v3 outputs.
2. Scaffold the new app with guided and lab modes, initially reading only
   precomputed results.
3. Port the ACIC stress test and reproduce its current ranking display.
4. Add NASA clean v3, binary metrics, DAG view, truth, distance curves, and the
   matched-background comparison.
5. Implement structural intervention propagation as a separately tested engine.
6. Cross-check Python results against the R common-random-number simulator and
   checked-in CSVs.
7. Add NASA-like v4 only after the clean method is frozen.
8. Package a one-command local launch and a deployed read-only demo.

## Acceptance criteria

- Guided mode reproduces checked-in paper numbers exactly without network
  access or random variation.
- Every displayed number has a bundle-relative source path and analysis version.
- TreeSHAP and matched-background ordinary/causal comparisons are labeled
  separately.
- The pedagogic stress test is visibly distinguished from source-aligned NASA
  analyses on every screen.
- App tests verify row manifests, feature order, target, DAG acyclicity,
  attribution efficiency, metric calculations, and bundle hashes.
- A full guided run completes in under 30 seconds on the recording machine;
  precomputed fallback mode is instantaneous.
- The recorded video can be recreated from a documented shot list without
  changing code or data.
