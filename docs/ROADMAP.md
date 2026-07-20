# Roadmap

## Companion toolkit (in place)

The public companion is built around a six-rung workflow ladder — vanilla SHAP →
causal discovery → complexity score → structural Causal SHAP → simulation
validation → iteration:

- `app/causal_shap/` now includes teaching DAGs, causal discovery (PC/GES live via
  causal-learn; LiNGAM/NOTEARS as precomputed appendix), a pluggable complexity
  score (PSCI v0), a Credence-style layered-parameter validation subpackage, and
  the homunculus/ladder figures.
- The Shiny app is rebuilt as the ladder, with live discovery and validation.
- A Quarto site (`site/`) carries the narrative, cheatsheets, and glossary.

Remaining toolkit tasks: swap in the authors' final complexity score at the
registry seam; complete the shinyapps.io deploy (upload was blocked by transient
S3 errors); optionally extend validation to multiple treatments and mediation.

## Publication-critical next steps

- Scale structural Causal SHAP to the locked clean-v3 evaluation and background
  manifests with a larger permutation budget.
- Quantify Monte Carlo error and paired uncertainty against the matched ordinary
  estimator.
- Repeat across simulation seeds and document stability.
- Run the longer-path `Loss of Mission Objectives` endpoint.
- Freeze the method before applying it to the NASA-like selection/missingness
  regime.
- Obtain domain review of graph version, coefficients, actionability, and
  intervention cost/difficulty.

## Subsequent extensions

- Add cost- and feasibility-aware intervention ranking.
- Add manifold-constrained counterfactual/recourse methods.
- Evaluate a second human-system-risk DAG when a reviewable graph is available.
- Produce the intervention-propagation animation and record the guided app
  walkthrough.
