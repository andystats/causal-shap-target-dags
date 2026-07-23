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
registry seam; enable GitHub Pages for the static site; optionally extend
validation to multiple treatments and mediation. The interactive app runs locally
by design (not hosted) — a hosted instance can't pin the result environment, and
the browser sandbox can't load the scientific stack.

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

## Explanation-to-intervention stack

- Validate LumaWarp as a prespecified candidate-depth diagnostic against known
  DAG depth and intervention truth; do not treat its score as causal proof.
- Add causally screened DiCE counterfactuals restricted to mutable,
  intervenable nodes and structurally possible descendant responses.
- Extend DiCE with decision-specific cost, burden, time, reversibility, and
  feasibility penalties.
- Compare the complete stack with traditional SHAP on causal influence,
  intervenability, feasibility, and recovery of a prespecified actionable
  target set.
- Add manifold-constrained counterfactual/recourse sensitivity analyses.

## Subsequent extensions

- Evaluate a second human-system-risk DAG when a reviewable graph is available.
- Produce the intervention-propagation animation and record the guided app
  walkthrough.
