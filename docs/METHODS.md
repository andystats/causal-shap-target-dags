# Methods

## Research question

Can a causal attribution procedure rank upstream intervention targets more
faithfully than predictive feature attribution when the data-generating DAG and
total-effect truth are known?

The primary worked example is a synthetic renal-stone risk system constructed
from NASA's published SA-07566 renal-stone DAG. The target is
`Nephrolithiasis`; the feature set contains its 28 pre-outcome ancestors.

## Structural simulation

The R simulator in `analysis/R/renal_stone_source_aligned_simcausal.R` converts
the published DAGitty specification into a `simcausal` structural model. The
clean-v3 data contain 10,000 synthetic records. A separate NASA-like-v4 regime
adds astronaut selection and informative measurement processes.

The locked simulator parent structure has Cohen's kappa 1.000 and zero edge
discrepancies against the declared source graph. Coefficients are simulation
parameters—not NASA effect estimates—and require domain review before substantive
interpretation.

## Frozen intervention truth

For each of the 28 ancestors, `analysis/06_compute_interventional_truth.R`
estimates a standardized absolute total risk difference with 50,000
common-random-number structural simulations. This truth is computed before and
independently of the attribution comparison.

## Prediction model

The prespecified first learner is XGBoost, trained once on clean v3 and evaluated
on a held-out test set. The fitted model is shared across all attribution methods.
The held-out AUC is 0.684; the true structural probability has AUC 0.701, showing
that the modest discrimination is largely a data-generating ceiling rather than
an obvious learner failure.

## Attribution estimators

The locked ordering-only comparison includes:

- exact TreeSHAP;
- unrestricted interventional SHAP using a fixed 128-record background and 128
  permutations;
- DAG-constrained asymmetric SHAP using the identical model, evaluation records,
  background, and number of permutations.

The Python prototype in `app/causal_shap/structural_value.py` changes the
coalition value function. It recovers background exogenous draws, applies
`do(X_S=x_S)`, propagates each intervention through descendants, and scores the
fixed fitted model. Only DAG-consistent permutations are used.

## Evaluation

Attribution importance is compared with the frozen total-effect truth using
Kendall tau, Spearman rho, top-five recovery, NDCG@5, mean directed distance to
the outcome, proximal mass, the Proximity Bias Index (PBI), and the Proximal
Over-attribution Area (POA). The locked ordering-only comparison uses 2,000
paired evaluation-record bootstrap draws.

The structural prototype currently uses 32 evaluation records, 32 background
draws, and 32 permutations. It is an implementation milestone, not yet a locked
publication result.
