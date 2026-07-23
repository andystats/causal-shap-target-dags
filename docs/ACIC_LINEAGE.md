# From the ACIC 2026 Causal SHAP project to Target DAGs

The NASA Target DAGs project follows the archived Tao RWD ACIC 2026 project,
[“SHAP has short memory. Causal SHAP remembers the
DAG.”](https://www.tao-rwd.com/acic-2026/causal-shap), and expands its scientific
test. That page—not the sparse local conference pointer—is the visual and
narrative source of truth for the earlier work.

## What carries forward

- Feature attribution becomes misleading when predictive proximity is treated
  as intervention leverage.
- A defensible DAG and domain expertise are explicit inputs, not decorations.
- Ordinary, DAG-constrained, and structurally informed attribution must be
  compared on the same fitted learner.
- Monte Carlo sensitivity and negative findings are part of the result.

## What the new project adds

- A public NASA Living DAG rather than a small generic simulation graph.
- Source-aligned clean and NASA-like structural simulation regimes.
- Frozen common-random-number total-effect truth for all eligible ancestors.
- A matched-background control showing that topological ordering alone does not
  produce a reliable recovery advantage.
- An explicit intervention-propagating coalition value function that distinguishes
  `do(X=x)` propagation from ordering constraints.
- DAG-distance recovery metrics, including PBI and POA.
- A deterministic paper/video app with separate narrative and audit views.

The small structural prototype is encouraging, but its scale and uncertainty do
not yet justify a manuscript claim. That guardrail is a direct continuation of
the original project's 50-versus-500-permutation sensitivity lesson.
