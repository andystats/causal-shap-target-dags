# Canonical project narrative

## One sentence

**SHAP explains the model's ears. Target DAGs look for the system's levers.**

## Thirty-second pitch

SHAP tells us which features a fitted model used. That is not automatically the
same as telling us which variable to change. In a causal DAG, the variables
nearest the outcome can screen off their ancestors for prediction, so a strong
predictor compresses onto the last measured mediator or proxy. SHAP then
faithfully concentrates credit there—even when an upstream intervention would
propagate through that node, or when the winning feature is a downstream proxy
with zero total effect.

This project tests whether attribution methods recover known intervention
targets. Its central empirical contrast is not “ordinary SHAP versus a method
with a causal name.” It is **static or ordering-only coalitions versus a value
function that lets descendants respond to `do(X=x)`**.

## The Markov explanation

Consider the linear structural causal model

\[
X=\varepsilon_X, \qquad
M=aX+\varepsilon_M, \qquad
Y=bM+\varepsilon_Y,
\]

with independent, mean-zero errors and graph
\(X\rightarrow M\rightarrow Y\).

The causal Markov condition gives

\[
Y\perp X\mid M.
\]

Therefore, when both features are available, the Bayes predictor is

\[
f(x,m)=\mathbb E[Y\mid X=x,M=m]
=\mathbb E[Y\mid M=m]
=bm.
\]

Once the mediator is observed, the predictor has no remaining use for the
upstream cause. For marginal or model-interventional SHAP, adding `X` changes no
coalition value, so the Shapley dummy property permits

\[
\phi_X^{\mathrm{model}}=0.
\]

But the intervention effect remains

\[
\mathbb E[Y\mid do(X=x)]=abx,
\qquad
\frac{\partial}{\partial x}\mathbb E[Y\mid do(X=x)]=ab.
\]

The mediator **screens off** the ancestor for prediction while still
**transmitting** its intervention effect. This is the mathematical core of the
“last-node splat.” It is an estimand mismatch, not a failure of the Shapley
axioms.

### When the exact splat weakens

Upstream variables can retain predictive attribution when they have direct
paths to the outcome, mediators are missing or noisy, the learner is finite or
misspecified, latent variables violate the proposed DAG, or conditional rather
than marginal background semantics redistribute observational information.
None of those cases turns predictive SHAP into a total-effect estimator.

## The demonstration sequence

### 1. Build a legible trap

Use the five-node teaching DAG:

`Diet / Climate → Hydration → Outcome → ClinicVisit`

The complete graph also includes `Climate → Outcome` and
`Diet → ClinicVisit`. `ClinicVisit` is highly predictive but has zero total
effect on the outcome.

### 2. Show the ordinary attribution

Normalized mean absolute ordinary SHAP:

| Variable | Share |
|---|---:|
| Diet | 8.8% |
| Climate | 15.4% |
| Hydration | 30.2% |
| **ClinicVisit** | **45.6%** |

The honest interpretation is: ordinary SHAP correctly reports what the fitted
predictor used.

### 3. Show known intervention truth

Normalized absolute total effects from the known structural equations:

| Variable | Share |
|---|---:|
| Diet | 27.6% |
| Climate | 37.9% |
| Hydration | 34.5% |
| **ClinicVisit** | **0.0%** |

The conclusion is deliberately narrow: **the most predictive feature is not
necessarily an intervention target**.

## The methodological move

The structural coalition value is

\[
v_{do}(S)=\mathbb E\!\left[f(X)\mid do(X_S=x_S)\right].
\]

Fix the in-coalition variables, then regenerate every unfixed descendant through
the structural model. The resulting Shapley value can contain both direct and
propagated indirect contribution.

Keep these two design choices separate:

- **Causal ordering** changes which feature permutations or coalitions are
  allowed.
- **Intervention propagation** changes what the modeled system does after an
  intervention.

The NASA-topology matched comparison is valuable precisely because ordering
alone is statistically tied with ordinary SHAP, while the small structural
prototype moves substantially closer to frozen total-effect truth.

## From attribution to intervenable recommendations

Structural attribution is one input to a recommendation, not the recommendation
itself. A general decision target is

\[
a^*=\arg\max_{a\in\mathcal A_{\mathrm{feasible}}}
\frac{\mathbb E[Y\mid do(a)]-\mathbb E[Y]}{C(a)}
\quad\text{subject to}\quad
\Pr\{\Delta_Y(a)>0\}\ge 1-\alpha.
\]

The three required filters are:

1. **Structural effect:** what changes downstream under `do(a)`?
2. **Feasibility and cost:** is the variable manipulable, ethical, affordable,
   and within an allowed range?
3. **Uncertainty:** does the preferred action survive plausible DAGs,
   mechanisms, and effect estimates?

The output is a **recommendation candidate worth testing**, not a causal
guarantee.

## Evidence hierarchy

### Established in this repository

- A zero-effect downstream proxy can dominate ordinary SHAP in a known toy SCM.
- On the NASA-topology simulation, matched ordinary and DAG-ordering SHAP are
  statistically tied.
- The intervention-propagating prototype is substantially closer to frozen
  total-effect truth.

### Still provisional

- Every outcome is synthetic; NASA and collaborator DAGs supply topology, not
  effect estimates.
- The structural NASA run is small and needs larger Monte Carlo budgets,
  repeated seeds, and paired uncertainty.
- Target recovery does not yet include real action costs, ethical constraints,
  or a validated intervention policy.

## Defensible novelty claim

Do not claim a new definition of Causal SHAP. Use:

> We provide a reproducible target-recovery demonstration showing why Markov
> screening can concentrate predictive SHAP near an outcome, why causal ordering
> alone can leave that failure intact, and why an intervention-propagating
> coalition value is the relevant bridge toward actionable target ranking.

## Phrases to use consistently

- predictive attribution versus intervention leverage;
- last-node splat;
- screening off for prediction, transmitting under intervention;
- ordering alone versus intervention propagation;
- target recovery against frozen total-effect truth;
- recommendation candidate, not causal promise.

Avoid “SHAP is wrong,” “NASA effect,” “causal importance proves actionability,”
or “the method recovers the true intervention” without the relevant qualifiers.
