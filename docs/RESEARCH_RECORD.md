# Research record — narrative, methods, results, limitations, roadmap

_Consolidated 2026-08-09 from: PROJECT_NARRATIVE METHODS RESULTS LIMITATIONS ROADMAP_

---

## Canonical project narrative

This project is the next step after the archived ACIC 2026 Causal SHAP
demonstration, [“SHAP has short memory. Causal SHAP remembers the
DAG.”](https://www.tao-rwd.com/acic-2026/causal-shap). The archive establishes
the attribution problem; Target DAGs asks how that causal account can support
the search for an intervention target.

### One sentence

**SHAP explains the model's ears. Target DAGs look for the system's levers.**

### Thirty-second pitch

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

### The Markov explanation

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

#### When the exact splat weakens

Upstream variables can retain predictive attribution when they have direct
paths to the outcome, mediators are missing or noisy, the learner is finite or
misspecified, latent variables violate the proposed DAG, or conditional rather
than marginal background semantics redistribute observational information.
None of those cases turns predictive SHAP into a total-effect estimator.

### The demonstration sequence

#### 1. Build a legible trap

Use the five-node teaching DAG:

`Diet / Climate → Hydration → Outcome → ClinicVisit`

The complete graph also includes `Climate → Outcome` and
`Diet → ClinicVisit`. `ClinicVisit` is highly predictive but has zero total
effect on the outcome.

#### 2. Show the ordinary attribution

Normalized mean absolute ordinary SHAP:

| Variable | Share |
|---|---:|
| Diet | 8.8% |
| Climate | 15.4% |
| Hydration | 30.2% |
| **ClinicVisit** | **45.6%** |

The honest interpretation is: ordinary SHAP correctly reports what the fitted
predictor used.

#### 3. Show known intervention truth

Normalized absolute total effects from the known structural equations:

| Variable | Share |
|---|---:|
| Diet | 27.6% |
| Climate | 37.9% |
| Hydration | 34.5% |
| **ClinicVisit** | **0.0%** |

The conclusion is deliberately narrow: **the most predictive feature is not
necessarily an intervention target**.

### The methodological move

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

### From attribution to intervenable recommendations

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

### Evidence hierarchy

#### Established in this repository

- A zero-effect downstream proxy can dominate ordinary SHAP in a known toy SCM.
- On the NASA-topology simulation, matched ordinary and DAG-ordering SHAP are
  statistically tied.
- The intervention-propagating prototype is substantially closer to frozen
  total-effect truth.

#### Still provisional

- Every outcome is synthetic; NASA and collaborator DAGs supply topology, not
  effect estimates.
- The structural NASA run is small and needs larger Monte Carlo budgets,
  repeated seeds, and paired uncertainty.
- Target recovery does not yet include real action costs, ethical constraints,
  or a validated intervention policy.

### Defensible novelty claim

Do not claim a new definition of Causal SHAP. Use:

> We provide a reproducible target-recovery demonstration showing why Markov
> screening can concentrate predictive SHAP near an outcome, why causal ordering
> alone can leave that failure intact, and why an intervention-propagating
> coalition value is the relevant bridge toward actionable target ranking.

### Phrases to use consistently

- predictive attribution versus intervention leverage;
- last-node splat;
- screening off for prediction, transmitting under intervention;
- ordering alone versus intervention propagation;
- target recovery against frozen total-effect truth;
- recommendation candidate, not causal promise.

Avoid “SHAP is wrong,” “NASA effect,” “causal importance proves actionability,”
or “the method recovers the true intervention” without the relevant qualifiers.

---

## Methods

### Research question

Can a causal attribution procedure rank upstream intervention targets more
faithfully than predictive feature attribution when the data-generating DAG and
total-effect truth are known?

The primary worked example is a synthetic renal-stone risk system constructed
from NASA's published SA-07566 renal-stone DAG. The target is
`Nephrolithiasis`; the feature set contains its 28 pre-outcome ancestors.

### Structural simulation

The R simulator in `analysis/R/renal_stone_source_aligned_simcausal.R` converts
the published DAGitty specification into a `simcausal` structural model. The
clean-v3 data contain 10,000 synthetic records. A separate NASA-like-v4 regime
adds astronaut selection and informative measurement processes.

The locked simulator parent structure has Cohen's kappa 1.000 and zero edge
discrepancies against the declared source graph. Coefficients are simulation
parameters—not NASA effect estimates—and require domain review before substantive
interpretation.

### Frozen intervention truth

For each of the 28 ancestors, `analysis/06_compute_interventional_truth.R`
estimates a standardized absolute total risk difference with 50,000
common-random-number structural simulations. This truth is computed before and
independently of the attribution comparison.

### Prediction model

The prespecified first learner is XGBoost, trained once on clean v3 and evaluated
on a held-out test set. The fitted model is shared across all attribution methods.
The held-out AUC is 0.684; the true structural probability has AUC 0.701, showing
that the modest discrimination is largely a data-generating ceiling rather than
an obvious learner failure.

### Attribution estimators

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

### Evaluation

Attribution importance is compared with the frozen total-effect truth using
Kendall tau, Spearman rho, top-five recovery, NDCG@5, mean directed distance to
the outcome, proximal mass, the Proximity Bias Index (PBI), and the Proximal
Over-attribution Area (POA). The locked ordering-only comparison uses 2,000
paired evaluation-record bootstrap draws.

The structural prototype currently uses 32 evaluation records, 32 background
draws, and 32 permutations. It is an implementation milestone, not yet a locked
publication result.

---

## Current Results

### Predictive ceiling

| Score | AUC |
| --- | ---: |
| True structural risk probability | 0.701 |
| XGBoost | 0.684 |

The low-looking AUC is expected for the current weak-signal structural model.
Correctly specified logistic models and a probability forest did not reveal a
substantially higher exploitable ceiling.

### Ordering alone does not solve the problem

| Method | Kendall tau | Top-5 recovery | PBI | Mass within 2 hops |
| --- | ---: | ---: | ---: | ---: |
| Exact TreeSHAP | 0.522 | 0.60 | 1.082 | 83.1% |
| Matched ordinary interventional SHAP | 0.506 | 0.60 | 1.051 | 81.6% |
| DAG-asymmetric SHAP | 0.528 | 0.60 | 1.051 | 81.2% |

Paired bootstrap intervals include zero for the differences in rank recovery,
PBI, POA, and proximal mass between matched ordinary and DAG-asymmetric SHAP.
Restricting feature order therefore does not, by itself, support a causal
recovery claim.

### Structural prototype

| Metric | Structural prototype | Ordering-only DAG-asymmetric |
| --- | ---: | ---: |
| Kendall tau | 0.794 | 0.528 |
| Spearman rho | 0.932 | 0.652 |
| Top-5 recovery | 1.00 | 0.60 |
| PBI | -0.113 | 1.051 |
| POA | -0.0226 | 0.210 |
| Mass within 2 hops | 38.1% | 81.2% |

The frozen truth places 42.1% of importance within two hops. The prototype is
slightly too upstream but far closer than the predictive estimators. These
numbers remain provisional until the structural estimator is scaled,
bootstrapped, and repeated across simulation seeds.

### Pedagogic stress test

The app also contains a deliberately dramatic mediator/proxy example. It is
useful for teaching the difference between prediction and intervention, but it
is not NASA evidence and is never used as the primary scientific result.

---

## Limitations and Guardrails

1. The data are synthetic and the structural coefficients are not NASA effect
   estimates.
2. The primary source graph still requires domain review for version choice,
   actionability labels, and intervention cost/difficulty.
3. A held-out AUC near 0.68 limits prediction-level separation, although the true
   structural score shows that this is mostly a simulation ceiling.
4. DAG-constrained ordering and structural intervention propagation answer
   different questions; they should not share the label “Causal SHAP” without a
   precise value-function definition.
5. The encouraging structural result is currently a 32×32×32 prototype without
   paired bootstrap uncertainty or a simulation-seed grid.
6. The pedagogic mediator/proxy example is intentionally dramatic and is not
   primary scientific evidence.
7. Intervention-target ranking is not a treatment recommendation. Feasibility,
   safety, timing, cost, and domain constraints are outside the current engine.

---

## Roadmap

### Research toolkit (in place)

The interactive app is built around a six-rung workflow ladder — vanilla SHAP
→ causal discovery → complexity score → structural Causal SHAP → simulation
validation → iteration. The public site now reduces that machinery to a
five-question conceptual hierarchy from prediction to recommendation:

- `app/causal_shap/` now includes teaching DAGs, causal discovery (PC/GES live via
  causal-learn; LiNGAM/NOTEARS as precomputed appendix), a pluggable complexity
  score (PSCI v0), a Credence-style layered-parameter validation subpackage, and
  the homunculus/ladder figures.
- The Shiny app is rebuilt as the ladder, with live discovery and validation.
- A single-page Quarto site (`site/`) carries the ACIC-to-Target-DAGs argument
  and five-rung hierarchy. Detailed methods, references, provenance, and
  limitations remain in `docs/`.

Remaining toolkit tasks: swap in the authors' final complexity score at the
registry seam and optionally extend validation to multiple treatments and
mediation. The interactive app runs locally by design (not hosted) — a hosted
instance can't pin the result environment, and the browser sandbox can't load
the scientific stack.

### Publication-critical next steps

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

### Explanation-to-intervention research program

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

### Subsequent extensions

- Evaluate a second human-system-risk DAG when a reviewable graph is available.
- Produce the intervention-propagation animation and record the guided app
  walkthrough.
