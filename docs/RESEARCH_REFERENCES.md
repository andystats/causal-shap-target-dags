# Annotated research references

These are the primary references behind the streamlined project narrative. The
annotations record what each source supports so citations do not drift beyond
the paper's actual contribution.

## Predictive Shapley attribution

### Lundberg & Lee (2017)

Scott M. Lundberg and Su-In Lee. “A Unified Approach to Interpreting Model
Predictions.” *Advances in Neural Information Processing Systems 30*.

- Primary source: <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>
- Supports: the modern additive feature-attribution/SHAP framework and its
  relationship to model explanations.
- Use for: defining ordinary SHAP as an explanation of a fitted prediction.
- Do not use for: claiming that SHAP values are causal effects or intervention
  recommendations.

## Markov screening and feature sufficiency

### Margaritis (2009)

Dimitris Margaritis. “Toward Provably Correct Feature Selection in Arbitrary
Domains.” *Advances in Neural Information Processing Systems 22*.

- Primary source: <https://proceedings.neurips.cc/paper_files/paper/2009/file/6da37dd3139aa4d9aa55b8d237ec5d4a-Paper.pdf>
- Supports: a Markov boundary as a minimal feature set that makes the target's
  distribution conditionally invariant to all other features.
- Use for: explaining why a predictor can lose nothing by ignoring upstream
  variables after observing an outcome-near sufficient set.
- Do not use for: identifying causal directions from a fitted predictive model.

## Observation versus intervention in feature relevance

### Janzing, Minorics & Blöbaum (2020)

Dominik Janzing, Lenon Minorics, and Patrick Blöbaum. “Feature Relevance
Quantification in Explainable AI: A Causal Problem.” *Proceedings of AISTATS*,
PMLR 108:2907–2916.

- Primary source: <https://proceedings.mlr.press/v108/janzing20a.html>
- Supports: the distinction between observational and interventional
  distributions when defining the value of dropped features.
- Use for: showing that background semantics encode a causal question and must
  be named explicitly.
- Important nuance: their interventional explanation of the prediction
  algorithm collapses dropped real-world dependencies to marginal sampling.
  Heskes et al. deliberately take the next step and model causal dependence
  among real-world features.

## Causal Shapley values and indirect effects

### Heskes, Sijben, Bucur & Claassen (2020)

Tom Heskes, Evi Sijben, Ioan Gabriel Bucur, and Tom Claassen. “Causal Shapley
Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex
Models.” *Advances in Neural Information Processing Systems 33*.

- Primary source: <https://proceedings.neurips.cc/paper/2020/file/32e54441e6382a7fbacbbbaf3c450059-Paper.pdf>
- Supports: the coalition value
  \(v(S)=\mathbb E[f(X)\mid do(X_S=x_S)]\), causal-DAG factorization, and a
  decomposition into direct and indirect Shapley contribution.
- Use for: the structural value function and the chain example in which a model
  ignores an upstream feature while causal Shapley recovers propagated
  contribution.
- Do not present this repository as inventing the do-based Causal Shapley
  definition.

## Causal ordering as a separate choice

### Frye, Rowat & Feige (2020)

Christopher Frye, Colin Rowat, and Ilya Feige. “Asymmetric Shapley Values:
Incorporating Causal Knowledge into Model-Agnostic Explainability.” *Advances in
Neural Information Processing Systems 33*.

- Primary source: <https://papers.nips.cc/paper_files/paper/2020/hash/0d770c496aa3da6d2c3f2bd19e7b9d6b-Abstract.html>
- Supports: restricting Shapley permutations so explanations respect a known
  partial or causal order.
- Use for: defining the ordering-only comparator.
- Do not imply that our ordering-only null refutes asymmetric Shapley values. It
  shows only that ordering did not recover total-effect targets in this matched
  simulation without intervention propagation.

## From explanation to action

### Karimi, Schölkopf & Valera (2021)

Amir-Hossein Karimi, Bernhard Schölkopf, and Isabel Valera. “Algorithmic
Recourse: from Counterfactual Explanations to Interventions.” *Proceedings of
FAccT '21*. <https://doi.org/10.1145/3442188.3445899>

- Open version: <https://arxiv.org/abs/2002.06278>
- Supports: the shift from nearest favorable counterfactual states to minimal
  interventions—moving from explanation to recommendation.
- Use for: the claim that actionability requires feasible actions and costs in
  addition to an explanation or attribution.

### Karimi, von Kügelgen, Schölkopf & Valera (2020)

Amir-Hossein Karimi, Julius von Kügelgen, Bernhard Schölkopf, and Isabel Valera.
“Algorithmic Recourse under Imperfect Causal Knowledge: a Probabilistic
Approach.” *Advances in Neural Information Processing Systems 33*.

- Primary source: <https://proceedings.neurips.cc/paper/2020/file/02a3c7fb3f489288ae6942498498db20-Paper.pdf>
- Supports: the impossibility of generally guaranteeing recourse without the
  true structural equations when interventions have descendants, plus
  probabilistic approaches under imperfect causal knowledge.
- Use for: the uncertainty/robustness requirement and the phrase
  “recommendation candidate, not causal promise.”

## Designed validation against known truth

### Parikh, Varjão, Xu & Tchetgen Tchetgen (2022)

Harsh Parikh, Carolina Varjão, Louise Xu, and Eric Tchetgen Tchetgen.
“Validating Causal Inference Methods.” *Proceedings of ICML*, PMLR 162.

- Primary source: <https://proceedings.mlr.press/v162/parikh22a.html>
- Supports: validation of causal estimators across designed data-generating
  processes where target quantities are known.
- Use for: the layered simulation philosophy and the separation between
  pedagogic stress tests and publication-facing evidence.
- Note: this repository uses the author's own implementation in the spirit of
  Credence; it redistributes no Credence repository code.

## Claim-to-citation map

| Claim | Cite |
|---|---|
| SHAP explains a fitted model prediction | Lundberg & Lee (2017) |
| An outcome-near feature set can screen off other predictors | Margaritis (2009) |
| Dropped-feature semantics require observation/intervention clarity | Janzing et al. (2020) |
| A do-based Shapley value can contain direct and indirect contribution | Heskes et al. (2020) |
| Causal ordering can be encoded by asymmetric permutations | Frye et al. (2020) |
| Recommendation requires actions, feasibility, and cost | Karimi et al. (2021) |
| Imperfect SCM knowledge prevents general recourse guarantees | Karimi et al. (2020) |
| Causal methods should be validated against designed known truth | Parikh et al. (2022) |
