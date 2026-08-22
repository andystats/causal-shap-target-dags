# Provenance and references — DAG/data lineage, ACIC lineage, Reynolds handoff, annotated sources

_Consolidated 2026-08-09 from: DATA_PROVENANCE ACIC_LINEAGE ROBERT_REYNOLDS_DAGS_2026-07-13 RESEARCH_REFERENCES_

---

## Data and DAG Provenance

### NASA renal-stone DAG

The source graph is the public NASA SA-07566 renal-stone risk DAG distributed in
DAGitty syntax. The repository retains the machine-readable text required to
reproduce the simulations and derived graph-validation outputs.

Public sources:

- NASA renal-stone risk page:
  <https://www.nasa.gov/directorates/esdmd/hhp/risk-of-renal-stone-formation/>
- NASA DAG guidance: <https://ntrs.nasa.gov/citations/20220006812>
- NASA HSRB DAG report: <https://ntrs.nasa.gov/citations/20220015709>
- Causal diagramming for human system risk:
  <https://doi.org/10.1038/s41526-024-00375-7>

The repository's renal-stone datasets are entirely synthetic. They do not
contain astronaut, patient, or participant records.

### Robert Reynolds DAG handoff

Robert Reynolds supplied two raw DAGitty files by email on 2026-07-13:

- `Renal Stone Risk Edge Work DAG CM Final - Errata 20220322`
- `SANS Risk Edge Work DAG CM Final - Errata 20220411`

The raw graph text is preserved under `references/robert-reynolds-2026-07-13/`.
`analysis/10_ingest_robert_dags.R` converts both graphs to canonical node and
edge tables, renders them, and checks lossless graph round trips. The associated
email itself is not stored in this public research repository; a concise record
of the scientific notes is in `PROVENANCE_AND_REFERENCES.md`.

The SANS graph currently establishes topology and provenance only. No SANS
coefficients, distributions, synthetic records, or attribution results are
claimed.

### Simulation regimes

- `source_aligned_clean_v3`: 10,000 synthetic records generated directly from
  the declared source-aligned structural model.
- `source_aligned_nasa_like_v4`: synthetic selection and informative measurement
  processes layered onto the same renal mechanism.
- `clean_v1` and `nasa_like_v2`: earlier exploratory models retained for
  provenance but not treated as source-exact.

### Pedagogic mediator/proxy data

`app/bundles/acic_proxy_stress_test/data.csv` is a synthetic teaching dataset.
The associated edge list and total effects are checked into the same bundle. It
is deliberately designed to make proxy over-credit visible and must not be
described as a NASA result.

### Interpretation boundary

Graph topology is source-aligned; numerical coefficients are simulation-design
parameters. Neither the coefficients nor the resulting risk estimates should be
reported as empirically estimated NASA quantities without domain calibration.

---

## From the ACIC 2026 Causal SHAP project to Target DAGs

The NASA Target DAGs project follows the archived Tao RWD ACIC 2026 project,
[“SHAP has short memory. Causal SHAP remembers the
DAG.”](https://www.tao-rwd.com/acic-2026/causal-shap), and expands its scientific
test. That page—not the sparse local conference pointer—is the visual and
narrative source of truth for the earlier work.

### What carries forward

- Feature attribution becomes misleading when predictive proximity is treated
  as intervention leverage.
- A defensible DAG and domain expertise are explicit inputs, not decorations.
- Ordinary, DAG-constrained, and structurally informed attribution must be
  compared on the same fitted learner.
- Monte Carlo sensitivity and negative findings are part of the result.

### What the new project adds

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

---

## Robert Reynolds DAG Handoff — 2026-07-13

### What arrived

Robert Reynolds sent Andy Wilson and Lexi Pasi two DAGitty source files in the
email thread `DAG code` on 2026-07-13:

- `Renal Stone Risk.txt`: *Renal Stone Risk Edge Work DAG CM Final - Errata
  20220322* — 53 nodes and 83 directed edges.
- `SANS Risk.txt`: *SANS Risk Edge Work DAG CM Final - Errata 20220411* — 50
  nodes and 89 directed edges.

Both parse as DAGs and round-trip exactly through the repository's canonical
node/edge CSV representation. The raw files are preserved in
`references/robert-reynolds-2026-07-13/`; derived artifacts are in
`analysis/output/dag_sources/`.

### Robert's modeling note

Some edges are definitional or “by design,” not empirical associations that
need literature-effect estimates. His examples were astronaut selection shaping
the crew's individual factors and an environmental-control system governing
spacecraft temperature or humidity.

For synthetic-data generation, this distinction should be explicit in an edge
or mechanism manifest. A by-design mechanism may be deterministic, but that
does **not** imply a correlation of 1.0 unless the child is literally an
identity or affine copy with no other causes. Deterministic structural equations
also need special handling in attribution experiments because they can create
redundant features and off-support coalitions.

Before fitting either graph with distributions, classify each edge as one of:

1. definitional/identity;
2. engineered or by-design control;
3. biological/behavioral causal relationship;
4. measurement/detection relationship; or
5. decision/treatment response.

Then specify which mechanisms are deterministic, stochastic, calibrated, or
stress-test assumptions.

### Renal-stone match to the repository graph

The match is close after accounting for label and abstraction differences, but
it is **not exact**.

#### Strict comparison

- Existing SA-07566 source: 51 nodes, 75 edges.
- Robert's renal graph: 53 nodes, 83 edges.
- Union-label comparison: Cohen's kappa 0.638, structural Hamming distance 56.

The strict result is dominated by granularity and naming. Robert splits `Bone
Remodeling` into `Bone Formation` and `Bone Resorption`, and expands `Medical
Illness` into renal colic, hydronephrosis, infection, sepsis, and renal failure.
He also uses updated labels for CO2, HSI, pharmaceutical effectiveness, and
urinary retention.

#### Semantic shared-node projection

After applying the documented crosswalk and collapsing Robert's expanded nodes:

- 48 shared nodes;
- 69 of 72 existing shared-node edges matched;
- no extra edges in Robert's normalized shared-node projection;
- three existing edges were absent from Robert's file;
- Cohen's kappa 0.978; precision 1.000; recall 0.958; structural Hamming distance
  3.

The three unmatched existing edges are:

- `Resistive Exercise -> Bone Remodeling`
- `Medical Prevention Capability -> Water Intake`
- `Medical Treatment Capability -> Water Intake`

Three upstream abstraction nodes exist only in the repository's previous graph
after semantic mapping: `Bone Fracture (Risk)`, `Food and Nutrition (Risk)`, and
`Microhost (Risk)`.

Therefore, the current source-aligned renal simulator remains exactly matched to
the previous 51-node/75-edge reference, not to Robert's expanded 53-node/83-edge
graph. Robert's version should be treated as a new candidate source revision
until the team decides whether the three unmatched edges and the expanded
complication/bone structure supersede the previous reference.

### Pipeline status and next decisions

`analysis/10_ingest_robert_dags.R` now:

- ingests both raw DAGitty files;
- exports nodes and edges;
- renders both graphs;
- verifies exact source-to-CSV round trips;
- runs strict and semantic renal concordance checks; and
- records the semantic crosswalk and discrepancies.

The SANS graph is ready for domain review and simulation design, but the pipeline
does not yet generate SANS data. The next decisions are:

1. confirm which renal graph is canonical for the next analysis;
2. ask Robert to adjudicate the three unmatched renal edges;
3. label definitional/by-design edges in both graphs;
4. choose the SANS target outcome and intervention questions;
5. specify distributions, structural equations, measurement processes, and
   coefficient sensitivity ranges; and
6. confirm Robert's authorship participation, which Andy asked about in the
   July 13 reply.

---

## Annotated research references

These are the primary references behind the streamlined project narrative. The
annotations record what each source supports so citations do not drift beyond
the paper's actual contribution.

### Predictive Shapley attribution

#### Lundberg & Lee (2017)

Scott M. Lundberg and Su-In Lee. “A Unified Approach to Interpreting Model
Predictions.” *Advances in Neural Information Processing Systems 30*.

- Primary source: <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>
- Supports: the modern additive feature-attribution/SHAP framework and its
  relationship to model explanations.
- Use for: defining ordinary SHAP as an explanation of a fitted prediction.
- Do not use for: claiming that SHAP values are causal effects or intervention
  recommendations.

### Markov screening and feature sufficiency

#### Margaritis (2009)

Dimitris Margaritis. “Toward Provably Correct Feature Selection in Arbitrary
Domains.” *Advances in Neural Information Processing Systems 22*.

- Primary source: <https://proceedings.neurips.cc/paper_files/paper/2009/file/6da37dd3139aa4d9aa55b8d237ec5d4a-Paper.pdf>
- Supports: a Markov boundary as a minimal feature set that makes the target's
  distribution conditionally invariant to all other features.
- Use for: explaining why a predictor can lose nothing by ignoring upstream
  variables after observing an outcome-near sufficient set.
- Do not use for: identifying causal directions from a fitted predictive model.

### Observation versus intervention in feature relevance

#### Janzing, Minorics & Blöbaum (2020)

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

### Causal Shapley values and indirect effects

#### Heskes, Sijben, Bucur & Claassen (2020)

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

### Causal ordering as a separate choice

#### Frye, Rowat & Feige (2020)

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

### From explanation to action

#### Karimi, Schölkopf & Valera (2021)

Amir-Hossein Karimi, Bernhard Schölkopf, and Isabel Valera. “Algorithmic
Recourse: from Counterfactual Explanations to Interventions.” *Proceedings of
FAccT '21*. <https://doi.org/10.1145/3442188.3445899>

- Open version: <https://arxiv.org/abs/2002.06278>
- Supports: the shift from nearest favorable counterfactual states to minimal
  interventions—moving from explanation to recommendation.
- Use for: the claim that actionability requires feasible actions and costs in
  addition to an explanation or attribution.

#### Karimi, von Kügelgen, Schölkopf & Valera (2020)

Amir-Hossein Karimi, Julius von Kügelgen, Bernhard Schölkopf, and Isabel Valera.
“Algorithmic Recourse under Imperfect Causal Knowledge: a Probabilistic
Approach.” *Advances in Neural Information Processing Systems 33*.

- Primary source: <https://proceedings.neurips.cc/paper/2020/file/02a3c7fb3f489288ae6942498498db20-Paper.pdf>
- Supports: the impossibility of generally guaranteeing recourse without the
  true structural equations when interventions have descendants, plus
  probabilistic approaches under imperfect causal knowledge.
- Use for: the uncertainty/robustness requirement and the phrase
  “recommendation candidate, not causal promise.”

### Designed validation against known truth

#### Parikh, Varjão, Xu & Tchetgen Tchetgen (2022)

Harsh Parikh, Carolina Varjão, Louise Xu, and Eric Tchetgen Tchetgen.
“Validating Causal Inference Methods.” *Proceedings of ICML*, PMLR 162.

- Primary source: <https://proceedings.mlr.press/v162/parikh22a.html>
- Supports: validation of causal estimators across designed data-generating
  processes where target quantities are known.
- Use for: the layered simulation philosophy and the separation between
  pedagogic stress tests and publication-facing evidence.
- Note: this repository uses the author's own implementation in the spirit of
  Credence; it redistributes no Credence repository code.

### Claim-to-citation map

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
