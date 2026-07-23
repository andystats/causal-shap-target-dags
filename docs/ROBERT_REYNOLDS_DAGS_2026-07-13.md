# Robert Reynolds DAG Handoff — 2026-07-13

## What arrived

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

## Robert's modeling note

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

## Renal-stone match to the repository graph

The match is close after accounting for label and abstraction differences, but
it is **not exact**.

### Strict comparison

- Existing SA-07566 source: 51 nodes, 75 edges.
- Robert's renal graph: 53 nodes, 83 edges.
- Union-label comparison: Cohen's kappa 0.638, structural Hamming distance 56.

The strict result is dominated by granularity and naming. Robert splits `Bone
Remodeling` into `Bone Formation` and `Bone Resorption`, and expands `Medical
Illness` into renal colic, hydronephrosis, infection, sepsis, and renal failure.
He also uses updated labels for CO2, HSI, pharmaceutical effectiveness, and
urinary retention.

### Semantic shared-node projection

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

## Pipeline status and next decisions

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
