# Data and DAG Provenance

## NASA renal-stone DAG

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

## Robert Reynolds DAG handoff

Robert Reynolds supplied two raw DAGitty files by email on 2026-07-13:

- `Renal Stone Risk Edge Work DAG CM Final - Errata 20220322`
- `SANS Risk Edge Work DAG CM Final - Errata 20220411`

The raw graph text is preserved under `references/robert-reynolds-2026-07-13/`.
`analysis/10_ingest_robert_dags.R` converts both graphs to canonical node and
edge tables, renders them, and checks lossless graph round trips. The associated
email itself is not stored in this public research repository; a concise record
of the scientific notes is in `ROBERT_REYNOLDS_DAGS_2026-07-13.md`.

The SANS graph currently establishes topology and provenance only. No SANS
coefficients, distributions, synthetic records, or attribution results are
claimed.

## Simulation regimes

- `source_aligned_clean_v3`: 10,000 synthetic records generated directly from
  the declared source-aligned structural model.
- `source_aligned_nasa_like_v4`: synthetic selection and informative measurement
  processes layered onto the same renal mechanism.
- `clean_v1` and `nasa_like_v2`: earlier exploratory models retained for
  provenance but not treated as source-exact.

## Pedagogic mediator/proxy data

`app/bundles/acic_proxy_stress_test/data.csv` is a synthetic teaching dataset.
The associated edge list and total effects are checked into the same bundle. It
is deliberately designed to make proxy over-credit visible and must not be
described as a NASA result.

## Interpretation boundary

Graph topology is source-aligned; numerical coefficients are simulation-design
parameters. Neither the coefficients nor the resulting risk estimates should be
reported as empirically estimated NASA quantities without domain calibration.
