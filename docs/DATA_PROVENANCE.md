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
