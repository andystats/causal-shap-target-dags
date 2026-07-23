# Renal Stone `simcausal` Sandbox

This folder contains source-aligned and exploratory renal-stone simulators,
graph-concordance tooling, and the first intervention-truth/SHAP comparison.

## Clean v3 Nephrolithiasis SHAP pilot

The first locked comparison is complete in
`output/shap_nephrolithiasis_clean_v3/`.

- `06_compute_interventional_truth.R` estimates standardized absolute total
  risk-difference truth for all 28 ancestors using 50,000 common-random-number
  structural simulations.
- `07_run_shap_comparison.R` fits one held-out XGBoost model and compares exact
  TreeSHAP, matched-background unrestricted ordinary SHAP, and DAG-constrained
  asymmetric SHAP on the same 64 evaluation records. The latter two use the same
  128-record background and 128 permutations.
- `08_bootstrap_shap_comparison.R` runs 2,000 paired evaluation-record
  bootstraps.
- `09_diagnose_predictive_signal.R` compares XGBoost with the true structural
  risk score, correctly specified logistic models, and a probability forest.
- `../app/causal_shap/structural_value.py` implements the tested Python
  intervention-propagating coalition value function; its checked-in 32×32×32
  prototype outputs are stored alongside the R results.
- `R/shap_distance_metrics.R` contains the reusable structural simulator,
  topological-order SHAP engine, and distance metrics.

Primary result: DAG-asymmetric SHAP looks modestly less proximal than exact
TreeSHAP, but it is essentially indistinguishable from matched-background
unrestricted ordinary SHAP (PBI 1.051 for both; POA 0.210 for both). Paired
bootstrap intervals include zero for rank, PBI, POA, and proximal-mass
differences. Ordering alone does not recover the distributed total-effect truth.
The small structural prototype is substantially closer to the frozen truth
(Kendall tau 0.794; top-five recovery 1.00; PBI -0.113), while remaining explicitly
exploratory pending scale-up and bootstrap uncertainty. See
`../docs/RESULTS.md` and `../docs/METHODS.md`.

## Canonical versions for the SHAP pilot

### Source-aligned clean v3

- Built programmatically from NASA's published SA-07566 DAGitty code.
- 51 source nodes and 75 source edges.
- Locked `simcausal` parent structure has Cohen's kappa 1.000 versus NASA, with zero false-positive or false-negative edges.
- Output: `output/source_aligned_clean/renal_stone_source_aligned_clean_v3.csv`.

### Source-aligned NASA-like v4

- Retains all 51 NASA nodes and 75 NASA edges.
- Adds eight observation-process nodes and ten edges for fitness/age selection and informative measurement.
- Full 59-node/85-edge specification has Cohen's kappa 1.000 versus the locked `simcausal` parent structure.
- Output: `output/source_aligned_nasa_like/renal_stone_source_aligned_nasa_like_v4.csv`.

These are the versions to use for the first definitive SHAP comparison.

## Earlier exploratory versions

### Clean v1

- 10,000 complete records.
- No selection; `selected_astronaut` is fixed to 1.
- Fully observed renal pathway and downstream outcomes.
- Output: `output/clean/renal_stone_clean_v1.csv`.

### NASA-like v2

- Generates a 50,000-person source population.
- Astronaut selection depends on baseline fitness, individual susceptibility, and age.
- Conditions on selection and retains 450 observations.
- Adds informative measurement sparsity for bone remodeling, hydration, urine concentration, urine chemistry, and mineralized renal material.
- Output: `output/nasa_like/renal_stone_nasa_like_v2.csv`.

Both versions use identical renal-mechanism equations. Version 2 changes only the observation/selection process.

The exploratory v1/v2 models predated discovery of NASA's machine-readable DAG code. They are retained for provenance, but their mapped source-projection kappas are approximately 0.81 and 0.80 and they should not be treated as source-exact.

## Run

The whole pipeline runs from one command (from anywhere in the repo):

```bash
Rscript analysis/run_all.R
```

Stages, in order: `generate.R` (all four datasets) → `03_validate_and_plot_dags.R`
→ `06_compute_interventional_truth.R` → `07_run_shap_comparison.R` →
`08_bootstrap_shap_comparison.R` → `09_diagnose_predictive_signal.R` →
`validate_outputs.R`. Each is also runnable on its own — via `Rscript` **or**
interactively (`source("analysis/generate.R"); generate_dataset("source_aligned_clean")`)
— because every entry point self-locates the repo through `R/paths.R` instead of
the old `--file=` header. The four numbered generators (`01/02/04/05`) are
consolidated into `generate.R`.

The Robert Reynolds DAG ingest runs separately from the main pipeline:

```bash
Rscript analysis/10_ingest_robert_dags.R
```

`10_ingest_robert_dags.R` preserves and parses Robert Reynolds's renal-stone
and SANS DAGitty files, exports canonical node/edge tables and plots, verifies
lossless round trips, and compares the renal graph with the repository's
previous SA-07566 source. It ingests SANS topology only; it does not invent
structural coefficients or a SANS data-generating mechanism.

The scripts require `simcausal`, `igraph`, `xgboost`, `dagitty`, `ggdag`, and
`ggplot2` (run under R 4.5.2 in this project).

## Current validation result

- Clean rows: 10,000.
- NASA-like rows: 450, all conditioned on `selected_astronaut == 1`.
- Clean nephrolithiasis prevalence: 9.57%.
- NASA-like nephrolithiasis prevalence in the retained sample: 6.67%.
- NASA-like urine-chemistry missingness: 50.89%.
- Selection moves mean baseline fitness from approximately 0.00 to 0.34 and mean susceptibility from approximately 0.00 to -0.31 in the selected pool.
- Conditioning induces a fitness-susceptibility correlation of about 0.16 in the selected pool from approximately 0.00 in the source population.

Exact results are seed-stable using the seeds in the generation scripts.

Graph-validation and source versioning are summarized in
`../docs/DATA_PROVENANCE.md` and checked in under `output/dag_validation/`.

## Important limitation

These are simulation-design parameters, not NASA estimates. The structure and
coefficients need domain review before substantive interpretation. See
`../docs/LIMITATIONS.md`.
