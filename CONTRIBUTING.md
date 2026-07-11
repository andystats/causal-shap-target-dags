# Contributing

This repository is an early research implementation. Issues and pull requests
that improve reproducibility, tests, documentation, causal estimands, or
simulation diagnostics are welcome.

Before opening a pull request:

1. Keep pedagogic stress tests visibly separate from source-aligned analyses.
2. Do not replace a null or diagnostic result by tuning the data-generating
   process for a more dramatic visual result.
3. Add or update tests for changes to the structural value function.
4. Run the Python tests and `Rscript analysis/validate_outputs.R`.
5. Do not commit private correspondence, unpublished collaborator material,
   credentials, or third-party PDFs.

Scientific claims should identify the estimand, intervention semantics,
background population, learner, evaluation records, Monte Carlo configuration,
and uncertainty procedure.
