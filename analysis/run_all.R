#!/usr/bin/env Rscript
# Run the whole R analysis pipeline in order, from one command:
#
#   Rscript analysis/run_all.R
#
# Stages: generate every dataset -> validate/plot the DAGs -> interventional
# truth -> SHAP comparison -> paired bootstrap -> predictive-signal diagnostics
# -> the release gate. Each stage is also runnable on its own (see the file
# headers); this is just the ordered driver.

local({                                   # locate + source paths.R from anywhere at/below the repo
  dir <- getwd()
  while (!file.exists(file.path(dir, "analysis", "R", "paths.R"))) {
    parent <- dirname(dir)
    if (identical(parent, dir)) stop("Run from inside the repository.")
    dir <- parent
  }
  source(file.path(dir, "analysis", "R", "paths.R"))
})

# Generators are function-based; the analysis stages run on source().
source(file.path(analysis_dir, "generate.R"))
for (variant in c("clean", "nasa_like", "source_aligned_clean", "source_aligned_nasa_like")) {
  generate_dataset(variant)
}
message("stage complete: generate")

for (stage in c("03_validate_and_plot_dags",
                "06_compute_interventional_truth",
                "07_run_shap_comparison",
                "08_bootstrap_shap_comparison",
                "09_diagnose_predictive_signal")) {
  source(file.path(analysis_dir, paste0(stage, ".R")))
  message("stage complete: ", stage)
}

source(file.path(analysis_dir, "validate_outputs.R"))
message("pipeline complete")
