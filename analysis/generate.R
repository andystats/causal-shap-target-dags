#!/usr/bin/env Rscript
# Generate every renal-stone dataset from one entry point — replaces the four
# numbered drivers 01/02/04/05.
#
#   Rscript analysis/generate.R                 # write all four datasets
#   Rscript analysis/generate.R source_aligned_clean   # just one variant
#   source("analysis/generate.R"); d <- generate_dataset("source_aligned_clean")
#
# Each variant keeps its exact n / seed / output paths, so the bytes are
# identical to the old drivers. The two structural models (hand-written vs
# source-derived) stay separate — this only consolidates the drivers.

local({                                   # locate + source paths.R from anywhere at/below the repo
  dir <- getwd()
  while (!file.exists(file.path(dir, "analysis", "R", "paths.R"))) {
    parent <- dirname(dir)
    if (identical(parent, dir)) stop("Run from inside the repository.")
    dir <- parent
  }
  source(file.path(dir, "analysis", "R", "paths.R"))
})
source(file.path(analysis_dir, "R", "renal_stone_simcausal.R"))
source(file.path(analysis_dir, "R", "dag_concordance.R"))
source(file.path(analysis_dir, "R", "renal_stone_source_aligned_simcausal.R"))

SOURCE_CODE_PATH <- file.path(project_dir, "references", "renal-stone-dag-code-SA-07566.txt")

generate_dataset <- function(variant = c("clean", "nasa_like", "source_aligned_clean", "source_aligned_nasa_like"),
                             write = TRUE) {
  variant <- match.arg(variant)
  out <- file.path(analysis_dir, "output")

  if (variant == "clean") {
    dir <- file.path(out, "clean")
    clean <- simulate_renal_stone(n = 10000L, seed = 20260710L, nasa_like = FALSE)
    clean$simulation_version <- "clean_v1"
    if (write) {
      dir.create(dir, recursive = TRUE, showWarnings = FALSE)
      write.csv(clean, file.path(dir, "renal_stone_clean_v1.csv"), row.names = FALSE, na = "")
      write.csv(summarize_simulation(clean, "clean_v1"),
                file.path(dir, "renal_stone_clean_v1_summary.csv"), row.names = FALSE, na = "")
    }
    return(invisible(clean))
  }

  if (variant == "nasa_like") {
    dir <- file.path(out, "nasa_like")
    source_population <- simulate_renal_stone(n = 50000L, seed = 20260710L, nasa_like = TRUE)
    selected_pool <- source_population[source_population$selected_astronaut == 1, , drop = FALSE]
    observed <- make_nasa_like_observed(source_data = source_population, n_observed = 450L, seed = 20260711L)
    observed$simulation_version <- "nasa_like_sparse_selection_v2"
    if (write) {
      dir.create(dir, recursive = TRUE, showWarnings = FALSE)
      write.csv(observed, file.path(dir, "renal_stone_nasa_like_v2.csv"), row.names = FALSE, na = "")
      write.csv(selection_diagnostics(source_population, selected_pool, observed),
                file.path(dir, "renal_stone_nasa_like_v2_diagnostics.csv"), row.names = FALSE, na = "")
    }
    return(invisible(observed))
  }

  if (variant == "source_aligned_clean") {
    dir <- file.path(out, "source_aligned_clean")
    data <- simulate_source_aligned(source_code_path = SOURCE_CODE_PATH, n = 10000L, seed = 20260710L, nasa_like = FALSE)
    variable_map <- attr(data, "variable_map"); attr(data, "variable_map") <- NULL
    data$simulation_version <- "source_aligned_clean_v3"
    if (write) {
      dir.create(dir, recursive = TRUE, showWarnings = FALSE)
      write.csv(data, file.path(dir, "renal_stone_source_aligned_clean_v3.csv"), row.names = FALSE, na = "")
      write.csv(variable_map, file.path(dir, "source_to_simulation_variable_map.csv"), row.names = FALSE)
    }
    return(invisible(data))
  }

  # source_aligned_nasa_like
  dir <- file.path(out, "source_aligned_nasa_like")
  source_population <- simulate_source_aligned(source_code_path = SOURCE_CODE_PATH, n = 50000L, seed = 20260710L, nasa_like = TRUE)
  variable_map <- attr(source_population, "variable_map"); attr(source_population, "variable_map") <- NULL
  selected_pool <- source_population[source_population$selected_into_observed_cohort == 1, , drop = FALSE]
  observed <- make_source_aligned_nasa_like_observed(source_data = source_population, n_observed = 450L, seed = 20260712L)
  observed$simulation_version <- "source_aligned_nasa_like_v4"
  if (write) {
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
    write.csv(observed, file.path(dir, "renal_stone_source_aligned_nasa_like_v4.csv"), row.names = FALSE, na = "")
    write.csv(source_aligned_diagnostics(source_population, selected_pool, observed),
              file.path(dir, "renal_stone_source_aligned_nasa_like_v4_diagnostics.csv"), row.names = FALSE, na = "")
    write.csv(variable_map, file.path(dir, "source_to_simulation_variable_map.csv"), row.names = FALSE)
  }
  invisible(observed)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  variants <- if (length(args)) args else c("clean", "nasa_like", "source_aligned_clean", "source_aligned_nasa_like")
  for (v in variants) { generate_dataset(v); message("generated: ", v) }
}
