script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")
analysis_dir <- dirname(script_path)
project_dir <- normalizePath(file.path(analysis_dir, ".."), winslash = "/")

source(file.path(analysis_dir, "R", "renal_stone_simcausal.R"))

output_dir <- file.path(analysis_dir, "output", "nasa_like")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

source_population <- simulate_renal_stone(
  n = 50000L,
  seed = 20260710L,
  nasa_like = TRUE
)

selected_pool <- source_population[
  source_population$selected_astronaut == 1,
  ,
  drop = FALSE
]

observed <- make_nasa_like_observed(
  source_data = source_population,
  n_observed = 450L,
  seed = 20260711L
)

observed$simulation_version <- "nasa_like_sparse_selection_v2"

write.csv(
  observed,
  file.path(output_dir, "renal_stone_nasa_like_v2.csv"),
  row.names = FALSE,
  na = ""
)

write.csv(
  selection_diagnostics(source_population, selected_pool, observed),
  file.path(output_dir, "renal_stone_nasa_like_v2_diagnostics.csv"),
  row.names = FALSE,
  na = ""
)

message("Wrote NASA-like sparse/selected simulation to: ", output_dir)
