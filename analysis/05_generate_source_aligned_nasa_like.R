script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")
analysis_dir <- dirname(script_path)
project_dir <- normalizePath(file.path(analysis_dir, ".."), winslash = "/")

source(file.path(analysis_dir, "R", "dag_concordance.R"))
source(file.path(analysis_dir, "R", "renal_stone_source_aligned_simcausal.R"))

source_code_path <- file.path(
  project_dir,
  "references",
  "renal-stone-dag-code-SA-07566.txt"
)
output_dir <- file.path(analysis_dir, "output", "source_aligned_nasa_like")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

source_population <- simulate_source_aligned(
  source_code_path = source_code_path,
  n = 50000L,
  seed = 20260710L,
  nasa_like = TRUE
)
variable_map <- attr(source_population, "variable_map")
attr(source_population, "variable_map") <- NULL

selected_pool <- source_population[
  source_population$selected_into_observed_cohort == 1,
  ,
  drop = FALSE
]
observed <- make_source_aligned_nasa_like_observed(
  source_data = source_population,
  n_observed = 450L,
  seed = 20260712L
)
observed$simulation_version <- "source_aligned_nasa_like_v4"

write.csv(
  observed,
  file.path(output_dir, "renal_stone_source_aligned_nasa_like_v4.csv"),
  row.names = FALSE,
  na = ""
)
write.csv(
  source_aligned_diagnostics(source_population, selected_pool, observed),
  file.path(output_dir, "renal_stone_source_aligned_nasa_like_v4_diagnostics.csv"),
  row.names = FALSE,
  na = ""
)
write.csv(
  variable_map,
  file.path(output_dir, "source_to_simulation_variable_map.csv"),
  row.names = FALSE
)

message("Wrote source-aligned NASA-like simulation to: ", output_dir)
