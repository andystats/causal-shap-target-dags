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
output_dir <- file.path(analysis_dir, "output", "source_aligned_clean")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data <- simulate_source_aligned(
  source_code_path = source_code_path,
  n = 10000L,
  seed = 20260710L,
  nasa_like = FALSE
)
variable_map <- attr(data, "variable_map")
attr(data, "variable_map") <- NULL
data$simulation_version <- "source_aligned_clean_v3"

write.csv(
  data,
  file.path(output_dir, "renal_stone_source_aligned_clean_v3.csv"),
  row.names = FALSE,
  na = ""
)
write.csv(
  variable_map,
  file.path(output_dir, "source_to_simulation_variable_map.csv"),
  row.names = FALSE
)

message("Wrote source-aligned clean simulation to: ", output_dir)
