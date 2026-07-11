script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")
analysis_dir <- dirname(script_path)
project_dir <- normalizePath(file.path(analysis_dir, ".."), winslash = "/")

source(file.path(analysis_dir, "R", "renal_stone_simcausal.R"))

output_dir <- file.path(analysis_dir, "output", "clean")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

clean <- simulate_renal_stone(
  n = 10000L,
  seed = 20260710L,
  nasa_like = FALSE
)

clean$simulation_version <- "clean_v1"

write.csv(
  clean,
  file.path(output_dir, "renal_stone_clean_v1.csv"),
  row.names = FALSE,
  na = ""
)

write.csv(
  summarize_simulation(clean, "clean_v1"),
  file.path(output_dir, "renal_stone_clean_v1_summary.csv"),
  row.names = FALSE,
  na = ""
)

message("Wrote clean renal-stone simulation to: ", output_dir)
