# Self-locate the project + analysis directories without relying on Rscript's
# `--file=` argument. Works under `Rscript script.R`, an interactive `source()`,
# and RStudio -- unlike the old commandArgs("^--file=") header, which crashed
# with "subscript out of bounds" whenever a script was sourced instead of run.
#
# Sourcing this file defines `project_dir` and `analysis_dir` in the caller.

.find_project_dir <- function(start = getwd()) {
  dir <- normalizePath(start, winslash = "/", mustWork = FALSE)
  marker <- file.path("references", "renal-stone-dag-code-SA-07566.txt")
  repeat {
    if (file.exists(file.path(dir, marker)) || dir.exists(file.path(dir, ".git"))) {
      return(dir)
    }
    parent <- dirname(dir)
    if (identical(parent, dir)) break
    dir <- parent
  }
  stop("Could not locate the project root -- run from inside the repository.")
}

project_dir <- .find_project_dir()
analysis_dir <- file.path(project_dir, "analysis")
