required <- c(
  "dagitty",
  "ggdag",
  "ggplot2",
  "igraph",
  "ranger",
  "simcausal",
  "xgboost"
)

missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

message("R analysis dependencies are available.")
