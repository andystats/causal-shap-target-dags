script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")
analysis_dir <- dirname(script_path)

source(file.path(analysis_dir, "R", "shap_distance_metrics.R"))

output_dir <- file.path(analysis_dir, "output", "shap_nephrolithiasis_clean_v3")
truth <- read.csv(file.path(output_dir, "interventional_truth.csv"), check.names = FALSE)
ancestors <- read.csv(file.path(output_dir, "target_ancestor_table.csv"), check.names = FALSE)
ordinary <- read.csv(
  file.path(output_dir, "ordinary_interventional_shap_values.csv"),
  check.names = FALSE
)
causal <- read.csv(file.path(output_dir, "dag_asymmetric_shap_values.csv"), check.names = FALSE)

features <- setdiff(names(ordinary), "source_row")
stopifnot(identical(features, setdiff(names(causal), "source_row")))
ordinary <- as.matrix(ordinary[, features, drop = FALSE])
causal <- as.matrix(causal[, features, drop = FALSE])

truth_values <- truth$absolute_total_effect[match(features, truth$variable)]
truth_values <- truth_values / sum(truth_values)
distances <- ancestors$distance[match(features, ancestors$variable)]
truth_mean_distance <- sum(truth_values * distances)
top_truth <- order(truth_values, decreasing = TRUE)[seq_len(5L)]
max_distance <- max(distances)
truth_curve <- vapply(
  seq_len(max_distance),
  function(k) sum(truth_values[distances <= k]),
  numeric(1)
)

summarize_one <- function(values) {
  importance <- colMeans(abs(values))
  importance <- importance / sum(importance)
  curve <- vapply(
    seq_len(max_distance),
    function(k) sum(importance[distances <= k]),
    numeric(1)
  )
  data.frame(
    kendall_tau = suppressWarnings(stats::cor(importance, truth_values, method = "kendall")),
    spearman_rho = suppressWarnings(stats::cor(importance, truth_values, method = "spearman")),
    top5_recovery = length(intersect(order(importance, decreasing = TRUE)[seq_len(5L)], top_truth)) / 5,
    ndcg_at_5 = normalized_discounted_cumulative_gain(truth_values, importance, 5L),
    pbi = truth_mean_distance - sum(importance * distances),
    poa = mean((curve - truth_curve)[seq_len(max_distance - 1L)]),
    proximal_mass_distance_le_2 = sum(importance[distances <= 2]),
    stringsAsFactors = FALSE
  )
}

n_bootstrap <- 2000L
bootstrap_seed <- 20260718L
set.seed(bootstrap_seed)

draws <- lapply(seq_len(n_bootstrap), function(iteration) {
  indices <- sample.int(nrow(ordinary), nrow(ordinary), replace = TRUE)
  ordinary_metrics <- summarize_one(ordinary[indices, , drop = FALSE])
  causal_metrics <- summarize_one(causal[indices, , drop = FALSE])

  data.frame(
    iteration = iteration,
    ordinary_kendall_tau = ordinary_metrics$kendall_tau,
    causal_kendall_tau = causal_metrics$kendall_tau,
    delta_kendall_tau = causal_metrics$kendall_tau - ordinary_metrics$kendall_tau,
    ordinary_spearman_rho = ordinary_metrics$spearman_rho,
    causal_spearman_rho = causal_metrics$spearman_rho,
    delta_spearman_rho = causal_metrics$spearman_rho - ordinary_metrics$spearman_rho,
    ordinary_top5_recovery = ordinary_metrics$top5_recovery,
    causal_top5_recovery = causal_metrics$top5_recovery,
    delta_top5_recovery = causal_metrics$top5_recovery - ordinary_metrics$top5_recovery,
    ordinary_ndcg_at_5 = ordinary_metrics$ndcg_at_5,
    causal_ndcg_at_5 = causal_metrics$ndcg_at_5,
    delta_ndcg_at_5 = causal_metrics$ndcg_at_5 - ordinary_metrics$ndcg_at_5,
    ordinary_pbi = ordinary_metrics$pbi,
    causal_pbi = causal_metrics$pbi,
    improvement_absolute_pbi = abs(ordinary_metrics$pbi) - abs(causal_metrics$pbi),
    ordinary_poa = ordinary_metrics$poa,
    causal_poa = causal_metrics$poa,
    improvement_absolute_poa = abs(ordinary_metrics$poa) - abs(causal_metrics$poa),
    ordinary_proximal_mass = ordinary_metrics$proximal_mass_distance_le_2,
    causal_proximal_mass = causal_metrics$proximal_mass_distance_le_2,
    reduction_proximal_mass =
      ordinary_metrics$proximal_mass_distance_le_2 -
      causal_metrics$proximal_mass_distance_le_2,
    stringsAsFactors = FALSE
  )
})
draws <- do.call(rbind, draws)

summary_columns <- c(
  "delta_kendall_tau",
  "delta_spearman_rho",
  "delta_top5_recovery",
  "delta_ndcg_at_5",
  "improvement_absolute_pbi",
  "improvement_absolute_poa",
  "reduction_proximal_mass"
)
bootstrap_summary <- do.call(rbind, lapply(summary_columns, function(metric) {
  values <- draws[[metric]]
  data.frame(
    metric = metric,
    bootstrap_mean = mean(values),
    percentile_2_5 = unname(stats::quantile(values, 0.025, type = 7)),
    percentile_97_5 = unname(stats::quantile(values, 0.975, type = 7)),
    probability_favors_causal = mean(values > 0),
    zero_fraction = mean(values == 0),
    stringsAsFactors = FALSE
  )
}))

write.csv(draws, file.path(output_dir, "paired_bootstrap_draws.csv"), row.names = FALSE)
write.csv(
  bootstrap_summary,
  file.path(output_dir, "paired_bootstrap_summary.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    bootstrap_replicates = n_bootstrap,
    bootstrap_seed = bootstrap_seed,
    resampling_unit = "evaluation record, paired across methods",
    comparison = "Matched-background unrestricted ordinary SHAP versus DAG-constrained asymmetric SHAP",
    interpretation = "Positive deltas/improvements favor DAG-constrained asymmetric SHAP",
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "paired_bootstrap_metadata.csv"),
  row.names = FALSE
)

stopifnot(
  nrow(draws) == n_bootstrap,
  all(is.finite(as.matrix(draws[, setdiff(names(draws), "iteration")]))),
  all(bootstrap_summary$probability_favors_causal >= 0 &
    bootstrap_summary$probability_favors_causal <= 1)
)

print(bootstrap_summary)
message("Wrote paired bootstrap results to: ", output_dir)
