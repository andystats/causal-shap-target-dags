script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")
analysis_dir <- dirname(script_path)
project_dir <- normalizePath(file.path(analysis_dir, ".."), winslash = "/")

source(file.path(analysis_dir, "R", "dag_concordance.R"))
source(file.path(analysis_dir, "R", "renal_stone_source_aligned_simcausal.R"))
source(file.path(analysis_dir, "R", "shap_distance_metrics.R"))

source_code_path <- file.path(project_dir, "references", "renal-stone-dag-code-SA-07566.txt")
data_path <- file.path(
  analysis_dir,
  "output",
  "source_aligned_clean",
  "renal_stone_source_aligned_clean_v3.csv"
)
output_dir <- file.path(analysis_dir, "output", "shap_nephrolithiasis_clean_v3")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

target_source_node <- "Nephrolithiasis"
target_variable <- "nephrolithiasis"
n_monte_carlo <- 50000L
truth_seed <- 20260713L

clean <- read.csv(data_path, check.names = FALSE)
built <- build_source_aligned_simcausal_dag(source_code_path, nasa_like = FALSE)
ancestor_table <- make_target_ancestor_table(
  source_nodes = built$source_nodes,
  source_edges = built$source_edges,
  variable_map = built$variable_map,
  target_source_node = target_source_node
)

ancestor_table$distribution <- vapply(
  ancestor_table$source_node,
  source_node_distribution,
  character(1)
)
ancestor_table$contrast_low <- vapply(seq_len(nrow(ancestor_table)), function(i) {
  variable <- ancestor_table$variable[[i]]
  if (ancestor_table$distribution[[i]] == "binary") 0 else {
    unname(stats::quantile(clean[[variable]], 0.25, type = 7))
  }
}, numeric(1))
ancestor_table$contrast_high <- vapply(seq_len(nrow(ancestor_table)), function(i) {
  variable <- ancestor_table$variable[[i]]
  if (ancestor_table$distribution[[i]] == "binary") 1 else {
    unname(stats::quantile(clean[[variable]], 0.75, type = 7))
  }
}, numeric(1))
ancestor_table$contrast_definition <- ifelse(
  ancestor_table$distribution == "binary",
  "do(X=1) versus do(X=0)",
  "do(X=observed Q75) versus do(X=observed Q25)"
)

structural_spec <- make_source_structural_spec(built$source_nodes, built$source_edges)
exogenous <- make_common_exogenous_draws(
  node_names = structural_spec$order,
  n = n_monte_carlo,
  seed = truth_seed
)

truth_rows <- lapply(seq_len(nrow(ancestor_table)), function(i) {
  source_node <- ancestor_table$source_node[[i]]
  low <- ancestor_table$contrast_low[[i]]
  high <- ancestor_table$contrast_high[[i]]

  low_data <- simulate_source_structural_crn(
    structural_spec,
    exogenous,
    intervention = stats::setNames(low, source_node)
  )
  high_data <- simulate_source_structural_crn(
    structural_spec,
    exogenous,
    intervention = stats::setNames(high, source_node)
  )
  paired_difference <- high_data[[target_source_node]] - low_data[[target_source_node]]

  data.frame(
    variable = ancestor_table$variable[[i]],
    source_node = source_node,
    distance = ancestor_table$distance[[i]],
    total_effect_risk_difference = mean(paired_difference),
    absolute_total_effect = abs(mean(paired_difference)),
    paired_mc_standard_error = stats::sd(paired_difference) / sqrt(n_monte_carlo),
    risk_high = mean(high_data[[target_source_node]]),
    risk_low = mean(low_data[[target_source_node]]),
    stringsAsFactors = FALSE
  )
})

truth <- do.call(rbind, truth_rows)
truth$normalized_truth <- truth$absolute_total_effect / sum(truth$absolute_total_effect)
truth <- truth[order(-truth$normalized_truth, truth$distance, truth$variable), , drop = FALSE]

metadata <- data.frame(
  target_source_node = target_source_node,
  target_variable = target_variable,
  feature_count = nrow(ancestor_table),
  distance_min = min(ancestor_table$distance),
  distance_max = max(ancestor_table$distance),
  monte_carlo_n = n_monte_carlo,
  common_random_number_seed = truth_seed,
  continuous_contrast = "Observed clean-v3 Q75 versus Q25",
  binary_contrast = "1 versus 0",
  truth_scale = "Absolute total risk difference, normalized across ancestors",
  stringsAsFactors = FALSE
)

stopifnot(
  nrow(ancestor_table) == 28L,
  all(sort(unique(ancestor_table$distance)) == 1:6),
  all(truth$absolute_total_effect >= 0),
  abs(sum(truth$normalized_truth) - 1) < 1e-10
)

write.csv(ancestor_table, file.path(output_dir, "target_ancestor_table.csv"), row.names = FALSE)
write.csv(
  ancestor_table[, c(
    "variable", "source_node", "distribution", "contrast_low", "contrast_high",
    "contrast_definition"
  )],
  file.path(output_dir, "intervention_contrasts.csv"),
  row.names = FALSE
)
write.csv(truth, file.path(output_dir, "interventional_truth.csv"), row.names = FALSE)
write.csv(metadata, file.path(output_dir, "interventional_truth_metadata.csv"), row.names = FALSE)

message("Wrote intervention truth for ", nrow(truth), " ancestors to: ", output_dir)
