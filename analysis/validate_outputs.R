local({                                   # locate + source paths.R from anywhere at/below the repo
  dir <- getwd()
  while (!file.exists(file.path(dir, "analysis", "R", "paths.R"))) {
    parent <- dirname(dir)
    if (identical(parent, dir)) stop("Run from inside the repository.")
    dir <- parent
  }
  source(file.path(dir, "analysis", "R", "paths.R"))
})

clean_path <- file.path(
  analysis_dir,
  "output",
  "clean",
  "renal_stone_clean_v1.csv"
)
nasa_like_path <- file.path(
  analysis_dir,
  "output",
  "nasa_like",
  "renal_stone_nasa_like_v2.csv"
)
diagnostics_path <- file.path(
  analysis_dir,
  "output",
  "nasa_like",
  "renal_stone_nasa_like_v2_diagnostics.csv"
)

source_aligned_clean_path <- file.path(
  analysis_dir,
  "output",
  "source_aligned_clean",
  "renal_stone_source_aligned_clean_v3.csv"
)
source_aligned_nasa_like_path <- file.path(
  analysis_dir,
  "output",
  "source_aligned_nasa_like",
  "renal_stone_source_aligned_nasa_like_v4.csv"
)
source_aligned_diagnostics_path <- file.path(
  analysis_dir,
  "output",
  "source_aligned_nasa_like",
  "renal_stone_source_aligned_nasa_like_v4_diagnostics.csv"
)
concordance_path <- file.path(
  analysis_dir,
  "output",
  "dag_validation",
  "dag_concordance_summary.csv"
)
shap_output_dir <- file.path(
  analysis_dir,
  "output",
  "shap_nephrolithiasis_clean_v3"
)
truth_path <- file.path(shap_output_dir, "interventional_truth.csv")
ancestor_path <- file.path(shap_output_dir, "target_ancestor_table.csv")
model_metrics_path <- file.path(shap_output_dir, "model_metrics.csv")
importance_path <- file.path(shap_output_dir, "importance_comparison.csv")
attribution_metrics_path <- file.path(shap_output_dir, "attribution_summary_metrics.csv")
bootstrap_summary_path <- file.path(shap_output_dir, "paired_bootstrap_summary.csv")
efficiency_path <- file.path(shap_output_dir, "shap_efficiency_checks.csv")
distance_plot_path <- file.path(shap_output_dir, "distance_concentration_curves.png")
predictive_signal_path <- file.path(shap_output_dir, "predictive_signal_diagnostics.csv")
predictive_signal_plot_path <- file.path(shap_output_dir, "predictive_signal_diagnostics.png")
structural_importance_path <- file.path(shap_output_dir, "structural_causal_shap_importance.csv")
structural_values_path <- file.path(shap_output_dir, "structural_causal_shap_values.csv")
structural_summary_path <- file.path(shap_output_dir, "structural_causal_shap_summary.json")

stopifnot(
  file.exists(clean_path),
  file.exists(nasa_like_path),
  file.exists(diagnostics_path),
  file.exists(source_aligned_clean_path),
  file.exists(source_aligned_nasa_like_path),
  file.exists(source_aligned_diagnostics_path),
  file.exists(concordance_path),
  file.exists(truth_path),
  file.exists(ancestor_path),
  file.exists(model_metrics_path),
  file.exists(importance_path),
  file.exists(attribution_metrics_path),
  file.exists(bootstrap_summary_path),
  file.exists(efficiency_path),
  file.exists(distance_plot_path),
  file.exists(predictive_signal_path),
  file.exists(predictive_signal_plot_path),
  file.exists(structural_importance_path),
  file.exists(structural_values_path),
  file.exists(structural_summary_path)
)

clean <- read.csv(clean_path, na.strings = "")
nasa_like <- read.csv(nasa_like_path, na.strings = "")
diagnostics <- read.csv(diagnostics_path, na.strings = "")
source_aligned_clean <- read.csv(source_aligned_clean_path, na.strings = "")
source_aligned_nasa_like <- read.csv(source_aligned_nasa_like_path, na.strings = "")
source_aligned_diagnostics <- read.csv(source_aligned_diagnostics_path, na.strings = "")
concordance <- read.csv(concordance_path, na.strings = "")
truth <- read.csv(truth_path, na.strings = "")
ancestors <- read.csv(ancestor_path, na.strings = "")
model_metrics <- read.csv(model_metrics_path, na.strings = "")
importance <- read.csv(importance_path, na.strings = "")
attribution_metrics <- read.csv(attribution_metrics_path, na.strings = "")
bootstrap_summary <- read.csv(bootstrap_summary_path, na.strings = "")
efficiency <- read.csv(efficiency_path, na.strings = "")
predictive_signal <- read.csv(predictive_signal_path, na.strings = "")
structural_importance <- read.csv(structural_importance_path, na.strings = "")
structural_values <- read.csv(structural_values_path, na.strings = "")

node_path <- file.path(
  analysis_dir,
  "..",
  "dag-candidates",
  "renal-stone-core-nodes.csv"
)
edge_path <- file.path(
  analysis_dir,
  "..",
  "dag-candidates",
  "renal-stone-core-edges.csv"
)

stopifnot(file.exists(node_path), file.exists(edge_path))

node_metadata <- read.csv(node_path, check.names = FALSE)
edges <- read.csv(edge_path, check.names = FALSE)

stopifnot(
  !anyDuplicated(node_metadata$node),
  !anyDuplicated(edges[c("from", "to")]),
  all(edges$from %in% node_metadata$node),
  all(edges$to %in% node_metadata$node)
)

# Kahn's algorithm: fail if the implemented edge manifest is cyclic.
graph_nodes <- unique(c(edges$from, edges$to))
remaining <- edges[c("from", "to")]
resolved <- character()

while (length(resolved) < length(graph_nodes)) {
  unresolved <- setdiff(graph_nodes, resolved)
  incoming_targets <- remaining$to[remaining$from %in% unresolved]
  roots <- setdiff(unresolved, incoming_targets)

  if (length(roots) == 0L) {
    stop("renal-stone-core-edges.csv contains a directed cycle")
  }

  resolved <- c(resolved, roots)
  remaining <- remaining[!remaining$from %in% roots, , drop = FALSE]
}

required_columns <- c(
  "altered_gravity",
  "bone_remodeling",
  "hydration",
  "urine_concentration",
  "urine_chemistry_risk",
  "mineralized_renal_material",
  "nephrolithiasis",
  "ureterolithiasis",
  "impaired_urine_flow",
  "medical_illness",
  "loss_mission_objectives"
)

stopifnot(
  nrow(clean) == 10000L,
  nrow(nasa_like) == 450L,
  all(required_columns %in% names(clean)),
  all(required_columns %in% names(nasa_like)),
  all(nasa_like$selected_astronaut == 1),
  anyNA(nasa_like$urine_chemistry_risk),
  anyNA(nasa_like$mineralized_renal_material),
  mean(clean$nephrolithiasis) > 0.01,
  mean(clean$nephrolithiasis) < 0.30,
  mean(clean$medical_illness) > 0.01,
  mean(clean$loss_mission_objectives) < 0.20,
  nrow(diagnostics) > 0,
  length(resolved) == length(graph_nodes),
  nrow(source_aligned_clean) == 10000L,
  nrow(source_aligned_nasa_like) == 450L,
  all(source_aligned_nasa_like$selected_into_observed_cohort == 1),
  anyNA(source_aligned_nasa_like$urine_chemistry),
  anyNA(source_aligned_nasa_like$mineralized_renal_material),
  mean(source_aligned_clean$nephrolithiasis) > 0.01,
  mean(source_aligned_clean$nephrolithiasis) < 0.30,
  nrow(source_aligned_diagnostics) > 0,
  all(concordance$exact_match[seq_len(5)]),
  all(concordance$cohen_kappa[seq_len(5)] == 1),
  all(concordance$false_positive[seq_len(5)] == 0),
  all(concordance$false_negative[seq_len(5)] == 0),
  nrow(truth) == 28L,
  nrow(ancestors) == 28L,
  all(table(ancestors$distance) == c(2, 2, 7, 9, 6, 2)),
  abs(sum(truth$normalized_truth) - 1) < 1e-10,
  nrow(model_metrics) == 1L,
  model_metrics$auc > 0.5,
  model_metrics$auc <= 1,
  nrow(importance) == 112L,
  all(abs(tapply(importance$normalized_importance, importance$method, sum) - 1) < 1e-10),
  nrow(attribution_metrics) == 3L,
  all(is.finite(attribution_metrics$pbi)),
  all(is.finite(attribution_metrics$poa)),
  nrow(bootstrap_summary) == 7L,
  all(bootstrap_summary$probability_favors_causal >= 0),
  all(bootstrap_summary$probability_favors_causal <= 1),
  max(efficiency$max_absolute_efficiency_error) < 1e-4,
  nrow(predictive_signal) == 5L,
  predictive_signal$auc[predictive_signal$learner == "Oracle structural P(Y|direct parents)"] < 0.75,
  predictive_signal$auc[predictive_signal$learner == "XGBoost: all 28 ancestors"] > 0.65,
  nrow(structural_importance) == 28L,
  abs(sum(structural_importance$normalized_importance) - 1) < 1e-10,
  nrow(structural_values) == 32L,
  ncol(structural_values) == 29L
)

cat("Validation passed.\n")
cat("Clean rows:", nrow(clean), "\n")
cat("NASA-like rows:", nrow(nasa_like), "\n")
cat("Clean nephrolithiasis prevalence:", round(mean(clean$nephrolithiasis), 4), "\n")
cat(
  "NASA-like nephrolithiasis prevalence:",
  round(mean(nasa_like$nephrolithiasis), 4),
  "\n"
)
cat(
  "NASA-like urine chemistry missingness:",
  round(mean(is.na(nasa_like$urine_chemistry_risk)), 4),
  "\n"
)
cat("Source-aligned clean rows:", nrow(source_aligned_clean), "\n")
cat("Source-aligned NASA-like rows:", nrow(source_aligned_nasa_like), "\n")
cat(
  "Source-aligned clean nephrolithiasis prevalence:",
  round(mean(source_aligned_clean$nephrolithiasis), 4),
  "\n"
)
cat(
  "Source-aligned NASA-like urine chemistry missingness:",
  round(mean(is.na(source_aligned_nasa_like$urine_chemistry)), 4),
  "\n"
)
cat("Source-aligned graph kappa:", concordance$cohen_kappa[[4]], "\n")
cat("Nephrolithiasis truth features:", nrow(truth), "\n")
cat("Held-out XGBoost AUC:", round(model_metrics$auc, 4), "\n")
cat(
  "TreeSHAP / matched ordinary / DAG-asymmetric PBI:",
  paste(round(attribution_metrics$pbi, 4), collapse = " / "),
  "\n"
)
cat(
  "Oracle / XGBoost AUC:",
  round(predictive_signal$auc[[1]], 4),
  "/",
  round(predictive_signal$auc[[5]], 4),
  "\n"
)
cat(
  "Structural prototype top-5 variables:",
  paste(
    structural_importance$variable[
      order(structural_importance$rank)[seq_len(5)]
    ],
    collapse = ", "
  ),
  "\n"
)
