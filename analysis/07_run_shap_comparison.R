script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")
analysis_dir <- dirname(script_path)
project_dir <- normalizePath(file.path(analysis_dir, ".."), winslash = "/")

suppressPackageStartupMessages({
  library(ggplot2)
  library(igraph)
  library(xgboost)
})

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
split_seed <- 20260714L
model_seed <- 20260715L
evaluation_seed <- 20260716L
causal_shap_seed <- 20260717L
ordinary_shap_seed <- 20260719L
n_evaluation <- 64L
n_background <- 128L
n_causal_permutations <- 128L

clean <- read.csv(data_path, check.names = FALSE)
truth <- read.csv(file.path(output_dir, "interventional_truth.csv"), check.names = FALSE)
ancestor_table <- read.csv(file.path(output_dir, "target_ancestor_table.csv"), check.names = FALSE)
built <- build_source_aligned_simcausal_dag(source_code_path, nasa_like = FALSE)

features <- ancestor_table$variable
features <- features[match(truth$variable, features, nomatch = 0L) > 0L]
features <- sort(features)
stopifnot(length(features) == 28L, all(features %in% names(clean)))

stratified_split <- function(y, seed) {
  set.seed(seed)
  assignments <- rep(NA_character_, length(y))
  for (class_value in sort(unique(y))) {
    indices <- sample(which(y == class_value))
    n <- length(indices)
    n_train <- floor(0.60 * n)
    n_validation <- floor(0.20 * n)
    assignments[indices[seq_len(n_train)]] <- "train"
    assignments[indices[n_train + seq_len(n_validation)]] <- "validation"
    assignments[indices[(n_train + n_validation + 1L):n]] <- "test"
  }
  assignments
}

split <- stratified_split(clean[[target_variable]], split_seed)
x <- as.matrix(clean[, features, drop = FALSE])
y <- clean[[target_variable]]

dtrain <- xgb.DMatrix(x[split == "train", , drop = FALSE], label = y[split == "train"])
dvalidation <- xgb.DMatrix(
  x[split == "validation", , drop = FALSE],
  label = y[split == "validation"]
)

set.seed(model_seed)
model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 3L,
    min_child_weight = 5,
    eta = 0.04,
    subsample = 0.85,
    colsample_bytree = 0.85,
    lambda = 1,
    alpha = 0,
    seed = model_seed,
    nthread = 4L
  ),
  data = dtrain,
  nrounds = 600L,
  evals = list(validation = dvalidation),
  early_stopping_rounds = 40L,
  verbose = 0
)

test_indices <- which(split == "test")
x_test <- x[test_indices, , drop = FALSE]
y_test <- y[test_indices]
test_probability <- predict(model, x_test)
test_probability_clipped <- pmin(pmax(test_probability, 1e-12), 1 - 1e-12)

model_metrics <- data.frame(
  dataset = "held_out_test",
  n = length(y_test),
  events = sum(y_test),
  event_fraction = mean(y_test),
  auc = binary_auc(y_test, test_probability),
  brier_score = mean((test_probability - y_test)^2),
  log_loss = -mean(
    y_test * log(test_probability_clipped) +
      (1 - y_test) * log(1 - test_probability_clipped)
  ),
  best_iteration = attributes(model)$early_stop$best_iteration,
  feature_count = length(features),
  stringsAsFactors = FALSE
)

set.seed(evaluation_seed)
positive_test <- test_indices[y[test_indices] == 1]
negative_test <- test_indices[y[test_indices] == 0]
n_positive_eval <- min(length(positive_test), max(8L, round(n_evaluation * mean(y[test_indices]))))
evaluation_indices <- c(
  sample(positive_test, n_positive_eval),
  sample(negative_test, n_evaluation - n_positive_eval)
)
evaluation_indices <- sample(evaluation_indices)
background_indices <- sample(which(split == "train"), n_background)

evaluation <- x[evaluation_indices, , drop = FALSE]
background <- x[background_indices, , drop = FALSE]
rownames(evaluation) <- as.character(evaluation_indices)

ordinary_raw <- predict(model, evaluation, predcontrib = TRUE, approxcontrib = FALSE)
ordinary_shap <- ordinary_raw[, features, drop = FALSE]

feature_source_nodes <- built$variable_map$source_node[
  match(features, built$variable_map$variable)
]
feature_source_edges <- built$source_edges[
  built$source_edges$from %in% feature_source_nodes &
    built$source_edges$to %in% feature_source_nodes,
  ,
  drop = FALSE
]
feature_edges <- data.frame(
  from = built$variable_map$variable[match(feature_source_edges$from, built$variable_map$source_node)],
  to = built$variable_map$variable[match(feature_source_edges$to, built$variable_map$source_node)],
  stringsAsFactors = FALSE
)

causal_shap <- compute_asymmetric_interventional_shap(
  model = model,
  evaluation = evaluation,
  background = background,
  feature_edges = feature_edges,
  n_permutations = n_causal_permutations,
  seed = causal_shap_seed
)

ordinary_interventional_shap <- compute_asymmetric_interventional_shap(
  model = model,
  evaluation = evaluation,
  background = background,
  feature_edges = data.frame(from = character(), to = character()),
  n_permutations = n_causal_permutations,
  seed = ordinary_shap_seed
)

ordinary_importance <- colMeans(abs(ordinary_shap))
ordinary_interventional_importance <- colMeans(abs(ordinary_interventional_shap))
causal_importance <- colMeans(abs(causal_shap))
truth_importance <- truth$absolute_total_effect[match(features, truth$variable)]
distance <- ancestor_table$distance[match(features, ancestor_table$variable)]
source_node <- ancestor_table$source_node[match(features, ancestor_table$variable)]
structural_role <- ancestor_table$structural_role[match(features, ancestor_table$variable)]

importance_table <- rbind(
  data.frame(
    method = "Interventional truth",
    variable = features,
    source_node = source_node,
    distance = distance,
    structural_role = structural_role,
    raw_importance = truth_importance,
    stringsAsFactors = FALSE
  ),
  data.frame(
    method = "Ordinary TreeSHAP",
    variable = features,
    source_node = source_node,
    distance = distance,
    structural_role = structural_role,
    raw_importance = ordinary_importance,
    stringsAsFactors = FALSE
  ),
  data.frame(
    method = "Ordinary interventional SHAP",
    variable = features,
    source_node = source_node,
    distance = distance,
    structural_role = structural_role,
    raw_importance = ordinary_interventional_importance,
    stringsAsFactors = FALSE
  ),
  data.frame(
    method = "DAG-constrained asymmetric SHAP",
    variable = features,
    source_node = source_node,
    distance = distance,
    structural_role = structural_role,
    raw_importance = causal_importance,
    stringsAsFactors = FALSE
  )
)
importance_table$normalized_importance <- ave(
  importance_table$raw_importance,
  importance_table$method,
  FUN = function(value) value / sum(value)
)
importance_table$rank <- ave(
  -importance_table$normalized_importance,
  importance_table$method,
  FUN = function(value) rank(value, ties.method = "average")
)

summary_metrics <- rbind(
  summarize_attribution_method(importance_table, "Ordinary TreeSHAP"),
  summarize_attribution_method(importance_table, "Ordinary interventional SHAP"),
  summarize_attribution_method(importance_table, "DAG-constrained asymmetric SHAP")
)
truth_mean_distance <- sum(
  importance_table$normalized_importance[
    importance_table$method == "Interventional truth"
  ] * importance_table$distance[importance_table$method == "Interventional truth"]
)
summary_metrics$truth_mean_distance <- truth_mean_distance

concentration <- make_distance_concentration(importance_table)
truth_curve <- concentration[
  concentration$method == "Interventional truth",
  c("hop_radius", "cumulative_importance"),
  drop = FALSE
]
names(truth_curve)[2] <- "truth_cumulative"
summary_metrics$poa <- NA_real_
for (i in seq_len(nrow(summary_metrics))) {
  method_curve <- concentration[concentration$method == summary_metrics$method[[i]], , drop = FALSE]
  aligned <- merge(method_curve, truth_curve, by = "hop_radius")
  summary_metrics$poa[i] <- mean(
    (aligned$cumulative_importance - aligned$truth_cumulative)[
      aligned$hop_radius < max(aligned$hop_radius)
    ]
  )
}
distance_calibration <- make_distance_calibration(importance_table)

ordinary_bias <- ordinary_raw[, ncol(ordinary_raw)]
ordinary_margin <- rowSums(ordinary_shap) + ordinary_bias
expected_margin <- predict(model, evaluation, outputmargin = TRUE)
causal_margin <- rowSums(causal_shap) + attr(causal_shap, "baseline_margin")
ordinary_interventional_margin <-
  rowSums(ordinary_interventional_shap) +
  attr(ordinary_interventional_shap, "baseline_margin")
efficiency <- data.frame(
  method = c(
    "Ordinary TreeSHAP",
    "Ordinary interventional SHAP",
    "DAG-constrained asymmetric SHAP"
  ),
  max_absolute_efficiency_error = c(
    max(abs(ordinary_margin - expected_margin)),
    max(abs(ordinary_interventional_margin - expected_margin)),
    max(abs(causal_margin - expected_margin))
  ),
  stringsAsFactors = FALSE
)

evaluation_manifest <- data.frame(
  source_row = evaluation_indices,
  outcome = y[evaluation_indices],
  predicted_probability = predict(model, evaluation),
  stringsAsFactors = FALSE
)
background_manifest <- data.frame(source_row = background_indices)

ordinary_output <- data.frame(source_row = evaluation_indices, ordinary_shap, check.names = FALSE)
causal_output <- data.frame(source_row = evaluation_indices, causal_shap, check.names = FALSE)
ordinary_interventional_output <- data.frame(
  source_row = evaluation_indices,
  ordinary_interventional_shap,
  check.names = FALSE
)

invisible(xgb.save(model, file.path(output_dir, "nephrolithiasis_xgboost_clean_v3.ubj")))
write.csv(model_metrics, file.path(output_dir, "model_metrics.csv"), row.names = FALSE)
write.csv(evaluation_manifest, file.path(output_dir, "evaluation_manifest.csv"), row.names = FALSE)
write.csv(background_manifest, file.path(output_dir, "background_manifest.csv"), row.names = FALSE)
write.csv(ordinary_output, file.path(output_dir, "ordinary_treeshap_values.csv"), row.names = FALSE)
write.csv(
  ordinary_interventional_output,
  file.path(output_dir, "ordinary_interventional_shap_values.csv"),
  row.names = FALSE
)
write.csv(causal_output, file.path(output_dir, "dag_asymmetric_shap_values.csv"), row.names = FALSE)
write.csv(importance_table, file.path(output_dir, "importance_comparison.csv"), row.names = FALSE)
write.csv(summary_metrics, file.path(output_dir, "attribution_summary_metrics.csv"), row.names = FALSE)
write.csv(concentration, file.path(output_dir, "distance_concentration_curves.csv"), row.names = FALSE)
write.csv(distance_calibration, file.path(output_dir, "distance_calibration.csv"), row.names = FALSE)
write.csv(efficiency, file.path(output_dir, "shap_efficiency_checks.csv"), row.names = FALSE)
write.csv(
  data.frame(
    split_seed = split_seed,
    model_seed = model_seed,
    evaluation_seed = evaluation_seed,
    causal_shap_seed = causal_shap_seed,
    ordinary_shap_seed = ordinary_shap_seed,
    n_evaluation = n_evaluation,
    n_background = n_background,
    n_causal_permutations = n_causal_permutations,
    model_output_scale = "raw log-odds margin for both SHAP methods",
    ordinary_method = paste(
      "Exact XGBoost TreeSHAP plus unrestricted-permutation interventional SHAP;",
      "the latter uses the same fixed empirical background as DAG-asymmetric SHAP"
    ),
    causal_method = paste(
      "Asymmetric interventional Shapley values averaged over random",
      "DAG-consistent topological orders with a fixed empirical background"
    ),
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "shap_run_metadata.csv"),
  row.names = FALSE
)

method_levels <- c(
  "Interventional truth",
  "Ordinary TreeSHAP",
  "Ordinary interventional SHAP",
  "DAG-constrained asymmetric SHAP"
)
importance_table$method <- factor(importance_table$method, levels = method_levels)
concentration$method <- factor(concentration$method, levels = method_levels)

truth_curve_plot <- concentration[
  concentration$method == "Interventional truth",
  c("hop_radius", "cumulative_importance"),
  drop = FALSE
]
ordinary_curve_plot <- concentration[
  concentration$method == "Ordinary interventional SHAP",
  c("hop_radius", "cumulative_importance"),
  drop = FALSE
]
curve_ribbon <- merge(
  truth_curve_plot,
  ordinary_curve_plot,
  by = "hop_radius",
  suffixes = c("_truth", "_ordinary")
)

concentration_plot <- ggplot(
  concentration,
  aes(x = hop_radius, y = cumulative_importance, color = method, linetype = method)
) +
  geom_ribbon(
    data = curve_ribbon,
    aes(
      x = hop_radius,
      ymin = pmin(cumulative_importance_truth, cumulative_importance_ordinary),
      ymax = pmax(cumulative_importance_truth, cumulative_importance_ordinary)
    ),
    inherit.aes = FALSE,
    fill = "#d97706",
    alpha = 0.10
  ) +
  geom_line(linewidth = 1.15) +
  geom_point(size = 2.2) +
  annotate(
    "text",
    x = 3.1,
    y = 0.53,
    hjust = 0,
    size = 3.3,
    label = sprintf(
      "PBI: matched ordinary %.3f; DAG-asymmetric %.3f\nPOA: matched ordinary %.3f; DAG-asymmetric %.3f",
      summary_metrics$pbi[summary_metrics$method == "Ordinary interventional SHAP"],
      summary_metrics$pbi[summary_metrics$method == "DAG-constrained asymmetric SHAP"],
      summary_metrics$poa[summary_metrics$method == "Ordinary interventional SHAP"],
      summary_metrics$poa[summary_metrics$method == "DAG-constrained asymmetric SHAP"]
    )
  ) +
  scale_x_continuous(breaks = sort(unique(concentration$hop_radius))) +
  scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"), limits = c(0, 1)) +
  scale_color_manual(values = c("#111827", "#6b7280", "#d97706", "#2563eb")) +
  scale_linetype_manual(values = c("solid", "dotted", "dashed", "dotdash")) +
  labs(
    title = "Attribution mass concentrated near Nephrolithiasis",
    subtitle = "Clean source-aligned v3; identical model, evaluation records, and feature set",
    x = "Directed hop radius from target",
    y = "Cumulative normalized importance within radius",
    color = NULL,
    linetype = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"))

top_features <- unique(
  unlist(lapply(method_levels, function(method) {
    rows <- importance_table[importance_table$method == method, , drop = FALSE]
    head(rows$variable[order(rows$normalized_importance, decreasing = TRUE)], 8L)
  }))
)
top_importance <- importance_table[importance_table$variable %in% top_features, , drop = FALSE]
top_importance$variable <- factor(
  top_importance$variable,
  levels = rev(unique(
    top_importance$variable[order(
      top_importance$normalized_importance[top_importance$method == "Interventional truth"],
      decreasing = TRUE
    )]
  ))
)

importance_plot <- ggplot(
  top_importance,
  aes(x = normalized_importance, y = variable, fill = method)
) +
  geom_col(position = position_dodge(width = 0.8), width = 0.72) +
  scale_x_continuous(labels = function(x) paste0(round(100 * x), "%")) +
  scale_fill_manual(values = c("#111827", "#6b7280", "#d97706", "#2563eb")) +
  labs(
    title = "Feature-level recovery of interventional truth",
    subtitle = "Union of the eight highest-ranked features from each method",
    x = "Normalized global importance",
    y = NULL,
    fill = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"))

nodes <- built$source_nodes
nodes <- nodes[nodes$node %in% c(target_source_node, source_node), , drop = FALSE]
ancestor_edges <- built$source_edges[
  built$source_edges$from %in% nodes$node & built$source_edges$to %in% nodes$node,
  ,
  drop = FALSE
]
edge_coordinates <- merge(
  ancestor_edges,
  nodes[, c("node", "x", "y")],
  by.x = "from",
  by.y = "node"
)
names(edge_coordinates)[names(edge_coordinates) %in% c("x", "y")] <- c("x_from", "y_from")
edge_coordinates <- merge(
  edge_coordinates,
  nodes[, c("node", "x", "y")],
  by.x = "to",
  by.y = "node"
)
names(edge_coordinates)[names(edge_coordinates) %in% c("x", "y")] <- c("x_to", "y_to")

map_nodes <- merge(
  importance_table,
  nodes[, c("node", "x", "y")],
  by.x = "source_node",
  by.y = "node"
)
target_rows <- data.frame(
  source_node = target_source_node,
  method = factor(method_levels, levels = method_levels),
  variable = target_variable,
  distance = 0,
  structural_role = "target",
  raw_importance = 0,
  normalized_importance = 0.012,
  rank = NA_real_,
  x = nodes$x[nodes$node == target_source_node],
  y = nodes$y[nodes$node == target_source_node],
  stringsAsFactors = FALSE
)
map_nodes <- rbind(map_nodes, target_rows)
map_nodes$method <- factor(map_nodes$method, levels = method_levels)
map_nodes$node_label <- ifelse(
  map_nodes$structural_role == "target" | (!is.na(map_nodes$rank) & map_nodes$rank <= 8),
  map_nodes$source_node,
  ""
)
map_edges <- do.call(rbind, lapply(method_levels, function(method) {
  transform(edge_coordinates, method = factor(method, levels = method_levels))
}))

ancestor_map_plot <- ggplot() +
  geom_segment(
    data = map_edges,
    aes(x = x_from, y = y_from, xend = x_to, yend = y_to),
    color = "grey72",
    linewidth = 0.35,
    arrow = grid::arrow(length = grid::unit(0.06, "inches"), type = "closed")
  ) +
  geom_point(
    data = map_nodes,
    aes(x = x, y = y, size = normalized_importance, color = factor(distance)),
    alpha = 0.88
  ) +
  geom_text(
    data = map_nodes,
    aes(x = x, y = y, label = node_label),
    size = 2.25,
    color = "#111827",
    vjust = -1.0,
    check_overlap = FALSE
  ) +
  facet_wrap(
    ~method,
    nrow = 1,
    labeller = as_labeller(c(
      "Interventional truth" = "Interventional truth",
      "Ordinary TreeSHAP" = "Ordinary TreeSHAP",
      "Ordinary interventional SHAP" = "Matched ordinary SHAP",
      "DAG-constrained asymmetric SHAP" = "DAG-asymmetric SHAP"
    ))
  ) +
  scale_size_area(max_size = 11, breaks = c(0.01, 0.05, 0.10, 0.20)) +
  scale_color_viridis_d(option = "D", end = 0.9) +
  coord_equal(clip = "off") +
  labs(
    title = "Where each method places importance in the ancestor DAG",
    subtitle = "Node area is normalized importance; color is directed distance to Nephrolithiasis",
    size = "Importance",
    color = "Hops"
  ) +
  theme_void(base_size = 10) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA),
    legend.key = element_rect(fill = "white", color = NA),
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold"),
    plot.margin = margin(10, 20, 10, 20)
  )

ggsave(
  file.path(output_dir, "distance_concentration_curves.png"),
  concentration_plot,
  width = 9,
  height = 6,
  dpi = 300,
  bg = "white"
)
ggsave(
  file.path(output_dir, "distance_concentration_curves.pdf"),
  concentration_plot,
  width = 9,
  height = 6,
  bg = "white"
)
ggsave(
  file.path(output_dir, "feature_importance_comparison.png"),
  importance_plot,
  width = 10,
  height = 7.5,
  dpi = 300,
  bg = "white"
)
ggsave(
  file.path(output_dir, "feature_importance_comparison.pdf"),
  importance_plot,
  width = 10,
  height = 7.5,
  bg = "white"
)
ggsave(
  file.path(output_dir, "ancestor_importance_maps.png"),
  ancestor_map_plot,
  width = 22,
  height = 7.5,
  dpi = 300,
  bg = "white"
)
ggsave(
  file.path(output_dir, "ancestor_importance_maps.pdf"),
  ancestor_map_plot,
  width = 22,
  height = 7.5,
  bg = "white"
)

stopifnot(
  all(is.finite(summary_metrics$pbi)),
  all(is.finite(summary_metrics$poa)),
  max(efficiency$max_absolute_efficiency_error) < 1e-4,
  abs(sum(importance_table$normalized_importance[importance_table$method == "Interventional truth"]) - 1) < 1e-10,
  abs(sum(importance_table$normalized_importance[importance_table$method == "Ordinary TreeSHAP"]) - 1) < 1e-10,
  abs(sum(importance_table$normalized_importance[importance_table$method == "Ordinary interventional SHAP"]) - 1) < 1e-10,
  abs(sum(importance_table$normalized_importance[importance_table$method == "DAG-constrained asymmetric SHAP"]) - 1) < 1e-10
)

print(model_metrics)
print(summary_metrics)
print(efficiency)
message("Wrote SHAP comparison to: ", output_dir)
