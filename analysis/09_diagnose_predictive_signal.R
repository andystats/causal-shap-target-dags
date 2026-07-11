script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")
analysis_dir <- dirname(script_path)
project_dir <- normalizePath(file.path(analysis_dir, ".."), winslash = "/")

suppressPackageStartupMessages({
  library(ggplot2)
  library(ranger)
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
shap_dir <- file.path(analysis_dir, "output", "shap_nephrolithiasis_clean_v3")
output_path <- file.path(shap_dir, "predictive_signal_diagnostics.csv")
plot_path <- file.path(shap_dir, "predictive_signal_diagnostics.png")
plot_pdf_path <- file.path(shap_dir, "predictive_signal_diagnostics.pdf")

target_source_node <- "Nephrolithiasis"
target_variable <- "nephrolithiasis"
split_seed <- 20260714L

clean <- read.csv(data_path, check.names = FALSE)
ancestors <- read.csv(file.path(shap_dir, "target_ancestor_table.csv"), check.names = FALSE)
built <- build_source_aligned_simcausal_dag(source_code_path, nasa_like = FALSE)
features <- sort(ancestors$variable)

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

score_predictions <- function(learner, observed, predicted) {
  clipped <- pmin(pmax(predicted, 1e-12), 1 - 1e-12)
  data.frame(
    learner = learner,
    auc = binary_auc(observed, predicted),
    brier_score = mean((predicted - observed)^2),
    log_loss = -mean(
      observed * log(clipped) + (1 - observed) * log(1 - clipped)
    ),
    stringsAsFactors = FALSE
  )
}

split <- stratified_split(clean[[target_variable]], split_seed)
train <- clean[split == "train", , drop = FALSE]
test <- clean[split == "test", , drop = FALSE]
observed <- test[[target_variable]]

parent_sources <- built$source_edges$from[
  built$source_edges$to == target_source_node
]
parent_variables <- built$variable_map$variable[
  match(parent_sources, built$variable_map$source_node)
]
parent_coefficients <- vapply(
  parent_sources,
  function(parent) edge_coefficient(parent, target_source_node),
  numeric(1)
)

oracle_linear_predictor <- rep(binary_intercept(target_source_node), nrow(test))
for (i in seq_along(parent_variables)) {
  oracle_linear_predictor <-
    oracle_linear_predictor +
    parent_coefficients[[i]] * test[[parent_variables[[i]]]]
}
oracle_probability <- stats::plogis(oracle_linear_predictor)

parent_formula <- stats::reformulate(parent_variables, response = target_variable)
parent_glm <- stats::glm(parent_formula, data = train, family = stats::binomial())
parent_glm_probability <- stats::predict(parent_glm, newdata = test, type = "response")

full_formula <- stats::reformulate(features, response = target_variable)
full_glm <- stats::glm(full_formula, data = train, family = stats::binomial())
full_glm_probability <- stats::predict(full_glm, newdata = test, type = "response")

ranger_train <- train[, c(target_variable, features), drop = FALSE]
ranger_train[[target_variable]] <- factor(ranger_train[[target_variable]], levels = c(0, 1))
set.seed(20260720L)
ranger_model <- ranger::ranger(
  formula = full_formula,
  data = ranger_train,
  probability = TRUE,
  num.trees = 500,
  mtry = max(2L, floor(sqrt(length(features)))),
  min.node.size = 10,
  seed = 20260720L,
  num.threads = 4L
)
ranger_probability <- predict(
  ranger_model,
  data = test[, features, drop = FALSE]
)$predictions[, "1"]

xgboost_model <- xgb.load(file.path(shap_dir, "nephrolithiasis_xgboost_clean_v3.ubj"))
xgboost_probability <- predict(
  xgboost_model,
  as.matrix(test[, features, drop = FALSE])
)

diagnostics <- rbind(
  score_predictions("Oracle structural P(Y|direct parents)", observed, oracle_probability),
  score_predictions("Logistic regression: direct parents", observed, parent_glm_probability),
  score_predictions("Logistic regression: all 28 ancestors", observed, full_glm_probability),
  score_predictions("Ranger probability forest: all 28 ancestors", observed, ranger_probability),
  score_predictions("XGBoost: all 28 ancestors", observed, xgboost_probability)
)
diagnostics$n_test <- nrow(test)
diagnostics$events_test <- sum(observed)
diagnostics$direct_parent_count <- length(parent_variables)
diagnostics$direct_parents <- paste(parent_variables, collapse = "; ")
diagnostics$interpretation <- c(
  "Upper discrimination available from the true simulated conditional risk score",
  "Correctly specified parametric model estimated from the training data",
  "Parametric benchmark with all eligible ancestors",
  "Flexible nonparametric benchmark",
  "Primary learner used for the SHAP comparison"
)

write.csv(diagnostics, output_path, row.names = FALSE)

plot_data <- diagnostics
plot_data$short_label <- c(
  "Oracle structural score",
  "Logistic: direct parents",
  "Logistic: all ancestors",
  "Ranger forest",
  "XGBoost"
)
plot_data$short_label <- factor(
  plot_data$short_label,
  levels = rev(plot_data$short_label[order(plot_data$auc, decreasing = TRUE)])
)
plot_data$auc_above_chance <- plot_data$auc - 0.50

signal_plot <- ggplot(
  plot_data,
  aes(x = auc_above_chance, y = short_label)
) +
  geom_col(fill = "#2563eb", width = 0.68) +
  geom_text(
    aes(label = sprintf("AUC %.3f", auc)),
    hjust = -0.08,
    size = 3.7
  ) +
  scale_x_continuous(
    limits = c(0, 0.225),
    breaks = seq(0, 0.20, 0.05),
    labels = function(x) sprintf("%.2f", x + 0.50),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "The Nephrolithiasis AUC ceiling is low by construction",
    subtitle = paste(
      "The first learner, XGBoost, is close to the true structural risk score.",
      "Flexible modeling cannot recover signal the simulation did not create.",
      sep = "\n"
    ),
    x = "AUC (bars begin at chance = 0.50)",
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.y = element_blank(),
    plot.title = element_text(face = "bold"),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  )

ggsave(plot_path, signal_plot, width = 9.5, height = 5.4, dpi = 300, bg = "white")
ggsave(plot_pdf_path, signal_plot, width = 9.5, height = 5.4, bg = "white")

stopifnot(
  nrow(diagnostics) == 5L,
  all(diagnostics$auc >= 0.5 & diagnostics$auc <= 1),
  diagnostics$learner[[1]] == "Oracle structural P(Y|direct parents)"
)

print(diagnostics)
message("Wrote predictive-signal diagnostics to: ", output_path)
