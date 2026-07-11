suppressPackageStartupMessages({
  library(igraph)
  library(xgboost)
})

make_target_ancestor_table <- function(
    source_nodes,
    source_edges,
    variable_map,
    target_source_node) {
  graph <- igraph::graph_from_data_frame(
    source_edges,
    directed = TRUE,
    vertices = data.frame(name = source_nodes$node, stringsAsFactors = FALSE)
  )

  ancestor_nodes <- setdiff(
    igraph::as_ids(igraph::subcomponent(graph, target_source_node, mode = "in")),
    target_source_node
  )
  distances <- as.numeric(igraph::distances(
    graph,
    v = ancestor_nodes,
    to = target_source_node,
    mode = "out"
  )[, 1])

  ancestor_graph <- igraph::induced_subgraph(
    graph,
    vids = c(ancestor_nodes, target_source_node)
  )
  ancestor_indegree <- igraph::degree(
    ancestor_graph,
    v = ancestor_nodes,
    mode = "in"
  )

  result <- data.frame(
    source_node = ancestor_nodes,
    variable = variable_map$variable[
      match(ancestor_nodes, variable_map$source_node)
    ],
    distance = distances,
    ancestor_subgraph_indegree = as.integer(ancestor_indegree),
    structural_role = ifelse(
      distances == 1,
      "direct_parent",
      ifelse(ancestor_indegree == 0, "upstream_root", "intermediate_ancestor")
    ),
    stringsAsFactors = FALSE
  )

  result[order(result$distance, result$source_node), , drop = FALSE]
}

make_common_exogenous_draws <- function(node_names, n, seed) {
  set.seed(seed)
  normal <- matrix(
    stats::rnorm(n * length(node_names)),
    nrow = n,
    ncol = length(node_names),
    dimnames = list(NULL, node_names)
  )
  uniform <- matrix(
    stats::runif(n * length(node_names)),
    nrow = n,
    ncol = length(node_names),
    dimnames = list(NULL, node_names)
  )
  list(normal = normal, uniform = uniform)
}

make_source_structural_spec <- function(source_nodes, source_edges) {
  graph <- igraph::graph_from_data_frame(
    source_edges,
    directed = TRUE,
    vertices = data.frame(name = source_nodes$node, stringsAsFactors = FALSE)
  )
  stopifnot(igraph::is_dag(graph))

  order <- igraph::as_ids(igraph::topo_sort(graph, mode = "out"))
  parents <- stats::setNames(
    lapply(order, function(node_name) source_edges$from[source_edges$to == node_name]),
    order
  )

  list(
    order = order,
    parents = parents,
    distribution = stats::setNames(
      vapply(order, source_node_distribution, character(1)),
      order
    )
  )
}

simulate_source_structural_crn <- function(
    structural_spec,
    exogenous,
    intervention = numeric()) {
  n <- nrow(exogenous$normal)
  values <- stats::setNames(vector("list", length(structural_spec$order)), structural_spec$order)

  for (node_name in structural_spec$order) {
    if (node_name %in% names(intervention)) {
      values[[node_name]] <- rep(unname(intervention[[node_name]]), n)
      next
    }

    parents <- structural_spec$parents[[node_name]]
    distribution <- structural_spec$distribution[[node_name]]

    if (length(parents) == 0L) {
      values[[node_name]] <- if (distribution == "continuous") {
        exogenous$normal[, node_name]
      } else {
        as.numeric(exogenous$uniform[, node_name] < 0.35)
      }
      next
    }

    coefficients <- vapply(
      parents,
      function(parent) edge_coefficient(parent, node_name),
      numeric(1)
    )
    linear_predictor <- Reduce(
      `+`,
      Map(function(parent, coefficient) values[[parent]] * coefficient, parents, coefficients)
    )

    values[[node_name]] <- if (distribution == "continuous") {
      linear_predictor + 0.75 * exogenous$normal[, node_name]
    } else {
      probability <- stats::plogis(binary_intercept(node_name) + linear_predictor)
      as.numeric(exogenous$uniform[, node_name] < probability)
    }
  }

  values
}

random_topological_order <- function(nodes, edges) {
  remaining_nodes <- nodes
  remaining_edges <- edges[edges$from %in% nodes & edges$to %in% nodes, , drop = FALSE]
  ordering <- character()

  while (length(remaining_nodes) > 0L) {
    available <- setdiff(remaining_nodes, remaining_edges$to)
    if (length(available) == 0L) stop("Feature DAG contains a directed cycle.")
    chosen <- sample(available, 1L)
    ordering <- c(ordering, chosen)
    remaining_nodes <- setdiff(remaining_nodes, chosen)
    remaining_edges <- remaining_edges[remaining_edges$from != chosen, , drop = FALSE]
  }

  ordering
}

compute_asymmetric_interventional_shap <- function(
    model,
    evaluation,
    background,
    feature_edges,
    n_permutations,
    seed) {
  feature_names <- colnames(evaluation)
  stopifnot(
    identical(feature_names, colnames(background)),
    all(feature_edges$from %in% feature_names),
    all(feature_edges$to %in% feature_names)
  )

  n_evaluation <- nrow(evaluation)
  n_background <- nrow(background)
  result <- matrix(
    0,
    nrow = n_evaluation,
    ncol = length(feature_names),
    dimnames = list(rownames(evaluation), feature_names)
  )

  background_margin <- predict(model, background, outputmargin = TRUE)
  baseline <- mean(background_margin)
  set.seed(seed)

  for (permutation_index in seq_len(n_permutations)) {
    ordering <- random_topological_order(feature_names, feature_edges)
    batch <- background[rep(seq_len(n_background), times = n_evaluation), , drop = FALSE]
    previous_value <- rep(baseline, n_evaluation)

    for (feature in ordering) {
      batch[, feature] <- rep(evaluation[, feature], each = n_background)
      margin <- predict(model, batch, outputmargin = TRUE)
      current_value <- rowMeans(
        matrix(margin, nrow = n_evaluation, ncol = n_background, byrow = TRUE)
      )
      result[, feature] <- result[, feature] + current_value - previous_value
      previous_value <- current_value
    }
  }

  result <- result / n_permutations
  attr(result, "baseline_margin") <- baseline
  result
}

binary_auc <- function(observed, predicted) {
  positive <- observed == 1
  n_positive <- sum(positive)
  n_negative <- sum(!positive)
  if (n_positive == 0L || n_negative == 0L) return(NA_real_)
  ranks <- rank(predicted, ties.method = "average")
  (sum(ranks[positive]) - n_positive * (n_positive + 1) / 2) /
    (n_positive * n_negative)
}

normalized_discounted_cumulative_gain <- function(truth, score, k) {
  k <- min(k, length(truth))
  gain <- function(relevance) {
    sum((2^relevance - 1) / log2(seq_along(relevance) + 1))
  }
  observed <- truth[order(score, decreasing = TRUE)][seq_len(k)]
  ideal <- sort(truth, decreasing = TRUE)[seq_len(k)]
  denominator <- gain(ideal)
  if (denominator == 0) return(NA_real_)
  gain(observed) / denominator
}

summarize_attribution_method <- function(importance_table, method, top_k = 5L) {
  method_rows <- importance_table[importance_table$method == method, , drop = FALSE]
  truth_rows <- importance_table[importance_table$method == "Interventional truth", , drop = FALSE]
  truth <- truth_rows$normalized_importance[match(method_rows$variable, truth_rows$variable)]
  score <- method_rows$normalized_importance
  top_truth <- truth_rows$variable[order(truth_rows$normalized_importance, decreasing = TRUE)][seq_len(top_k)]
  top_method <- method_rows$variable[order(method_rows$normalized_importance, decreasing = TRUE)][seq_len(top_k)]

  data.frame(
    method = method,
    kendall_tau_vs_truth = suppressWarnings(stats::cor(score, truth, method = "kendall")),
    spearman_rho_vs_truth = suppressWarnings(stats::cor(score, truth, method = "spearman")),
    top5_recovery = length(intersect(top_truth, top_method)) / top_k,
    ndcg_at_5 = normalized_discounted_cumulative_gain(truth, score, top_k),
    normalized_mean_distance = sum(score * method_rows$distance),
    pbi = sum(truth * method_rows$distance) - sum(score * method_rows$distance),
    proximal_mass_distance_le_2 = sum(score[method_rows$distance <= 2]),
    upstream_mass_distance_ge_3 = sum(score[method_rows$distance >= 3]),
    proximal_to_upstream_ratio =
      sum(score[method_rows$distance <= 2]) /
      sum(score[method_rows$distance >= 3]),
    stringsAsFactors = FALSE
  )
}

make_distance_concentration <- function(importance_table) {
  max_distance <- max(importance_table$distance)
  methods <- unique(importance_table$method)

  do.call(rbind, lapply(methods, function(method) {
    rows <- importance_table[importance_table$method == method, , drop = FALSE]
    data.frame(
      method = method,
      hop_radius = seq_len(max_distance),
      cumulative_importance = vapply(
        seq_len(max_distance),
        function(k) sum(rows$normalized_importance[rows$distance <= k]),
        numeric(1)
      ),
      stringsAsFactors = FALSE
    )
  }))
}

make_distance_calibration <- function(importance_table, epsilon = 1e-8) {
  truth <- importance_table[
    importance_table$method == "Interventional truth",
    c("distance", "normalized_importance"),
    drop = FALSE
  ]
  truth_by_distance <- stats::aggregate(normalized_importance ~ distance, truth, sum)
  names(truth_by_distance)[2] <- "truth_mass"

  methods <- setdiff(unique(importance_table$method), "Interventional truth")
  do.call(rbind, lapply(methods, function(method) {
    rows <- importance_table[
      importance_table$method == method,
      c("distance", "normalized_importance"),
      drop = FALSE
    ]
    method_by_distance <- stats::aggregate(normalized_importance ~ distance, rows, sum)
    names(method_by_distance)[2] <- "method_mass"
    result <- merge(truth_by_distance, method_by_distance, by = "distance", all = TRUE)
    result$method <- method
    result$calibration_ratio <-
      (result$method_mass + epsilon) / (result$truth_mass + epsilon)
    result[, c("method", "distance", "truth_mass", "method_mass", "calibration_ratio")]
  }))
}
