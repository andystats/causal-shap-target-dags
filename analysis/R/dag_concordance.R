suppressPackageStartupMessages({
  library(dagitty)
  library(ggdag)
  library(ggplot2)
})

read_dagitty_source <- function(path) {
  code <- paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = " ")
  # Source files arrive in two equivalent forms: raw DAGitty (`dag {`) and
  # Pandoc-escaped text (`dag \\{`). Accept both so the ingestion pipeline can
  # preserve collaborators' original attachments without rewriting them.
  starts <- c(
    regexpr("dag {", code, fixed = TRUE)[[1]],
    regexpr("dag \\{", code, fixed = TRUE)[[1]]
  )
  starts <- starts[starts >= 0L]

  if (length(starts) == 0L) {
    stop("Could not find a DAGitty graph in the source file.")
  }

  code <- substring(code, min(starts))
  code <- gsub("\\{", "{", code, fixed = TRUE)
  code <- gsub("\\}", "}", code, fixed = TRUE)
  code <- gsub("{[}", "[", code, fixed = TRUE)
  code <- gsub("{]}", "]", code, fixed = TRUE)
  code <- gsub("-\\textgreater{}", "->", code, fixed = TRUE)
  code <- gsub("[[:space:]]+", " ", code)

  dagitty(code)
}

# Backward-compatible name retained for the renal-stone simulation scripts.
read_nasa_dag_code <- function(path) {
  read_dagitty_source(path)
}

dag_edges <- function(dag) {
  edge_data <- dagitty::edges(dag)
  edge_data <- edge_data[edge_data$e == "->", c("v", "w"), drop = FALSE]
  names(edge_data) <- c("from", "to")
  edge_data <- unique(edge_data)
  edge_data[order(edge_data$from, edge_data$to), , drop = FALSE]
}

dag_nodes <- function(dag) {
  coords <- dagitty::coordinates(dag)
  node_names <- sort(unique(c(names(coords$x), names(coords$y))))

  data.frame(
    node = node_names,
    x = unname(coords$x[node_names]),
    y = unname(coords$y[node_names]),
    exposure = node_names %in% dagitty::exposures(dag),
    outcome = node_names %in% dagitty::outcomes(dag),
    latent = node_names %in% dagitty::latents(dag),
    stringsAsFactors = FALSE
  )
}

simcausal_edges <- function(dag) {
  parent_list <- attr(dag, "parents")

  rows <- lapply(names(parent_list), function(child) {
    parents <- parent_list[[child]]
    if (length(parents) == 0L) return(NULL)
    data.frame(from = parents, to = child, stringsAsFactors = FALSE)
  })

  edges <- do.call(rbind, rows)
  edges[] <- lapply(edges, function(x) gsub(".", "_", x, fixed = TRUE))
  unique(edges[order(edges$from, edges$to), , drop = FALSE])
}

quote_dagitty_id <- function(x) {
  paste0('"', gsub('"', '\\\\"', x), '"')
}

dag_from_edges <- function(nodes, edges, coords = NULL, layout = FALSE) {
  node_statements <- quote_dagitty_id(nodes)
  edge_statements <- paste(
    quote_dagitty_id(edges$from),
    "->",
    quote_dagitty_id(edges$to)
  )

  dag <- dagitty(paste0(
    "dag { ",
    paste(c(node_statements, edge_statements), collapse = " "),
    " }"
  ), layout = layout)

  if (!is.null(coords)) {
    coords <- coords[match(nodes, coords$node), , drop = FALSE]
    dagitty::coordinates(dag) <- list(
      x = stats::setNames(coords$x, coords$node),
      y = stats::setNames(coords$y, coords$node)
    )
  }

  dag
}

compare_directed_edges <- function(reference_edges, candidate_edges, nodes, comparison) {
  nodes <- sort(unique(nodes))
  universe <- expand.grid(
    from = nodes,
    to = nodes,
    stringsAsFactors = FALSE
  )
  universe <- universe[universe$from != universe$to, , drop = FALSE]

  edge_key <- function(data) paste(data$from, data$to, sep = "\r")

  reference <- edge_key(universe) %in% edge_key(reference_edges)
  candidate <- edge_key(universe) %in% edge_key(candidate_edges)

  tp <- sum(reference & candidate)
  tn <- sum(!reference & !candidate)
  fp <- sum(!reference & candidate)
  fn <- sum(reference & !candidate)
  n <- length(reference)

  observed_agreement <- (tp + tn) / n
  reference_positive <- mean(reference)
  candidate_positive <- mean(candidate)
  expected_agreement <-
    reference_positive * candidate_positive +
    (1 - reference_positive) * (1 - candidate_positive)

  kappa <- if (expected_agreement == 1) {
    if (observed_agreement == 1) 1 else NA_real_
  } else {
    (observed_agreement - expected_agreement) / (1 - expected_agreement)
  }

  precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
  recall <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  f1 <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
    NA_real_
  } else {
    2 * precision * recall / (precision + recall)
  }

  summary <- data.frame(
    comparison = comparison,
    node_count = length(nodes),
    directed_pair_count = n,
    reference_edge_count = sum(reference),
    candidate_edge_count = sum(candidate),
    true_positive = tp,
    true_negative = tn,
    false_positive = fp,
    false_negative = fn,
    observed_agreement = observed_agreement,
    cohen_kappa = kappa,
    precision = precision,
    recall = recall,
    f1 = f1,
    structural_hamming_distance = fp + fn,
    exact_match = fp == 0 && fn == 0,
    stringsAsFactors = FALSE
  )

  discrepancy <- universe[reference != candidate, , drop = FALSE]
  if (nrow(discrepancy) == 0L) {
    discrepancy <- data.frame(
      comparison = character(),
      status = character(),
      from = character(),
      to = character(),
      stringsAsFactors = FALSE
    )
  } else {
    discrepancy$comparison <- comparison
    discrepancy$status <- ifelse(
      reference[reference != candidate],
      "missing_from_candidate",
      "extra_in_candidate"
    )
    discrepancy <- discrepancy[c("comparison", "status", "from", "to")]
  }

  list(summary = summary, discrepancy = discrepancy)
}

plot_dag_with_labels <- function(dag, title, subtitle = NULL) {
  ggdag::ggdag(
    dag,
    edge_type = "link",
    node_size = 5,
    text = FALSE
  ) +
    ggdag::geom_dag_text_repel(
      aes(label = name),
      size = 2.35,
      max.overlaps = Inf,
      box.padding = 0.25,
      point.padding = 0.15,
      min.segment.length = 0
    ) +
    ggdag::theme_dag_blank() +
    labs(title = title, subtitle = subtitle) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10, color = "grey30"),
      plot.margin = margin(18, 18, 18, 18)
    )
}
