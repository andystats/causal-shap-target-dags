script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")
analysis_dir <- dirname(script_path)
project_dir <- normalizePath(file.path(analysis_dir, ".."), winslash = "/")

source(file.path(analysis_dir, "R", "dag_concordance.R"))

reference_dir <- file.path(
  project_dir,
  "references",
  "robert-reynolds-2026-07-13"
)
output_dir <- file.path(analysis_dir, "output", "dag_sources")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

source_specs <- data.frame(
  graph_id = c("robert_renal_stone_20220322", "robert_sans_20220411"),
  risk = c("renal_stone", "sans"),
  file = c("Renal Stone Risk.txt", "SANS Risk.txt"),
  supplied_by = "Robert Reynolds",
  received_date = "2026-07-13",
  source_version = c(
    "Renal Stone Risk Edge Work DAG CM Final - Errata 20220322",
    "SANS Risk Edge Work DAG CM Final - Errata 20220411"
  ),
  stringsAsFactors = FALSE
)

graphs <- list()
inventory <- list()
roundtrip_summaries <- list()

for (i in seq_len(nrow(source_specs))) {
  spec <- source_specs[i, ]
  path <- file.path(reference_dir, spec$file)
  dag <- read_dagitty_source(path)
  nodes <- dag_nodes(dag)
  edges <- dag_edges(dag)
  graphs[[spec$risk]] <- dag

  write.csv(
    nodes,
    file.path(output_dir, paste0(spec$graph_id, "_nodes.csv")),
    row.names = FALSE
  )
  write.csv(
    edges,
    file.path(output_dir, paste0(spec$graph_id, "_edges.csv")),
    row.names = FALSE
  )

  roundtrip <- dag_from_edges(
    nodes = nodes$node,
    edges = edges,
    coords = nodes,
    layout = FALSE
  )
  roundtrip_summaries[[i]] <- compare_directed_edges(
    reference_edges = edges,
    candidate_edges = dag_edges(roundtrip),
    nodes = nodes$node,
    comparison = paste(spec$graph_id, "source vs canonical CSV round-trip")
  )$summary

  inventory[[i]] <- data.frame(
    graph_id = spec$graph_id,
    risk = spec$risk,
    supplied_by = spec$supplied_by,
    received_date = spec$received_date,
    source_version = spec$source_version,
    source_file = file.path("references", "robert-reynolds-2026-07-13", spec$file),
    node_count = nrow(nodes),
    edge_count = nrow(edges),
    exposure_count = sum(nodes$exposure),
    outcome_count = sum(nodes$outcome),
    latent_count = sum(nodes$latent),
    stringsAsFactors = FALSE
  )

  plot <- plot_dag_with_labels(
    dag,
    title = if (spec$risk == "renal_stone") {
      "Robert Reynolds renal-stone risk DAG"
    } else {
      "Robert Reynolds SANS risk DAG"
    },
    subtitle = paste0(spec$source_version, "; ingested 2026-07-13")
  )
  ggsave(
    file.path(output_dir, paste0(spec$graph_id, ".png")),
    plot = plot,
    width = 16,
    height = 12,
    dpi = 160,
    bg = "white"
  )
}

write.csv(
  do.call(rbind, inventory),
  file.path(output_dir, "robert_dag_inventory.csv"),
  row.names = FALSE
)
write.csv(
  do.call(rbind, roundtrip_summaries),
  file.path(output_dir, "robert_dag_roundtrip_summary.csv"),
  row.names = FALSE
)

# Compare Robert's renal-stone file with the repository's previously canonical
# public SA-07566 DAG. The strict comparison retains all original labels and
# graph granularity.
existing_path <- file.path(
  project_dir,
  "references",
  "renal-stone-dag-code-SA-07566.txt"
)
existing_renal <- read_dagitty_source(existing_path)
robert_renal <- graphs$renal_stone

existing_nodes <- dag_nodes(existing_renal)
existing_edges <- dag_edges(existing_renal)
robert_nodes <- dag_nodes(robert_renal)
robert_edges <- dag_edges(robert_renal)

strict_nodes <- union(existing_nodes$node, robert_nodes$node)
strict_comparison <- compare_directed_edges(
  reference_edges = existing_edges,
  candidate_edges = robert_edges,
  nodes = strict_nodes,
  comparison = "Existing SA-07566 source vs Robert renal DAG (strict labels)"
)

# Robert's graph expands two nodes used by the repository's graph and renames
# four others. Collapse those known differences before comparing the shared
# scientific structure. This crosswalk is descriptive; it does not replace
# either raw source graph.
semantic_aliases <- c(
  "Bone Formation" = "Bone Remodeling",
  "Bone Resorption" = "Bone Remodeling",
  "CO2 Physiologic Changes" = "CO2 (Risk)",
  "HSI Processes" = "HSIA (Risk)",
  "Hydronephrosis" = "Medical Illness",
  "Infection" = "Medical Illness",
  "Renal Colic" = "Medical Illness",
  "Renal Failure" = "Medical Illness",
  "Sepsis" = "Medical Illness",
  "Pharmaceutical Effectiveness" = "Pharm (Risk)",
  "Urinary Retention" = "Urinary Retention (Risk)"
)

normalize_robert_edges <- function(edges, aliases) {
  edges$from <- ifelse(
    edges$from %in% names(aliases),
    unname(aliases[edges$from]),
    edges$from
  )
  edges$to <- ifelse(
    edges$to %in% names(aliases),
    unname(aliases[edges$to]),
    edges$to
  )
  edges <- edges[edges$from != edges$to, , drop = FALSE]
  unique(edges[order(edges$from, edges$to), , drop = FALSE])
}

normalized_robert_edges <- normalize_robert_edges(robert_edges, semantic_aliases)
normalized_robert_nodes <- sort(unique(c(
  normalized_robert_edges$from,
  normalized_robert_edges$to
)))
shared_nodes <- intersect(existing_nodes$node, normalized_robert_nodes)
existing_shared_edges <- existing_edges[
  existing_edges$from %in% shared_nodes & existing_edges$to %in% shared_nodes,
  ,
  drop = FALSE
]
robert_shared_edges <- normalized_robert_edges[
  normalized_robert_edges$from %in% shared_nodes &
    normalized_robert_edges$to %in% shared_nodes,
  ,
  drop = FALSE
]
semantic_comparison <- compare_directed_edges(
  reference_edges = existing_shared_edges,
  candidate_edges = robert_shared_edges,
  nodes = shared_nodes,
  comparison = "Existing SA-07566 vs Robert renal DAG (semantic shared-node projection)"
)

write.csv(
  rbind(strict_comparison$summary, semantic_comparison$summary),
  file.path(output_dir, "robert_renal_concordance_summary.csv"),
  row.names = FALSE
)
write.csv(
  rbind(strict_comparison$discrepancy, semantic_comparison$discrepancy),
  file.path(output_dir, "robert_renal_edge_discrepancies.csv"),
  row.names = FALSE
)
write.csv(
  data.frame(
    robert_node = names(semantic_aliases),
    existing_node = unname(semantic_aliases),
    mapping_type = ifelse(
      duplicated(unname(semantic_aliases)) |
        duplicated(unname(semantic_aliases), fromLast = TRUE),
      "many-to-one abstraction",
      "label alias"
    ),
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "robert_renal_semantic_crosswalk.csv"),
  row.names = FALSE
)
node_discrepancy_rows <- function(graph, nodes) {
  data.frame(
    graph = rep(graph, length(nodes)),
    node = nodes,
    stringsAsFactors = FALSE
  )
}

write.csv(
  rbind(
    node_discrepancy_rows(
      "existing_only_after_semantic_mapping",
      setdiff(existing_nodes$node, normalized_robert_nodes)
    ),
    node_discrepancy_rows(
      "robert_only_after_semantic_mapping",
      setdiff(normalized_robert_nodes, existing_nodes$node)
    )
  ),
  file.path(output_dir, "robert_renal_node_discrepancies.csv"),
  row.names = FALSE
)

cat("Ingested Robert Reynolds DAGs:\n")
print(do.call(rbind, inventory)[, c("graph_id", "node_count", "edge_count")])
cat("\nRenal-stone concordance:\n")
print(rbind(strict_comparison$summary, semantic_comparison$summary))
