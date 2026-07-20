local({                                   # locate + source paths.R from anywhere at/below the repo
  dir <- getwd()
  while (!file.exists(file.path(dir, "analysis", "R", "paths.R"))) {
    parent <- dirname(dir)
    if (identical(parent, dir)) stop("Run from inside the repository.")
    dir <- parent
  }
  source(file.path(dir, "analysis", "R", "paths.R"))
})

source(file.path(analysis_dir, "R", "renal_stone_simcausal.R"))
source(file.path(analysis_dir, "R", "dag_concordance.R"))
source(file.path(analysis_dir, "R", "renal_stone_source_aligned_simcausal.R"))

output_dir <- file.path(analysis_dir, "output", "dag_validation")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

source_code_path <- file.path(
  project_dir,
  "references",
  "renal-stone-dag-code-SA-07566.txt"
)

nasa_dag <- read_nasa_dag_code(source_code_path)
nasa_nodes <- dag_nodes(nasa_dag)
nasa_edges <- dag_edges(nasa_dag)

write.csv(
  nasa_nodes,
  file.path(output_dir, "nasa_sa07566_source_nodes.csv"),
  row.names = FALSE
)
write.csv(
  nasa_edges,
  file.path(output_dir, "nasa_sa07566_source_edges.csv"),
  row.names = FALSE
)

# Round-trip the source code through canonical node/edge CSVs. This is the
# strongest machine-verifiable check that the rendered reference is structurally
# identical to NASA's published DAG code.
roundtrip_dag <- dag_from_edges(
  nodes = nasa_nodes$node,
  edges = nasa_edges,
  coords = nasa_nodes,
  layout = FALSE
)
roundtrip_edges <- dag_edges(roundtrip_dag)
source_roundtrip <- compare_directed_edges(
  reference_edges = nasa_edges,
  candidate_edges = roundtrip_edges,
  nodes = nasa_nodes$node,
  comparison = "NASA source code vs canonical CSV round-trip"
)

# Canonical validated graph specifications. Clean is structurally identical to
# the source. NASA-like preserves the entire source graph and adds only an
# observation-process layer; concordance is evaluated after removing that layer.
validated_clean_nodes <- nasa_nodes
validated_clean_edges <- nasa_edges

augmentation_nodes <- data.frame(
  node = c(
    "Baseline Fitness",
    "Age",
    "Selected into Observed Cohort",
    "Observe Bone Remodeling",
    "Observe Hydration",
    "Observe Urine Concentration",
    "Observe Urine Chemistry",
    "Observe Mineralized Renal Material"
  ),
  x = c(-0.25, -0.10, 0.05, -0.32, -0.16, 0.00, 0.16, 0.32),
  y = c(-0.56, -0.56, -0.56, 0.50, 0.50, 0.50, 0.50, 0.50),
  exposure = FALSE,
  outcome = FALSE,
  latent = FALSE,
  stringsAsFactors = FALSE
)

augmentation_edges <- data.frame(
  from = c(
    "Baseline Fitness",
    "Individual Factors",
    "Age",
    "Medical Illness",
    "Medical Illness",
    "Medical Illness",
    "Nephrolithiasis",
    "Medical Illness",
    "Nephrolithiasis",
    "Medical Illness"
  ),
  to = c(
    "Selected into Observed Cohort",
    "Selected into Observed Cohort",
    "Selected into Observed Cohort",
    "Observe Bone Remodeling",
    "Observe Hydration",
    "Observe Urine Concentration",
    "Observe Urine Chemistry",
    "Observe Urine Chemistry",
    "Observe Mineralized Renal Material",
    "Observe Mineralized Renal Material"
  ),
  stringsAsFactors = FALSE
)

validated_nasa_like_nodes <- rbind(nasa_nodes, augmentation_nodes)
validated_nasa_like_edges <- unique(rbind(nasa_edges, augmentation_edges))

write.csv(
  validated_clean_nodes,
  file.path(output_dir, "validated_clean_source_nodes.csv"),
  row.names = FALSE
)
write.csv(
  validated_clean_edges,
  file.path(output_dir, "validated_clean_source_edges.csv"),
  row.names = FALSE
)
write.csv(
  validated_nasa_like_nodes,
  file.path(output_dir, "validated_nasa_like_nodes.csv"),
  row.names = FALSE
)
write.csv(
  validated_nasa_like_edges,
  file.path(output_dir, "validated_nasa_like_edges.csv"),
  row.names = FALSE
)

validated_clean_comparison <- compare_directed_edges(
  reference_edges = nasa_edges,
  candidate_edges = validated_clean_edges,
  nodes = nasa_nodes$node,
  comparison = "NASA source vs validated clean DAG"
)

validated_nasa_core_edges <- validated_nasa_like_edges[
  validated_nasa_like_edges$from %in% nasa_nodes$node &
    validated_nasa_like_edges$to %in% nasa_nodes$node,
  ,
  drop = FALSE
]
validated_nasa_like_comparison <- compare_directed_edges(
  reference_edges = nasa_edges,
  candidate_edges = validated_nasa_core_edges,
  nodes = nasa_nodes$node,
  comparison = "NASA source vs validated NASA-like source layer"
)

# Verify the parent structures of the actual source-aligned simcausal DAGs, not
# only the CSV specifications.
source_aligned_clean_built <- build_source_aligned_simcausal_dag(
  source_code_path,
  nasa_like = FALSE
)
source_aligned_nasa_built <- build_source_aligned_simcausal_dag(
  source_code_path,
  nasa_like = TRUE
)

source_aligned_clean_raw_edges <- simcausal_edges(source_aligned_clean_built$dag)
source_aligned_nasa_raw_edges <- simcausal_edges(source_aligned_nasa_built$dag)

source_aligned_map <- source_aligned_clean_built$variable_map[c("variable", "source_node")]
augmentation_map <- data.frame(
  variable = c(
    "baseline_fitness",
    "age_z",
    "selected_into_observed_cohort",
    "observe_bone_remodeling",
    "observe_hydration",
    "observe_urine_concentration",
    "observe_urine_chemistry",
    "observe_mineralized_renal_material"
  ),
  source_node = augmentation_nodes$node,
  stringsAsFactors = FALSE
)

map_source_aligned_edges <- function(edges, mapping) {
  mapped <- data.frame(
    from = mapping$source_node[match(edges$from, mapping$variable)],
    to = mapping$source_node[match(edges$to, mapping$variable)],
    stringsAsFactors = FALSE
  )
  mapped <- mapped[!is.na(mapped$from) & !is.na(mapped$to), , drop = FALSE]
  unique(mapped[order(mapped$from, mapped$to), , drop = FALSE])
}

actual_source_aligned_clean_edges <- map_source_aligned_edges(
  source_aligned_clean_raw_edges,
  source_aligned_map
)
actual_source_aligned_nasa_edges <- map_source_aligned_edges(
  source_aligned_nasa_raw_edges,
  rbind(source_aligned_map, augmentation_map)
)

actual_source_aligned_clean_comparison <- compare_directed_edges(
  reference_edges = nasa_edges,
  candidate_edges = actual_source_aligned_clean_edges,
  nodes = nasa_nodes$node,
  comparison = "NASA source vs actual source-aligned clean simcausal DAG"
)
actual_source_aligned_nasa_comparison <- compare_directed_edges(
  reference_edges = validated_nasa_like_edges,
  candidate_edges = actual_source_aligned_nasa_edges,
  nodes = validated_nasa_like_nodes$node,
  comparison = "Validated NASA-like spec vs actual source-aligned simcausal DAG"
)

write.csv(
  actual_source_aligned_clean_edges,
  file.path(output_dir, "actual_source_aligned_clean_simcausal_edges.csv"),
  row.names = FALSE
)
write.csv(
  actual_source_aligned_nasa_edges,
  file.path(output_dir, "actual_source_aligned_nasa_like_simcausal_edges.csv"),
  row.names = FALSE
)

clean_dag <- build_renal_stone_dag(nasa_like = FALSE)
nasa_like_dag <- build_renal_stone_dag(nasa_like = TRUE)
clean_edges <- simcausal_edges(clean_dag)
nasa_like_edges <- simcausal_edges(nasa_like_dag)

write.csv(clean_edges, file.path(output_dir, "clean_simcausal_edges.csv"), row.names = FALSE)
write.csv(nasa_like_edges, file.path(output_dir, "nasa_like_simcausal_edges.csv"), row.names = FALSE)

node_map <- data.frame(
  simulation_node = c(
    "individual_susceptibility", "selected_astronaut", "altered_gravity",
    "hostile_closed_environment", "co2_exposure", "urinary_retention",
    "dietary_risk", "microbiome_risk", "resistive_exercise",
    "bisphosphonate", "adequate_water_intake", "potassium_citrate",
    "thiazide", "bone_remodeling", "hydration", "urine_concentration",
    "urine_chemistry_risk", "mineralized_renal_material",
    "nephrolithiasis", "ureterolithiasis", "impaired_urine_flow",
    "renal_colic", "hydronephrosis", "infection", "sepsis",
    "renal_failure", "medical_illness", "individual_readiness_loss",
    "evacuation", "crew_capability_loss", "task_performance_loss",
    "loss_mission_objectives", "loss_mission", "loss_crew_life",
    "long_term_health_outcome"
  ),
  source_node = c(
    "Individual Factors", "Astronaut Selection", "Altered Gravity",
    "Hostile Closed Environment", "CO2 (Risk)", "Urinary Retention (Risk)",
    "Nutrients", "Microbiome", "Resistive Exercise",
    "Bisphosphonates", "Water Intake", "K+ Citrate",
    "Thiazides", "Bone Remodeling", "Hydration", "Urine Concentration",
    "Urine Chemistry", "Mineralized Renal Material",
    "Nephrolithiasis", "Ureterolithiasis", "Urine Flow",
    "Medical Illness", "Medical Illness", "Medical Illness", "Medical Illness",
    "Medical Illness", "Medical Illness", "Individual Readiness",
    "Evacuation", "Crew Capability", "Task Performance",
    "Loss of Mission Objectives", "Loss of Mission", "Loss of Crew Life",
    "Long Term Health Outcomes"
  ),
  stringsAsFactors = FALSE
)

write.csv(
  node_map,
  file.path(output_dir, "simulation_to_nasa_node_map.csv"),
  row.names = FALSE
)

map_edges <- function(edges, node_map) {
  from_match <- match(edges$from, node_map$simulation_node)
  to_match <- match(edges$to, node_map$simulation_node)
  mapped <- data.frame(
    from = node_map$source_node[from_match],
    to = node_map$source_node[to_match],
    stringsAsFactors = FALSE
  )
  mapped <- mapped[!is.na(mapped$from) & !is.na(mapped$to), , drop = FALSE]
  mapped <- mapped[mapped$from != mapped$to, , drop = FALSE]
  unique(mapped[order(mapped$from, mapped$to), , drop = FALSE])
}

mapped_clean_edges <- map_edges(clean_edges, node_map)
mapped_nasa_like_edges <- map_edges(nasa_like_edges, node_map)
comparison_nodes <- sort(unique(node_map$source_node))
reference_projection <- nasa_edges[
  nasa_edges$from %in% comparison_nodes & nasa_edges$to %in% comparison_nodes,
  ,
  drop = FALSE
]

clean_projection <- compare_directed_edges(
  reference_edges = reference_projection,
  candidate_edges = mapped_clean_edges,
  nodes = comparison_nodes,
  comparison = "NASA source projection vs clean simcausal model"
)
nasa_like_projection <- compare_directed_edges(
  reference_edges = reference_projection,
  candidate_edges = mapped_nasa_like_edges,
  nodes = comparison_nodes,
  comparison = "NASA source projection vs NASA-like simcausal model"
)

summary <- rbind(
  source_roundtrip$summary,
  validated_clean_comparison$summary,
  validated_nasa_like_comparison$summary,
  actual_source_aligned_clean_comparison$summary,
  actual_source_aligned_nasa_comparison$summary,
  clean_projection$summary,
  nasa_like_projection$summary
)
discrepancies <- rbind(
  source_roundtrip$discrepancy,
  validated_clean_comparison$discrepancy,
  validated_nasa_like_comparison$discrepancy,
  actual_source_aligned_clean_comparison$discrepancy,
  actual_source_aligned_nasa_comparison$discrepancy,
  clean_projection$discrepancy,
  nasa_like_projection$discrepancy
)

write.csv(summary, file.path(output_dir, "dag_concordance_summary.csv"), row.names = FALSE)
write.csv(discrepancies, file.path(output_dir, "dag_edge_discrepancies.csv"), row.names = FALSE)

source_plot <- plot_dag_with_labels(
  nasa_dag,
  title = "NASA Risk of Renal Stone Formation DAG",
  subtitle = "Parsed directly from NASA SA-07566 DAG code; published coordinates retained"
)

clean_plot_dag <- dag_from_edges(
  nodes = unique(c(clean_edges$from, clean_edges$to)),
  edges = clean_edges,
  layout = TRUE
)
clean_plot <- plot_dag_with_labels(
  clean_plot_dag,
  title = "Clean renal-stone simcausal DAG",
  subtitle = "Actual parent structure extracted from the locked simcausal DAG"
)

nasa_like_plot_dag <- dag_from_edges(
  nodes = unique(c(nasa_like_edges$from, nasa_like_edges$to)),
  edges = nasa_like_edges,
  layout = TRUE
)
nasa_like_plot <- plot_dag_with_labels(
  nasa_like_plot_dag,
  title = "NASA-like renal-stone simcausal DAG",
  subtitle = "Source mechanism plus astronaut-selection and measurement-process nodes"
)

validated_nasa_like_plot_dag <- dag_from_edges(
  nodes = validated_nasa_like_nodes$node,
  edges = validated_nasa_like_edges,
  coords = validated_nasa_like_nodes,
  layout = FALSE
)
validated_nasa_like_plot <- plot_dag_with_labels(
  validated_nasa_like_plot_dag,
  title = "Validated NASA-like renal-stone DAG specification",
  subtitle = "NASA SA-07566 retained exactly; selection and informative measurement added as a separate layer"
)

ggsave(file.path(output_dir, "nasa_source_dag.png"), source_plot, width = 16, height = 10, dpi = 300)
ggsave(file.path(output_dir, "nasa_source_dag.pdf"), source_plot, width = 16, height = 10)
ggsave(file.path(output_dir, "clean_simcausal_dag.png"), clean_plot, width = 16, height = 11, dpi = 300)
ggsave(file.path(output_dir, "clean_simcausal_dag.pdf"), clean_plot, width = 16, height = 11)
ggsave(file.path(output_dir, "nasa_like_simcausal_dag.png"), nasa_like_plot, width = 17, height = 12, dpi = 300)
ggsave(file.path(output_dir, "nasa_like_simcausal_dag.pdf"), nasa_like_plot, width = 17, height = 12)
ggsave(file.path(output_dir, "validated_nasa_like_dag.png"), validated_nasa_like_plot, width = 18, height = 13, dpi = 300)
ggsave(file.path(output_dir, "validated_nasa_like_dag.pdf"), validated_nasa_like_plot, width = 18, height = 13)

print(summary)
