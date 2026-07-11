suppressPackageStartupMessages({
  library(simcausal)
  library(igraph)
})

options(simcausal.verbose = FALSE)

source_node_to_variable <- function(x) {
  x <- gsub("+", " plus ", x, fixed = TRUE)
  x <- gsub("(Risk)", " risk ", x, fixed = TRUE)
  x <- tolower(x)
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  x <- gsub("_+", "_", x)
  x
}

source_node_to_simcausal <- function(x) {
  gsub("_", ".", source_node_to_variable(x), fixed = TRUE)
}

edge_coefficient <- function(from, to) {
  negative_edges <- paste(
    c(
      "Astronaut Selection",
      "Medical Prevention Capability",
      "Resistive Exercise",
      "Bisphosphonates",
      "Medical Prevention Capability",
      "K+ Citrate",
      "Thiazides",
      "Ultrasound Manipulation",
      "Water Intake",
      "Tamsulosin",
      "Percutaneous Nephrostomy",
      rep("Medications", 5)
    ),
    c(
      "Individual Factors",
      "Bone Remodeling",
      "Bone Remodeling",
      "Bone Remodeling",
      "Urine Chemistry",
      "Urine Chemistry",
      "Urine Chemistry",
      "Ureterolithiasis",
      "Urine Flow",
      "Urine Flow",
      "Medical Illness",
      "Individual Readiness",
      "Long Term Health Outcomes",
      "Loss of Crew Life",
      "Medical Illness",
      "Evacuation"
    ),
    sep = "\r"
  )

  if (paste(from, to, sep = "\r") %in% negative_edges) -0.70 else 0.60
}

source_node_distribution <- function(node_name) {
  continuous_nodes <- c(
    "Individual Factors",
    "Nutrients",
    "Microbiome",
    "Bone Remodeling",
    "Hydration",
    "Urine Concentration",
    "Urine Chemistry",
    "Mineralized Renal Material",
    "Urine Flow"
  )

  if (node_name %in% continuous_nodes) "continuous" else "binary"
}

binary_intercept <- function(node_name) {
  severe_outcomes <- c(
    "Loss of Mission",
    "Loss of Crew Life",
    "Loss of Mission Objectives",
    "Evacuation",
    "Long Term Health Outcomes"
  )
  clinical_outcomes <- c(
    "Nephrolithiasis",
    "Ureterolithiasis",
    "Medical Illness",
    "Individual Readiness",
    "Crew Capability",
    "Task Performance"
  )

  if (node_name %in% severe_outcomes) return(-4.00)
  if (node_name %in% clinical_outcomes) return(-2.35)
  -0.65
}

build_source_aligned_simcausal_dag <- function(source_code_path, nasa_like = FALSE) {
  nasa_dag <- read_nasa_dag_code(source_code_path)
  source_nodes <- dag_nodes(nasa_dag)
  source_edges <- dag_edges(nasa_dag)

  variable_map <- data.frame(
    source_node = source_nodes$node,
    variable = source_node_to_variable(source_nodes$node),
    simcausal_node = source_node_to_simcausal(source_nodes$node),
    stringsAsFactors = FALSE
  )

  stopifnot(
    !anyDuplicated(variable_map$variable),
    !anyDuplicated(variable_map$simcausal_node)
  )

  graph <- igraph::graph_from_data_frame(
    source_edges,
    directed = TRUE,
    vertices = data.frame(name = source_nodes$node, stringsAsFactors = FALSE)
  )
  stopifnot(igraph::is_dag(graph))
  order <- igraph::as_ids(igraph::topo_sort(graph, mode = "out"))

  dag <- DAG.empty()

  for (source_node in order) {
    sim_node <- variable_map$simcausal_node[
      match(source_node, variable_map$source_node)
    ]
    parent_sources <- source_edges$from[source_edges$to == source_node]
    parent_sim <- variable_map$simcausal_node[
      match(parent_sources, variable_map$source_node)
    ]

    if (length(parent_sources) == 0L) {
      if (source_node_distribution(source_node) == "continuous") {
        dag <- add.nodes(
          dag,
          node(name = sim_node, distr = "rnorm", mean = 0, sd = 1)
        )
      } else {
        dag <- add.nodes(
          dag,
          node(name = sim_node, distr = "rbern", prob = 0.35)
        )
      }
      next
    }

    coefficients <- vapply(
      seq_along(parent_sources),
      function(i) edge_coefficient(parent_sources[[i]], source_node),
      numeric(1)
    )
    terms <- paste0(
      sprintf("%.2f", coefficients),
      " * ",
      parent_sim
    )

    if (source_node_distribution(source_node) == "continuous") {
      mean_formula <- paste(c("0", terms), collapse = " + ")
      dag <- add.nodes(
        dag,
        node(
          name = sim_node,
          distr = "rnorm",
          params = list(mean = mean_formula, sd = 0.75)
        )
      )
    } else {
      probability_formula <- paste0(
        "plogis(",
        paste(c(sprintf("%.2f", binary_intercept(source_node)), terms), collapse = " + "),
        ")"
      )
      dag <- add.nodes(
        dag,
        node(
          name = sim_node,
          distr = "rbern",
          params = list(prob = probability_formula)
        )
      )
    }
  }

  if (nasa_like) {
    individual_factors_node <- variable_map$simcausal_node[
      variable_map$source_node == "Individual Factors"
    ]
    medical_illness_node <- variable_map$simcausal_node[
      variable_map$source_node == "Medical Illness"
    ]
    nephrolithiasis_node <- variable_map$simcausal_node[
      variable_map$source_node == "Nephrolithiasis"
    ]

    dag <- dag +
      node("baseline.fitness", distr = "rnorm", mean = 0, sd = 1) +
      node("age.z", distr = "rnorm", mean = 0, sd = 1) +
      node(
        "selected.into.observed.cohort",
        distr = "rbern",
        params = list(
          prob = paste0(
            "plogis(0.40 + 1.20 * baseline.fitness - 0.95 * ",
            individual_factors_node,
            " - 0.35 * age.z)"
          )
        )
      ) +
      node(
        "observe.bone.remodeling",
        distr = "rbern",
        params = list(prob = paste0("plogis(0.25 + 0.55 * ", medical_illness_node, ")"))
      ) +
      node(
        "observe.hydration",
        distr = "rbern",
        params = list(prob = paste0("plogis(0.80 + 0.35 * ", medical_illness_node, ")"))
      ) +
      node(
        "observe.urine.concentration",
        distr = "rbern",
        params = list(prob = paste0("plogis(0.10 + 0.65 * ", medical_illness_node, ")"))
      ) +
      node(
        "observe.urine.chemistry",
        distr = "rbern",
        params = list(
          prob = paste0(
            "plogis(-0.35 + 0.85 * ",
            nephrolithiasis_node,
            " + 0.65 * ",
            medical_illness_node,
            ")"
          )
        )
      ) +
      node(
        "observe.mineralized.renal.material",
        distr = "rbern",
        params = list(
          prob = paste0(
            "plogis(-1.00 + 1.25 * ",
            nephrolithiasis_node,
            " + 0.80 * ",
            medical_illness_node,
            ")"
          )
        )
      )
  }

  list(
    dag = set.DAG(dag),
    variable_map = variable_map,
    source_dag = nasa_dag,
    source_nodes = source_nodes,
    source_edges = source_edges
  )
}

simulate_source_aligned <- function(source_code_path, n, seed, nasa_like = FALSE) {
  built <- build_source_aligned_simcausal_dag(
    source_code_path = source_code_path,
    nasa_like = nasa_like
  )
  data <- sim(DAG = built$dag, n = n, rndseed = seed)
  names(data) <- gsub(".", "_", names(data), fixed = TRUE)
  attr(data, "variable_map") <- built$variable_map
  data
}

make_source_aligned_nasa_like_observed <- function(
    source_data,
    n_observed = 450L,
    seed = 20260712L) {
  selected <- source_data[
    source_data$selected_into_observed_cohort == 1,
    ,
    drop = FALSE
  ]

  if (nrow(selected) < n_observed) {
    stop("Selected cohort is smaller than n_observed.")
  }

  set.seed(seed)
  observed <- selected[
    sample.int(nrow(selected), n_observed, replace = FALSE),
    ,
    drop = FALSE
  ]

  mask_map <- c(
    bone_remodeling = "observe_bone_remodeling",
    hydration = "observe_hydration",
    urine_concentration = "observe_urine_concentration",
    urine_chemistry = "observe_urine_chemistry",
    mineralized_renal_material = "observe_mineralized_renal_material"
  )

  for (variable in names(mask_map)) {
    indicator <- unname(mask_map[[variable]])
    observed[observed[[indicator]] == 0, variable] <- NA
  }

  rownames(observed) <- NULL
  observed
}

source_aligned_diagnostics <- function(source_data, selected_data, observed_data) {
  variables <- c(
    "baseline_fitness",
    "age_z",
    "individual_factors",
    "bone_remodeling",
    "hydration",
    "urine_concentration",
    "urine_chemistry",
    "mineralized_renal_material",
    "nephrolithiasis",
    "medical_illness",
    "loss_of_mission_objectives"
  )

  summarize_one <- function(data, dataset) {
    do.call(rbind, lapply(intersect(variables, names(data)), function(variable) {
      data.frame(
        dataset = dataset,
        variable = variable,
        mean = mean(data[[variable]], na.rm = TRUE),
        missing_fraction = mean(is.na(data[[variable]])),
        stringsAsFactors = FALSE
      )
    }))
  }

  rbind(
    summarize_one(source_data, "source_population"),
    summarize_one(selected_data, "selected_pool"),
    summarize_one(observed_data, "observed_sparse_sample")
  )
}
