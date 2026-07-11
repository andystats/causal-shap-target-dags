suppressPackageStartupMessages(library(simcausal))

options(simcausal.verbose = FALSE)

# First-pass structural model for the renal-stone analysis sandbox.
#
# The mechanistic spine follows Figure 11.6 and the NASA renal-stone DAG
# narrative:
#   altered gravity -> bone remodeling -> urine chemistry
#   hostile closed environment -> hydration -> urine concentration
#   urine chemistry -> mineralized renal material -> nephrolithiasis
#   nephrolithiasis -> ureterolithiasis -> impaired urine flow
#   impaired urine flow -> renal complications -> mission consequences
#
# Coefficients are deliberately transparent placeholders. They are not NASA
# estimates and must be replaced or calibrated with domain experts.

build_renal_stone_dag <- function(nasa_like = FALSE) {
  dag <- DAG.empty()

  dag <- dag +
    node("baseline.fitness", distr = "rnorm", mean = 0, sd = 1) +
    node("individual.susceptibility", distr = "rnorm", mean = 0, sd = 1) +
    node("age.z", distr = "rnorm", mean = 0, sd = 1)

  if (nasa_like) {
    dag <- dag +
      node(
        "selected.astronaut",
        distr = "rbern",
        prob = plogis(
          0.50 +
            1.20 * baseline.fitness -
            1.10 * individual.susceptibility -
            0.35 * age.z
        )
      )
  } else {
    dag <- dag + node("selected.astronaut", distr = "rconst", const = 1)
  }

  dag <- dag +
    # Spaceflight hazards and background contributors.
    node("altered.gravity", distr = "rbern", prob = 0.70) +
    node("hostile.closed.environment", distr = "rbern", prob = 0.65) +
    node("co2.exposure", distr = "rbern", prob = 0.35) +
    node(
      "urinary.retention",
      distr = "rbern",
      prob = plogis(-2.20 + 0.45 * hostile.closed.environment)
    ) +
    node(
      "dietary.risk",
      distr = "rnorm",
      mean = 0.15 * hostile.closed.environment,
      sd = 1
    ) +
    node("microbiome.risk", distr = "rnorm", mean = 0, sd = 1) +

    # First-pass prevention/countermeasure levers.
    node("resistive.exercise", distr = "rbern", prob = 0.65) +
    node("bisphosphonate", distr = "rbern", prob = 0.18) +
    node("adequate.water.intake", distr = "rbern", prob = 0.60) +
    node("potassium.citrate", distr = "rbern", prob = 0.15) +
    node("thiazide", distr = "rbern", prob = 0.10) +

    # Mechanistic renal-stone pathway.
    node(
      "bone.remodeling",
      distr = "rnorm",
      mean = 0.90 * altered.gravity +
        0.20 * age.z +
        0.20 * individual.susceptibility -
        0.55 * resistive.exercise -
        0.65 * bisphosphonate,
      sd = 0.70
    ) +
    node(
      "hydration",
      distr = "rnorm",
      mean = 0.25 * baseline.fitness -
        0.75 * hostile.closed.environment +
        0.80 * adequate.water.intake,
      sd = 0.65
    ) +
    node(
      "urine.concentration",
      distr = "rnorm",
      mean = -0.95 * hydration + 0.20 * hostile.closed.environment,
      sd = 0.60
    ) +
    node(
      "urine.chemistry.risk",
      distr = "rnorm",
      mean = 0.80 * bone.remodeling +
        0.75 * urine.concentration +
        0.30 * dietary.risk +
        0.25 * microbiome.risk +
        0.35 * co2.exposure +
        0.20 * individual.susceptibility -
        0.65 * potassium.citrate -
        0.45 * thiazide,
      sd = 0.80
    ) +
    node(
      "mineralized.renal.material",
      distr = "rbern",
      prob = plogis(
        -2.85 +
          1.10 * urine.chemistry.risk +
          0.35 * individual.susceptibility
      )
    ) +
    node(
      "nephrolithiasis",
      distr = "rbern",
      prob = plogis(
        -2.70 +
          1.55 * mineralized.renal.material +
          0.45 * individual.susceptibility
      )
    ) +
    node(
      "ureterolithiasis",
      distr = "rbern",
      prob = plogis(-3.00 + 2.25 * nephrolithiasis)
    ) +
    node(
      "impaired.urine.flow",
      distr = "rbern",
      prob = plogis(
        -2.70 +
          2.10 * ureterolithiasis +
          1.10 * urinary.retention
      )
    ) +

    # Expanded medical-illness block from the Chapter 11 version of the DAG.
    node(
      "renal.colic",
      distr = "rbern",
      prob = plogis(
        -3.20 +
          1.80 * ureterolithiasis +
          0.85 * impaired.urine.flow
      )
    ) +
    node(
      "hydronephrosis",
      distr = "rbern",
      prob = plogis(-3.35 + 2.00 * impaired.urine.flow)
    ) +
    node(
      "infection",
      distr = "rbern",
      prob = plogis(
        -3.45 +
          1.45 * impaired.urine.flow +
          0.90 * ureterolithiasis
      )
    ) +
    node(
      "sepsis",
      distr = "rbern",
      prob = plogis(-4.70 + 2.45 * infection)
    ) +
    node(
      "renal.failure",
      distr = "rbern",
      prob = plogis(
        -4.70 +
          1.70 * hydronephrosis +
          2.10 * sepsis
      )
    ) +
    node(
      "medical.illness",
      distr = "rconst",
      const = ifelse(
        renal.colic + hydronephrosis + infection + sepsis + renal.failure > 0,
        1,
        0
      )
    ) +

    # Readiness and mission consequences.
    node(
      "individual.readiness.loss",
      distr = "rbern",
      prob = plogis(
        -3.20 +
          1.90 * medical.illness +
          0.80 * renal.colic +
          0.90 * renal.failure
      )
    ) +
    node(
      "evacuation",
      distr = "rbern",
      prob = plogis(
        -5.20 +
          1.35 * medical.illness +
          2.00 * sepsis +
          2.20 * renal.failure
      )
    ) +
    node(
      "crew.capability.loss",
      distr = "rbern",
      prob = plogis(
        -3.35 +
          2.20 * individual.readiness.loss +
          0.75 * medical.illness
      )
    ) +
    node(
      "task.performance.loss",
      distr = "rbern",
      prob = plogis(
        -3.25 +
          2.15 * crew.capability.loss +
          1.15 * evacuation
      )
    ) +
    node(
      "loss.mission.objectives",
      distr = "rbern",
      prob = plogis(
        -4.10 +
          2.35 * task.performance.loss +
          1.35 * evacuation
      )
    ) +
    node(
      "loss.mission",
      distr = "rbern",
      prob = plogis(-5.10 + 2.45 * loss.mission.objectives)
    ) +
    node(
      "loss.crew.life",
      distr = "rbern",
      prob = plogis(
        -7.00 +
          2.10 * sepsis +
          2.70 * renal.failure +
          1.80 * evacuation
      )
    ) +
    node(
      "long.term.health.outcome",
      distr = "rbern",
      prob = plogis(
        -4.20 +
          1.30 * medical.illness +
          2.10 * renal.failure
      )
    )

  if (nasa_like) {
    dag <- dag +
      # Informative and uneven measurement creates column-wise sparsity.
      node(
        "observe.bone.remodeling",
        distr = "rbern",
        prob = plogis(0.25 + 0.55 * medical.illness)
      ) +
      node(
        "observe.hydration",
        distr = "rbern",
        prob = plogis(0.80 + 0.35 * medical.illness)
      ) +
      node(
        "observe.urine.concentration",
        distr = "rbern",
        prob = plogis(0.10 + 0.65 * medical.illness)
      ) +
      node(
        "observe.urine.chemistry",
        distr = "rbern",
        prob = plogis(
          -0.35 +
            0.85 * nephrolithiasis +
            0.65 * medical.illness
        )
      ) +
      node(
        "observe.mineralized.material",
        distr = "rbern",
        prob = plogis(
          -1.00 +
            1.25 * nephrolithiasis +
            0.80 * medical.illness
        )
      )
  }

  set.DAG(dag)
}

simulate_renal_stone <- function(n, seed, nasa_like = FALSE) {
  dag <- build_renal_stone_dag(nasa_like = nasa_like)
  data <- sim(DAG = dag, n = n, rndseed = seed)
  names(data) <- gsub(".", "_", names(data), fixed = TRUE)
  data
}

make_nasa_like_observed <- function(source_data, n_observed = 450L, seed = 20260710L) {
  selected_pool <- source_data[source_data$selected_astronaut == 1, , drop = FALSE]

  if (nrow(selected_pool) < n_observed) {
    stop("The selected astronaut pool is smaller than n_observed.")
  }

  set.seed(seed)
  keep <- sample.int(nrow(selected_pool), size = n_observed, replace = FALSE)
  observed <- selected_pool[keep, , drop = FALSE]

  mask_map <- c(
    bone_remodeling = "observe_bone_remodeling",
    hydration = "observe_hydration",
    urine_concentration = "observe_urine_concentration",
    urine_chemistry_risk = "observe_urine_chemistry",
    mineralized_renal_material = "observe_mineralized_material"
  )

  for (variable in names(mask_map)) {
    indicator <- unname(mask_map[[variable]])
    observed[observed[[indicator]] == 0, variable] <- NA
  }

  rownames(observed) <- NULL
  observed
}

summarize_simulation <- function(data, dataset) {
  binary_variables <- c(
    "selected_astronaut",
    "altered_gravity",
    "hostile_closed_environment",
    "co2_exposure",
    "urinary_retention",
    "resistive_exercise",
    "bisphosphonate",
    "adequate_water_intake",
    "potassium_citrate",
    "thiazide",
    "mineralized_renal_material",
    "nephrolithiasis",
    "ureterolithiasis",
    "impaired_urine_flow",
    "renal_colic",
    "hydronephrosis",
    "infection",
    "sepsis",
    "renal_failure",
    "medical_illness",
    "individual_readiness_loss",
    "evacuation",
    "crew_capability_loss",
    "task_performance_loss",
    "loss_mission_objectives",
    "loss_mission",
    "loss_crew_life",
    "long_term_health_outcome"
  )

  variables <- intersect(
    c(
      "baseline_fitness",
      "individual_susceptibility",
      "bone_remodeling",
      "hydration",
      "urine_concentration",
      "urine_chemistry_risk",
      binary_variables
    ),
    names(data)
  )

  do.call(
    rbind,
    lapply(variables, function(variable) {
      values <- data[[variable]]
      data.frame(
        dataset = dataset,
        variable = variable,
        statistic = if (variable %in% binary_variables) "proportion_1" else "mean",
        value = mean(values, na.rm = TRUE),
        missing_fraction = mean(is.na(values)),
        stringsAsFactors = FALSE
      )
    })
  )
}

selection_diagnostics <- function(source_data, selected_data, observed_data) {
  source_summary <- summarize_simulation(source_data, "source_population")
  selected_summary <- summarize_simulation(selected_data, "selected_pool")
  observed_summary <- summarize_simulation(observed_data, "observed_sparse_sample")

  summaries <- rbind(source_summary, selected_summary, observed_summary)

  correlations <- data.frame(
    dataset = c("source_population", "selected_pool", "observed_sparse_sample"),
    variable = "cor_baseline_fitness_individual_susceptibility",
    statistic = "correlation",
    value = c(
      cor(source_data$baseline_fitness, source_data$individual_susceptibility),
      cor(selected_data$baseline_fitness, selected_data$individual_susceptibility),
      cor(observed_data$baseline_fitness, observed_data$individual_susceptibility)
    ),
    missing_fraction = 0,
    stringsAsFactors = FALSE
  )

  rbind(summaries, correlations)
}
