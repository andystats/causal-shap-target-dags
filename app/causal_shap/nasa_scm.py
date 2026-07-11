"""Construct the source-aligned NASA renal-stone structural model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .structural_value import LinearLogisticSCM, NodeSpec


CONTINUOUS_SOURCE_NODES = {
    "Individual Factors",
    "Nutrients",
    "Microbiome",
    "Bone Remodeling",
    "Hydration",
    "Urine Concentration",
    "Urine Chemistry",
    "Mineralized Renal Material",
    "Urine Flow",
}

SEVERE_OUTCOMES = {
    "Loss of Mission",
    "Loss of Crew Life",
    "Loss of Mission Objectives",
    "Evacuation",
    "Long Term Health Outcomes",
}

CLINICAL_OUTCOMES = {
    "Nephrolithiasis",
    "Ureterolithiasis",
    "Medical Illness",
    "Individual Readiness",
    "Crew Capability",
    "Task Performance",
}

NEGATIVE_EDGES = {
    ("Astronaut Selection", "Individual Factors"),
    ("Medical Prevention Capability", "Bone Remodeling"),
    ("Resistive Exercise", "Bone Remodeling"),
    ("Bisphosphonates", "Bone Remodeling"),
    ("Medical Prevention Capability", "Urine Chemistry"),
    ("K+ Citrate", "Urine Chemistry"),
    ("Thiazides", "Urine Chemistry"),
    ("Ultrasound Manipulation", "Ureterolithiasis"),
    ("Water Intake", "Urine Flow"),
    ("Tamsulosin", "Urine Flow"),
    ("Percutaneous Nephrostomy", "Medical Illness"),
    ("Medications", "Individual Readiness"),
    ("Medications", "Long Term Health Outcomes"),
    ("Medications", "Loss of Crew Life"),
    ("Medications", "Medical Illness"),
    ("Medications", "Evacuation"),
}


def binary_intercept(source_node: str) -> float:
    if source_node in SEVERE_OUTCOMES:
        return -4.0
    if source_node in CLINICAL_OUTCOMES:
        return -2.35
    return -0.65


def edge_coefficient(source: str, target: str) -> float:
    return -0.70 if (source, target) in NEGATIVE_EDGES else 0.60


def build_nasa_renal_scm(edges_path: Path, variable_map_path: Path) -> LinearLogisticSCM:
    edges = pd.read_csv(edges_path)
    variable_map = pd.read_csv(variable_map_path)
    source_to_variable = dict(zip(variable_map["source_node"], variable_map["variable"]))
    source_nodes = list(variable_map["source_node"])
    specs: list[NodeSpec] = []

    for source_node in source_nodes:
        parent_sources = list(edges.loc[edges["to"] == source_node, "from"])
        parents = tuple(source_to_variable[parent] for parent in parent_sources)
        coefficients = tuple(edge_coefficient(parent, source_node) for parent in parent_sources)
        kind = "continuous" if source_node in CONTINUOUS_SOURCE_NODES else "binary"
        specs.append(
            NodeSpec(
                name=source_to_variable[source_node],
                kind=kind,
                parents=parents,
                coefficients=coefficients,
                intercept=0.0 if kind == "continuous" else binary_intercept(source_node),
                noise_sd=1.0 if not parents else 0.75,
                root_probability=0.35,
            )
        )

    return LinearLogisticSCM(specs)
