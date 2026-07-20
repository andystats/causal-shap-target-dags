"""Credence-style, spec-driven simulation validation with layered parameters.

Torch-free by default: MVN/bootstrap generators, spec, estimands, and pipeline
import without torch. The CVAE generator lazily imports torch when used.
"""

from .estimands import naive_difference_in_means, true_estimands
from .generators import BootstrapGenerator, CVAEGenerator, MVNGenerator
from .pipeline import (
    Scenario,
    SimulatedDataset,
    ValidationSuite,
    fit_generator,
    generate_validation_suite,
    reference_covariates,
    simulate,
)
from .spec import (
    ConfounderSpec,
    SimulationSpec,
    additive_bias,
    constant_tau,
    fit_baseline,
    interaction_tau,
    linear_tau,
)

__all__ = [
    "BootstrapGenerator",
    "CVAEGenerator",
    "ConfounderSpec",
    "MVNGenerator",
    "Scenario",
    "SimulatedDataset",
    "SimulationSpec",
    "ValidationSuite",
    "additive_bias",
    "constant_tau",
    "fit_baseline",
    "fit_generator",
    "generate_validation_suite",
    "interaction_tau",
    "linear_tau",
    "naive_difference_in_means",
    "reference_covariates",
    "simulate",
    "true_estimands",
]
