"""Small pretend DAGs for the tutorial's lower rungs.

These are linear-Gaussian structural models built from the same ``NodeSpec``
engine as the NASA analysis, sized so that causal discovery and structural
Causal SHAP both run live in the app. Each DAG deliberately contains a node
that is predictive of the outcome yet has zero total causal effect on it — the
collider ``ClinicVisit`` and the outcome-adjacent proxies — so that ordinary
SHAP visibly over-credits it (the "homunculus" failure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import networkx as nx
import numpy as np
import pandas as pd

from .structural_value import ExogenousDraws, LinearLogisticSCM, NodeSpec


@dataclass(frozen=True)
class TeachingDAG:
    name: str
    specs: tuple[NodeSpec, ...]
    outcome: str
    features: tuple[str, ...]
    true_total_effects: Mapping[str, float]

    def __post_init__(self) -> None:
        names = {spec.name for spec in self.specs}
        if self.outcome not in names:
            raise ValueError(f"Outcome {self.outcome} is not a node of {self.name}")
        if set(self.features) - names:
            raise ValueError(f"Unknown feature(s) in {self.name}: {set(self.features) - names}")

    @property
    def graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(spec.name for spec in self.specs)
        for spec in self.specs:
            graph.add_edges_from((parent, spec.name) for parent in spec.parents)
        return graph

    def scm(self) -> LinearLogisticSCM:
        return LinearLogisticSCM(self.specs)


def _linear_total_effects(
    specs: tuple[NodeSpec, ...], outcome: str, order: tuple[str, ...]
) -> dict[str, float]:
    """Exact total effect of each node on the outcome for a linear SCM.

    For a linear system the total effect is the sum over directed paths of the
    product of edge coefficients, computed here by the chain rule in reverse
    topological order (``order`` from the SCM): te(outcome)=1,
    te(node)=sum_children coef * te(child).
    """
    children: dict[str, list[tuple[str, float]]] = {spec.name: [] for spec in specs}
    for spec in specs:
        for parent, coefficient in zip(spec.parents, spec.coefficients):
            children[parent].append((spec.name, coefficient))

    effects: dict[str, float] = {}
    for node in reversed(order):
        if node == outcome:
            effects[node] = 1.0
        else:
            effects[node] = sum(coef * effects[child] for child, coef in children[node])
    return effects


def _build(name: str, outcome: str, specs: tuple[NodeSpec, ...]) -> TeachingDAG:
    scm = LinearLogisticSCM(specs)  # validates acyclicity and parent references
    total_effects = _linear_total_effects(specs, outcome, scm.order)
    features = tuple(spec.name for spec in specs if spec.name != outcome)
    true_total_effects = {name: total_effects[name] for name in features}
    return TeachingDAG(
        name=name,
        specs=specs,
        outcome=outcome,
        features=features,
        true_total_effects=true_total_effects,
    )


def toy_chain_fork_collider() -> TeachingDAG:
    """Five-node blackboard DAG: a chain, a fork, and a collider proxy.

    Diet -> Hydration -> Y (chain), Climate -> {Hydration, Y} (fork), and
    Diet -> ClinicVisit <- Y (collider). ClinicVisit is caused by Y, so it is
    strongly predictive but has zero total effect on Y.
    """
    specs = (
        NodeSpec("Diet", "continuous", noise_sd=1.0),
        NodeSpec("Climate", "continuous", noise_sd=1.0),
        NodeSpec(
            "Hydration",
            "continuous",
            parents=("Diet", "Climate"),
            coefficients=(0.8, 0.6),
            noise_sd=0.5,
        ),
        NodeSpec(
            "Y",
            "continuous",
            parents=("Hydration", "Climate"),
            coefficients=(1.0, 0.5),
            noise_sd=0.5,
        ),
        NodeSpec(
            "ClinicVisit",
            "continuous",
            parents=("Diet", "Y"),
            coefficients=(0.7, 0.9),
            noise_sd=0.5,
        ),
    )
    return _build("toy_chain_fork_collider", "Y", specs)


def layered_ladder() -> TeachingDAG:
    """Nine-node, three-tier DAG bridging the toy graph and NASA/ACIC.

    Distal causes (Genetics, Diet, Environment) feed mediators (Metabolism,
    Inflammation, Perfusion) that feed the outcome Y; two outcome-adjacent
    proxies (LabProxy, MonitorProxy) are caused by Y and carry zero effect.
    """
    specs = (
        NodeSpec("Genetics", "continuous", noise_sd=1.0),
        NodeSpec("Diet", "continuous", noise_sd=1.0),
        NodeSpec("Environment", "continuous", noise_sd=1.0),
        NodeSpec(
            "Metabolism",
            "continuous",
            parents=("Genetics", "Diet"),
            coefficients=(0.7, 0.5),
            noise_sd=0.5,
        ),
        NodeSpec(
            "Inflammation",
            "continuous",
            parents=("Diet", "Environment"),
            coefficients=(0.6, 0.5),
            noise_sd=0.5,
        ),
        NodeSpec(
            "Perfusion",
            "continuous",
            parents=("Environment", "Genetics"),
            coefficients=(0.7, 0.4),
            noise_sd=0.5,
        ),
        NodeSpec(
            "Y",
            "continuous",
            parents=("Metabolism", "Inflammation", "Perfusion"),
            coefficients=(1.0, 0.8, 0.6),
            noise_sd=0.5,
        ),
        NodeSpec(
            "LabProxy",
            "continuous",
            parents=("Y",),
            coefficients=(0.9,),
            noise_sd=0.5,
        ),
        NodeSpec(
            "MonitorProxy",
            "continuous",
            parents=("Y",),
            coefficients=(0.8,),
            noise_sd=0.5,
        ),
    )
    return _build("layered_ladder", "Y", specs)


def simulate_dataframe(dag: TeachingDAG, n: int, seed: int) -> pd.DataFrame:
    """Draw an observational sample (all nodes) from a teaching DAG."""
    scm = dag.scm()
    rng = np.random.default_rng(seed)
    exogenous = ExogenousDraws({spec.name: rng.standard_normal(n) for spec in dag.specs})
    simulated = scm.simulate(exogenous)
    return pd.DataFrame({name: simulated[name] for name in scm.order})
