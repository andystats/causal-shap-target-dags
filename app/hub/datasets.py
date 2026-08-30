"""Bundled demo datasets for the hub, with their truth and their price sheets.

Everything computational is keyed by the data's own column names; prettier
display names are a rendering concern only. Each dataset also carries the
default decision framing (exposure, outcome, direction, confidence floor) and a
prefilled cost sheet, because a cost editor full of zeros demonstrates nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping

import networkx as nx
import pandas as pd

from causal_shap.action_costs import ActionSpec
from causal_shap.graph_state import GraphProvenance, GraphState
from causal_shap.graphs import load_edges_csv
from causal_shap.nasa_scm import build_nasa_renal_scm
from causal_shap.seeds import SEED_HUB_DEMO
from causal_shap.teaching_dags import simulate_dataframe, toy_chain_fork_collider

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKBENCH_DATA = REPO_ROOT / "app" / "workbench" / "data"
RENAL_DATA = REPO_ROOT / "analysis" / "output" / "source_aligned_clean"
RENAL_EDGES = REPO_ROOT / "analysis" / "output" / "dag_validation" / "validated_clean_source_edges.csv"
RENAL_TRUTH = REPO_ROOT / "analysis" / "output" / "shap_nephrolithiasis_clean_v3" / "interventional_truth.csv"


@dataclass(frozen=True)
class HubDataset:
    """One bundled demonstration, self-describing enough to drive every stage."""

    id: str
    label: str
    flags_id: str                                   # key into frozen detector tables
    load_data: Callable[[], pd.DataFrame]
    truth_graph: Callable[[], GraphState | None]
    truth_effects: Callable[[], dict[str, float] | None]
    default_exposure: str
    default_outcome: str
    direction: str = "increase"
    default_alpha: float = 0.05
    excluded_columns: tuple[str, ...] = ()
    cost_specs: Callable[[], dict[str, ActionSpec]] = lambda: {}
    display_names: Callable[[], Mapping[str, str]] = lambda: {}
    note: str = ""


def _bundle_provenance() -> GraphProvenance:
    return GraphProvenance(source="bundle")


# ---------------------------------------------------------------------------
# Toy trap: the manuscript's five-node teaching DAG
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _toy_data() -> pd.DataFrame:
    return simulate_dataframe(toy_chain_fork_collider(), n=2000, seed=SEED_HUB_DEMO)


def _toy_truth_graph() -> GraphState:
    return GraphState.from_digraph(toy_chain_fork_collider().graph, _bundle_provenance())


def _toy_truth_effects() -> dict[str, float]:
    return dict(toy_chain_fork_collider().true_total_effects)


def _toy_cost_specs() -> dict[str, ActionSpec]:
    """The exact fixture behind the budget-sweep story (test_policy.py).

    Fixed costs 0 / 0.3 / 0.5 make the optimum flip Hydration -> Climate as the
    budget crosses 1.5. ClinicVisit is deliberately manipulable so the ancestor
    screen, not the mutability flag, is what removes it.
    """
    return {
        "Hydration": ActionSpec("Hydration", True, -1.0, 1.0, fixed_cost=0.0, unit_cost=1.0),
        "Diet": ActionSpec("Diet", True, -1.0, 1.0, fixed_cost=0.3, unit_cost=1.0),
        "Climate": ActionSpec("Climate", True, -1.0, 1.0, fixed_cost=0.5, unit_cost=1.0),
        "ClinicVisit": ActionSpec("ClinicVisit", True, -1.0, 1.0, fixed_cost=0.0, unit_cost=1.0),
        "Y": ActionSpec("Y", manipulable=False),
    }


# ---------------------------------------------------------------------------
# Clinical teaching fixture: the frozen simcausal workbench sample
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _simcausal_data() -> pd.DataFrame:
    return pd.read_csv(WORKBENCH_DATA / "simcausal_train.csv")


def _simcausal_truth_graph() -> GraphState:
    graph = load_edges_csv(WORKBENCH_DATA / "ground_truth_edges.csv")
    return GraphState.from_digraph(graph, _bundle_provenance())


def _simcausal_truth_effects() -> dict[str, float]:
    raw = json.loads((WORKBENCH_DATA / "true_total_effects.json").read_text(encoding="utf-8"))
    return {name: float(value) for name, value in raw.items() if not name.startswith("_")}


def sd_cost_specs(data: pd.DataFrame, outcome: str) -> dict[str, ActionSpec]:
    """Continuous columns get +/- 1 SD shifts; binary levers stay set aside.

    Policy actions are shifts, and a +1 shift on a 0/1 column produces values
    outside its support; rather than fake set-point semantics, binary levers
    are marked non-manipulable and labelled future work. Non-ancestors are NOT
    pre-hidden here — letting the ancestor screen refuse them, with its reason
    on display, is the point of the screening table.
    """
    import numpy as np

    specs: dict[str, ActionSpec] = {}
    for node in data.columns:
        series = data[node].dropna()
        spread = float(series.std()) if len(series) > 1 else float("nan")
        if node == outcome:
            specs[node] = ActionSpec(node, manipulable=False)
        elif set(series.unique()) <= {0, 1}:
            specs[node] = ActionSpec(
                node, manipulable=False,
                ethical_note="binary lever: set-point actions are future work",
            )
        elif not np.isfinite(spread) or spread <= 0:
            specs[node] = ActionSpec(
                node, manipulable=False, ethical_note="no usable spread in the data"
            )
        else:
            specs[node] = ActionSpec(node, True, -spread, spread, fixed_cost=0.0, unit_cost=1.0)
    return specs


def _simcausal_cost_specs() -> dict[str, ActionSpec]:
    return sd_cost_specs(_simcausal_data(), "Outcome")


# ---------------------------------------------------------------------------
# Renal NASA-topology simulation (synthetic coefficients, source-aligned graph)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _renal_data() -> pd.DataFrame:
    return pd.read_csv(RENAL_DATA / "renal_stone_source_aligned_clean_v3.csv")


@lru_cache(maxsize=1)
def _renal_scm_graph() -> nx.DiGraph:
    # Already snake_case, matching the data columns exactly (verified): the SCM
    # builder does the source->variable mapping once, so no runtime join.
    scm = build_nasa_renal_scm(RENAL_EDGES, RENAL_DATA / "source_to_simulation_variable_map.csv")
    return scm.graph


def _renal_truth_graph() -> GraphState:
    return GraphState.from_digraph(_renal_scm_graph(), _bundle_provenance())


def _renal_truth_effects() -> dict[str, float]:
    truth = pd.read_csv(RENAL_TRUTH)
    return dict(zip(truth["variable"], truth["total_effect_risk_difference"].astype(float)))


def _renal_cost_specs() -> dict[str, ActionSpec]:
    data = _renal_data().drop(columns=["ID", "simulation_version"], errors="ignore")
    return sd_cost_specs(data, "nephrolithiasis")


@lru_cache(maxsize=1)
def _renal_display_names() -> Mapping[str, str]:
    mapping = pd.read_csv(RENAL_DATA / "source_to_simulation_variable_map.csv")
    return dict(zip(mapping["variable"], mapping["source_node"]))


DATASETS: dict[str, HubDataset] = {
    dataset.id: dataset
    for dataset in (
        HubDataset(
            id="toy_trap",
            label="Toy trap — chain, fork, collider (5 nodes)",
            flags_id="toy",
            load_data=_toy_data,
            truth_graph=_toy_truth_graph,
            truth_effects=_toy_truth_effects,
            default_exposure="Hydration",
            default_outcome="Y",
            cost_specs=_toy_cost_specs,
            note="The manuscript's teaching trap: ClinicVisit is caused by the "
                 "outcome, so it predicts strongly and moves nothing.",
        ),
        HubDataset(
            id="clinical_teaching",
            label="Clinical teaching sample (12 variables)",
            flags_id="simcausal",
            load_data=_simcausal_data,
            truth_graph=_simcausal_truth_graph,
            truth_effects=_simcausal_truth_effects,
            default_exposure="Treatment",
            default_outcome="Outcome",
            cost_specs=_simcausal_cost_specs,
            note="Frozen synthetic fixture with a 27-edge answer key; the "
                 "generator itself was not preserved, so it is a fixed "
                 "demonstration, not a regenerable result.",
        ),
        HubDataset(
            id="renal_nasa",
            label="Renal stones — NASA-topology simulation (51 nodes)",
            flags_id="renal",
            load_data=_renal_data,
            truth_graph=_renal_truth_graph,
            truth_effects=_renal_truth_effects,
            default_exposure="altered_gravity",
            default_outcome="nephrolithiasis",
            direction="decrease",
            default_alpha=0.999,
            excluded_columns=("ID", "simulation_version"),
            cost_specs=_renal_cost_specs,
            display_names=_renal_display_names,
            note="Source-aligned topology; every coefficient is a simulation "
                 "parameter, not a NASA estimate. Outcome is rare, so the "
                 "benefit-confidence floor defaults to permissive.",
        ),
    )
}

UPLOAD_ID = "upload"
