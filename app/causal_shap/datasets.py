"""Shared dataset registry for the build stages.

Each entry knows how to produce its observational data, its ground-truth graph,
its outcome, and where its bundle lives, so the build stages iterate uniformly
instead of duplicating per-dataset wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx
import pandas as pd

from .graphs import load_edges_csv
from .seeds import SEED_TEACHING_LADDER, SEED_TEACHING_TOY
from .teaching_dags import (
    TeachingDAG,
    layered_ladder,
    simulate_dataframe,
    toy_chain_fork_collider,
)

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent

BUNDLES_DIR = APP_DIR / "bundles"
TEACHING_ROWS = 6000


@dataclass(frozen=True)
class Dataset:
    bundle_id: str
    outcome: str
    live_discovery: bool  # small enough for live PC/GES in the app
    load_data: Callable[[], pd.DataFrame]
    load_graph: Callable[[], nx.DiGraph]
    load_true_effects: Callable[[], dict[str, float]]

    @property
    def bundle_dir(self) -> Path:
        return BUNDLES_DIR / self.bundle_id

    @property
    def stage_dir(self) -> Path:
        return self.bundle_dir / "stages"


def _teaching_dataset(dag: TeachingDAG, bundle_id: str, seed: int) -> Dataset:
    def load_data() -> pd.DataFrame:
        path = BUNDLES_DIR / bundle_id / "data.csv"
        if path.exists():
            return pd.read_csv(path)
        return simulate_dataframe(dag, TEACHING_ROWS, seed)

    return Dataset(
        bundle_id=bundle_id,
        outcome=dag.outcome,
        live_discovery=True,
        load_data=load_data,
        load_graph=lambda: dag.graph,
        load_true_effects=lambda: dict(dag.true_total_effects),
    )


def _acic_dataset() -> Dataset:
    bundle_dir = BUNDLES_DIR / "acic_proxy_stress_test"

    def load_data() -> pd.DataFrame:
        return pd.read_csv(bundle_dir / "data.csv")

    def load_graph() -> nx.DiGraph:
        return load_edges_csv(bundle_dir / "edges.csv")

    def load_true_effects() -> dict[str, float]:
        from .pedagogic import TRUE_TOTAL_EFFECTS

        return dict(TRUE_TOTAL_EFFECTS)

    return Dataset(
        bundle_id="acic_proxy_stress_test",
        outcome="AcuteRisk",
        live_discovery=True,
        load_data=load_data,
        load_graph=load_graph,
        load_true_effects=load_true_effects,
    )


def teaching_datasets() -> list[Dataset]:
    return [
        _teaching_dataset(toy_chain_fork_collider(), "toy_chain_fork_collider", SEED_TEACHING_TOY),
        _teaching_dataset(layered_ladder(), "layered_ladder", SEED_TEACHING_LADDER),
    ]


def discovery_datasets() -> list[Dataset]:
    """Datasets small enough to run live discovery on (teaching + ACIC)."""
    return teaching_datasets() + [_acic_dataset()]
