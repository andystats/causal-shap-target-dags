"""Choosing which intervention to actually pay for.

Attribution says where influence sits; this module says which affordable action
is worth testing. Two rules keep it honest.

First, benefit is never read off a Shapley value. Every candidate is scored by
running the structural model under ``do(a)`` and differencing against the same
exogenous draws, so the number reported is an intervention effect rather than a
credit allocation. Common random numbers come free: ``simulate`` replaces an
intervened node's equation while every other node reuses its abducted draw, so
the contrast is paired and its Monte Carlo error is tiny.

Second, the objective is a constraint, not a ratio. Maximising
``benefit / cost`` looks natural and is degenerate under linear costs: benefit
and cost both scale with the dose, so the ratio is dose-invariant and every
infinitesimally small cheap action ties for first. The optimum here is the
largest expected benefit that fits the budget and clears a probability-of-benefit
floor; the ratio is still reported, as a diagnostic column, because the two
disagree in informative ways.

Nodes that are not ancestors of the outcome are screened out *by construction*,
not by estimation. That is what removes an outcome-adjacent proxy which ordinary
attribution over-credits: its total effect is structurally zero, so no amount of
money spent on it can move the outcome.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .action_costs import CostModel
from .seeds import SEED_ACTION_ABDUCTION, SEED_ACTION_SEARCH
from .structural_value import ExogenousDraws, LinearLogisticSCM
from .validation.estimands import _summarize

DIRECTIONS = ("increase", "decrease")

NOT_MANIPULABLE = "not-manipulable"
NOT_AN_ANCESTOR = "not-an-ancestor"
IS_OUTCOME = "is-outcome"
NO_ALLOWED_SHIFT = "no-allowed-shift"
OVER_BUDGET = "over-budget"
BELOW_BENEFIT_CONFIDENCE = "below-benefit-confidence"


@dataclass(frozen=True)
class ScreenedNode:
    node: str
    screened_out: str


@dataclass(frozen=True)
class InterventionProblem:
    """The decision: whose outcome, moved which way, under what price sheet."""

    scm: LinearLogisticSCM
    outcome: str
    cost_model: CostModel
    graph: nx.DiGraph | None = None
    direction: str = "increase"
    alpha: float = 0.05
    n_undirected_pairs: int = 0

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"Unknown direction: {self.direction!r}")
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(f"alpha must lie in [0, 1): {self.alpha}")
        if self.outcome not in self.scm.specs:
            raise ValueError(f"Outcome {self.outcome} is not a node of the model")

    def structure(self) -> nx.DiGraph:
        return self.scm.graph if self.graph is None else self.graph


@dataclass(frozen=True)
class ActionEvaluation:
    """One candidate action, priced and scored."""

    action: Mapping[str, float]
    touched: tuple[str, ...]
    benefit: float
    benefit_se: float
    cost: float
    ratio: float
    p_unit_benefit: float
    p_mean_benefit: float
    feasible: bool
    screened_out: str = ""

    @property
    def label(self) -> str:
        return ", ".join(f"{node}{shift:+g}" for node, shift in sorted(self.action.items()))


@dataclass(frozen=True)
class ActionRanking:
    """Every candidate considered, plus what was refused and why."""

    evaluations: tuple[ActionEvaluation, ...]
    screened: tuple[ScreenedNode, ...]
    baseline_outcome_mean: float
    n_units: int
    search_mode: str
    n_undirected_pairs: int
    budget: float | None
    alpha: float
    cost_provenance: str
    seed: int
    direction: str = "increase"

    @property
    def n_candidates_evaluated(self) -> int:
        return len(self.evaluations)

    def feasible(self) -> tuple[ActionEvaluation, ...]:
        return tuple(item for item in self.evaluations if item.feasible)

    def best(self) -> ActionEvaluation | None:
        """The affordable action with the largest expected benefit."""
        feasible = self.feasible()
        return feasible[0] if feasible else None

    def pareto_frontier(self) -> tuple[ActionEvaluation, ...]:
        """Affordable actions no other affordable action beats on both axes."""
        feasible = sorted(self.feasible(), key=lambda item: (item.cost, -item.benefit))
        frontier: list[ActionEvaluation] = []
        best_benefit = -math.inf
        for item in feasible:
            if item.benefit > best_benefit:
                frontier.append(item)
                best_benefit = item.benefit
        return tuple(frontier)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "action": item.label,
                    "touched": ", ".join(item.touched),
                    "benefit": item.benefit,
                    "benefit_se": item.benefit_se,
                    "cost": item.cost,
                    "ratio": item.ratio,
                    "p_unit_benefit": item.p_unit_benefit,
                    "p_mean_benefit": item.p_mean_benefit,
                    "feasible": item.feasible,
                    "screened_out": item.screened_out,
                }
                for item in self.evaluations
            ]
        )

    def screened_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"node": item.node, "screened_out": item.screened_out} for item in self.screened]
        )


def screen_nodes(
    problem: InterventionProblem,
    *,
    screen_non_ancestors: bool = True,
) -> tuple[tuple[str, ...], tuple[ScreenedNode, ...]]:
    """Split the cost sheet into nodes worth searching and nodes refused, with reasons.

    Nothing is dropped silently: every refusal is returned with the rule that
    caused it, so a reviewer can see that the outcome-adjacent proxy was excluded
    for having no causal path rather than for being expensive.
    """
    graph = problem.structure()
    ancestors = nx.ancestors(graph, problem.outcome) if problem.outcome in graph else set()

    eligible: list[str] = []
    screened: list[ScreenedNode] = []
    for node in sorted(problem.cost_model.specs):
        spec = problem.cost_model.specs[node]
        if node == problem.outcome:
            screened.append(ScreenedNode(node, IS_OUTCOME))
        elif not spec.manipulable:
            screened.append(ScreenedNode(node, NOT_MANIPULABLE))
        elif screen_non_ancestors and node not in ancestors:
            screened.append(ScreenedNode(node, NOT_AN_ANCESTOR))
        elif spec.min_shift == 0.0 and spec.max_shift == 0.0:
            screened.append(ScreenedNode(node, NO_ALLOWED_SHIFT))
        else:
            eligible.append(node)
    return tuple(eligible), tuple(screened)


def default_grid(
    problem: InterventionProblem, nodes: Sequence[str]
) -> dict[str, tuple[float, ...]]:
    """One level per allowed direction of travel; deterministic and tiny."""
    grid: dict[str, tuple[float, ...]] = {}
    for node in nodes:
        spec = problem.cost_model.specs[node]
        levels = sorted({spec.max_shift, spec.min_shift} - {0.0})
        if levels:
            grid[node] = tuple(levels)
    return grid


def evaluate_action(
    problem: InterventionProblem,
    exogenous: ExogenousDraws,
    action: Mapping[str, float],
    *,
    baseline: Mapping[str, np.ndarray] | None = None,
) -> ActionEvaluation:
    """Score one action by simulating ``do(a)`` against the same exogenous draws."""
    touched = tuple(sorted(node for node, shift in action.items() if shift != 0.0))
    base = dict(baseline) if baseline is not None else problem.scm.simulate(exogenous)

    interventions = {node: base[node] + action[node] for node in touched}
    post = problem.scm.simulate(exogenous, interventions=interventions)

    delta = post[problem.outcome] - base[problem.outcome]
    if problem.direction == "decrease":
        delta = -delta

    summary = _summarize(delta, "action")
    benefit = float(summary["value"])
    benefit_se = float(summary["mc_std_error"])
    cost = problem.cost_model.cost(action)

    p_unit_benefit = float(np.mean(delta > 0.0))
    if benefit_se > 0.0:
        p_mean_benefit = _normal_cdf(benefit / benefit_se)
    else:
        p_mean_benefit = 1.0 if benefit > 0.0 else 0.0

    within_budget = problem.cost_model.budget is None or cost <= problem.cost_model.budget
    clears_floor = p_unit_benefit >= 1.0 - problem.alpha
    if not within_budget:
        screened_out = OVER_BUDGET
    elif not clears_floor:
        screened_out = BELOW_BENEFIT_CONFIDENCE
    else:
        screened_out = ""

    return ActionEvaluation(
        action=dict(action),
        touched=touched,
        benefit=benefit,
        benefit_se=benefit_se,
        cost=cost,
        ratio=benefit / cost if cost > 0 else math.inf,
        p_unit_benefit=p_unit_benefit,
        p_mean_benefit=p_mean_benefit,
        feasible=not screened_out,
        screened_out=screened_out,
    )


def rank_actions(
    problem: InterventionProblem,
    exogenous: ExogenousDraws,
    *,
    grid: Mapping[str, Sequence[float]] | None = None,
    max_arity: int = 1,
    screen_non_ancestors: bool = True,
    seed: int = SEED_ACTION_SEARCH,
) -> ActionRanking:
    """Exhaustively price a deterministic grid of actions and rank the affordable ones.

    Ranking is by expected benefit among actions that fit the budget and clear
    the probability-of-benefit floor. Infeasible candidates are kept in the
    result, carrying the constraint that rejected them.
    """
    if max_arity < 1:
        raise ValueError(f"max_arity must be at least 1: {max_arity}")

    eligible, screened = screen_nodes(problem, screen_non_ancestors=screen_non_ancestors)
    levels = (
        {node: tuple(values) for node, values in grid.items() if node in eligible}
        if grid is not None
        else default_grid(problem, eligible)
    )
    searchable = tuple(node for node in eligible if levels.get(node))

    baseline = problem.scm.simulate(exogenous)
    evaluations: list[ActionEvaluation] = []
    for arity in range(1, max_arity + 1):
        for combination in itertools.combinations(searchable, arity):
            for choice in itertools.product(*(levels[node] for node in combination)):
                action = dict(zip(combination, choice))
                evaluations.append(
                    evaluate_action(problem, exogenous, action, baseline=baseline)
                )

    evaluations.sort(key=lambda item: (not item.feasible, -item.benefit, item.cost, item.label))

    return ActionRanking(
        evaluations=tuple(evaluations),
        screened=screened,
        baseline_outcome_mean=float(np.mean(baseline[problem.outcome])),
        n_units=exogenous.n,
        search_mode="exhaustive",
        n_undirected_pairs=problem.n_undirected_pairs,
        budget=problem.cost_model.budget,
        alpha=problem.alpha,
        cost_provenance=problem.cost_model.provenance,
        seed=seed,
        direction=problem.direction,
    )


def abduct(
    scm: LinearLogisticSCM,
    data: pd.DataFrame,
    *,
    seed: int = SEED_ACTION_ABDUCTION,
) -> ExogenousDraws:
    """Recover the exogenous draws behind an observed cohort.

    Binary abduction samples a latent uniform consistent with the observed 0/1,
    so it is seed-dependent; repeat across seeds and report the spread before
    treating any single binary-heavy result as settled.
    """
    return scm.recover_exogenous(data, seed=seed)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
