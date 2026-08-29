from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from causal_shap.action_costs import ActionSpec, CostModel
from causal_shap.policy import (
    IS_OUTCOME,
    NOT_AN_ANCESTOR,
    OVER_BUDGET,
    InterventionProblem,
    abduct,
    rank_actions,
)
from causal_shap.seeds import SEED_ACTION_ABDUCTION
from causal_shap.teaching_dags import simulate_dataframe, toy_chain_fork_collider

# Toy DAG total effects, exact for a linear SCM (teaching_dags._linear_total_effects):
#   te(Hydration)   = 1.0                    Hydration -> Y
#   te(Diet)        = 0.8 * 1.0     = 0.8    Diet -> Hydration -> Y
#   te(Climate)     = 0.6 * 1.0 + 0.5 = 1.1  Climate -> Hydration -> Y, Climate -> Y
#   te(ClinicVisit) = 0.0                    caused by Y; no children
UNIT_SHIFT = 1.0
GRID = {
    "Hydration": (UNIT_SHIFT,),
    "Diet": (UNIT_SHIFT,),
    "Climate": (UNIT_SHIFT,),
    "ClinicVisit": (UNIT_SHIFT,),
}

# Prices chosen so that cost ordering and effect ordering deliberately disagree:
#   Hydration  benefit 1.0  cost 0.0 + 1.0 = 1.0   ratio 1.000
#   Diet       benefit 0.8  cost 0.3 + 1.0 = 1.3   ratio 0.615
#   Climate    benefit 1.1  cost 0.5 + 1.0 = 1.5   ratio 0.733
COST_SPECS = {
    "Hydration": ActionSpec("Hydration", True, -1.0, 1.0, fixed_cost=0.0, unit_cost=1.0),
    "Diet": ActionSpec("Diet", True, -1.0, 1.0, fixed_cost=0.3, unit_cost=1.0),
    "Climate": ActionSpec("Climate", True, -1.0, 1.0, fixed_cost=0.5, unit_cost=1.0),
    # Deliberately manipulable, so the ancestor screen -- not the mutability
    # flag -- is what removes the zero-effect collider proxy.
    "ClinicVisit": ActionSpec("ClinicVisit", True, -1.0, 1.0, fixed_cost=0.0, unit_cost=1.0),
    "Y": ActionSpec("Y", manipulable=False),
}


def toy_problem(budget: float | None, *, direction: str = "increase") -> InterventionProblem:
    dag = toy_chain_fork_collider()
    return InterventionProblem(
        scm=dag.scm(),
        outcome=dag.outcome,
        cost_model=CostModel(specs=dict(COST_SPECS), budget=budget),
        direction=direction,
    )


class PolicyBenefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dag = toy_chain_fork_collider()
        cls.data = simulate_dataframe(dag, n=2000, seed=SEED_ACTION_ABDUCTION)
        cls.exogenous = abduct(dag.scm(), cls.data)

    def rank(self, budget: float | None, **kwargs: object) -> object:
        return rank_actions(toy_problem(budget), self.exogenous, grid=GRID, **kwargs)

    def benefits(self, ranking) -> dict[str, float]:
        return {item.label: item.benefit for item in ranking.evaluations}

    def test_benefit_reproduces_the_exact_linear_total_effects(self) -> None:
        benefits = self.benefits(self.rank(None))
        self.assertAlmostEqual(benefits["Hydration+1"], 1.0, places=12)
        self.assertAlmostEqual(benefits["Diet+1"], 0.8, places=12)
        self.assertAlmostEqual(benefits["Climate+1"], 1.1, places=12)

    def test_zero_effect_proxy_is_screened_for_having_no_causal_path(self) -> None:
        ranking = self.rank(None)
        reasons = {item.node: item.screened_out for item in ranking.screened}
        self.assertEqual(reasons["ClinicVisit"], NOT_AN_ANCESTOR)
        # The outcome is refused for being the outcome, which is more
        # informative than refusing it for being immovable.
        self.assertEqual(reasons["Y"], IS_OUTCOME)
        self.assertNotIn("ClinicVisit+1", self.benefits(ranking))

    def test_forced_proxy_buys_exactly_nothing(self) -> None:
        ranking = self.rank(None, screen_non_ancestors=False)
        self.assertEqual(self.benefits(ranking)["ClinicVisit+1"], 0.0)

    def test_probability_of_benefit_is_one_for_a_deterministic_shift(self) -> None:
        # A linear SCM under common random numbers shifts every unit equally,
        # so the whole population benefits and the MC error is ~0.
        best = self.rank(None).best()
        self.assertEqual(best.p_unit_benefit, 1.0)
        self.assertAlmostEqual(best.benefit_se, 0.0, places=12)

    def test_decrease_direction_negates_benefit_and_reverses_the_ranking(self) -> None:
        ranking = rank_actions(
            toy_problem(None, direction="decrease"), self.exogenous, grid=GRID
        )
        benefits = self.benefits(ranking)
        self.assertAlmostEqual(benefits["Hydration+1"], -1.0, places=12)
        self.assertAlmostEqual(benefits["Climate+1"], -1.1, places=12)
        # Nothing helps, so nothing clears the probability-of-benefit floor.
        self.assertEqual(ranking.feasible(), ())


class PolicyBudgetTests(unittest.TestCase):
    """The price thesis: the biggest effect wins only once it is affordable."""

    @classmethod
    def setUpClass(cls) -> None:
        dag = toy_chain_fork_collider()
        cls.data = simulate_dataframe(dag, n=2000, seed=SEED_ACTION_ABDUCTION)
        cls.exogenous = abduct(dag.scm(), cls.data)

    def rank(self, budget: float | None):
        return rank_actions(toy_problem(budget), self.exogenous, grid=GRID)

    def test_budget_1_2_affords_only_hydration(self) -> None:
        ranking = self.rank(1.2)
        self.assertEqual([item.label for item in ranking.feasible()], ["Hydration+1"])
        self.assertEqual(ranking.best().label, "Hydration+1")
        self.assertAlmostEqual(ranking.best().benefit, 1.0, places=12)

    def test_budget_1_4_adds_diet_but_hydration_still_wins(self) -> None:
        ranking = self.rank(1.4)
        self.assertEqual(
            sorted(item.label for item in ranking.feasible()), ["Diet+1", "Hydration+1"]
        )
        self.assertEqual(ranking.best().label, "Hydration+1")

    def test_budget_1_5_finally_affords_climate_and_the_optimum_flips(self) -> None:
        ranking = self.rank(1.5)
        self.assertEqual(len(ranking.feasible()), 3)
        self.assertEqual(ranking.best().label, "Climate+1")
        self.assertAlmostEqual(ranking.best().benefit, 1.1, places=12)

    def test_over_budget_candidates_are_kept_with_their_reason(self) -> None:
        ranking = self.rank(1.2)
        rejected = {
            item.label: item.screened_out for item in ranking.evaluations if not item.feasible
        }
        self.assertEqual(rejected, {"Diet+1": OVER_BUDGET, "Climate+1": OVER_BUDGET})

    def test_ratio_and_constrained_optimum_disagree(self) -> None:
        # Exactly why benefit/cost is reported as a diagnostic and not optimised:
        # at budget 1.5 the ratio still prefers Hydration while the constrained
        # optimum is Climate.
        ranking = self.rank(1.5)
        by_ratio = max(ranking.feasible(), key=lambda item: item.ratio)
        self.assertEqual(by_ratio.label, "Hydration+1")
        self.assertEqual(ranking.best().label, "Climate+1")

    def test_pareto_frontier_trades_cost_against_benefit(self) -> None:
        frontier = [item.label for item in self.rank(1.5).pareto_frontier()]
        # Diet costs more than Hydration and buys less, so it is dominated.
        self.assertEqual(frontier, ["Hydration+1", "Climate+1"])

    def test_ranking_is_deterministic(self) -> None:
        pd.testing.assert_frame_equal(self.rank(1.5).to_frame(), self.rank(1.5).to_frame())


class CostModelTests(unittest.TestCase):
    def test_cost_is_fixed_plus_unit_times_distance_moved(self) -> None:
        model = CostModel(specs=dict(COST_SPECS))
        # Diet: 0.3 + 1.0*|0.5| = 0.8;  Climate: 0.5 + 1.0*|-1.0| = 1.5
        self.assertAlmostEqual(model.cost({"Diet": 0.5, "Climate": -1.0}), 2.3, places=12)

    def test_untouched_nodes_are_free(self) -> None:
        model = CostModel(specs=dict(COST_SPECS))
        self.assertEqual(model.cost({"Diet": 0.0}), 0.0)

    def test_rejects_immovable_node_and_out_of_range_shift(self) -> None:
        model = CostModel(specs=dict(COST_SPECS))
        with self.assertRaisesRegex(ValueError, "not manipulable"):
            model.cost({"Y": 1.0})
        with self.assertRaisesRegex(ValueError, "outside the allowed range"):
            model.cost({"Diet": 5.0})

    def test_rejects_inverted_range_and_negative_cost(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds max_shift"):
            ActionSpec("A", True, min_shift=1.0, max_shift=-1.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ActionSpec("A", True, fixed_cost=-1.0)

    def test_from_csv_round_trip(self) -> None:
        rows = "node,manipulable,min_shift,max_shift,fixed_cost,unit_cost\n"
        rows += "Hydration,true,-1,1,0,1\n"
        rows += "Y,false,0,0,0,0\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "costs.csv"
            path.write_text(rows, encoding="utf-8")
            model = CostModel.from_csv(path, budget=2.0)
        self.assertEqual(model.manipulable_nodes(), ("Hydration",))
        self.assertAlmostEqual(model.cost({"Hydration": 1.0}), 1.0, places=12)
        self.assertEqual(model.budget, 2.0)
        self.assertIn("ILLUSTRATIVE", model.provenance)


if __name__ == "__main__":
    unittest.main()
