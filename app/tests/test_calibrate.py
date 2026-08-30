from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from causal_shap.calibrate import fit_linear_logistic_scm
from causal_shap.seeds import SEED_SCM_CALIBRATION
from causal_shap.structural_value import ExogenousDraws, NodeSpec
from causal_shap.teaching_dags import (
    TeachingDAG,
    simulate_dataframe,
    toy_chain_fork_collider,
)


def toy_fit(n: int = 20000):
    dag = toy_chain_fork_collider()
    data = simulate_dataframe(dag, n=n, seed=SEED_SCM_CALIBRATION)
    return dag, data, fit_linear_logistic_scm(data, dag.graph, seed=SEED_SCM_CALIBRATION)


def binary_dag() -> TeachingDAG:
    """Binary root feeding one continuous and one binary child."""
    specs = (
        NodeSpec("Exposure", "binary", root_probability=0.35),
        NodeSpec("Y", "continuous", parents=("Exposure",), coefficients=(1.5,), noise_sd=0.5),
        NodeSpec(
            "Event", "binary", parents=("Exposure", "Y"),
            coefficients=(0.8, 0.6), intercept=-1.0,
        ),
    )
    return TeachingDAG(
        name="binary_fixture", specs=specs, outcome="Y",
        features=("Exposure", "Event"), true_total_effects={"Exposure": 1.5, "Event": 0.0},
    )


class CalibrateToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dag, cls.data, cls.fitted = toy_fit()
        cls.specs = cls.fitted.scm.specs

    def test_recovers_the_known_coefficients_within_mc_error(self) -> None:
        # True equations (teaching_dags.py): Hydration = 0.8*Diet + 0.6*Climate + 0.5*eps,
        # Y = 1.0*Hydration + 0.5*Climate + 0.5*eps. OLS SEs at n=20000 are ~0.004,
        # so a 0.03 tolerance is ~7 sigma.
        hydration = self.specs["Hydration"]
        self.assertEqual(hydration.parents, ("Climate", "Diet"))  # sorted parent order
        by_name = dict(zip(hydration.parents, hydration.coefficients))
        self.assertAlmostEqual(by_name["Diet"], 0.8, delta=0.03)
        self.assertAlmostEqual(by_name["Climate"], 0.6, delta=0.03)
        self.assertAlmostEqual(hydration.noise_sd, 0.5, delta=0.03)

        y = dict(zip(self.specs["Y"].parents, self.specs["Y"].coefficients))
        self.assertAlmostEqual(y["Hydration"], 1.0, delta=0.03)
        self.assertAlmostEqual(y["Climate"], 0.5, delta=0.03)

    def test_diagnostics_report_the_population_r2_per_node(self) -> None:
        # Var(Hydration) = .64*1 + .36*1 + .25 = 1.25 -> R2 = 1.00/1.25 = 0.800
        # Var(lin Y) = 1*1.25 + .25*1 + 2*.5*.6 = 2.10; Var(Y) = 2.35 -> R2 = 0.894
        r2 = self.fitted.diagnostics.set_index("node")["fit_value"]
        self.assertAlmostEqual(r2["Hydration"], 0.800, delta=0.02)
        self.assertAlmostEqual(r2["Y"], 0.894, delta=0.02)

    def test_roots_are_fit_as_marginals_with_means_recorded(self) -> None:
        # The engine ignores root intercepts, so calibration must not pretend
        # otherwise: sd goes into noise_sd, the mean is informational only.
        self.assertEqual(self.specs["Diet"].parents, ())
        self.assertAlmostEqual(self.specs["Diet"].noise_sd, 1.0, delta=0.03)
        self.assertIn("Diet", self.fitted.root_means)
        self.assertAlmostEqual(self.fitted.root_means["Diet"], 0.0, delta=0.03)

    def test_replay_reproduces_the_observed_data_exactly(self) -> None:
        # The whole point of grade="replay": abduction absorbs root locations,
        # so abduct-then-simulate round-trips every continuous column.
        scm = self.fitted.scm
        exogenous = scm.recover_exogenous(self.data, seed=SEED_SCM_CALIBRATION)
        replayed = scm.simulate(exogenous)
        for node in scm.order:
            np.testing.assert_allclose(
                replayed[node], self.data[node].to_numpy(dtype=float), atol=1e-9
            )

    def test_grade_and_assumptions_travel_with_the_result(self) -> None:
        self.assertEqual(self.fitted.grade, "replay")
        self.assertTrue(any("generation" in a for a in self.fitted.assumptions))


class CalibrateBinaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dag = binary_dag()
        cls.data = simulate_dataframe(dag, n=20000, seed=SEED_SCM_CALIBRATION)
        cls.fitted = fit_linear_logistic_scm(cls.data, dag.graph, seed=SEED_SCM_CALIBRATION)

    def test_binary_root_rate_is_recovered(self) -> None:
        self.assertAlmostEqual(self.specs()["Exposure"].root_probability, 0.35, delta=0.02)

    def test_binary_child_coefficients_are_recovered(self) -> None:
        # Logistic fits are noisier than OLS: loose, directional tolerances.
        event = dict(zip(self.specs()["Event"].parents, self.specs()["Event"].coefficients))
        self.assertAlmostEqual(event["Exposure"], 0.8, delta=0.15)
        self.assertAlmostEqual(event["Y"], 0.6, delta=0.15)
        self.assertAlmostEqual(self.specs()["Event"].intercept, -1.0, delta=0.15)

    def specs(self):
        return self.fitted.scm.specs


class CalibrateValidationTests(unittest.TestCase):
    def test_rejects_two_valued_columns_not_coded_0_1(self) -> None:
        # {-1, 1} would calibrate cleanly and replay wrongly, since abduction
        # treats only the literal 1 as positive. Fail loudly instead.
        dag = toy_chain_fork_collider()
        data = simulate_dataframe(dag, n=200, seed=SEED_SCM_CALIBRATION)
        data["Diet"] = np.where(data["Diet"] > 0, 1.0, -1.0)
        with self.assertRaisesRegex(ValueError, "not coded 0/1"):
            fit_linear_logistic_scm(data, dag.graph, seed=SEED_SCM_CALIBRATION)

    def test_rejects_cyclic_graph_and_missing_columns(self) -> None:
        import networkx as nx

        dag = toy_chain_fork_collider()
        data = simulate_dataframe(dag, n=200, seed=SEED_SCM_CALIBRATION)
        with self.assertRaisesRegex(ValueError, "acyclic"):
            fit_linear_logistic_scm(data, nx.DiGraph([("A", "B"), ("B", "A")]),
                                    seed=SEED_SCM_CALIBRATION)
        with self.assertRaisesRegex(ValueError, "missing graph nodes"):
            fit_linear_logistic_scm(data.drop(columns=["Climate"]), dag.graph,
                                    seed=SEED_SCM_CALIBRATION)

    def test_constant_column_is_floored_with_a_warning(self) -> None:
        dag = toy_chain_fork_collider()
        data = simulate_dataframe(dag, n=200, seed=SEED_SCM_CALIBRATION)
        data["Diet"] = 3.7
        fitted = fit_linear_logistic_scm(data, dag.graph, seed=SEED_SCM_CALIBRATION)
        self.assertTrue(any("constant" in warning for warning in fitted.warnings))

    def test_single_class_binary_child_gets_zero_effects_not_a_crash(self) -> None:
        # A rare child can go single-class after complete-case filtering;
        # logistic regression refuses one-class targets, so the fit falls back
        # to zero parent effects with a clipped-logit intercept.
        dag = binary_dag()
        data = simulate_dataframe(dag, n=400, seed=SEED_SCM_CALIBRATION)
        data["Event"] = 0.0
        fitted = fit_linear_logistic_scm(data, dag.graph, seed=SEED_SCM_CALIBRATION)
        event = fitted.scm.specs["Event"]
        self.assertEqual(event.coefficients, (0.0, 0.0))
        self.assertLess(event.intercept, -10.0)  # logit of a clipped near-zero rate
        self.assertTrue(any("single class" in warning for warning in fitted.warnings))

    def test_collinear_parents_are_flagged_as_not_identified(self) -> None:
        # Minimum-norm least squares happily splits one true coefficient
        # across two identical columns; the warning is what stops those
        # numbers being read as intervention effects.
        import networkx as nx

        rng = np.random.default_rng(SEED_SCM_CALIBRATION)
        a = rng.standard_normal(500)
        data = pd.DataFrame({"A": a, "B": a, "C": 1.5 * a + 0.1 * rng.standard_normal(500)})
        graph = nx.DiGraph([("A", "C"), ("B", "C")])
        fitted = fit_linear_logistic_scm(data, graph, seed=SEED_SCM_CALIBRATION)
        self.assertTrue(any("collinear" in warning for warning in fitted.warnings))


if __name__ == "__main__":
    unittest.main()
