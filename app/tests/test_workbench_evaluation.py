"""Focused tests for the Workbench evaluation and prediction boundary."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from causal_shap import evaluation
from workbench.attribution import adjustment_model_features, prediction_callable


APP_DIR = Path(__file__).resolve().parents[1]
WORKBENCH_DIR = APP_DIR / "workbench"


class WorkbenchBatteryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data_dir = WORKBENCH_DIR / "data"
        cls.data = pd.read_csv(data_dir / "simcausal_train.csv")
        edges = pd.read_csv(data_dir / "ground_truth_edges.csv")
        cls.graph_true = nx.DiGraph(
            edges[["from", "to"]].itertuples(index=False, name=None)
        )
        cls.exposure = "Treatment"
        cls.outcome = "Outcome"
        cls.graph_learned = cls.graph_true.copy()
        cls.graph_learned.remove_edge("Treatment", "Inflammation")
        cls.graph_learned.add_edge("Inflammation", "Treatment")
        cls.graph_learned.remove_edge("Age", "Creatinine")
        cls.graph_learned.add_edge("Sex", "Outcome")
        cls.true_effect = json.loads(
            (data_dir / "true_total_effects.json").read_text(encoding="utf-8")
        )["Treatment"]

    def test_m1_m2_and_m3_match_the_teaching_perturbation(self) -> None:
        m1 = evaluation.m1_concordance(self.graph_learned, self.graph_true)
        self.assertEqual(m1["tp"], 25)
        self.assertAlmostEqual(m1["precision"], 25 / 27)
        self.assertEqual(m1["shd"], 3)
        self.assertAlmostEqual(m1["skeleton_f1"], 26 / 27)

        m2 = evaluation.m2_target_pathway(
            self.graph_learned,
            self.graph_true,
            self.exposure,
            self.outcome,
        )
        self.assertEqual(m2["n_true_pathway"], 6)
        self.assertEqual(m2["edges"]["Treatment->Inflammation"], "reversed")
        self.assertEqual(m2["n_correct"], 5)
        self.assertEqual(m2["n_reversed"], 1)

        m3 = evaluation.m3_sufficiency_transfer(
            self.graph_learned,
            self.graph_true,
            self.exposure,
            self.outcome,
        )
        self.assertFalse(m3["valid_in_true"])
        self.assertIn("Inflammation", m3["descendant_offenders"])

    def test_m4_recovers_the_frozen_total_effect_under_true_adjustment(self) -> None:
        true_adjustment = evaluation.minimal_adjustment_set(
            self.graph_true, self.exposure, self.outcome
        )["set"]
        result = evaluation.m4_parameter_fidelity(
            self.data,
            self.exposure,
            self.outcome,
            true_adjustment,
            true_effect=self.true_effect,
            z_true=true_adjustment,
        )
        self.assertLess(
            abs(result["estimate_under_z_true"] - self.true_effect), 1.5
        )

    def test_m5_excludes_orientation_that_adds_an_unshielded_collider(self) -> None:
        cpdag = nx.DiGraph()
        cpdag.add_nodes_from(["A", "B", "C"])
        result = evaluation.m5_identification_honesty(
            cpdag,
            [("A", "B"), ("B", "C")],
            "A",
            "C",
            [],
        )
        self.assertEqual(result["mode"], "exhaustive")
        self.assertEqual(result["n_possible_orientations"], 4)
        self.assertEqual(result["n_extensions"], 3)
        self.assertEqual(result["n_inconsistent_orientations"], 1)
        self.assertEqual(result["n_valid"], 1)

    def test_m5_rejects_adjusting_for_an_exposure_descendant(self) -> None:
        graph = nx.DiGraph([("D", "M"), ("M", "Y")])
        result = evaluation.m5_identification_honesty(
            graph, [], "D", "Y", ["M"]
        )
        self.assertEqual(result["n_extensions"], 1)
        self.assertEqual(result["n_valid"], 0)
        self.assertEqual(result["fraction_valid"], 0.0)

    def test_m5_preserves_an_existing_unshielded_collider(self) -> None:
        cpdag = nx.DiGraph([("A", "C"), ("B", "C")])
        cpdag.add_node("D")
        result = evaluation.m5_identification_honesty(
            cpdag, [("C", "D")], "A", "D", []
        )
        self.assertEqual(result["n_possible_orientations"], 2)
        self.assertEqual(result["n_extensions"], 1)
        self.assertEqual(result["n_inconsistent_orientations"], 1)

    def test_m5_capped_mode_samples_unique_orientations(self) -> None:
        cpdag = nx.DiGraph()
        pairs = [(f"A{index}", f"B{index}") for index in range(9)]
        cpdag.add_nodes_from(node for pair in pairs for node in pair)
        result = evaluation.m5_identification_honesty(
            cpdag, pairs, "A0", "B0", [], cap=32, seed=17
        )
        self.assertEqual(result["mode"], "capped_monte_carlo")
        self.assertTrue(result["capped"])
        self.assertEqual(result["n_orientations_evaluated"], 32)
        self.assertEqual(result["n_extensions"], 32)
        self.assertEqual(result["n_inconsistent_orientations"], 0)


class WorkbenchPredictionTests(unittest.TestCase):
    def test_binary_classifier_uses_positive_class_probability(self) -> None:
        features = pd.DataFrame(
            {"x1": [-2.0, -1.0, 0.0, 1.0, 2.0], "x2": [0, 1, 0, 1, 1]}
        )
        outcome = np.array([0, 0, 0, 1, 1])
        model = LogisticRegression(random_state=0).fit(features, outcome)
        predict = prediction_callable(model, features.columns)
        scores = predict(features.to_numpy())
        np.testing.assert_allclose(scores, model.predict_proba(features)[:, 1])
        self.assertTrue(np.all((scores > 0.0) & (scores < 1.0)))
        self.assertFalse(np.array_equal(scores, model.predict(features)))

    def test_adjustment_model_always_includes_treatment(self) -> None:
        features = adjustment_model_features(
            ["treatment", "confounder", "outcome"],
            "outcome",
            "treatment",
            ["confounder", "treatment", "outcome", "confounder"],
        )
        self.assertEqual(features, ["treatment", "confounder"])

    def test_empty_adjustment_set_means_treatment_only(self) -> None:
        features = adjustment_model_features(
            ["treatment", "outcome"], "outcome", "treatment", []
        )
        self.assertEqual(features, ["treatment"])


if __name__ == "__main__":
    unittest.main()
