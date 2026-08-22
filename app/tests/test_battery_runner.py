from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from analysis import run_m1_m5_battery as battery  # noqa: E402
from causal_shap.graphs import PDAG  # noqa: E402


class BatteryInputTests(unittest.TestCase):
    def test_toy_loading_is_independent_of_current_directory(self) -> None:
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                example = battery.load_toy_example()
            finally:
                os.chdir(original)

        self.assertEqual(example.data.shape, (6000, 5))
        self.assertEqual(example.exposure, "Hydration")
        self.assertEqual(example.outcome, "Y")
        self.assertEqual(example.true_effect, 1.0)

    def test_renal_edges_are_derived_from_committed_sources(self) -> None:
        mapped = battery.derive_renal_edges()
        self.assertEqual(len(mapped), 75)
        self.assertFalse(mapped.isna().any().any())
        self.assertFalse(mapped.duplicated().any())
        edges = set(mapped.itertuples(index=False, name=None))
        self.assertIn(("altered_gravity", "bone_remodeling"), edges)
        self.assertIn(("mineralized_renal_material", "nephrolithiasis"), edges)

    def test_normalized_hash_is_stable_across_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            lf_path = directory / "lf.csv"
            crlf_path = directory / "crlf.csv"
            lf_path.write_bytes(b"a,b\n1,2\n")
            crlf_path.write_bytes(b"a,b\r\n1,2\r\n")
            self.assertEqual(
                battery._normalized_text_sha256(lf_path),
                battery._normalized_text_sha256(crlf_path),
            )


class ConsistentExtensionTests(unittest.TestCase):
    def test_extension_is_deterministic_and_preserves_skeleton(self) -> None:
        pdag = PDAG(
            nodes=("A", "B", "C"),
            directed_edges=frozenset(),
            undirected_edges=frozenset({("A", "B"), ("B", "C")}),
        )
        first = battery.deterministic_consistent_extension(pdag)
        second = battery.deterministic_consistent_extension(pdag)
        self.assertEqual(set(first.edges), {("B", "A"), ("C", "B")})
        self.assertEqual(set(first.edges), set(second.edges))
        self.assertTrue(nx.is_directed_acyclic_graph(first))
        self.assertEqual(
            {tuple(sorted(edge)) for edge in first.edges},
            set(pdag.skeleton),
        )

    def test_extension_preserves_an_existing_unshielded_collider(self) -> None:
        pdag = PDAG(
            nodes=("A", "B", "C"),
            directed_edges=frozenset({("A", "B"), ("C", "B")}),
            undirected_edges=frozenset(),
        )
        extension = battery.deterministic_consistent_extension(pdag)
        self.assertEqual(set(extension.edges), set(pdag.directed_edges))


class BatteryProcessAndOutputTests(unittest.TestCase):
    def test_spawn_timeout_returns_without_unix_signals(self) -> None:
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(rng.normal(size=(40, 3)), columns=["A", "B", "C"])
        result = battery.run_discovery_with_timeout("pc", frame, timeout_seconds=0.0)
        self.assertEqual(result["status"], "timeout")

    def test_outputs_are_separate_and_do_not_touch_legacy_json(self) -> None:
        scientific = {"schema_version": 2, "examples": {"toy": {"status": "ok"}}}
        metadata = {"schema_version": 2, "run": {"selection": "toy"}}
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            legacy = output_dir / "stage_results.json"
            legacy.write_text("legacy sentinel\n", encoding="utf-8")
            results_path, metadata_path = battery._write_artifacts(
                output_dir,
                scientific,
                metadata,
                mapped_edges=None,
            )

            self.assertTrue(results_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(legacy.read_text(encoding="utf-8"), "legacy sentinel\n")
            self.assertNotIn("generated_at", results_path.read_text(encoding="utf-8"))
            self.assertIn(
                battery.RESULTS_FILENAME,
                metadata_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
