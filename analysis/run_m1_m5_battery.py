"""Run the portable M1--M5 discovery battery on committed synthetic data.

The scientific result is written separately from run metadata so repeated
runs can be compared without timestamps or wall-clock timings getting in the
way.  The legacy ``analysis/output/battery_v1/stage_results.json`` is never
read or overwritten by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from causal_shap import discovery, evaluation  # noqa: E402
from causal_shap.discovery import deterministic_consistent_extension  # noqa: E402
from causal_shap.graphs import PDAG  # noqa: E402


SCHEMA_VERSION = 2
DISCOVERY_SEED = 20260813
ALGORITHMS = ("pc", "ges", "direct_lingam")
ALGORITHM_LABELS = {
    "pc": "PC",
    "ges": "GES",
    "direct_lingam": "DirectLiNGAM",
}
TIMEOUTS_SECONDS = {
    "toy": {"pc": 120, "ges": 300, "direct_lingam": 120},
    "renal": {"pc": 900, "ges": 900, "direct_lingam": 600},
}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "output" / "battery_v2"
RESULTS_FILENAME = "battery_results.json"
METADATA_FILENAME = "battery_run_metadata.json"
RENAL_EDGES_FILENAME = "renal_edges_mapped.csv"

TOY_DIR = REPO_ROOT / "app" / "bundles" / "toy_chain_fork_collider"
RENAL_DATA_PATH = (
    REPO_ROOT
    / "analysis"
    / "output"
    / "source_aligned_clean"
    / "renal_stone_source_aligned_clean_v3.csv"
)
RENAL_SOURCE_EDGES_PATH = (
    REPO_ROOT
    / "analysis"
    / "output"
    / "dag_validation"
    / "validated_clean_source_edges.csv"
)
RENAL_CROSSWALK_PATH = (
    REPO_ROOT
    / "analysis"
    / "output"
    / "source_aligned_clean"
    / "source_to_simulation_variable_map.csv"
)
RENAL_TRUTH_PATH = (
    REPO_ROOT
    / "analysis"
    / "output"
    / "shap_nephrolithiasis_clean_v3"
    / "interventional_truth.csv"
)
RENAL_TRUTH_METADATA_PATH = RENAL_TRUTH_PATH.with_name("interventional_truth_metadata.csv")


@dataclass(frozen=True)
class Example:
    """One sealed-truth battery example."""

    name: str
    data: pd.DataFrame
    true_graph: nx.DiGraph
    exposure: str
    outcome: str
    true_effect: float
    input_paths: dict[str, Path]
    mapped_edges: pd.DataFrame | None = None


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _validate_example(example: Example) -> Example:
    if example.data.columns.duplicated().any():
        duplicates = example.data.columns[example.data.columns.duplicated()].tolist()
        raise ValueError(f"{example.name} has duplicate data columns: {duplicates}")
    if example.exposure not in example.data or example.outcome not in example.data:
        raise ValueError(
            f"{example.name} must contain exposure={example.exposure!r} "
            f"and outcome={example.outcome!r}"
        )
    if example.data.isna().any().any():
        raise ValueError(f"{example.name} contains missing values")
    try:
        example.data.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{example.name} contains nonnumeric discovery columns") from error

    data_nodes = set(example.data.columns)
    graph_nodes = set(example.true_graph.nodes)
    if graph_nodes != data_nodes:
        raise ValueError(
            f"{example.name} graph/data nodes differ: "
            f"graph_only={sorted(graph_nodes - data_nodes)}, "
            f"data_only={sorted(data_nodes - graph_nodes)}"
        )
    if not nx.is_directed_acyclic_graph(example.true_graph):
        raise ValueError(f"{example.name} truth graph is cyclic")
    if not math.isfinite(example.true_effect):
        raise ValueError(f"{example.name} true effect is not finite")
    return example


def load_toy_example() -> Example:
    """Load the committed five-node teaching example."""

    data_path = TOY_DIR / "data.csv"
    edges_path = TOY_DIR / "edges.csv"
    effects_path = TOY_DIR / "true_effects.json"
    data = pd.read_csv(data_path)
    edges = pd.read_csv(edges_path)
    _require_columns(edges, ("from", "to"), "toy edges")
    effects = json.loads(effects_path.read_text(encoding="utf-8"))

    exposure = "Hydration"
    outcome = "Y"
    if effects.get("outcome") != outcome:
        raise ValueError("toy truth outcome does not match the battery outcome")
    try:
        true_effect = float(effects["true_total_effects"][exposure])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"toy truth is missing the effect for {exposure}") from error

    graph = nx.DiGraph()
    graph.add_nodes_from(data.columns)
    graph.add_edges_from(edges[["from", "to"]].itertuples(index=False, name=None))
    return _validate_example(
        Example(
            name="toy",
            data=data,
            true_graph=graph,
            exposure=exposure,
            outcome=outcome,
            true_effect=true_effect,
            input_paths={
                "data": data_path,
                "edges": edges_path,
                "true_effects": effects_path,
            },
        )
    )


def derive_renal_edges() -> pd.DataFrame:
    """Map the committed NASA source edges to clean-v3 variable names."""

    source_edges = pd.read_csv(RENAL_SOURCE_EDGES_PATH)
    crosswalk = pd.read_csv(RENAL_CROSSWALK_PATH)
    _require_columns(source_edges, ("from", "to"), "renal source edges")
    _require_columns(crosswalk, ("source_node", "variable"), "renal crosswalk")

    if crosswalk["source_node"].duplicated().any():
        raise ValueError("renal crosswalk has duplicate source_node values")
    if crosswalk["variable"].duplicated().any():
        raise ValueError("renal crosswalk has duplicate variable values")

    source_to_variable = dict(zip(crosswalk["source_node"], crosswalk["variable"]))
    endpoints = set(source_edges["from"]) | set(source_edges["to"])
    missing = sorted(endpoints - set(source_to_variable))
    if missing:
        raise ValueError(f"renal crosswalk is missing source nodes: {missing}")

    mapped = pd.DataFrame(
        {
            "from": source_edges["from"].map(source_to_variable),
            "to": source_edges["to"].map(source_to_variable),
        }
    )
    if mapped.isna().any().any():
        raise ValueError("renal edge mapping produced missing endpoints")
    if mapped.duplicated().any():
        raise ValueError("renal edge mapping produced duplicate edges")
    if len(mapped) != 75:
        raise ValueError(f"expected 75 source-aligned renal edges, found {len(mapped)}")
    return mapped.sort_values(["from", "to"], kind="stable").reset_index(drop=True)


def load_renal_example() -> Example:
    """Load the committed source-aligned renal simulation and frozen truth."""

    raw_data = pd.read_csv(RENAL_DATA_PATH)
    data = raw_data.drop(columns=["ID", "simulation_version"], errors="ignore")
    mapped_edges = derive_renal_edges()
    truth = pd.read_csv(RENAL_TRUTH_PATH)
    truth_metadata = pd.read_csv(RENAL_TRUTH_METADATA_PATH)
    _require_columns(
        truth,
        ("variable", "total_effect_risk_difference"),
        "renal interventional truth",
    )
    _require_columns(
        truth_metadata,
        ("monte_carlo_n", "common_random_number_seed", "truth_scale"),
        "renal interventional-truth metadata",
    )
    if len(truth_metadata) != 1 or int(truth_metadata.iloc[0]["monte_carlo_n"]) != 50_000:
        raise ValueError("renal truth metadata must identify the locked 50,000-draw run")

    exposure = "altered_gravity"
    outcome = "nephrolithiasis"
    truth_row = truth.loc[truth["variable"] == exposure]
    if len(truth_row) != 1:
        raise ValueError(
            f"expected exactly one renal truth row for {exposure}, found {len(truth_row)}"
        )
    true_effect = float(truth_row.iloc[0]["total_effect_risk_difference"])

    endpoints = set(mapped_edges["from"]) | set(mapped_edges["to"])
    missing_from_data = sorted(endpoints - set(data.columns))
    if missing_from_data:
        raise ValueError(f"mapped renal graph nodes absent from data: {missing_from_data}")

    graph = nx.DiGraph()
    graph.add_nodes_from(data.columns)
    graph.add_edges_from(mapped_edges.itertuples(index=False, name=None))
    return _validate_example(
        Example(
            name="renal",
            data=data,
            true_graph=graph,
            exposure=exposure,
            outcome=outcome,
            true_effect=true_effect,
            input_paths={
                "data": RENAL_DATA_PATH,
                "source_edges": RENAL_SOURCE_EDGES_PATH,
                "crosswalk": RENAL_CROSSWALK_PATH,
                "interventional_truth": RENAL_TRUTH_PATH,
                "interventional_truth_metadata": RENAL_TRUTH_METADATA_PATH,
            },
            mapped_edges=mapped_edges,
        )
    )


def load_examples(selection: str) -> list[Example]:
    """Load one or both examples without depending on the current directory."""

    if selection == "toy":
        return [load_toy_example()]
    if selection == "renal":
        return [load_renal_example()]
    if selection == "all":
        return [load_toy_example(), load_renal_example()]
    raise ValueError(f"unknown example selection: {selection}")


def _discovery_worker(
    result_connection: Any,
    algorithm: str,
    matrix: np.ndarray,
    columns: list[str],
    seed: int,
) -> None:
    """Spawn-process entry point; return only serializable discovery state."""

    try:
        np.random.seed(seed)
        frame = pd.DataFrame(matrix, columns=columns)
        started = time.perf_counter()
        if algorithm == "pc":
            result = discovery.run_pc(frame, alpha=0.05)
        elif algorithm == "ges":
            result = discovery.run_ges(frame)
        elif algorithm == "direct_lingam":
            result = discovery.run_direct_lingam(frame, random_state=seed)
        else:
            raise ValueError(f"unknown discovery algorithm: {algorithm}")
        elapsed = time.perf_counter() - started
        result_connection.send(
            {
                "status": "ok",
                "seconds": elapsed,
                "algorithm": result.algorithm,
                "params": result.params,
                "pdag": {
                    "nodes": list(result.pdag.nodes),
                    "directed_edges": [list(edge) for edge in sorted(result.pdag.directed_edges)],
                    "undirected_edges": [list(edge) for edge in sorted(result.pdag.undirected_edges)],
                },
            }
        )
    except Exception as error:  # pragma: no cover - exercised through parent process
        result_connection.send(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
    finally:
        result_connection.close()


def run_discovery_with_timeout(
    algorithm: str,
    data: pd.DataFrame,
    timeout_seconds: float,
    seed: int = DISCOVERY_SEED,
) -> dict[str, Any]:
    """Run one discovery algorithm under a cross-platform hard time limit."""

    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be nonnegative")
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_discovery_worker,
        args=(
            send_connection,
            algorithm,
            data.to_numpy(dtype=float),
            list(data.columns),
            seed,
        ),
    )
    started = time.perf_counter()
    process.start()
    send_connection.close()

    if receive_connection.poll(timeout_seconds):
        try:
            result = receive_connection.recv()
        except EOFError:
            result = {
                "status": "error",
                "error_type": "WorkerExitError",
                "error_message": "worker closed its result pipe without a payload",
            }
        process.join(5)
        if process.is_alive():  # result arrived, but do not leave a stray worker
            process.terminate()
            process.join(5)
    else:
        elapsed = time.perf_counter() - started
        process.terminate()
        process.join(5)
        if process.is_alive():  # pragma: no cover - terminate normally suffices
            process.kill()
            process.join(5)
        result = {"status": "timeout", "seconds": elapsed}
    receive_connection.close()
    result.setdefault("seconds", time.perf_counter() - started)
    return result


def _pdag_from_payload(payload: dict[str, Any]) -> PDAG:
    return PDAG(
        nodes=tuple(payload["nodes"]),
        directed_edges=frozenset(tuple(edge) for edge in payload["directed_edges"]),
        undirected_edges=frozenset(tuple(edge) for edge in payload["undirected_edges"]),
    )


def _json_safe(value: Any) -> Any:
    """Convert NumPy and non-finite values to strict deterministic JSON data."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value)]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _scientific_example(example: Example) -> tuple[dict[str, Any], dict[str, Any]]:
    scientific: dict[str, Any] = {
        "exposure": example.exposure,
        "outcome": example.outcome,
        "true_effect": example.true_effect,
        "n_rows": len(example.data),
        "n_variables": len(example.data.columns),
        "algorithms": {},
    }
    runtime: dict[str, Any] = {}

    for algorithm in ALGORITHMS:
        label = ALGORITHM_LABELS[algorithm]
        timeout_seconds = TIMEOUTS_SECONDS[example.name][algorithm]
        run = run_discovery_with_timeout(algorithm, example.data, timeout_seconds)
        runtime[label] = {
            key: _json_safe(value)
            for key, value in run.items()
            if key in {"status", "seconds", "error_type", "error_message"}
        }
        if run["status"] != "ok":
            scientific["algorithms"][label] = {"status": run["status"]}
            continue

        try:
            pdag = _pdag_from_payload(run["pdag"])
            extension = deterministic_consistent_extension(pdag)
            metrics = evaluation.evaluate_battery(
                extension,
                example.true_graph,
                example.exposure,
                example.outcome,
                data=example.data,
                true_effect=example.true_effect,
                undirected_pairs=sorted(pdag.undirected_edges),
            )
        except Exception as error:
            runtime[label].update(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            scientific["algorithms"][label] = {"status": "error"}
            continue
        scientific["algorithms"][label] = {
            "status": "ok",
            "discovery": {
                "algorithm": run["algorithm"],
                "parameters": run["params"],
                "seed": DISCOVERY_SEED,
                "directed_edges": [list(edge) for edge in sorted(pdag.directed_edges)],
                "undirected_edges": [list(edge) for edge in sorted(pdag.undirected_edges)],
                "consistent_extension_edges": [list(edge) for edge in sorted(extension.edges)],
            },
            "metrics": metrics,
        }
    return _json_safe(scientific), _json_safe(runtime)


def _normalized_text_sha256(path: Path) -> str:
    """Hash text with LF newlines so Git autocrlf does not change the digest."""

    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return "<external-output>"


def _input_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": _relative_path(path),
        "size_bytes": path.stat().st_size,
        "sha256_lf_normalized": _normalized_text_sha256(path),
    }


def _git_value(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    packages = (
        "causal-learn",
        "networkx",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_artifacts(
    output_dir: Path,
    scientific: dict[str, Any],
    metadata: dict[str, Any],
    mapped_edges: pd.DataFrame | None,
) -> tuple[Path, Path]:
    results_path = output_dir / RESULTS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    results_text = _json_text(scientific)
    metadata = dict(metadata)
    metadata["outputs"] = {
        RESULTS_FILENAME: {
            "sha256": hashlib.sha256(results_text.encode("utf-8")).hexdigest(),
        }
    }

    if mapped_edges is not None:
        edges_text = mapped_edges.to_csv(index=False, lineterminator="\n")
        metadata["outputs"][RENAL_EDGES_FILENAME] = {
            "sha256": hashlib.sha256(edges_text.encode("utf-8")).hexdigest(),
        }
        _atomic_write_text(output_dir / RENAL_EDGES_FILENAME, edges_text)

    _atomic_write_text(results_path, results_text)
    _atomic_write_text(metadata_path, _json_text(metadata))
    return results_path, metadata_path


def run_battery(selection: str, output_dir: Path, allow_partial: bool) -> tuple[Path, Path]:
    """Run the selected examples and write separated result/metadata artifacts."""

    examples = load_examples(selection)
    scientific: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design": {
            "battery": "M1-M5",
            "discovery_seed": DISCOVERY_SEED,
            "cpdag_adapter": "dor_tarsi_lexicographic_v1",
            "examples": [example.name for example in examples],
            "exposure_note": {
                "renal": (
                    "clean-v3 has no mission-duration variable; this battery uses "
                    "altered_gravity because frozen intervention truth is available"
                )
            },
        },
        "examples": {},
    }
    runtimes: dict[str, Any] = {}
    failed: list[str] = []
    all_inputs: dict[str, Any] = {}
    mapped_edges: pd.DataFrame | None = None

    for example in examples:
        result, runtime = _scientific_example(example)
        scientific["examples"][example.name] = result
        runtimes[example.name] = runtime
        for label, run in result["algorithms"].items():
            if run["status"] != "ok":
                failed.append(f"{example.name}:{label}:{run['status']}")
        for role, path in example.input_paths.items():
            all_inputs[f"{example.name}.{role}"] = _input_fingerprint(path)
        if example.mapped_edges is not None:
            mapped_edges = example.mapped_edges

    if failed and not allow_partial:
        joined = ", ".join(failed)
        raise RuntimeError(
            f"battery incomplete ({joined}); no canonical artifacts were written. "
            "Use --allow-partial only for diagnostic output."
        )

    git_status = _git_value("status", "--porcelain")
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": _input_fingerprint(Path(__file__)),
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": None if git_status is None else bool(git_status),
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "thread_limits": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
        "run": {
            "selection": selection,
            "allow_partial": allow_partial,
            "timeouts_seconds": {
                example.name: TIMEOUTS_SECONDS[example.name] for example in examples
            },
            "algorithms": runtimes,
        },
        "inputs": all_inputs,
    }
    return _write_artifacts(output_dir.resolve(), scientific, metadata, mapped_edges)


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example",
        choices=("toy", "renal", "all"),
        default="all",
        help="example(s) to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="artifact directory (default: analysis/output/battery_v2)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write diagnostic artifacts even if an algorithm errors or times out",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = _parse_args(arguments)
    try:
        results_path, metadata_path = run_battery(
            args.example,
            args.output_dir,
            args.allow_partial,
        )
    except Exception as error:
        print(f"battery failed: {error}", file=sys.stderr)
        return 1
    print(f"wrote {results_path}")
    print(f"wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
