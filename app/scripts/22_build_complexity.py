"""Score every dataset's problem complexity (PSCI v0) and freeze the reports.

Discovery datasets feed their cross-algorithm disagreement into the score; NASA
is scored from its known DAG in degraded mode (single graph, no disagreement).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from _datasets import discovery_datasets  # noqa: E402
from causal_shap import build_nasa_renal_scm  # noqa: E402
from causal_shap.complexity import ComplexityInputs, get_score  # noqa: E402

ANALYSIS = PROJECT_DIR / "analysis" / "output"
NASA_EDGES = ANALYSIS / "dag_validation" / "validated_clean_source_edges.csv"
NASA_MAP = ANALYSIS / "source_aligned_clean" / "source_to_simulation_variable_map.csv"
NASA_OUTCOME = "nephrolithiasis"


def _report_dict(report) -> dict[str, object]:
    return {
        "score_name": report.score_name,
        "score_version": report.score_version,
        "provisional": report.provisional,
        "total": report.total,
        "band": report.band,
        "subscores": {
            name: {"value": sub.value, "rationale": sub.rationale, "available": sub.available}
            for name, sub in report.subscores.items()
        },
        "recommendations": list(report.recommendations),
    }


def _write(bundle_dir: Path, report) -> None:
    stage_dir = bundle_dir / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "complexity.json").write_text(json.dumps(_report_dict(report), indent=2), encoding="utf-8")


def main() -> None:
    score = get_score("PSCI")
    summary = []

    for dataset in discovery_datasets():
        discovery_path = dataset.stage_dir / "discovery.json"
        disagreement = None
        if discovery_path.exists():
            disagreement = json.loads(discovery_path.read_text())["cross_algorithm_disagreement"]
        report = score.compute(
            ComplexityInputs(graph=dataset.load_graph(), outcome=dataset.outcome, disagreement=disagreement)
        )
        _write(dataset.bundle_dir, report)
        summary.append({"bundle": dataset.bundle_id, "total": round(report.total, 1), "band": report.band})

    nasa_scm = build_nasa_renal_scm(NASA_EDGES, NASA_MAP)
    nasa_report = score.compute(ComplexityInputs(graph=nasa_scm.graph, outcome=NASA_OUTCOME))
    _write(APP_DIR / "bundles" / "nasa_renal_clean_v3", nasa_report)
    summary.append({"bundle": "nasa_renal_clean_v3", "total": round(nasa_report.total, 1), "band": nasa_report.band})

    print(json.dumps({"status": "complexity", "datasets": summary}, indent=2))


if __name__ == "__main__":
    main()
