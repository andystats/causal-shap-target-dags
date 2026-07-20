"""Release gate for the Python bundle pipeline (mirrors validate_outputs.R).

Asserts three invariants: the frozen analysis/output tree is byte-for-byte
unchanged (via a committed baseline hash manifest), the bundle manifest resolves
with no missing files, and every expected stage artifact exists.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..bundles import BundleRepository
from .common import APP_DIR, PROJECT_DIR, print_status

ANALYSIS_OUTPUT = PROJECT_DIR / "analysis" / "output"
BASELINE_PATH = APP_DIR / "bundles" / "analysis_output_baseline_hashes.json"
MANIFEST_PATH = APP_DIR / "bundles" / "manifest.json"

EXPECTED_STAGE_ARTIFACTS = {
    "toy_chain_fork_collider": ["discovery.json", "complexity.json", "attribution.csv"],
    "layered_ladder": ["discovery.json", "complexity.json", "attribution.csv"],
    "acic_proxy_stress_test": ["discovery.json", "complexity.json"],
    "nasa_renal_clean_v3": ["complexity.json", "validation_scorecard.csv", "validation_estimands.csv"],
}


def _hash_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _check_frozen_output(errors: list[str]) -> None:
    current = _hash_tree(ANALYSIS_OUTPUT)
    if not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  baseline created: {len(current)} files under analysis/output")
        return
    baseline = json.loads(BASELINE_PATH.read_text())
    for name in sorted(set(baseline) - set(current)):
        errors.append(f"analysis/output: file removed: {name}")
    for name in sorted(set(current) - set(baseline)):
        errors.append(f"analysis/output: file added: {name}")
    for name in sorted(set(baseline) & set(current)):
        if baseline[name] != current[name]:
            errors.append(f"analysis/output: file changed: {name}")


def validate_bundles() -> None:
    errors: list[str] = []
    _check_frozen_output(errors)

    errors.extend(BundleRepository(APP_DIR, MANIFEST_PATH).validate())

    for bundle_id, artifacts in EXPECTED_STAGE_ARTIFACTS.items():
        stage_dir = APP_DIR / "bundles" / bundle_id / "stages"
        for artifact in artifacts:
            if not (stage_dir / artifact).exists():
                errors.append(f"{bundle_id}: missing stage artifact {artifact}")

    if errors:
        print_status({"status": "FAIL", "errors": errors})
        raise SystemExit(1)
    print_status({"status": "OK", "checked": sorted(EXPECTED_STAGE_ARTIFACTS)})
