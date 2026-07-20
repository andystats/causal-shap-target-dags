"""Versioned local bundle loading for deterministic app playback.

Schema v1 records carry a flat ``paths`` map. Schema v2 adds per-stage path
groups (discovery, complexity, causal_shap, validation) plus ``description`` and
``provenance``; v1 records still load so the migration can happen incrementally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_SCHEMA_VERSIONS = {1, 2}


@dataclass(frozen=True)
class Bundle:
    id: str
    label: str
    kind: str
    target: str
    paths: dict[str, Path]
    stages: dict[str, dict[str, Path]] = field(default_factory=dict)
    description: str = ""
    provenance: str = ""

    def stage(self, name: str) -> dict[str, Path]:
        if name not in self.stages:
            raise KeyError(f"Bundle {self.id} has no stage: {name}")
        return self.stages[name]


class BundleRepository:
    def __init__(self, project_dir: Path, manifest_path: Path):
        self.project_dir = project_dir.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("Unsupported bundle manifest schema")
        self._bundles: dict[str, Bundle] = {}
        for record in manifest["bundles"]:
            self._bundles[record["id"]] = Bundle(
                id=record["id"],
                label=record["label"],
                kind=record["kind"],
                target=record["target"],
                paths=self._resolve(record.get("paths", {})),
                stages={
                    stage: self._resolve(group)
                    for stage, group in record.get("stages", {}).items()
                },
                description=record.get("description", ""),
                provenance=record.get("provenance", ""),
            )

    def _resolve(self, group: dict[str, str]) -> dict[str, Path]:
        return {name: (self.project_dir / relative).resolve() for name, relative in group.items()}

    def choices(self) -> dict[str, str]:
        return {bundle.id: bundle.label for bundle in self._bundles.values()}

    def get(self, bundle_id: str) -> Bundle:
        if bundle_id not in self._bundles:
            raise KeyError(f"Unknown bundle: {bundle_id}")
        return self._bundles[bundle_id]

    def validate(self) -> list[str]:
        errors: list[str] = []
        for bundle in self._bundles.values():
            for name, path in bundle.paths.items():
                if not path.exists():
                    errors.append(f"{bundle.id}.{name}: missing {path}")
            for stage, group in bundle.stages.items():
                for name, path in group.items():
                    if not path.exists():
                        errors.append(f"{bundle.id}.{stage}.{name}: missing {path}")
        return errors


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
