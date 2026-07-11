"""Versioned local bundle loading for deterministic app playback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Bundle:
    id: str
    label: str
    kind: str
    target: str
    paths: dict[str, Path]


class BundleRepository:
    def __init__(self, project_dir: Path, manifest_path: Path):
        self.project_dir = project_dir.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported bundle manifest schema")
        self._bundles: dict[str, Bundle] = {}
        for record in manifest["bundles"]:
            paths = {
                name: (self.project_dir / relative).resolve()
                for name, relative in record["paths"].items()
            }
            self._bundles[record["id"]] = Bundle(
                id=record["id"],
                label=record["label"],
                kind=record["kind"],
                target=record["target"],
                paths=paths,
            )

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
        return errors


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
