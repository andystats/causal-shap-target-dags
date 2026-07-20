"""``python -m causal_shap.build <stage>`` — run one build stage (or all).

Collapses the former numbered ``scripts/2x`` build scripts into one CLI. ``all``
runs the safe, reproducible teaching pipeline in dependency order; it excludes
the pre-frozen ``acic`` and ``nasa-structural`` bundles, matching the numbered
20-29 flow, which never regenerated them.
"""

from __future__ import annotations

import argparse

from . import stages
from .validate import validate_bundles

# Hyphenated CLI stage name -> the function that runs it.
STAGES = {
    "teaching-data": stages.teaching_data,
    "discovery": stages.discovery,
    "complexity": stages.complexity,
    "causal-shap": stages.causal_shap,
    "validation": stages.validation,
    "figures": stages.figures,
    "glossary": stages.glossary,
    "acic": stages.acic,
    "nasa-structural": stages.nasa_structural,
    "validate": validate_bundles,
}

# The safe build order: the teaching pipeline plus the release gate.
ALL_ORDER = [
    "teaching-data",
    "discovery",
    "complexity",
    "causal-shap",
    "validation",
    "figures",
    "glossary",
    "validate",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m causal_shap.build",
        description="Run a Causal SHAP build stage (or the full teaching pipeline).",
    )
    parser.add_argument(
        "stage",
        choices=[*STAGES, "all"],
        help="Build stage to run; 'all' runs the safe teaching pipeline in order.",
    )
    args = parser.parse_args(argv)

    if args.stage == "all":
        for name in ALL_ORDER:
            STAGES[name]()
        return
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
