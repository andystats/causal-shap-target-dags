"""Build pipeline: one CLI (``python -m causal_shap.build``) over every stage.

The stages here are the in-package home of the former numbered ``scripts/2x``
build scripts; each stage function carries the same logic, seeds, and output
paths so the frozen artifacts stay byte-for-byte reproducible.
"""
