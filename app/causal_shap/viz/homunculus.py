"""The signature homunculus figures.

Ordinary SHAP over-credits outcome-adjacent nodes, so a DAG whose node areas
scale with attribution looks like a homunculus: bloated proxies near the
outcome, withered distal causes. Node area encodes attribution magnitude; a
blue-to-amber diverging fill (neutral gray at truth) encodes over- vs
under-attribution relative to the interventional truth.
"""

from __future__ import annotations

from typing import Mapping

import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from ..graphs import undirected_distance_to_outcome

matplotlib.use("Agg")

# Diverging pair validated CVD-safe (blue under-credit ↔ amber over-credit),
# neutral gray at the truth midpoint.
_UNDER = "#2563eb"
_MID = "#e5e7eb"
_OVER = "#d97706"
DIVERGING = LinearSegmentedColormap.from_list("attribution_deviation", [_UNDER, _MID, _OVER])

_MIN_MARKER = 120.0
_MAX_MARKER = 2600.0


def _shares(weights: Mapping[str, float], features: list[str]) -> dict[str, float]:
    total = sum(abs(weights.get(name, 0.0)) for name in features)
    if total <= 0:
        return {name: 0.0 for name in features}
    return {name: abs(weights.get(name, 0.0)) / total for name in features}


def _layered_layout(graph: nx.DiGraph, outcome: str, features: list[str]) -> dict[str, tuple[float, float]]:
    """Place nodes by undirected distance to the outcome (deeper = left)."""
    distances = undirected_distance_to_outcome(graph, outcome)
    max_distance = max(distances.values(), default=1)
    layers: dict[int, list[str]] = {}
    for name in features:
        depth = distances.get(name, max_distance + 1)
        layers.setdefault(depth, []).append(name)
    positions: dict[str, tuple[float, float]] = {outcome: (max_distance + 0.6, 0.0)}
    for depth, members in layers.items():
        x = max_distance - depth
        for index, name in enumerate(sorted(members)):
            y = index - (len(members) - 1) / 2
            positions[name] = (float(x), float(y))
    return positions


def _draw(
    ax,
    graph: nx.DiGraph,
    outcome: str,
    features: list[str],
    positions: dict[str, tuple[float, float]],
    sizes: dict[str, float],
    colors: dict[str, float],
    title: str,
) -> None:
    for source, target in graph.edges():
        if source in positions and target in positions:
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#cbd5e1", lw=1.0, shrinkA=8, shrinkB=10),
                zorder=1,
            )
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    xs = [positions[name][0] for name in features]
    ys = [positions[name][1] for name in features]
    ax.scatter(
        xs,
        ys,
        s=[sizes[name] for name in features],
        c=[colors[name] for name in features],
        cmap=DIVERGING,
        norm=norm,
        edgecolors="#334155",
        linewidths=1.0,
        zorder=2,
    )
    for name in features:
        x, y = positions[name]
        ax.text(x, y, name, ha="center", va="center", fontsize=7, zorder=3)
    ox, oy = positions[outcome]
    ax.scatter([ox], [oy], s=_MAX_MARKER, facecolors="none", edgecolors="#111827", linewidths=2.0, zorder=2)
    ax.text(ox, oy, outcome, ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def _sizes_from_shares(shares: dict[str, float], reference: float) -> dict[str, float]:
    scale = reference if reference > 0 else 1.0
    return {name: _MIN_MARKER + (_MAX_MARKER - _MIN_MARKER) * min(share / scale, 1.0) for name, share in shares.items()}


def _deviation_colors(shares: dict[str, float], reference_shares: dict[str, float], features: list[str]) -> dict[str, float]:
    """Signed over/under-attribution vs truth, scaled to [-1, 1] for the diverging map."""
    deviations = {name: shares[name] - reference_shares[name] for name in features}
    spread = max((abs(value) for value in deviations.values()), default=1.0) or 1.0
    return {name: deviations[name] / spread for name in features}


def homunculus_figure(
    graph: nx.DiGraph,
    outcome: str,
    attribution: Mapping[str, float],
    truth: Mapping[str, float],
    title: str = "Attribution homunculus",
):
    """One DAG with node area ∝ attribution and diverging over/under fill."""
    features = [node for node in graph.nodes() if node != outcome]
    attribution_shares = _shares(attribution, features)
    truth_shares = _shares(truth, features)
    positions = _layered_layout(graph, outcome, features)
    reference = max(attribution_shares.values(), default=1.0)
    sizes = _sizes_from_shares(attribution_shares, reference)
    colors = _deviation_colors(attribution_shares, truth_shares, features)

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="white")
    _draw(ax, graph, outcome, features, positions, sizes, colors, title)
    fig.tight_layout()
    return fig


def homunculus_pair(
    graph: nx.DiGraph,
    outcome: str,
    truth: Mapping[str, float],
    vanilla: Mapping[str, float],
    structural: Mapping[str, float],
):
    """Truth vs vanilla SHAP vs structural Causal SHAP, on a shared layout."""
    features = [node for node in graph.nodes() if node != outcome]
    positions = _layered_layout(graph, outcome, features)
    truth_shares = _shares(truth, features)
    reference = max(truth_shares.values(), default=1.0)

    panels = [
        ("Interventional truth", truth_shares, truth_shares),
        ("Ordinary SHAP", _shares(vanilla, features), truth_shares),
        ("Structural Causal SHAP", _shares(structural, features), truth_shares),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="white")
    for ax, (label, shares, reference_shares) in zip(axes, panels):
        sizes = _sizes_from_shares(shares, reference)
        colors = _deviation_colors(shares, reference_shares, features)
        _draw(ax, graph, outcome, features, positions, sizes, colors, label)
    fig.suptitle("Node area ∝ attribution share · fill = over/under vs truth", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def distortion_profile(
    depths: Mapping[str, int],
    method_shares: Mapping[str, Mapping[str, float]],
    truth_key: str = "Interventional truth",
):
    """Attribution mass by DAG depth, per method, with the truth as a step outline."""
    features = list(depths.keys())
    max_depth = max(depths.values(), default=1)
    bins = list(range(max_depth + 1))

    def mass_by_depth(shares: Mapping[str, float]) -> np.ndarray:
        totals = np.zeros(len(bins))
        normed = _shares(shares, features)
        for name in features:
            totals[depths[name]] += normed[name]
        return totals

    methods = [key for key in method_shares if key != truth_key]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    width = 0.8 / max(len(methods), 1)
    colors = ["#111827", _OVER, _UNDER, "#059669"]
    for index, method in enumerate(methods):
        offset = (index - (len(methods) - 1) / 2) * width
        ax.bar(np.array(bins) + offset, mass_by_depth(method_shares[method]), width=width, label=method, color=colors[index % len(colors)])
    if truth_key in method_shares:
        truth_mass = mass_by_depth(method_shares[truth_key])
        ax.step(bins, truth_mass, where="mid", color="#111827", linewidth=2.0, linestyle="--", label=truth_key)
    ax.set_xlabel("Distance to outcome in hops (0 = adjacent)")
    ax.set_ylabel("Share of attribution mass")
    ax.set_xticks(bins)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig
