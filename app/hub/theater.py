"""The surgery theater: an inline-SVG DAG you can operate on.

Pyvis renders into a sealed data-URI iframe with no channel back to Shiny, so
the theater draws its own SVG in the parent document and reuses the one
click-to-Shiny bridge this repo already trusts (the workbench schematic's
``Shiny.setInputValue`` pattern). Every node and edge carries a ``data-id``;
one delegated listener reports clicks as ``theater_pick``.

Layout is deterministic: topological layers by longest path from the roots,
the outcome pinned to the final layer, one barycenter pass to reduce edge
crossings. No physics, so the graph never squirms between renders.
"""

from __future__ import annotations

import html
from typing import Mapping

import networkx as nx

from causal_shap.graph_state import GraphState

INK = "#111111"
AMBER = "#b45309"
AMBER_SOFT = "#fdf3e7"
BLUE = "#1e4d8c"
MUTED = "#94a3b8"

HALO_COLORS = {"h0": AMBER, "h1": BLUE, "eig": MUTED}

NODE_HEIGHT = 26
LAYER_GAP = 168
ROW_GAP = 16
CHAR_WIDTH = 6.6
PADDING = 14


def layered_layout(graph: nx.DiGraph, outcome: str) -> dict[str, tuple[float, float]]:
    """Longest-path layering; every edge flows strictly left to right.

    The outcome is deliberately NOT pinned to the last layer: in the proxy
    story the outcome has descendants, and pinning it past them would draw its
    outgoing edges backwards — the one thing this layout must never do.
    """
    layer: dict[str, int] = {}
    for node in nx.topological_sort(graph):
        parents = list(graph.predecessors(node))
        layer[node] = 1 + max((layer[p] for p in parents), default=-1)

    columns: dict[int, list[str]] = {}
    for node, depth in layer.items():
        columns.setdefault(depth, []).append(node)

    # One barycenter pass: order each layer by the mean row of its parents.
    order: dict[str, float] = {}
    for depth in sorted(columns):
        nodes = sorted(columns[depth])
        if depth > 0:
            nodes.sort(
                key=lambda n: (
                    sum(order.get(p, 0.0) for p in graph.predecessors(n))
                    / max(1, sum(1 for _ in graph.predecessors(n))),
                    n,
                )
            )
        for row, node in enumerate(nodes):
            order[node] = float(row)
        columns[depth] = nodes

    positions: dict[str, tuple[float, float]] = {}
    tallest = max(len(nodes) for nodes in columns.values())
    for depth, nodes in columns.items():
        span = len(nodes) * (NODE_HEIGHT + ROW_GAP)
        offset = (tallest * (NODE_HEIGHT + ROW_GAP) - span) / 2.0
        for row, node in enumerate(nodes):
            positions[node] = (
                60.0 + depth * LAYER_GAP,
                40.0 + offset + row * (NODE_HEIGHT + ROW_GAP),
            )
    return positions


def _label(name: str, display_names: Mapping[str, str]) -> str:
    text = str(display_names.get(name, name))
    return text if len(text) <= 22 else text[:20] + "…"


def sorted_edges(state: GraphState) -> list[tuple[str, str]]:
    """The canonical edge order shared by the renderer and the click handler.

    Elements are addressed by index, never by name: uploaded column names are
    untrusted text and must not travel through DOM attributes or be parsed
    back out of a click payload.
    """
    return sorted(state.directed_edges)


def _node_width(name: str, display_names: Mapping[str, str]) -> float:
    return PADDING + CHAR_WIDTH * len(_label(name, display_names))


def render_theater(
    state: GraphState,
    exposure: str,
    outcome: str,
    *,
    halos: Mapping[str, str] | None = None,
    display_names: Mapping[str, str] | None = None,
    selected: str = "",
    height: int = 460,
) -> str:
    """Return the theater as self-contained HTML (SVG + bridge + pan/zoom)."""
    halos = halos or {}
    display_names = display_names or {}
    graph = state.digraph()
    positions = layered_layout(graph, outcome)
    unresolved = set(state.undirected_pairs)

    widths = {node: _node_width(node, display_names) for node in graph.nodes}
    max_x = max((x + widths[n] for n, (x, y) in positions.items()), default=400) + 60
    max_y = max((y for _, y in positions.values()), default=200) + 60

    parts: list[str] = []
    parts.append(
        '<defs><marker id="th-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>'
        '<marker id="th-arrow-sel" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{AMBER}"/></marker></defs>'
    )

    for index, (a, b) in enumerate(sorted_edges(state)):
        xa, ya = positions[a]
        xb, yb = positions[b]
        x1, y1 = xa + widths[a] / 2, ya
        x2, y2 = xb - widths[b] / 2 - 3, yb
        bend = max(30.0, (x2 - x1) * 0.45)
        path = f"M {x1:.0f} {y1:.0f} C {x1 + bend:.0f} {y1:.0f}, {x2 - bend:.0f} {y2:.0f}, {x2:.0f} {y2:.0f}"
        is_selected = selected == f"edge:{index}"
        dashed = tuple(sorted((a, b))) in unresolved
        stroke = AMBER if is_selected else INK
        width = 2.6 if is_selected else 1.4
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        marker = "th-arrow-sel" if is_selected else "th-arrow"
        tooltip = html.escape(f"{a} → {b}") + (
            " (orientation chosen, not identified)" if dashed else ""
        )
        parts.append(
            f'<g class="th-edge" data-id="edge:{index}" style="cursor:pointer">'
            f'<path d="{path}" fill="none" stroke="transparent" stroke-width="12"/>'
            f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="{width}"'
            f'{dash} marker-end="url(#{marker})"/>'
            f"<title>{tooltip}</title></g>"
        )

    for node_index, node in enumerate(state.nodes):
        x, y = positions[node]
        width = widths[node]
        left, top = x - width / 2, y - NODE_HEIGHT / 2
        is_selected = selected == f"node:{node_index}"
        fill = AMBER_SOFT if node == outcome else "#ffffff"
        border = AMBER if node == outcome else (BLUE if node == exposure else INK)
        border_width = 2.4 if node in (outcome, exposure) or is_selected else 1.3
        if is_selected:
            border = AMBER
        halo = ""
        channel = halos.get(node)
        if channel:
            halo = (
                f'<rect x="{left - 4:.0f}" y="{top - 4:.0f}" width="{width + 8:.0f}" '
                f'height="{NODE_HEIGHT + 8}" rx="8" fill="none" '
                f'stroke="{HALO_COLORS[channel]}" stroke-width="2" stroke-dasharray="3 3"/>'
            )
        parts.append(
            f'<g class="th-node" data-id="node:{node_index}" style="cursor:pointer">{halo}'
            f'<rect x="{left:.0f}" y="{top:.0f}" width="{width:.0f}" height="{NODE_HEIGHT}" '
            f'rx="6" fill="{fill}" stroke="{border}" stroke-width="{border_width}"/>'
            f'<text x="{x:.0f}" y="{y + 4:.0f}" text-anchor="middle" '
            f'font-family="Georgia, serif" font-size="11.5" fill="{INK}">'
            f"{html.escape(_label(node, display_names))}</text>"
            f"<title>{html.escape(str(display_names.get(node, node)))}</title></g>"
        )

    svg = (
        f'<svg id="theater-svg" viewBox="0 0 {max_x:.0f} {max_y:.0f}" '
        f'style="width:100%;height:{height}px;background:#fdfcfa;border:1px solid #e2ddd6;'
        f'border-radius:6px" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'
    )

    script = """
<script>
(function() {
  const svg = document.getElementById('theater-svg');
  if (!svg) return;
  svg.addEventListener('click', function(event) {
    const target = event.target.closest('[data-id]');
    if (target && window.Shiny) {
      Shiny.setInputValue('theater_pick', target.dataset.id, {priority: 'event'});
    }
  });
  let view = svg.viewBox.baseVal;
  svg.addEventListener('wheel', function(event) {
    event.preventDefault();
    const scale = event.deltaY > 0 ? 1.12 : 0.89;
    const cx = view.x + view.width / 2, cy = view.y + view.height / 2;
    view.width *= scale; view.height *= scale;
    view.x = cx - view.width / 2; view.y = cy - view.height / 2;
  }, {passive: false});
  let drag = null;
  svg.addEventListener('mousedown', function(event) {
    drag = {x: event.clientX, y: event.clientY, vx: view.x, vy: view.y};
  });
  window.addEventListener('mousemove', function(event) {
    if (!drag) return;
    const k = view.width / svg.clientWidth;
    view.x = drag.vx - (event.clientX - drag.x) * k;
    view.y = drag.vy - (event.clientY - drag.y) * k;
  });
  window.addEventListener('mouseup', function() { drag = null; });
})();
</script>"""
    return f'<div id="theater-wrap">{svg}{script}</div>'


def apply_surgery(
    state: GraphState,
    action: str,
    edge: tuple[str, str],
    rationale: str,
) -> GraphState:
    """One operation on the current graph, with honest pair bookkeeping.

    The rules encode the ``graph_state`` invariant that every unresolved pair
    keeps a directed representative: touching an edge whose pair is unresolved
    adjudicates that pair, so the pair leaves ``undirected_pairs`` and the
    ledger records what the human decided. A cycle-creating flip raises from
    ``GraphState`` itself, naming the cycle.
    """
    from causal_shap.graph_state import ConstraintEntry

    a, b = edge
    if (a, b) not in state.directed_edges:
        raise ValueError(f"No such edge: {a} → {b}")
    pair = tuple(sorted(edge))
    was_unresolved = pair in set(state.undirected_pairs)
    directed = set(state.directed_edges)
    pairs = tuple(p for p in state.undirected_pairs if p != pair)

    if action == "flip":
        directed.discard((a, b))
        directed.add((b, a))
        ledger = (ConstraintEntry((b, a), "required", "post_hoc", rationale),)
    elif action == "require":
        ledger = (ConstraintEntry((a, b), "required", "post_hoc", rationale),)
    elif action == "remove":
        directed.discard((a, b))
        ledger = (
            ConstraintEntry((a, b), "forbidden", "post_hoc", rationale),
            ConstraintEntry((b, a), "forbidden", "post_hoc", rationale),
        )
    elif action == "forbid":
        # Forbidding the shown orientation of an unresolved pair resolves it
        # the other way; forbidding a compelled edge removes it outright.
        directed.discard((a, b))
        if was_unresolved:
            directed.add((b, a))
            ledger = (
                ConstraintEntry((a, b), "forbidden", "post_hoc", rationale),
                ConstraintEntry((b, a), "required", "post_hoc", rationale),
            )
        else:
            ledger = (ConstraintEntry((a, b), "forbidden", "post_hoc", rationale),)
    else:
        raise ValueError(f"Unknown surgery action: {action!r}")

    return state.with_constraints(frozenset(directed), pairs, ledger)
