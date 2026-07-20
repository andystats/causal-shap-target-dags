"""The six-rung workflow ladder as a single SVG source.

Used verbatim by the site navigation art and the app header so the ladder looks
identical everywhere.
"""

from __future__ import annotations

RUNGS = (
    ("0", "Vanilla SHAP", "The homunculus: proximity bias"),
    ("1", "Causal discovery", "Learn structure; a tool, not an oracle"),
    ("2", "Complexity score", "How much care does this problem need?"),
    ("3", "Causal SHAP", "Propagate do(X=x) through descendants"),
    ("4", "Simulation validation", "Credence-style layered truth"),
    ("5", "Thoughtful iteration", "Refine the DAG; report uncertainty"),
)

_PRIMARY = "#2563eb"
_INK = "#111827"
_MUTED = "#64748b"
_SURFACE = "#ffffff"
_BORDER = "#cbd5e1"


def ladder_svg(active: int | None = None, width: int = 960) -> str:
    """Render the ladder; if `active` is given, that rung is highlighted."""
    rung_height = 74
    gap = 12
    height = len(RUNGS) * (rung_height + gap) + gap
    rows: list[str] = []
    for index, (number, title, subtitle) in enumerate(RUNGS):
        y = gap + index * (rung_height + gap)
        is_active = active is not None and index == active
        fill = "#eff6ff" if is_active else _SURFACE
        stroke = _PRIMARY if is_active else _BORDER
        rows.append(
            f'<g transform="translate(0,{y})">'
            f'<rect x="1" y="1" width="{width - 2}" height="{rung_height}" rx="10" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if is_active else 1}"/>'
            f'<circle cx="42" cy="{rung_height / 2}" r="20" fill="{_PRIMARY}"/>'
            f'<text x="42" y="{rung_height / 2 + 6}" text-anchor="middle" '
            f'font-family="Inter,sans-serif" font-size="20" font-weight="700" fill="#ffffff">{number}</text>'
            f'<text x="82" y="{rung_height / 2 - 4}" font-family="Inter,sans-serif" '
            f'font-size="18" font-weight="700" fill="{_INK}">{title}</text>'
            f'<text x="82" y="{rung_height / 2 + 18}" font-family="Inter,sans-serif" '
            f'font-size="13" fill="{_MUTED}">{subtitle}</text>'
            f"</g>"
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-label="Causal SHAP workflow ladder">'
        f"{''.join(rows)}</svg>"
    )
