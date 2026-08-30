"""Hub look and small rendering helpers: patent-paper skin, ink and amber."""

from __future__ import annotations

import html
from typing import Mapping, Sequence

HUB_CSS = """
<style>
:root {
  --ink: #111111; --muted: #6b6b6b; --line: #e2ddd6; --paper: #fdfcfa;
  --amber: #b45309; --amber-soft: #fdf3e7; --blue: #1e4d8c; --blue-soft: #eaf0f9;
  --green: #1a7f4b; --green-soft: #e8f5ee; --red: #a32020; --red-soft: #fbeaea;
}
body { background: var(--paper); color: var(--ink); font-family: Georgia, serif; }
.hub-header { border-bottom: 2px solid var(--ink); padding: 10px 4px 12px; margin-bottom: 10px; }
.hub-header h1 { font-size: 21px; margin: 0; letter-spacing: -0.01em; }
.hub-header .sub { color: var(--muted); font-size: 12.5px; font-family: ui-monospace, Consolas, monospace; }
.hub-card { background: #ffffff; border: 1px solid var(--line); border-radius: 6px;
  padding: 14px 16px; margin-bottom: 12px; }
.hub-card h4 { font-size: 12px; font-family: ui-monospace, Consolas, monospace;
  text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); margin: 0 0 8px; }
.pill { display: inline-block; font-family: ui-monospace, Consolas, monospace; font-size: 11px;
  padding: 2px 9px; border-radius: 12px; border: 1px solid var(--line); margin: 0 4px 4px 0; }
.pill.ok { background: var(--green-soft); color: var(--green); border-color: transparent; }
.pill.warn { background: var(--amber-soft); color: var(--amber); border-color: transparent; }
.pill.bad { background: var(--red-soft); color: var(--red); border-color: transparent; }
.pill.info { background: var(--blue-soft); color: var(--blue); border-color: transparent; }
.note { background: var(--amber-soft); border-left: 2px solid var(--amber);
  padding: 8px 12px; border-radius: 0 4px 4px 0; font-size: 13.5px; margin: 8px 0; }
.error-box { background: var(--red-soft); border-left: 2px solid var(--red);
  padding: 8px 12px; border-radius: 0 4px 4px 0; font-size: 13.5px; margin: 8px 0;
  font-family: ui-monospace, Consolas, monospace; white-space: pre-wrap; }
.hub-table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 6px 0; }
.hub-table th { font-family: ui-monospace, Consolas, monospace; font-size: 10.5px;
  text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
  text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--ink); }
.hub-table td { padding: 5px 8px; border-bottom: 1px solid var(--line); }
.hub-table tr.dim td { color: var(--muted); }
.map-strip { display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0 10px; }
.map-station { font-family: ui-monospace, Consolas, monospace; font-size: 11px;
  border: 1px solid var(--line); border-radius: 5px; padding: 5px 10px;
  background: #ffffff; cursor: pointer; }
.map-station b { display: block; font-size: 11.5px; }
.map-station.on { border-color: var(--amber); background: var(--amber-soft); }
.nav-tabs .nav-link { color: var(--muted); font-family: Georgia, serif; }
.nav-tabs .nav-link.active { color: var(--ink); font-weight: 700;
  border-bottom: 2px solid var(--amber); }
.figure-img { max-width: 100%; border: 1px solid var(--line); border-radius: 4px; }
</style>
"""


def card(title: str, *body: str) -> str:
    inner = "".join(body)
    return f'<div class="hub-card"><h4>{html.escape(title)}</h4>{inner}</div>'


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill {kind}">{html.escape(text)}</span>'


def note(text: str) -> str:
    return f'<div class="note">{text}</div>'


def error_box(message: str, trace: str = "") -> str:
    detail = f"\n{html.escape(trace)}" if trace else ""
    return f'<div class="error-box">{html.escape(message)}{detail}</div>'


def table(rows: Sequence[Mapping[str, object]], columns: Sequence[str],
          *, dim_when: str | None = None, limit: int = 40) -> str:
    if not rows:
        return '<p style="color:var(--muted);font-size:13px">nothing to show</p>'
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body: list[str] = []
    for row in list(rows)[:limit]:
        dim = ' class="dim"' if dim_when and row.get(dim_when) else ""
        cells = "".join(f"<td>{_cell(row.get(c, ''))}</td>" for c in columns)
        body.append(f"<tr{dim}>{cells}</tr>")
    more = (
        f'<p style="color:var(--muted);font-size:12px">… and {len(rows) - limit} more</p>'
        if len(rows) > limit else ""
    )
    return f'<table class="hub-table"><tr>{head}</tr>{"".join(body)}</table>{more}'


def _cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, bool):
        return "✓" if value else "—"
    return html.escape(str(value))


def figure(b64: str | None, alt: str) -> str:
    if not b64:
        return note("chart unavailable (matplotlib missing)")
    return f'<img class="figure-img" alt="{html.escape(alt)}" src="data:image/png;base64,{b64}"/>'
