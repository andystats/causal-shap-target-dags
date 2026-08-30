"""Hub look and small rendering helpers.

The design system is the patent-paper mockup (docs/workbench_mockup.html):
ink on paper with one amber, Georgia for prose, Courier for apparatus. Cards
are 1px ink boxes with ruled mono titles; notes are left-ruled italics; the
station strip across the top doubles as the simplified workflow ladder.
"""

from __future__ import annotations

import html
from typing import Mapping, Sequence

HUB_CSS = """
<style>
:root {
  --ink: #111111; --muted: #666666; --line: #d9d4cc; --paper: #ffffff;
  --amber: #b45309; --amber-soft: rgba(180,83,9,.12);
  --green: #1a7f4b; --green-soft: #e8f5ee; --red: #a32020; --red-soft: #fbeaea;
  --blue: #1e4d8c; --blue-soft: #eaf0f9;
  --serif: Georgia, 'Times New Roman', serif;
  --mono: 'Courier New', Courier, monospace;
}
body { background: var(--paper); color: var(--ink); font-family: var(--serif);
  line-height: 1.45; }
.hub-header { border-bottom: 2px solid var(--ink); padding: 8px 2px 10px;
  margin-bottom: 12px; display: flex; justify-content: space-between;
  align-items: baseline; }
.hub-header h1 { font-size: 1.35rem; font-weight: normal; letter-spacing: .04em;
  margin: 0; }
.hub-header h1 b { font-weight: bold; }
.hub-header .sub { font-family: var(--mono); font-size: .68rem; color: var(--muted);
  text-align: right; letter-spacing: .04em; }
.figtitle { font-family: var(--mono); font-size: .7rem; letter-spacing: .12em;
  color: var(--muted); text-transform: uppercase; margin: 2px 0 8px; }

/* the station strip IS the simplified ladder */
.map-strip { display: flex; gap: 0; margin: 0 0 14px; flex-wrap: wrap; }
.map-station { font-family: var(--mono); border: 1px solid var(--ink);
  border-right: none; padding: 7px 12px 6px; background: var(--paper);
  cursor: pointer; min-width: 118px; }
.map-station:last-child { border-right: 1px solid var(--ink); }
.map-station.optional { border-style: dashed; background: var(--paper); }
.map-station:hover { background: var(--amber-soft); }
.map-station b { display: block; font-size: .68rem; letter-spacing: .08em; }
.map-station .why { display: block; font-family: var(--serif); font-style: italic;
  font-size: .68rem; color: var(--muted); margin: 2px 0 3px; }
.pill { display: inline-block; font-family: var(--mono); font-size: .62rem;
  padding: 1px 7px; border: 1px solid var(--line); letter-spacing: .04em;
  margin: 0 4px 4px 0; }
.pill.ok { background: var(--green-soft); color: var(--green); border-color: transparent; }
.pill.warn { background: var(--amber-soft); color: var(--amber); border-color: transparent; }
.pill.bad { background: var(--red-soft); color: var(--red); border-color: transparent; }
.pill.info { background: var(--blue-soft); color: var(--blue); border-color: transparent; }

.hub-card { border: 1px solid var(--ink); padding: 13px 15px; margin-bottom: 13px;
  background: var(--paper); }
.hub-card h4 { font-family: var(--mono); font-size: .68rem; letter-spacing: .14em;
  text-transform: uppercase; border-bottom: 1px solid var(--ink);
  padding-bottom: 6px; margin: 0 0 10px; font-weight: normal; }
.note { border-left: 3px solid var(--ink); padding: 5px 10px; font-size: .82rem;
  color: var(--muted); font-style: italic; margin: 9px 0; }
.error-box { border-left: 3px solid var(--red); padding: 6px 10px;
  font-size: .78rem; color: var(--red); margin: 9px 0;
  font-family: var(--mono); white-space: pre-wrap; }
.hub-table { width: 100%; border-collapse: collapse; font-family: var(--mono);
  font-size: .74rem; margin: 6px 0; }
.hub-table th { text-align: left; border-bottom: 2px solid var(--ink);
  padding: 4px 8px; font-weight: normal; letter-spacing: .1em; font-size: .66rem;
  text-transform: uppercase; }
.hub-table td { border-bottom: 1px solid var(--line); padding: 4px 8px; }
.hub-table tr.dim td { color: var(--muted); }

.nav-tabs { border-bottom: 1px solid var(--ink) !important; margin-bottom: 14px; }
.nav-tabs .nav-link { font-family: var(--mono); font-size: .74rem;
  letter-spacing: .06em; color: var(--ink); border: none !important;
  border-radius: 0 !important; padding: 8px 14px; }
.nav-tabs .nav-link.active { background: var(--ink) !important;
  color: var(--paper) !important; }
.nav-tabs .nav-link:hover:not(.active) { background: var(--amber-soft); }

.btn, .btn-sm, button.btn { font-family: var(--mono) !important;
  font-size: .74rem !important; border: 1px solid var(--ink) !important;
  border-radius: 0 !important; background: var(--paper) !important;
  color: var(--ink) !important; letter-spacing: .06em; padding: 6px 12px !important; }
.btn:hover, .btn-sm:hover { background: var(--amber-soft) !important; }
.form-select, .form-control { font-family: var(--mono); font-size: .76rem;
  border: 1px solid var(--ink); border-radius: 0; }

.figure-img { max-width: 100%; border: 1px solid var(--ink); }
.code-details { margin: 8px 0; }
.code-details summary { font-family: var(--mono); font-size: .66rem;
  text-transform: uppercase; letter-spacing: .1em; color: var(--muted);
  cursor: pointer; }
.code-block { background: #1b1a16; color: #e8e4da; padding: 10px 12px;
  font-family: var(--mono); font-size: .72rem; line-height: 1.5;
  overflow-x: auto; margin: 6px 0 0; white-space: pre; }
.theater-static svg { background: var(--paper); }
</style>
"""


def card(title: str, *body: str, annotation: str = "") -> str:
    inner = "".join(body)
    right = f"<i style='font-style:italic;color:var(--muted)'>{html.escape(annotation)}</i>" if annotation else ""
    return (
        f'<div class="hub-card"><h4 style="display:flex;justify-content:space-between">'
        f"<span>{html.escape(title)}</span>{right}</h4>{inner}</div>"
    )


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill {kind}">{html.escape(text)}</span>'


def note(text: str) -> str:
    return f'<div class="note">{text}</div>'


def error_box(message: str, trace: str = "") -> str:
    detail = f"\n{html.escape(trace)}" if trace else ""
    return f'<div class="error-box">{html.escape(message)}{detail}</div>'


def figtitle(text: str) -> str:
    return f'<div class="figtitle">{html.escape(text)}</div>'


def table(rows: Sequence[Mapping[str, object]], columns: Sequence[str],
          *, dim_when: str | None = None, limit: int = 40) -> str:
    if not rows:
        return '<p style="color:var(--muted);font-size:.8rem">nothing to show</p>'
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body: list[str] = []
    for row in list(rows)[:limit]:
        dim = ' class="dim"' if dim_when and row.get(dim_when) else ""
        cells = "".join(f"<td>{_cell(row.get(c, ''))}</td>" for c in columns)
        body.append(f"<tr{dim}>{cells}</tr>")
    more = (
        f'<p style="color:var(--muted);font-size:.7rem">… and {len(rows) - limit} more</p>'
        if len(rows) > limit else ""
    )
    return f'<table class="hub-table"><tr>{head}</tr>{"".join(body)}</table>{more}'


def _cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, bool):
        return "✓" if value else "—"
    return html.escape(str(value))


def code_card(snippet: str) -> str:
    """Collapsible 'what actually ran' block; snippet text is escaped."""
    if not snippet:
        return ""
    return (
        '<details class="code-details"><summary>Show the code that ran</summary>'
        f'<pre class="code-block">{html.escape(snippet)}</pre></details>'
    )


def figure(b64: str | None, alt: str) -> str:
    if not b64:
        return note("chart unavailable (matplotlib missing)")
    return f'<img class="figure-img" alt="{html.escape(alt)}" src="data:image/png;base64,{b64}"/>'
