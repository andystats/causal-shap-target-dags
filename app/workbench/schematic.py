"""Map tab: the technical-paper schematic, embedded and clickable.

Stations are <g class="station" data-tab="..."> groups; clicking one fires
Shiny.setInputValue('goto_tab', tab) and the server switches the nav panel.
Station numbering matches the six-step teaching flow in the Workbench.
"""

from shiny import ui

INK = "#111111"
AMBER = "#b45309"


def _station(x, y, w, h, num, title, sub, tab, double=False):
    inner = (f'<rect x="{x+3}" y="{y+3}" width="{w-6}" height="{h-6}" '
             f'fill="none" stroke="{INK}" stroke-width="0.7"/>' if double else "")
    sub_html = (f'<text x="{x+w/2}" y="{y+h-14}" text-anchor="middle" '
                f'class="st-sub">{sub}</text>' if sub else "")
    return f'''
    <g class="station" data-tab="{tab}">
      <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white"
            stroke="{INK}" stroke-width="1.4" class="st-box"/>
      {inner}
      <text x="{x+w/2}" y="{y+26}" text-anchor="middle" class="st-title">{title}</text>
      {sub_html}
      <text x="{x+w-8}" y="{y+h-7}" text-anchor="end" class="st-num">{num}</text>
    </g>'''


def _arrow(x1, y1, x2, y2, dashed=False):
    dash = 'stroke-dasharray="6 4"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK}" '
            f'stroke-width="1.2" {dash} marker-end="url(#arr)"/>')


def schematic_svg():
    s = []
    # sealed-world strip (informational, not clickable)
    s.append(f'<rect x="20" y="18" width="920" height="74" fill="none" '
             f'stroke="{INK}" stroke-width="0.8" stroke-dasharray="3 3"/>')
    s.append(f'<text x="34" y="40" class="st-strip">(0)&#160;&#160;THE SEALED WORLD '
             f'&#8212; ground truths ship with the bundles and never touch the learners</text>')
    s.append(f'<text x="34" y="62" class="st-sub">simcausal true DAG + frozen total effects '
             f'&#183; your own upload&#8217;s separate reference graph</text>')
    s.append(f'<text x="34" y="80" class="st-sub" fill="{AMBER}">the answer key drops into '
             f'station 4 only &#8212; evaluation, never learning</text>')

    # main flow, two rows of three
    s.append(_station(20, 130, 280, 96, "1", "DATA", "load a bundle; choose variables", "Data"))
    s.append(_station(340, 130, 280, 96, "2", "DISCOVER", "PC &#183; DirectLiNGAM &#183; GES + constraint ledger", "Discover"))
    s.append(_station(660, 130, 280, 96, "3", "GRAPH", "inspect nodes; forge quick constraints", "Graph", double=True))
    s.append(_station(660, 300, 280, 96, "4", "EVALUATE", "M1&#8211;M2 concordance &#183; M3&#8211;M5 structural importance", "Evaluate"))
    s.append(_station(340, 300, 280, 96, "5", "EXPORT", "Shrier&#8211;Platt sets &#183; DAGitty", "Export"))
    s.append(_station(20, 300, 280, 96, "6", "CAUSAL SHAP", "knowing vs setting: do()-attribution", "Causal SHAP"))

    # flow arrows
    s.append(_arrow(300, 178, 336, 178))
    s.append(_arrow(620, 178, 656, 178))
    s.append(_arrow(800, 226, 800, 296))
    s.append(_arrow(656, 348, 624, 348))
    s.append(_arrow(336, 348, 304, 348))
    # expert loop-back: graph -> discover (dashed)
    s.append(f'<path d="M 700 226 L 700 262 L 480 262 L 480 230" fill="none" '
             f'stroke="{INK}" stroke-width="1.1" stroke-dasharray="5 4" '
             f'marker-end="url(#arr)"/>')
    s.append('<text x="590" y="256" text-anchor="middle" class="st-lab">revise constraints; re-run</text>')
    # sealed answer key into evaluate
    s.append(_arrow(880, 92, 880, 296, dashed=True))
    s.append(f'<text x="872" y="270" text-anchor="end" class="st-lab" fill="{AMBER}">the sealed answer key</text>')
    # caption
    s.append(f'<text x="480" y="440" text-anchor="middle" class="st-cap">FIG. W1 '
             f'&#8212; the Workbench is the machine it maps. Click a station to enter it.</text>')

    return f'''
<svg viewBox="0 0 960 456" style="width:100%;max-width:1060px;display:block;margin:0 auto;">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
            markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/>
    </marker>
  </defs>
  <style>
    .st-title {{ font-family: Georgia, 'Times New Roman', serif; font-size: 17px;
                 font-weight: bold; letter-spacing: 0.06em; fill: {INK}; }}
    .st-sub   {{ font-family: Georgia, serif; font-style: italic; font-size: 11.5px;
                 fill: #444; }}
    .st-num   {{ font-family: 'Courier New', monospace; font-style: italic;
                 font-size: 13px; fill: {INK}; }}
    .st-strip {{ font-family: Georgia, serif; font-weight: bold; font-size: 13px;
                 letter-spacing: 0.05em; fill: {INK}; }}
    .st-lab   {{ font-family: Georgia, serif; font-style: italic; font-size: 11.5px;
                 fill: #444; }}
    .st-cap   {{ font-family: Georgia, serif; font-size: 13px; font-weight: bold;
                 fill: {INK}; }}
    .station  {{ cursor: pointer; }}
    .station:hover .st-box {{ stroke: {AMBER}; stroke-width: 2.2; }}
    .station:hover .st-title {{ fill: {AMBER}; }}
  </style>
  {''.join(s)}
</svg>
<script>
(function attach() {{
  document.querySelectorAll('.station').forEach(function (el) {{
    el.addEventListener('click', function () {{
      if (window.Shiny) {{
        Shiny.setInputValue('goto_tab', el.dataset.tab, {{priority: 'event'}});
      }}
    }});
  }});
}})();
</script>'''


def map_panel():
    return ui.nav_panel(
        "Map",
        ui.HTML('<div class="card" style="padding:28px 20px;">'
                + schematic_svg() +
                '</div>'),
        ui.HTML('''<div class="info-box">
            The dashed strip is Phase 0. The bundled simulation carries a
            sealed truth (a known DAG and frozen effects) that only Evaluate
            may open; uploaded data require a separate reference edge list.
        </div>'''),
    )
