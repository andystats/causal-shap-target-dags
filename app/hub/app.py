"""The guided causal-discovery hub.

Seven stages, one storyline: what the model listened to, what the data can say
about structure, where a detector says to look twice, what a human decides in
the theater, what attribution looks like once interventions propagate, and
which affordable action survives a budget. Every expensive computation runs in
an extended task off the main thread; renders only read.
"""

from __future__ import annotations

import asyncio
import html
import os
import traceback
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path

import pandas as pd
from shiny import App, reactive, render, req, ui

from causal_shap.action_costs import CostModel
from causal_shap.seeds import SEED_ACTION_ABDUCTION, SEED_HUB_DEMO
from hub import education, report, snippets, stages, state, theater
from hub import ui as skin
from hub.datasets import DATASETS, UPLOAD_ID, HubDataset, sd_cost_specs

# The public hub ships a vendor-neutral detector explainer; a local, excluded
# module supplies the specific story on machines that have the runtime.
if find_spec("detector_docs_local"):
    from detector_docs_local import DETECTOR_DOCS_HTML
else:
    DETECTOR_DOCS_HTML = ""

# The station strip is the simplified workflow ladder: names and status only,
# the narration lives in each tab's Learn dropdown.
STATIONS = (
    ("Data", "data"),
    ("Naive SHAP", "naive"),
    ("Discover", "discover"),
    ("Surgical Prep", "flags"),
    ("Graph Surgery", "surgery"),
    ("Causal SHAP", "attribute"),
    ("Price & Dice", "policy"),
)

THEATER_KEY = """
<div class="hub-card" style="padding:9px 15px"><h4>KEY</h4>
<div style="font-size:.78rem;display:flex;gap:16px;flex-wrap:wrap;align-items:center">
<span><span style="display:inline-block;width:16px;height:11px;background:#fdf3e7;border:1.5px solid #b45309;vertical-align:-1px"></span> outcome</span>
<span><span style="display:inline-block;width:16px;height:11px;background:#fff;border:1.5px solid #1e4d8c;vertical-align:-1px"></span> nominated lever (last node clicked)</span>
<span><span style="display:inline-block;width:16px;height:11px;background:#fff;border:2.5px solid #b45309;vertical-align:-1px"></span> selected for surgery</span>
<span><span style="display:inline-block;width:22px;border-bottom:2px dashed #111;vertical-align:3px"></span> orientation chosen, not identified (unresolved pair)</span>
<span><span style="display:inline-block;width:12px;height:12px;border:2px dashed #b45309;border-radius:50%;vertical-align:-2px"></span><span style="display:inline-block;width:12px;height:12px;border:2px dashed #1e4d8c;border-radius:50%;vertical-align:-2px;margin-left:2px"></span><span style="display:inline-block;width:12px;height:12px;border:2px dashed #94a3b8;border-radius:50%;vertical-align:-2px;margin-left:2px"></span>
 dashed halo rings = Surgical Prep flags by channel (amber = h0, blue = h1, grey = eig): "look here again", never a causal claim</span>
</div></div>
"""

MAP_JS = """
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<script>
// Sliders initialized inside a collapsed <details> get zero width; nudge them.
document.addEventListener('toggle', function() {
  window.dispatchEvent(new Event('resize'));
}, true);
document.addEventListener('click', function(event) {
  const station = event.target.closest('.map-station');
  if (station && window.Shiny) {
    Shiny.setInputValue('goto_stage', station.dataset.stage, {priority: 'event'});
  }
});
</script>
"""

app_ui = ui.page_fluid(
    ui.HTML(skin.HUB_CSS),
    ui.HTML(MAP_JS),
    ui.HTML(
        '<div class="hub-header"><h1><b>CAUSAL SHAP</b> HUB</h1>'
        '<div class="sub">GUIDED CAUSAL DISCOVERY · v0.2<br>the machine is the map</div></div>'
    ),
    ui.output_ui("map_strip"),
    ui.div(
        ui.download_button("download_report", "Export session report",
                           class_="btn-sm"),
        ui.input_action_button("goto_next", "Next", class_="btn-sm", disabled=True),
        style="margin:0 0 10px;display:flex;gap:8px",
    ),
    ui.navset_tab(
        ui.nav_panel(
            "Data",
            ui.layout_columns(
                ui.div(
                    ui.input_radio_buttons(
                        "dataset_choice",
                        "Dataset",
                        choices={
                            **{d.id: d.label for d in DATASETS.values()},
                            UPLOAD_ID: "Upload a CSV…",
                        },
                        selected="toy_trap",
                    ),
                    ui.panel_conditional(
                        f"input.dataset_choice === '{UPLOAD_ID}'",
                        ui.input_file("data_upload", "Data CSV (drag & drop works)",
                                      accept=[".csv"], multiple=False),
                        ui.input_file("dict_upload",
                                      "Optional data dictionary (column,description)",
                                      accept=[".csv"], multiple=False),
                    ),
                    ui.output_ui("pickers"),
                ),
                ui.div(
                    ui.output_ui("data_summary"),
                    ui.output_ui("dict_table"),
                    ui.div(
                        ui.input_text_area(
                            "sandbox_code", "Local sandbox (runs on this machine only)",
                            value="df.describe().T.round(2)", rows=4, width="100%"),
                        ui.input_action_button("run_sandbox_code", "Run in sandbox",
                                               class_="btn-sm"),
                        ui.output_ui("sandbox_out"),
                        ui.HTML(education.learn("sandbox", title="Learn: the sandbox")),
                        style="border:1px solid var(--ink);padding:13px 15px;margin-bottom:13px",
                    ),
                ),
                col_widths=(5, 7),
            ),
        ),
        ui.nav_panel(
            "Naive SHAP",
            ui.layout_columns(
                ui.div(
                    ui.input_action_button("run_naive", "Run naive SHAP",
                                           class_="btn-sm btn-go"),
                    ui.tags.details(
                        ui.tags.summary("Advanced settings"),
                        ui.input_select("model_type", "Model",
                                        {"gbm": "Gradient boosting",
                                         "rf": "Random forest",
                                         "linear": "Linear / logistic"}),
                        class_="code-details",
                    ),
                    ui.HTML(skin.note(
                        "The benchmark: what the fitted predictor listened to. "
                        "The causal stages will argue with this chart."
                    )),
                    ui.HTML(education.learn("shap", "naive", "holdout")),
                ),
                ui.output_ui("naive_body"),
                col_widths=(4, 8),
            ),
        ),
        ui.nav_panel(
            "Discover",
            ui.layout_columns(
                ui.div(
                    ui.input_radio_buttons("algorithm", "Algorithm",
                                           {"pc": "PC (Fisher-Z)", "ges": "GES (BIC)"}),
                    ui.input_action_button("run_discover", "Discover structure",
                                           class_="btn-sm btn-go"),
                    ui.tags.details(
                        ui.tags.summary("Advanced settings"),
                        ui.input_slider("alpha", "PC significance α", min=0.01,
                                        max=0.2, value=0.05, step=0.01),
                        class_="code-details",
                    ),
                    ui.output_ui("adopt_truth_control"),
                    ui.HTML(education.learn("pc", "ges", "alpha_pc", "cpdag")),
                ),
                ui.div(
                    ui.output_ui("discover_body"),
                    ui.output_ui("truth_view"),
                ),
                col_widths=(4, 8),
            ),
        ),
        ui.nav_panel(
            "Surgical Prep",
            ui.layout_columns(
                ui.div(
                    ui.HTML(skin.pill("optional station", "info")),
                    ui.input_select("flag_provider", "Detector",
                                    {"auto": "auto-select", "null": "none (honest empty)",
                                     "precomputed": "frozen tables", "local": "local live"}),
                    ui.input_action_button("run_flags", "Run depth detector",
                                           class_="btn-sm btn-go"),
                    ui.HTML(skin.note(
                        "Optional: gather node-level evidence from ANY source (a "
                        "depth detector, the literature, expert priors) to decide "
                        "where the surgeon looks first. Advisory, never a causal "
                        "claim; skipping this stage blocks nothing downstream."
                    )),
                    ui.HTML(education.learn("prep", "channels")),
                ),
                ui.output_ui("flags_body"),
                col_widths=(4, 8),
            ),
            ui.output_ui("detector_docs"),
        ),
        ui.nav_panel(
            "Graph Surgery",
            ui.output_ui("theater_view"),
            ui.HTML(THEATER_KEY),
            ui.layout_columns(
                ui.div(
                    ui.output_ui("selection_info"),
                    ui.input_text("rationale", "Rationale (goes into the ledger)",
                                  placeholder="why this judgement"),
                    ui.div(
                        ui.input_action_button("act_flip", "Flip", class_="btn-sm"),
                        ui.input_action_button("act_require", "Require", class_="btn-sm"),
                        ui.input_action_button("act_forbid", "Forbid", class_="btn-sm"),
                        ui.input_action_button("act_remove", "Remove", class_="btn-sm"),
                        ui.input_action_button("act_reset", "Reset graph", class_="btn-sm"),
                        style="display:flex;gap:6px;flex-wrap:wrap;margin-top:4px",
                    ),
                    ui.output_ui("add_edge_form"),
                    ui.HTML(skin.code_card(snippets.surgery_snippet())),
                    ui.HTML(education.learn("surgery", "ledger", "cpdag")),
                ),
                ui.output_ui("scorecard"),
                col_widths=(5, 7),
            ),
        ),
        ui.nav_panel(
            "Causal SHAP",
            ui.layout_columns(
                ui.div(
                    ui.input_radio_buttons("arm", "Attribution arm",
                                           {"structural": "structural (do-propagation)",
                                            "nonparametric": "nonparametric (conditional models)"}),
                    ui.input_action_button("run_shap", "Run causal SHAP",
                                           class_="btn-sm btn-go"),
                    ui.tags.details(
                        ui.tags.summary("Advanced settings"),
                        ui.input_slider("n_perms", "Permutations", 8, 128, 32, step=8),
                        ui.input_slider("n_instances", "Explained rows", 8, 128, 32,
                                        step=8),
                        ui.input_slider("n_background", "Background draws", 8, 64, 16,
                                        step=8),
                        ui.HTML(skin.note(
                            "Monte Carlo budget, not science settings; the defaults "
                            "are the proven demo scale and rerun reproducibly. Scale "
                            "them up off-stage for smoother numbers."
                        )),
                        class_="code-details",
                    ),
                    ui.output_ui("shap_preflight"),
                    ui.HTML(education.learn("arms", "ancestors", "knobs")),
                ),
                ui.output_ui("shap_body"),
                col_widths=(4, 8),
            ),
        ),
        ui.nav_panel(
            "Price & Dice",
            ui.layout_columns(
                ui.div(
                    ui.output_ui("policy_controls"),
                    ui.input_action_button("run_policy", "Rank affordable actions",
                                           class_="btn-sm btn-go"),
                    ui.tags.details(
                        ui.tags.summary("Advanced settings"),
                        ui.output_ui("policy_advanced"),
                        ui.input_file("cost_upload_file", "Replace cost sheet (CSV)",
                                      accept=[".csv"], multiple=False),
                        class_="code-details",
                    ),
                    ui.output_ui("cost_preview"),
                    ui.HTML(skin.note(
                        "Each lever appears twice: it is priced in both allowed "
                        "directions (its +shift and its -shift), and the losing "
                        "direction keeps its refusal reason rather than vanishing."
                    )),
                    ui.HTML(education.learn("cost_sheet", "grid", "budget",
                                            "alpha_floor", "benefit")),
                ),
                ui.output_ui("policy_body"),
                col_widths=(4, 8),
            ),
        ),
        id="stage_nav",
    ),
)


def server(input, output, session):  # noqa: C901 - the wiring hub
    upload_frame = reactive.Value(None)
    upload_gen = reactive.Value(0)          # bumps per upload: same-schema files differ
    dictionary = reactive.Value({})
    current_graph = reactive.Value(None)
    baseline_graph = reactive.Value(None)   # what Reset restores; dataset-scoped
    graph_gen = reactive.Value(0)           # bumps when the graph story changes underfoot
    discover_launch = reactive.Value((-1, ""))
    picked = reactive.Value("")
    focus_node = reactive.Value(None)       # candidate lever under interrogation
    flags_result = reactive.Value(state.StageResult())
    cost_override = reactive.Value(None)
    fps = {name: reactive.Value("") for name in ("naive", "discover", "attribute", "policy")}
    # "What actually ran": captured at launch from the same values the worker
    # receives, so the card can never drift from the computation.
    snips = {name: reactive.Value("")
             for name in ("naive", "discover", "flags", "attribute", "policy")}

    # ------------------------------------------------------------------ data
    @reactive.calc
    def bundle() -> HubDataset | None:
        return DATASETS.get(input.dataset_choice())

    @reactive.calc
    def raw_data() -> pd.DataFrame | None:
        chosen = bundle()
        if chosen is not None:
            return chosen.load_data()
        return upload_frame.get()

    @reactive.calc
    def numeric_columns() -> list[str]:
        data = raw_data()
        req(data is not None)
        excluded = set(bundle().excluded_columns) if bundle() else set()
        columns = data.select_dtypes(include="number").columns
        keep = [c for c in columns if c not in excluded]
        # Integer columns with a distinct value per row are IDs, not measurements;
        # a continuous column is legitimately all-unique and must survive.
        return [
            c for c in keep
            if not (pd.api.types.is_integer_dtype(data[c]) and data[c].nunique() == len(data))
        ]

    @reactive.calc
    def analysis_data() -> pd.DataFrame:
        # NOT complete-cased here: one sparse, deselected column must not
        # delete the cohort. Every stage drops rows on its own required
        # columns only.
        data = raw_data()
        req(data is not None)
        return data[numeric_columns()]

    @reactive.calc
    def data_fp() -> str:
        data = raw_data()
        req(data is not None)
        return state.fingerprint(
            input.dataset_choice(), data.shape, tuple(data.columns), upload_gen.get()
        )

    @reactive.effect
    @reactive.event(input.data_upload)
    def _load_upload():
        files = input.data_upload()
        if files:
            upload_frame.set(pd.read_csv(files[0]["datapath"]))
            upload_gen.set(upload_gen.get() + 1)
            current_graph.set(None)
            baseline_graph.set(None)
            focus_node.set(None)
            graph_gen.set(graph_gen.get() + 1)
            picked.set("")
            cost_override.set(None)
            dictionary.set({})
            flags_result.set(state.StageResult())

    @reactive.effect
    @reactive.event(input.dict_upload)
    def _load_dictionary():
        files = input.dict_upload()
        if not files:
            return
        sheet = pd.read_csv(files[0]["datapath"])
        lower = {c.lower(): c for c in sheet.columns}
        if "column" in lower and "description" in lower:
            entries = {
                str(column).strip(): str(description)
                for column, description in zip(
                    sheet[lower["column"]], sheet[lower["description"]]
                )
                if pd.notna(description) and str(description).strip()
            }
            dictionary.set(entries)
        else:
            ui.notification_show(
                "Dictionary needs 'column' and 'description' columns", type="warning"
            )

    @reactive.effect
    @reactive.event(input.dataset_choice)
    def _reset_on_dataset_change():
        current_graph.set(None)
        baseline_graph.set(None)
        focus_node.set(None)
        graph_gen.set(graph_gen.get() + 1)
        picked.set("")
        cost_override.set(None)
        dictionary.set({})
        flags_result.set(state.StageResult())

    @render.ui
    def pickers():
        columns = numeric_columns()
        if not columns:
            return ui.HTML(skin.note("No usable numeric columns in this data."))
        chosen = bundle()
        outcome = chosen.default_outcome if chosen else columns[-1]
        features = [c for c in columns if c != outcome]
        # Deliberately no lever/exposure picker here: which node is worth
        # acting on is what the pipeline DISCOVERS. Candidate levers are
        # interrogated by clicking nodes in the theater, and price-and-dice
        # searches every manipulable ancestor without being told one.
        return ui.div(
            ui.input_select("outcome", "Outcome (the target)", columns, selected=outcome),
            ui.input_checkbox_group("features", f"Features ({len(features)})",
                                    columns, selected=features),
        )

    @reactive.calc
    def features() -> tuple[str, ...]:
        selected = tuple(c for c in input.features() if c != input.outcome())
        req(selected)
        return selected

    @render.ui
    def data_summary():
        data = raw_data()
        if data is None:
            return ui.HTML(skin.note("Pick a bundled dataset or drop a CSV to begin."))
        chosen = bundle()
        pills = [
            skin.pill(f"{len(data):,} rows", "info"),
            skin.pill(f"{len(numeric_columns())} usable columns", "info"),
            skin.pill("cross-sectional v1 — one row per unit, no time ordering", "warn"),
        ]
        if chosen and chosen.truth_graph() is not None:
            pills.append(skin.pill("answer key available", "ok"))
        note = chosen.note if chosen else "Uploaded data: no answer key, no detector table."
        return ui.HTML(skin.card("This dataset", "".join(pills), skin.note(note)))

    @render.ui
    def dict_table():
        entries = dictionary.get()
        if not entries:
            return ui.HTML("")
        rows = [{"column": k, "description": v} for k, v in entries.items()]
        return ui.HTML(skin.card("Data dictionary", skin.table(rows, ["column", "description"])))

    sandbox_result = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.run_sandbox_code)
    def _run_sandbox():
        frame = raw_data()
        if frame is None:
            ui.notification_show("Load a dataset first.", type="warning")
            return
        sandbox_result.set(stages.run_sandbox(input.sandbox_code(), frame))

    @render.ui
    def sandbox_out():
        result = sandbox_result.get()
        if result is None:
            return ui.HTML("")
        parts = []
        if not result["ok"]:
            parts.append(skin.pill(result["text"], "bad"))
        else:
            parts.append(
                '<pre style="font-size:.72rem;white-space:pre-wrap;'
                'max-height:280px;overflow:auto;margin:8px 0 0">'
                f'{html.escape(result["text"])}</pre>'
            )
            parts.extend(
                f'<img src="data:image/png;base64,{figure}" style="max-width:100%">'
                for figure in result["figures"]
            )
        return ui.HTML("".join(parts))

    @render.ui
    def detector_docs():
        return ui.HTML(DETECTOR_DOCS_HTML or education.GENERIC_DETECTOR_DOCS)

    # ----------------------------------------------------------- extended tasks
    @reactive.extended_task
    async def naive_task(kwargs: dict):
        return await asyncio.to_thread(lambda: stages.run_naive_shap(**kwargs))

    @reactive.extended_task
    async def discover_task(kwargs: dict):
        return await asyncio.to_thread(lambda: stages.run_discovery(**kwargs))

    @reactive.extended_task
    async def flags_task(kwargs: dict):
        return await asyncio.to_thread(lambda: stages.run_flags(**kwargs))

    @reactive.extended_task
    async def shap_task(kwargs: dict):
        return await asyncio.to_thread(lambda: stages.run_causal_shap(**kwargs))

    @reactive.extended_task
    async def policy_task(kwargs: dict):
        return await asyncio.to_thread(lambda: stages.run_policy(**kwargs))

    def _launch(task, name: str, fingerprint: str, kwargs: dict):
        if task.status() == "running":
            ui.notification_show("Already running — one at a time.", type="warning")
            return
        fps[name].set(fingerprint)
        task.invoke(kwargs)

    # ------------------------------------------------------------- naive stage
    @reactive.effect
    @reactive.event(input.run_naive)
    def _run_naive():
        kwargs = dict(
            data=analysis_data(), features=features(), outcome=input.outcome(),
            model_type=input.model_type(),
        )
        snips["naive"].set(snippets.naive_snippet(
            features(), input.outcome(), input.model_type(), 100, SEED_HUB_DEMO))
        _launch(naive_task, "naive",
                state.fingerprint(data_fp(), features(), input.outcome(), input.model_type()),
                kwargs)

    @render.ui
    def naive_body():
        return _task_body(naive_task, "Run naive SHAP to open the story.", _naive_html,
                          code=snips["naive"].get())

    def _naive_html(payload: dict) -> str:
        fit = payload["fit"]
        shares = sorted(payload["shares"].items(), key=lambda p: -p[1])
        top = ", ".join(
            f"{html.escape(str(name))} {share:.1f}%" for name, share in shares[:3]
        )
        return skin.card(
            "What the model listened to",
            skin.pill(f"{fit.model_type} · {fit.task}", "info"),
            skin.pill(f"{fit.stat_name} {fit.stat_value:.3f}", "ok"),
            skin.figure(payload["plot"], "naive SHAP"),
            skin.note(f"Top credit: {top}. Prediction credit is not an intervention target."),
        )

    # ---------------------------------------------------------- discover stage
    @reactive.effect
    @reactive.event(input.run_discover)
    def _run_discover():
        chosen = bundle()
        kwargs = dict(
            data=analysis_data(), features=features(), outcome=input.outcome(),
            algorithm=input.algorithm(), alpha=float(input.alpha()),
            truth=chosen.truth_graph() if chosen else None,
        )
        fingerprint = state.fingerprint(data_fp(), features(), input.outcome(),
                                        input.algorithm(), input.alpha())
        snips["discover"].set(snippets.discover_snippet(
            features(), input.outcome(), input.algorithm(), float(input.alpha())))
        discover_launch.set((graph_gen.get(), fingerprint))
        _launch(discover_task, "discover", fingerprint, kwargs)

    @reactive.effect
    def _adopt_discovered():
        # Adopt a finished discovery only if nothing changed underfoot: same
        # dataset/inputs, and no surgery or truth adoption raced the run.
        # A stale success stays in the task (the map chip says stale); it must
        # never clobber a newer graph and silently discard its ledger.
        if discover_task.status() != "success":
            return
        launch_gen, launch_fp = discover_launch.get()
        if launch_gen != graph_gen.get() or launch_fp != _live_fp("discover"):
            return
        adopted = discover_task.value.get()["graph"]
        current_graph.set(adopted)
        baseline_graph.set(adopted)
        picked.set("")

    @render.ui
    def adopt_truth_control():
        chosen = bundle()
        if chosen is None or chosen.truth_graph() is None:
            return ui.HTML("")
        return ui.div(
            ui.input_action_button("adopt_truth", "Use the known truth graph",
                                   class_="btn-sm"),
            style="margin-top:8px",
        )

    @reactive.effect
    @reactive.event(input.adopt_truth)
    def _adopt_truth():
        chosen = bundle()
        if chosen and chosen.truth_graph() is not None:
            truth = chosen.truth_graph()
            current_graph.set(truth)
            baseline_graph.set(truth)
            graph_gen.set(graph_gen.get() + 1)
            picked.set("")
            ui.notification_show("Adopted the answer-key graph.", type="message")

    @render.ui
    def discover_body():
        return _task_body(discover_task, "Run discovery, or adopt the truth graph.",
                          _discover_html, code=snips["discover"].get())

    def _discover_html(payload: dict) -> str:
        graph = payload["graph"]
        chosen = bundle()
        parts = [
            skin.pill(f"{len(graph.directed_edges)} edges", "info"),
            skin.pill(f"{graph.n_undirected_pairs} pairs undirected", "warn"),
        ]
        banner = skin.note(
            "The algorithm returns an equivalence class. Dashed edges are one "
            "deterministic representative — choices, not findings."
        )
        m1 = payload.get("m1")
        m1_html = _m1_html(m1, graph.n_undirected_pairs) if m1 else ""
        discovered_svg = theater.render_theater(
            graph, None, input.outcome(),
            display_names=dict(chosen.display_names()) if chosen else {},
            tooltips=dictionary.get(),
            interactive=False, height=280,
        )
        return skin.card(
            "Discovered structure", "".join(parts), banner, m1_html, discovered_svg,
            skin.note("Operate on this graph in Graph Surgery."),
        )

    @render.ui
    def truth_view():
        chosen = bundle()
        if chosen is None or chosen.truth_graph() is None:
            return ui.HTML("")
        truth = chosen.truth_graph()
        svg = theater.render_theater(
            truth, None, input.outcome(),
            display_names=dict(chosen.display_names()),
            interactive=False, height=300,
        )
        return ui.HTML(
            "<details class='code-details'><summary>Sealed answer key (known truth)"
            "</summary>" + skin.note(
                "The graph the data were generated from. Discovery never sees it; "
                "it exists so every claim in this session is checkable."
            ) + svg + "</details>"
        )

    def _m1_html(m1: dict, n_undirected: int) -> str:
        caveat = " (representative-dependent)" if n_undirected else ""
        return (
            skin.pill(f"skeleton F1 {m1['skeleton_f1']:.2f}", "ok")
            + skin.pill(f"directed F1 {m1['f1']:.2f}{caveat}", "warn" if n_undirected else "ok")
            + skin.pill(f"SHD {m1['shd']}", "info")
        )

    # ------------------------------------------------------------- flags stage
    flags_launch_fp = reactive.Value("")

    @reactive.effect
    @reactive.event(input.run_flags)
    def _run_flags():
        if flags_task.status() == "running":
            ui.notification_show("Detector already running.", type="warning")
            return
        chosen = bundle()
        root = os.environ.get("CAUSAL_SHAP_BLOCK_ROOT", "")
        preference = None if input.flag_provider() == "auto" else input.flag_provider()
        snips["flags"].set(snippets.flags_snippet(
            chosen.flags_id if chosen else "upload", input.outcome(), preference))
        flags_launch_fp.set(_flags_fp())
        flags_task.invoke(dict(
            flags_id=chosen.flags_id if chosen else "upload",
            outcome=input.outcome(), feature_names=features(),
            block_root=Path(root) if root else None,
            preference=preference,
            data=analysis_data(),   # a live detector trains on it
        ))

    @reactive.effect
    def _harvest_flags():
        status = flags_task.status()
        if status == "success":
            flags_result.set(state.StageResult(
                status=state.OK, payload=flags_task.value.get(),
                fingerprint=flags_launch_fp.get(),
            ))
        elif status == "error":
            error = flags_task.error.get()
            flags_result.set(state.StageResult(
                status=state.ERROR, error=f"{type(error).__name__}: {error}",
            ))

    @render.ui
    def flags_body():
        if flags_task.status() == "running":
            return ui.HTML(
                skin.card("Working…",
                          skin.pill("detector running on a worker", "warn"),
                          skin.note("A live run trains a model from scratch; "
                                    "minutes, not seconds."))
                + skin.code_card(snips["flags"].get())
            )
        result = flags_result.get()
        if result.status == state.EMPTY:
            return ui.HTML(skin.note("Run the detector to see per-node depth flags."))
        if result.status == state.ERROR:
            return ui.HTML(skin.error_box(result.error, result.traceback))
        payload = result.payload
        if payload["status"] != "ok":
            return ui.HTML(skin.card(
                "Detector", skin.pill(payload["status"], "warn"),
                skin.note(html.escape(payload["message"] or "no detector ran for this dataset")),
            ))
        columns = ["feature", "h0_z", "h0_flagged", "h1_z", "h1_flagged",
                   "eig_z", "eig_flagged", "complexity", "flagged"]
        body = [
            skin.pill(payload["provider"], "info"),
            skin.pill("channels differ — read them separately", "warn"),
        ]
        if not payload["cleared"]:
            body.append(skin.pill("not cleared for circulation", "bad"))
        body.append(skin.table(payload["records"], columns))
        body.append(skin.note("Flags carry into the theater as halo rings."))
        return ui.HTML(skin.card("Depth flags", *body))

    # ----------------------------------------------------------- theater stage
    def _flags_fp() -> str:
        try:
            return state.fingerprint(data_fp(), input.outcome(), features(),
                                     input.flag_provider())
        except Exception:
            return "unready"

    @render.ui
    def theater_view():
        graph = current_graph.get()
        if graph is None:
            return ui.HTML(skin.note(
                "No graph yet — discover one, or adopt the truth graph on the "
                "Discover stage."
            ))
        chosen = bundle()
        flags = flags_result.get()
        # Halos only while the flag run still matches the live inputs; stale
        # rings on a different dataset's graph would be a quiet lie.
        halos_live = (
            flags.status == state.OK and flags.payload
            and flags.fingerprint == _flags_fp()
        )
        halos = flags.payload["halos"] if halos_live else {}
        return ui.HTML(theater.render_theater(
            graph, live_focus(), input.outcome(),
            halos=halos,
            display_names=dict(chosen.display_names()) if chosen else {},
            tooltips=dictionary.get(),
            selected=picked.get(),
        ))

    @reactive.effect
    @reactive.event(input.theater_pick)
    def _pick():
        picked.set(input.theater_pick())
        # Clicking a node nominates it as the candidate lever under
        # interrogation; the scorecard then tells ITS identification story.
        node = _selected_node()
        if node is not None:
            focus_node.set(node)

    @reactive.calc
    def live_focus() -> str | None:
        graph = current_graph.get()
        node = focus_node.get()
        return node if graph is not None and node in graph.nodes else None

    def _selected_edge() -> tuple[str, str] | None:
        # Selections are indices into the canonical edge order, never names:
        # uploaded column names are untrusted and may contain anything.
        selection = picked.get()
        graph = current_graph.get()
        if graph is None or not selection.startswith("edge:"):
            return None
        try:
            index = int(selection[5:])
            return theater.sorted_edges(graph)[index]
        except (ValueError, IndexError):
            return None

    def _selected_node() -> str | None:
        selection = picked.get()
        graph = current_graph.get()
        if graph is None or not selection.startswith("node:"):
            return None
        try:
            return graph.nodes[int(selection[5:])]
        except (ValueError, IndexError):
            return None

    @render.ui
    def selection_info():
        node = _selected_node()
        if node is not None:
            graph = current_graph.get()
            digraph = graph.digraph()
            parents = sorted(digraph.predecessors(node))
            children = sorted(digraph.successors(node))
            unresolved = sorted(
                other for pair in graph.undirected_pairs if node in pair
                for other in pair if other != node
            )
            lines = (
                '<p style="font-family:var(--mono);font-size:.74rem;line-height:2;margin:6px 0">'
                f"parents&nbsp;&nbsp;: {html.escape(', '.join(parents) or '—')}<br>"
                f"children&nbsp;: {html.escape(', '.join(children) or '—')}<br>"
                f"unresolved: {html.escape(', '.join(unresolved) or '—')}</p>"
            )
            fingerprint = ""
            flags = flags_result.get()
            if flags.status == state.OK and flags.payload:
                record = next(
                    (r for r in flags.payload.get("records", [])
                     if str(r.get("feature")) == node), None,
                )
                if record:
                    fingerprint = "".join(
                        skin.pill(f"{ch} z {record[f'{ch}_z']:+.2f}"
                                  + (" ⚑" if record[f"{ch}_flagged"] else ""),
                                  "warn" if record[f"{ch}_flagged"] else "info")
                        for ch in ("h0", "h1", "eig")
                    )
            return ui.HTML(skin.card(
                "Node inspector",
                skin.pill("candidate lever", "info"),
                lines,
                fingerprint,
                skin.note("The scorecard tells this node's identification "
                          "story. Click another node to switch."),
                annotation=node,
            ))
        edge = _selected_edge()
        if edge is not None:
            return ui.HTML(skin.card("Selected edge",
                                     skin.pill(f"{edge[0]} → {edge[1]}", "warn")))
        return ui.HTML(skin.note(
            "Click an edge to operate on it, or a node to interrogate it as a "
            "candidate lever."
        ))

    def _surgery(action: str):
        graph = current_graph.get()
        edge = _selected_edge()
        if graph is None or edge is None:
            ui.notification_show("Select an edge first.", type="warning")
            return
        try:
            revised = theater.apply_surgery(graph, action, edge, input.rationale())
        except ValueError as error:
            ui.notification_show(str(error), type="error", duration=6)
            return
        current_graph.set(revised)
        graph_gen.set(graph_gen.get() + 1)
        picked.set("")
        ui.notification_show(f"{action}: recorded in the ledger.", type="message")

    @reactive.effect
    @reactive.event(input.act_flip)
    def _flip():
        _surgery("flip")

    @reactive.effect
    @reactive.event(input.act_require)
    def _require():
        _surgery("require")

    @reactive.effect
    @reactive.event(input.act_forbid)
    def _forbid():
        _surgery("forbid")

    @reactive.effect
    @reactive.event(input.act_remove)
    def _remove():
        _surgery("remove")

    @render.ui
    def add_edge_form():
        graph = current_graph.get()
        if graph is None:
            return ui.HTML("")
        nodes = list(graph.nodes)
        return ui.div(
            ui.HTML('<h4 style="font-family:ui-monospace,Consolas,monospace;'
                    'font-size:11px;text-transform:uppercase;letter-spacing:.07em;'
                    'color:var(--muted);margin:14px 0 4px">Assert a missing edge</h4>'),
            ui.div(
                ui.input_select("add_from", None, nodes, width="42%"),
                ui.HTML('<span style="align-self:center">→</span>'),
                ui.input_select("add_to", None, nodes,
                                selected=nodes[1] if len(nodes) > 1 else nodes[0],
                                width="42%"),
                style="display:flex;gap:6px",
            ),
            ui.input_action_button("act_add", "Add edge", class_="btn-sm"),
        )

    @reactive.effect
    @reactive.event(input.act_add)
    def _add_edge():
        graph = current_graph.get()
        if graph is None:
            return
        try:
            revised = theater.apply_surgery(
                graph, "add", (input.add_from(), input.add_to()), input.rationale()
            )
        except ValueError as error:
            ui.notification_show(str(error), type="error", duration=6)
            return
        current_graph.set(revised)
        graph_gen.set(graph_gen.get() + 1)
        picked.set("")
        ui.notification_show(
            f"added {input.add_from()} → {input.add_to()}: recorded in the ledger.",
            type="message",
        )

    @reactive.effect
    @reactive.event(input.act_reset)
    def _reset():
        # Restore this dataset's own baseline, never whatever discovery last
        # succeeded on some other dataset.
        restored = baseline_graph.get()
        if restored is None:
            ui.notification_show("Nothing to reset to yet.", type="warning")
            return
        current_graph.set(restored)
        graph_gen.set(graph_gen.get() + 1)
        picked.set("")

    @render.ui
    def scorecard():
        graph = current_graph.get()
        if graph is None:
            return ui.HTML("")
        chosen = bundle()
        card = stages.surgery_scorecard(
            graph, live_focus(), input.outcome(),
            chosen.truth_graph() if chosen else None,
        )
        source_kind = "ok" if card["source"] == "recovered" else "info"
        parts = [
            skin.pill(card["source"].upper(), source_kind),
            skin.pill(f"{card['n_edges']} edges", "info"),
            skin.pill(f"{card['n_undirected']} pairs still undirected", "warn"),
        ]
        if "adjustment" in card:
            adjustment = ", ".join(card["adjustment"]) or "∅"
            valid = "ok" if card["adjustment_valid"] else "bad"
            parts.append(skin.pill(f"lever {card['focus']} → adjust for: {adjustment}", valid))
        else:
            parts.append(skin.pill("click a node to interrogate a candidate lever", "info"))
        if "m1" in card:
            parts.append(_m1_html(card["m1"], card["n_undirected"]))
        if "m3_valid_in_true" in card:
            verdict = "ok" if card["m3_valid_in_true"] else "bad"
            parts.append(skin.pill("sufficiency holds in truth"
                                   if card["m3_valid_in_true"]
                                   else "sufficiency FAILS in truth", verdict))
        ledger = card["ledger"]
        ledger_html = (
            "<ol style='font-size:12.5px;margin:6px 0 0 18px'>"
            + "".join(f"<li>{html.escape(entry)}</li>" for entry in ledger)
            + "</ol>"
            if ledger else skin.note("No surgeries yet — the graph is as discovered.")
        )
        return ui.HTML(skin.card("Scorecard", *parts, ledger_html))

    # -------------------------------------------------------- causal SHAP stage
    @reactive.effect
    @reactive.event(input.run_shap)
    def _run_shap():
        graph = current_graph.get()
        if graph is None:
            ui.notification_show("Need a graph first — visit Discover.", type="warning")
            return
        chosen = bundle()
        kwargs = dict(
            data=analysis_data(), features=features(), outcome=input.outcome(),
            graph=graph, arm=input.arm(), model_type=input.model_type(),
            truth_effects=chosen.truth_effects() if chosen else None,
            n_perms=int(input.n_perms()), n_background=int(input.n_background()),
            n_instances=int(input.n_instances()),
        )
        snips["attribute"].set(snippets.shap_snippet(
            input.arm(), features(), input.outcome(), input.model_type(),
            int(input.n_perms()), int(input.n_background()), int(input.n_instances()),
            SEED_HUB_DEMO,
            graph_source=graph.provenance.source.upper(),
            graph_fingerprint=graph.fingerprint(),
            n_edges=len(graph.directed_edges),
            n_ledger=len(graph.provenance.constraint_ledger)))
        _launch(shap_task, "attribute",
                state.fingerprint(data_fp(), features(), input.outcome(),
                                  graph.fingerprint(), input.arm(), input.model_type(),
                                  input.n_perms(), input.n_background(),
                                  input.n_instances()),
                kwargs)

    @render.ui
    def shap_preflight():
        n_features = len(features()) if raw_data() is not None else 0
        cost = int(input.n_perms()) * (n_features + 1)
        return ui.HTML(skin.note(
            f"≈ {cost:,} score-function calls over "
            f"{int(input.n_instances()) * int(input.n_background())} rows."
        ))

    @render.ui
    def shap_body():
        return _task_body(shap_task, "Run causal SHAP once a graph exists.", _shap_html,
                          code=snips["attribute"].get())

    def _shap_html(payload: dict) -> str:
        comparison = payload["comparison"]
        pills = [
            skin.pill(f"arm: {payload['arm']}", "info"),
            skin.pill(f"τ naive-vs-causal {comparison['kendall_tau']:.2f}", "info"),
        ]
        excluded = payload.get("excluded") or ()
        if excluded:
            pills.append(skin.pill(
                f"structurally zero under this DAG: {', '.join(excluded)}", "warn"
            ))
        tau_naive = comparison.get("tau_vs_truth_standard")
        tau_causal = comparison.get("tau_vs_truth_causal")
        if tau_naive is not None:
            pills.append(skin.pill(f"τ vs truth: naive {tau_naive:.2f}", "warn"))
            pills.append(skin.pill(f"causal {tau_causal:.2f}", "ok"))
        changes = sorted(
            comparison["rank_changes"].items(), key=lambda p: -abs(p[1]["change"])
        )
        truth = comparison.get("true_effects") or {}
        truth_total = sum(abs(v) for v in truth.values()) or None
        rows = []
        for name, detail in changes[:12]:
            row = {
                "feature": name,
                "naive rank": detail["standard_rank"],
                "causal rank": detail["causal_rank"],
                "Δ": detail["change"],
            }
            if truth_total and name in truth:
                row["true effect %"] = 100.0 * abs(truth[name]) / truth_total
            rows.append(row)
        columns = ["feature", "naive rank", "causal rank", "Δ"]
        semantics = ""
        if truth_total:
            columns.append("true effect %")
            semantics = skin.note(
                "The graph governs eligibility: a feature with no directed path "
                "to the outcome under the current DAG is zero by construction, "
                "exactly as in the frozen record. Flip an edge in Graph Surgery and "
                "eligibility changes with it — the attribution is conditional on "
                "the hypothesis. Price &amp; Dice then prices the survivors on the "
                "outcome itself."
            )
        return skin.card(
            "Attribution under the current graph",
            "".join(pills),
            skin.note(html.escape(payload["arm_note"])),
            skin.figure(payload["plot"], "comparison"),
            skin.table(rows, columns),
            semantics,
        )

    # ------------------------------------------------------------- policy stage
    def _specs_fp(specs) -> str:
        # Sorted keys alone would miss a price edit on an existing node; every
        # decision-relevant field belongs in the fingerprint.
        return state.fingerprint(tuple(
            (s.node, s.manipulable, s.min_shift, s.max_shift, s.fixed_cost, s.unit_cost)
            for s in (specs[k] for k in sorted(specs))
        ))

    @reactive.calc
    def cost_specs():
        override = cost_override.get()
        if override is not None:
            return override
        chosen = bundle()
        if chosen is not None:
            return chosen.cost_specs()
        return sd_cost_specs(analysis_data(), input.outcome())

    @reactive.effect
    @reactive.event(input.cost_upload_file)
    def _load_cost_sheet():
        files = input.cost_upload_file()
        if files:
            try:
                cost_override.set(dict(CostModel.from_csv(Path(files[0]["datapath"])).specs))
                ui.notification_show("Cost sheet replaced.", type="message")
            except Exception as error:
                ui.notification_show(f"Bad cost sheet: {error}", type="error", duration=8)

    @render.ui
    def policy_controls():
        chosen = bundle()
        direction = chosen.direction if chosen else "increase"
        alpha = chosen.default_alpha if chosen else 0.05
        budget_default = 1.2 if chosen and chosen.id == "toy_trap" else 2.0
        return ui.div(
            ui.input_slider("budget", "Budget", min=0.2, max=6.0,
                            value=budget_default, step=0.1),
            ui.input_radio_buttons("direction", "Beneficial direction",
                                   {"increase": "increase outcome",
                                    "decrease": "decrease outcome"},
                                   selected=direction),
        )

    @render.ui
    def policy_advanced():
        chosen = bundle()
        alpha = chosen.default_alpha if chosen else 0.05
        return ui.input_slider("policy_alpha", "Confidence floor α", min=0.001,
                               max=0.999, value=alpha, step=0.001)

    @render.ui
    def cost_preview():
        specs = cost_specs()
        rows = [
            {
                "node": s.node,
                "manipulable": s.manipulable,
                "fixed": s.fixed_cost,
                "per-unit": s.unit_cost,
                "note": s.ethical_note,
            }
            for s in specs.values()
        ]
        return ui.HTML(skin.card(
            "Cost sheet",
            skin.pill("ILLUSTRATIVE — not domain-reviewed", "bad"),
            skin.table(rows, ["node", "manipulable", "fixed", "per-unit", "note"],
                       limit=60),
        ))

    @reactive.effect
    @reactive.event(input.run_policy)
    def _run_policy():
        graph = current_graph.get()
        if graph is None:
            ui.notification_show("Need a graph first — visit Discover.", type="warning")
            return
        kwargs = dict(
            data=analysis_data(), graph=graph, outcome=input.outcome(),
            specs=cost_specs(), budget=float(input.budget()),
            direction=input.direction(), alpha=float(input.policy_alpha()),
        )
        snips["policy"].set(snippets.policy_snippet(
            input.outcome(), float(input.budget()), input.direction(),
            float(input.policy_alpha()), SEED_ACTION_ABDUCTION))
        _launch(policy_task, "policy",
                state.fingerprint(data_fp(), graph.fingerprint(), input.outcome(),
                                  input.budget(), input.direction(),
                                  input.policy_alpha(), _specs_fp(cost_specs())),
                kwargs)

    @render.ui
    def policy_body():
        return _task_body(policy_task, "Price the levers once a graph exists.", _policy_html,
                          code=snips["policy"].get())

    def _policy_html(payload: dict) -> str:
        ranking = payload["ranking"]
        best = ranking.best()
        fitted = payload["calibration"]
        pills = [
            skin.pill(f"{ranking.n_candidates_evaluated} candidates", "info"),
            skin.pill(f"direction: {ranking.direction}", "info"),
            skin.pill(f"SCM grade: {fitted.grade}", "warn"),
            skin.pill(f"{ranking.n_undirected_pairs} pairs undirected", "warn"),
        ]
        if best is not None:
            pills.insert(0, skin.pill(
                f"best: {best.label} (benefit {best.benefit:.3g}, cost {best.cost:.3g})", "ok"
            ))
        else:
            pills.insert(0, skin.pill("nothing feasible under these constraints", "bad"))
        table_rows = payload["table"].to_dict("records")
        screened_rows = payload["screened"].to_dict("records")
        return skin.card(
            "Affordable actions",
            "".join(pills),
            skin.figure(payload["pareto_plot"], "pareto"),
            skin.table(table_rows,
                       ["action", "benefit", "cost", "ratio", "p_unit_benefit",
                        "feasible", "screened_out"]),
        ) + skin.card(
            "Screened before pricing — nothing is dropped silently",
            skin.table(screened_rows, ["node", "screened_out"], limit=60),
        )

    # ---------------------------------------------------------------- map strip
    @render.ui
    def map_strip():
        graph = current_graph.get()
        flags = flags_result.get()
        if flags_task.status() == "running":
            flags_status = "running"
        elif flags.ok and flags.fingerprint != _flags_fp():
            flags_status = "stale"
        else:
            flags_status = flags.status
        statuses = {
            "data": state.OK if raw_data() is not None else state.EMPTY,
            "naive": _task_status(naive_task, "naive"),
            "discover": _task_status(discover_task, "discover"),
            "flags": flags_status,
            "surgery": (
                state.OK if graph is not None and graph.provenance.source == "recovered"
                else (state.EMPTY if graph is None else "info")
            ),
            "attribute": _task_status(shap_task, "attribute"),
            "policy": _task_status(policy_task, "policy"),
        }
        chips = {
            state.OK: ("done", "ok"), "running": ("running…", "warn"),
            state.ERROR: ("error", "bad"), "stale": ("stale", "warn"),
            "info": ("as discovered", "info"), state.EMPTY: ("·", "info"),
        }
        stations = []
        for index, (label, key) in enumerate(STATIONS, start=1):
            text, kind = chips.get(statuses[key], ("·", "info"))
            optional = " optional" if key == "flags" else ""
            stations.append(
                f'<div class="map-station{optional}" data-stage="{html.escape(label)}">'
                f"<b>{index} · {html.escape(label).upper()}</b>"
                f"{skin.pill(text, kind)}</div>"
            )
        return ui.HTML(f'<div class="map-strip">{"".join(stations)}</div>')

    def _task_status(task, name: str) -> str:
        status = task.status()
        if status == "success":
            return "stale" if fps[name].get() != _live_fp(name) else state.OK
        if status == "running":
            return "running"
        if status == "error":
            return state.ERROR
        return state.EMPTY

    def _live_fp(name: str) -> str:
        try:
            if name == "naive":
                return state.fingerprint(data_fp(), features(), input.outcome(),
                                         input.model_type())
            if name == "discover":
                return state.fingerprint(data_fp(), features(), input.outcome(),
                                         input.algorithm(), input.alpha())
            graph = current_graph.get()
            if graph is None:
                return "no-graph"
            if name == "attribute":
                return state.fingerprint(data_fp(), features(), input.outcome(),
                                         graph.fingerprint(), input.arm(),
                                         input.model_type(), input.n_perms(),
                                         input.n_background(), input.n_instances())
            return state.fingerprint(data_fp(), graph.fingerprint(), input.outcome(),
                                     input.budget(), input.direction(),
                                     input.policy_alpha(), _specs_fp(cost_specs()))
        except Exception:
            return "unready"

    @reactive.effect
    @reactive.event(input.goto_stage)
    def _goto():
        ui.update_navs("stage_nav", selected=input.goto_stage())

    @reactive.effect
    def _sync_next_button():
        """Light the Next button when the current stage has what the next needs."""
        labels = [label for label, _ in STATIONS]
        current = input.stage_nav()
        if current not in labels or current == labels[-1]:
            ui.update_action_button("goto_next", label="Next", disabled=True)
            return
        key = dict(STATIONS)[current]
        graph_ready = current_graph.get() is not None
        ready = {
            "data": raw_data() is not None,
            "naive": _task_status(naive_task, "naive") == state.OK,
            "discover": graph_ready,
            "flags": graph_ready,      # optional station: a graph is all Surgery needs
            "surgery": graph_ready,
            "attribute": _task_status(shap_task, "attribute") == state.OK,
        }[key]
        ui.update_action_button(
            "goto_next", label=f"Next: {labels[labels.index(current) + 1]} →",
            disabled=not ready,
        )

    @reactive.effect
    @reactive.event(input.goto_next)
    def _goto_next():
        labels = [label for label, _ in STATIONS]
        current = input.stage_nav()
        if current in labels and current != labels[-1]:
            ui.update_navs("stage_nav", selected=labels[labels.index(current) + 1])

    # ----------------------------------------------------------- session report
    def _task_payload(task):
        return task.value.get() if task.status() == "success" else None

    def _assemble_report() -> str:
        chosen = bundle()
        data = raw_data()
        graph = current_graph.get()
        display = dict(chosen.display_names()) if chosen else {}
        outcome = input.outcome()

        graph_summary = None
        current_svg = ""
        if graph is not None:
            graph_summary = stages.surgery_scorecard(
                graph, live_focus(), outcome,
                chosen.truth_graph() if chosen else None,
            )
            current_svg = theater.render_theater(
                graph, live_focus(), outcome, display_names=display,
                tooltips=dictionary.get(), interactive=False, height=360,
            )
        truth_svg = ""
        if chosen is not None and chosen.truth_graph() is not None:
            truth_svg = theater.render_theater(
                chosen.truth_graph(), None, outcome, display_names=display,
                interactive=False, height=360,
            )

        discover_payload = _task_payload(discover_task) or {}
        flags = flags_result.get()
        m1 = (graph_summary or {}).get("m1") or discover_payload.get("m1")

        return report.build_report(
            dataset_label=chosen.label if chosen else "Uploaded CSV",
            dataset_note=chosen.note if chosen else "",
            n_rows=len(data) if data is not None else 0,
            outcome=outcome,
            features=features() if data is not None else (),
            naive=_task_payload(naive_task),
            discover_m1=m1,
            graph_summary=graph_summary,
            truth_svg=truth_svg,
            current_svg=current_svg,
            flags=flags.payload if flags.status == state.OK else None,
            shap=_task_payload(shap_task),
            policy=_task_payload(policy_task),
            code_appendix=[
                ("Naive SHAP", snips["naive"].get()),
                ("Discovery", snips["discover"].get()),
                ("Depth flags", snips["flags"].get()),
                ("Graph surgery", snippets.surgery_snippet()
                 if graph is not None and graph.provenance.source == "recovered" else ""),
                ("Causal SHAP", snips["attribute"].get()),
                ("Price and dice", snips["policy"].get()),
            ],
        )

    @render.download(
        filename=lambda: (
            f"hub_report_{input.dataset_choice()}_{datetime.now():%Y%m%d_%H%M}.html"
        )
    )
    def download_report():
        yield _assemble_report()

    # ------------------------------------------------------------ shared render
    def _task_body(task, placeholder: str, renderer, code: str = ""):
        status = task.status()
        if status == "initial":
            return ui.HTML(skin.note(placeholder))
        if status == "running":
            return ui.HTML(
                skin.card("Working…", skin.pill("running on a worker", "warn"))
                + skin.code_card(code)
            )
        if status == "error":
            error = task.error.get()
            return ui.HTML(
                skin.error_box(f"{type(error).__name__}: {error}") + skin.code_card(code)
            )
        if status == "cancelled":
            return ui.HTML(skin.note("Cancelled."))
        return ui.HTML(renderer(task.value.get()) + skin.code_card(code))


app = App(app_ui, server)
