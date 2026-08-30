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
from pathlib import Path

import pandas as pd
from shiny import App, reactive, render, req, ui

from causal_shap.action_costs import CostModel
from hub import stages, state, theater
from hub import ui as skin
from hub.datasets import DATASETS, UPLOAD_ID, HubDataset, sd_cost_specs

STATIONS = (
    ("Data", "data"),
    ("Naive SHAP", "naive"),
    ("Discover", "discover"),
    ("Flags", "flags"),
    ("Theater", "surgery"),
    ("Causal SHAP", "attribute"),
    ("Price & Dice", "policy"),
)

MAP_JS = """
<script>
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
        '<div class="hub-header"><h1>Guided Causal Discovery Hub</h1>'
        '<div class="sub">naive attribution → discovery → depth flags → surgery '
        "→ causal attribution → priced interventions</div></div>"
    ),
    ui.output_ui("map_strip"),
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
                ),
                col_widths=(5, 7),
            ),
        ),
        ui.nav_panel(
            "Naive SHAP",
            ui.layout_columns(
                ui.div(
                    ui.input_select("model_type", "Model",
                                    {"gbm": "Gradient boosting", "rf": "Random forest",
                                     "linear": "Linear / logistic"}),
                    ui.input_action_button("run_naive", "Run naive SHAP",
                                           class_="btn-sm"),
                    ui.HTML(skin.note(
                        "The benchmark: what the fitted predictor listened to. "
                        "The causal stages will argue with this chart."
                    )),
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
                    ui.input_slider("alpha", "PC significance α", min=0.01, max=0.2,
                                    value=0.05, step=0.01),
                    ui.input_action_button("run_discover", "Discover structure",
                                           class_="btn-sm"),
                    ui.output_ui("adopt_truth_control"),
                ),
                ui.output_ui("discover_body"),
                col_widths=(4, 8),
            ),
        ),
        ui.nav_panel(
            "Flags",
            ui.layout_columns(
                ui.div(
                    ui.input_select("flag_provider", "Detector",
                                    {"auto": "auto-select", "null": "none (honest empty)",
                                     "precomputed": "frozen tables", "local": "local live"}),
                    ui.input_action_button("run_flags", "Run depth detector",
                                           class_="btn-sm"),
                    ui.HTML(skin.note(
                        "A per-node depth signal: where to distrust the current "
                        "attribution and look again. Advisory, never a causal claim."
                    )),
                ),
                ui.output_ui("flags_body"),
                col_widths=(4, 8),
            ),
        ),
        ui.nav_panel(
            "Theater",
            ui.output_ui("theater_view"),
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
                    ui.input_slider("n_perms", "Permutations", 8, 128, 32, step=8),
                    ui.input_slider("n_instances", "Explained rows", 8, 128, 32, step=8),
                    ui.input_slider("n_background", "Background draws", 8, 64, 16, step=8),
                    ui.input_action_button("run_shap", "Run causal SHAP", class_="btn-sm"),
                    ui.output_ui("shap_preflight"),
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
                    ui.input_file("cost_upload_file", "Replace cost sheet (CSV)",
                                  accept=[".csv"], multiple=False),
                    ui.input_action_button("run_policy", "Rank affordable actions",
                                           class_="btn-sm"),
                    ui.output_ui("cost_preview"),
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

    # ----------------------------------------------------------- extended tasks
    @reactive.extended_task
    async def naive_task(kwargs: dict):
        return await asyncio.to_thread(lambda: stages.run_naive_shap(**kwargs))

    @reactive.extended_task
    async def discover_task(kwargs: dict):
        return await asyncio.to_thread(lambda: stages.run_discovery(**kwargs))

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
        _launch(naive_task, "naive",
                state.fingerprint(data_fp(), features(), input.outcome(), input.model_type()),
                kwargs)

    @render.ui
    def naive_body():
        return _task_body(naive_task, "Run naive SHAP to open the story.", _naive_html)

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
                          _discover_html)

    def _discover_html(payload: dict) -> str:
        graph = payload["graph"]
        parts = [
            skin.pill(f"{len(graph.directed_edges)} edges", "info"),
            skin.pill(f"{graph.n_undirected_pairs} pairs undirected", "warn"),
        ]
        banner = skin.note(
            "The algorithm returns an equivalence class. Dashed edges in the "
            "theater are one deterministic representative — choices, not findings."
        )
        m1 = payload.get("m1")
        m1_html = _m1_html(m1, graph.n_undirected_pairs) if m1 else ""
        return skin.card("Discovered structure", "".join(parts), banner, m1_html)

    def _m1_html(m1: dict, n_undirected: int) -> str:
        caveat = " (representative-dependent)" if n_undirected else ""
        return (
            skin.pill(f"skeleton F1 {m1['skeleton_f1']:.2f}", "ok")
            + skin.pill(f"directed F1 {m1['f1']:.2f}{caveat}", "warn" if n_undirected else "ok")
            + skin.pill(f"SHD {m1['shd']}", "info")
        )

    # ------------------------------------------------------------- flags stage
    @reactive.effect
    @reactive.event(input.run_flags)
    def _run_flags():
        chosen = bundle()
        root = os.environ.get("CAUSAL_SHAP_BLOCK_ROOT", "")
        preference = None if input.flag_provider() == "auto" else input.flag_provider()
        try:
            payload = stages.run_flags(
                chosen.flags_id if chosen else "upload",
                input.outcome(), features(),
                block_root=Path(root) if root else None,
                preference=preference,
            )
            flags_result.set(state.StageResult(status=state.OK, payload=payload,
                                               fingerprint=_flags_fp()))
        except Exception as error:  # a broken provider must not take the app down
            flags_result.set(state.StageResult(status=state.ERROR, error=str(error),
                                               traceback=traceback.format_exc(limit=4)))

    @render.ui
    def flags_body():
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
            display_names=dict(chosen.display_names()) if chosen else dictionary.get(),
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
            return ui.HTML(skin.card(
                "Candidate lever", skin.pill(node, "info"),
                skin.note("The scorecard now tells this node's identification "
                          "story. Click another node to interrogate it instead."),
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
        return _task_body(shap_task, "Run causal SHAP once a graph exists.", _shap_html)

    def _shap_html(payload: dict) -> str:
        comparison = payload["comparison"]
        pills = [
            skin.pill(f"arm: {payload['arm']}", "info"),
            skin.pill(f"τ naive-vs-causal {comparison['kendall_tau']:.2f}", "info"),
        ]
        tau_naive = comparison.get("tau_vs_truth_standard")
        tau_causal = comparison.get("tau_vs_truth_causal")
        if tau_naive is not None:
            pills.append(skin.pill(f"τ vs truth: naive {tau_naive:.2f}", "warn"))
            pills.append(skin.pill(f"causal {tau_causal:.2f}", "ok"))
        changes = sorted(
            comparison["rank_changes"].items(), key=lambda p: -abs(p[1]["change"])
        )
        rows = [
            {
                "feature": name,
                "naive rank": detail["standard_rank"],
                "causal rank": detail["causal_rank"],
                "Δ": detail["change"],
            }
            for name, detail in changes[:12]
        ]
        return skin.card(
            "Attribution under the current graph",
            "".join(pills),
            skin.note(html.escape(payload["arm_note"])),
            skin.figure(payload["plot"], "comparison"),
            skin.table(rows, ["feature", "naive rank", "causal rank", "Δ"]),
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
            ui.input_slider("policy_alpha", "Confidence floor α", min=0.001, max=0.999,
                            value=alpha, step=0.001),
        )

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
        _launch(policy_task, "policy",
                state.fingerprint(data_fp(), graph.fingerprint(), input.outcome(),
                                  input.budget(), input.direction(),
                                  input.policy_alpha(), _specs_fp(cost_specs())),
                kwargs)

    @render.ui
    def policy_body():
        return _task_body(policy_task, "Price the levers once a graph exists.", _policy_html)

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
        flags_status = (
            "stale" if flags.ok and flags.fingerprint != _flags_fp() else flags.status
        )
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
            stations.append(
                f'<div class="map-station" data-stage="{html.escape(label)}">'
                f"<b>{index} · {html.escape(label)}</b>"
                f'{skin.pill(text, kind)}</div>'
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

    # ------------------------------------------------------------ shared render
    def _task_body(task, placeholder: str, renderer):
        status = task.status()
        if status == "initial":
            return ui.HTML(skin.note(placeholder))
        if status == "running":
            return ui.HTML(skin.card("Working…", skin.pill("running on a worker", "warn")))
        if status == "error":
            error = task.error.get()
            return ui.HTML(skin.error_box(f"{type(error).__name__}: {error}"))
        if status == "cancelled":
            return ui.HTML(skin.note("Cancelled."))
        return ui.HTML(renderer(task.value.get()))


app = App(app_ui, server)
