"""
Causal SHAP Workbench
Ported 2026-08-12 from the ACIC workshop chassis (Causal Discovery
Playground, after Molak ch. 13). Technical-paper skin,
clickable schematic home (Map), M1-M5 structural-importance battery,
and embedded study guide.
"""

from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import networkx as nx
import os
import warnings
import base64
import time
import json
import sys
from html import escape as html_escape
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = APP_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    Network = None
    PYVIS_AVAILABLE = False

warnings.filterwarnings("ignore")

# Causal SHAP imports
SHAP_AVAILABLE = False
try:
    try:
        from .attribution import (
            compute_causal_shap_fast, compute_standard_shap,
            compute_adjustment_set_shap, compare_shap_rankings,
            mediator_inflation_ratio, mean_abs_shap,
            dag_from_adjacency,
            shap_bar_plot_to_base64, comparison_bar_plot_to_base64,
            rank_change_table_html, SIMCAUSAL_TRUE_TOTAL_EFFECTS,
        )
    except ImportError:
        from attribution import (
            compute_causal_shap_fast, compute_standard_shap,
            compute_adjustment_set_shap, compare_shap_rankings,
            mediator_inflation_ratio, mean_abs_shap,
            dag_from_adjacency,
            shap_bar_plot_to_base64, comparison_bar_plot_to_base64,
            rank_change_table_html, SIMCAUSAL_TRUE_TOTAL_EFFECTS,
        )
    SHAP_AVAILABLE = True
    print("[INIT] Causal SHAP module loaded")
except ImportError as e:
    print(f"[INIT] Causal SHAP module not available: {e}")

from causal_shap import evaluation
from causal_shap.discovery import (
    apply_constraints,
    causal_learn_graph_to_pdag,
    deterministic_consistent_extension,
    direct_lingam_adjacency_to_pdag,
)
try:
    from . import guide, schematic
except ImportError:
    import guide
    import schematic

# sklearn for SHAP tab model fitting
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

CASTLE_AVAILABLE = False
try:
    from causallearn.search.ConstraintBased.PC import pc as pc_algorithm
    from causallearn.search.FCMBased import lingam
    from causallearn.search.ScoreBased.GES import ges as ges_algorithm
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.graph.GraphNode import GraphNode
    CASTLE_AVAILABLE = True
    print("[INIT] causal-learn loaded successfully")
except ImportError as e:
    print(f"[INIT] causal-learn not available: {e}")

SAMPLE_FILE = os.path.join(APP_DIR, "data", "simcausal_train.csv")
GROUND_TRUTH_FILE = os.path.join(APP_DIR, "data", "ground_truth_edges.csv")
TRUE_EFFECTS_FILE = os.path.join(APP_DIR, "data", "true_total_effects.json")

print(f"[INIT] App directory: {APP_DIR}")
print(f"[INIT] Sample file exists: {os.path.exists(SAMPLE_FILE)}")
print(f"[INIT] Ground truth file exists: {os.path.exists(GROUND_TRUTH_FILE)}")
print(f"[INIT] causal-learn available: {CASTLE_AVAILABLE}")
print(f"[INIT] SHAP available: {SHAP_AVAILABLE}")
print(f"[INIT] pyvis available: {PYVIS_AVAILABLE}")


def safe_text(value):
    """Escape values originating in uploaded column or edge names."""

    return html_escape(str(value), quote=True)


def safe_join(values):
    return ", ".join(safe_text(value) for value in values)

# ============================================================================
# MODERN MINIMALIST CSS
# ============================================================================
MODERN_CSS = """
<style>
/* == technical paper: ink #111 on white, one amber accent, Georgia + Courier == */
:root {
    --primary: #b45309;          /* amber — the single accent */
    --primary-light: #d97706;
    --primary-dark: #92400e;
    --success: #1a5632;
    --warning: #92400e;
    --danger: #7f1d1d;
    --gray-50: #faf9f7;
    --gray-100: #f2f0ec;
    --gray-200: #d9d5cd;
    --gray-300: #b8b2a7;
    --gray-400: #8c8578;
    --gray-500: #6b655a;
    --gray-600: #4f4a41;
    --gray-700: #37332c;
    --gray-800: #201d18;
    --gray-900: #111111;
    --white: #ffffff;
    --ink: #111111;
    --shadow-sm: none;
    --shadow: none;
    --shadow-md: none;
    --shadow-lg: none;
}

* { box-sizing: border-box; }

body {
    background: var(--white) !important;
    color: var(--ink) !important;
    font-family: Georgia, 'Times New Roman', serif !important;
    line-height: 1.55;
}

.app-header {
    background: var(--white);
    border-bottom: 2px solid var(--ink);
    padding: 22px 0 14px 0;
    margin-bottom: 8px;
}

.app-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--ink);
    margin: 0;
    letter-spacing: 0.12em;
    font-variant: small-caps;
}

.app-subtitle {
    font-size: 0.9rem;
    color: var(--gray-600);
    margin: 2px 0 0 0;
    font-style: italic;
}

.nav-tabs {
    border-bottom: 1px solid var(--ink) !important;
    background: var(--white);
    padding: 0 8px;
    gap: 0;
}

.nav-tabs .nav-link {
    color: var(--gray-600) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Courier New', Courier, monospace;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    padding: 13px 16px !important;
    margin: 0 !important;
    border-radius: 0 !important;
}

.nav-tabs .nav-link:hover {
    color: var(--ink) !important;
    border-bottom-color: var(--gray-300) !important;
}

.nav-tabs .nav-link.active {
    color: var(--primary) !important;
    background: transparent !important;
    border-bottom: 2px solid var(--primary) !important;
}

.tab-content { background: transparent; border: none; padding: 20px 0; }

.card {
    background: var(--white);
    border: 1px solid var(--ink);
    border-radius: 2px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: none;
}

.card-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--ink);
    font-variant: small-caps;
    letter-spacing: 0.1em;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--ink);
}

.info-box, .warning-box, .success-box, .error-box {
    background: var(--white);
    border-radius: 0;
    padding: 10px 14px;
    margin: 12px 0;
    font-size: 0.875rem;
}

.info-box    { color: var(--gray-700); border-left: 3px solid var(--ink); }
.warning-box { color: var(--warning);  border-left: 3px solid var(--primary);
               border-top: 1px dashed var(--gray-300);
               border-bottom: 1px dashed var(--gray-300); }
.success-box { color: var(--success);  border-left: 3px solid var(--success); }
.error-box   { color: var(--danger);   border-left: 3px solid var(--danger); }

.btn-primary {
    background: var(--white) !important;
    border: 2px solid var(--ink) !important;
    color: var(--ink) !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
    padding: 9px 18px !important;
    border-radius: 2px !important;
}

.btn-primary:hover {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
    color: var(--white) !important;
    transform: none;
    box-shadow: none !important;
}

.btn-secondary {
    background: var(--white) !important;
    border: 1px solid var(--ink) !important;
    color: var(--ink) !important;
    font-family: 'Courier New', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 9px 18px !important;
    border-radius: 2px !important;
}

.btn-secondary:hover { background: var(--gray-100) !important; }

.form-control, .form-select {
    background: var(--white) !important;
    border: 1px solid var(--gray-600) !important;
    border-radius: 2px !important;
    padding: 8px 10px !important;
    font-size: 0.9rem !important;
    font-family: 'Courier New', monospace !important;
    color: var(--ink) !important;
}

.form-control:focus, .form-select:focus {
    border-color: var(--primary) !important;
    box-shadow: none !important;
    outline: 1px solid var(--primary) !important;
}

.form-check-input:checked {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
}

label, .form-label {
    color: var(--ink) !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    margin-bottom: 5px !important;
}

table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }

table th {
    background: var(--white) !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
    font-variant: small-caps;
    letter-spacing: 0.06em;
    font-size: 0.75rem;
    padding: 7px 10px !important;
    border-bottom: 2px solid var(--ink) !important;
    white-space: nowrap;
}

table td {
    padding: 5px 10px !important;
    border-bottom: 1px solid var(--gray-200) !important;
    color: var(--gray-800);
    white-space: nowrap;
    max-width: 140px;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: 'Courier New', monospace;
    font-size: 0.8rem;
}

table tr:hover td { background: var(--gray-50); }

.table-container {
    overflow-x: auto;
    max-height: 280px;
    overflow-y: auto;
    border: 1px solid var(--ink);
    border-radius: 2px;
}

.table-container table { margin: 0; }

.progress-container {
    background: var(--gray-50);
    border: 1px dashed var(--gray-300);
    border-radius: 2px;
    padding: 14px;
    margin: 12px 0;
}

.progress-bar-wrapper {
    background: var(--gray-100);
    border-radius: 0;
    height: 6px;
    overflow: hidden;
    margin-top: 8px;
}

.progress-bar-fill {
    background: var(--primary);
    height: 100%;
    animation: progress-animation 2s ease-in-out infinite;
}

@keyframes progress-animation {
    0% { width: 0%; } 50% { width: 70%; } 100% { width: 100%; }
}

.stat-pill {
    display: inline-flex;
    align-items: center;
    background: var(--white);
    border: 1px solid var(--ink);
    padding: 4px 10px;
    border-radius: 2px;
    font-family: 'Courier New', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--gray-700);
    margin-right: 8px;
    margin-bottom: 8px;
}

.stat-pill .value { color: var(--ink); margin-left: 4px; }
.stat-pill.primary { border-color: var(--primary); color: var(--primary); }
.stat-pill.primary .value { color: var(--primary); }

.node-inspector {
    background: var(--white);
    border: 1px dashed var(--gray-300);
    border-radius: 2px;
    padding: 14px;
}

.node-inspector h5 {
    font-size: 1rem;
    font-weight: 700;
    color: var(--ink);
    font-variant: small-caps;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
}

.node-inspector .parents { color: var(--ink); }
.node-inspector .children { color: var(--primary); }

.edge-list {
    max-height: 200px;
    overflow-y: auto;
    font-size: 0.8rem;
    font-family: 'Courier New', Courier, monospace;
    background: var(--white);
    border: 1px solid var(--gray-200);
    padding: 10px;
    border-radius: 2px;
    line-height: 1.8;
}

.edge-list .edge { padding: 2px 0; color: var(--gray-700); }

.code-block {
    background: var(--white);
    color: var(--ink);
    border: 1px solid var(--ink);
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.8rem;
    padding: 14px;
    border-radius: 2px;
    overflow-x: auto;
    line-height: 1.6;
    white-space: pre;
}

.footer {
    text-align: center;
    padding: 20px;
    color: var(--gray-500);
    font-size: 0.8rem;
    font-style: italic;
    border-top: 2px solid var(--ink);
    margin-top: 32px;
}

.form-check {
    padding: 10px 14px;
    border: 1px solid var(--gray-200);
    border-radius: 2px;
    margin-bottom: 8px;
}

.form-check:hover { border-color: var(--ink); }
.form-check-input:checked ~ .form-check-label { color: var(--primary); }

.stat-card {
    background: var(--white);
    border: 1px solid var(--ink);
    border-radius: 2px;
    padding: 14px;
    text-align: center;
}
.stat-label { font-size: 0.75rem; color: var(--gray-600);
              font-variant: small-caps; letter-spacing: 0.06em; }
.stat-value { font-size: 1.5rem; font-weight: 700; color: var(--ink);
              font-family: 'Courier New', monospace; }

details summary { cursor: pointer; color: var(--gray-500);
                  font-size: 0.8rem; font-style: italic; }
.teach { border-top: 1px dashed var(--gray-300); margin-top: 10px;
         padding-top: 8px; font-size: 0.83rem; color: var(--gray-700); }
</style>
"""

MATHJAX_HEAD = (
    '<script>window.MathJax={tex:{inlineMath:[["\\(","\\)"]]}};</script>'
    '<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">'
    '</script>')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_ancestors(G, node):
    """Get all ancestors of a node in the DAG."""
    ancestors = set()
    to_visit = list(G.predecessors(node))
    while to_visit:
        current = to_visit.pop()
        if current not in ancestors:
            ancestors.add(current)
            to_visit.extend(G.predecessors(current))
    return ancestors


def get_descendants(G, node):
    """Get all descendants of a node in the DAG."""
    descendants = set()
    to_visit = list(G.successors(node))
    while to_visit:
        current = to_visit.pop()
        if current not in descendants:
            descendants.add(current)
            to_visit.extend(G.successors(current))
    return descendants


def shrier_platt_check(G, exposure, outcome, adjustment_set):
    """
    Shrier-Platt 6-step algorithm to check if adjustment set blocks all backdoor paths.

    Reference: Shrier I, Platt RW. Reducing bias through directed acyclic graphs.
    BMC Medical Research Methodology. 2008;8:70.

    Steps:
    1. Keep only ancestors of A, Y, and Z plus themselves
    2. Remove arrows coming out of A (focus on backdoors)
    3. Moralize colliders (connect parents of each node with multiple parents)
    4. Make all edges undirected
    5. Remove adjustment set Z nodes from the graph
    6. Check if A and Y are still connected - if connected, backdoor path exists
    """
    nodes = list(G.nodes())
    if exposure not in nodes or outcome not in nodes:
        return {"valid": False, "message": "Exposure or outcome not in graph", "steps": []}

    adjustment_set = [z for z in adjustment_set if z in nodes]

    # Step 1: Keep ancestors of A, Y, Z plus themselves
    keep = set([exposure, outcome] + adjustment_set)
    keep.update(get_ancestors(G, exposure))
    keep.update(get_ancestors(G, outcome))
    for z in adjustment_set:
        keep.update(get_ancestors(G, z))

    # Build subgraph with kept nodes
    edges_step1 = [(u, v) for u, v in G.edges() if u in keep and v in keep]

    # Step 2: Remove arrows out of A (exposure)
    edges_step2 = [(u, v) for u, v in edges_step1 if u != exposure]

    # Step 3: Moralize colliders (connect parents of nodes with multiple parents)
    # Find nodes with multiple parents
    parent_counts = {}
    for u, v in edges_step2:
        parent_counts[v] = parent_counts.get(v, []) + [u]

    moralized_edges = list(edges_step2)
    for node, parents in parent_counts.items():
        if len(parents) >= 2:
            # Connect all pairs of parents
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    moralized_edges.append((parents[i], parents[j]))
                    moralized_edges.append((parents[j], parents[i]))

    # Step 4: Make undirected (create both directions)
    undirected_edges = set()
    for u, v in moralized_edges:
        undirected_edges.add((min(u, v), max(u, v)))

    # Step 5: Remove Z nodes
    if adjustment_set:
        filtered_edges = [(u, v) for u, v in undirected_edges
                         if u not in adjustment_set and v not in adjustment_set]
    else:
        filtered_edges = list(undirected_edges)

    # Step 6: Check connectivity between A and Y
    remaining_nodes = keep - set(adjustment_set)
    if exposure not in remaining_nodes or outcome not in remaining_nodes:
        blocks_all = True
    else:
        # Build undirected graph for connectivity check
        G_undirected = nx.Graph()
        G_undirected.add_nodes_from(remaining_nodes)
        for u, v in filtered_edges:
            if u in remaining_nodes and v in remaining_nodes:
                G_undirected.add_edge(u, v)

        blocks_all = not nx.has_path(G_undirected, exposure, outcome) if exposure in G_undirected and outcome in G_undirected else True

    steps = [
        f"1. Kept {len(keep)} ancestor nodes",
        f"2. Removed {len(edges_step1) - len(edges_step2)} arrows out of {exposure}",
        f"3. Moralized: added {len(moralized_edges) - len(edges_step2)} edges connecting co-parents",
        f"4. Made {len(undirected_edges)} undirected edges",
        f"5. Removed adjustment nodes: {', '.join(adjustment_set) if adjustment_set else 'none'}",
        f"6. {exposure}↔{outcome}: {'BLOCKED' if blocks_all else 'CONNECTED (unblocked backdoor!)'}"
    ]

    return {
        "valid": blocks_all,
        "message": "All backdoor paths blocked" if blocks_all else "Unblocked backdoor path exists",
        "steps": steps
    }


def identify_adjustment_sets(G, exposure, outcome, cols):
    """
    Identify various adjustment set criteria.

    Returns dict with different adjustment set recommendations.
    """
    nodes = list(G.nodes())
    exp_idx = cols.index(exposure) if exposure in cols else None
    out_idx = cols.index(outcome) if outcome in cols else None

    if exp_idx is None or out_idx is None:
        return {}

    # Get parents (direct causes)
    parents_exp = set(G.predecessors(exp_idx))
    parents_out = set(G.predecessors(out_idx))

    # Get descendants of exposure (should NOT adjust for these)
    descendants_exp = get_descendants(G, exp_idx)

    # Convert indices to names
    def idx_to_names(idx_set):
        return [cols[i] for i in idx_set if i < len(cols)]

    # 1. Traditional confounders: causes of BOTH exposure and outcome
    traditional = parents_exp & parents_out

    # 2. Disjunctive cause criterion (VanderWeele & Shpitser, 2011)
    # Adjust for any cause of exposure OR outcome (excluding descendants of exposure)
    disjunctive = (parents_exp | parents_out) - descendants_exp - {exp_idx}

    # 3. Pre-treatment covariates that are parents of exposure or outcome
    # (excluding outcome and descendants of exposure)
    valid_adjustment = (parents_exp | parents_out) - descendants_exp - {exp_idx, out_idx}

    # 4. Minimal sufficient set: just the confounders
    minimal = traditional - descendants_exp - {exp_idx, out_idx}

    # Check each set with Shrier-Platt
    results = {}

    # Build named graph for Shrier-Platt
    G_named = nx.DiGraph()
    for i, col in enumerate(cols):
        G_named.add_node(col)
    for u, v in G.edges():
        G_named.add_edge(cols[u], cols[v])

    # Traditional confounders
    trad_names = idx_to_names(traditional)
    trad_check = shrier_platt_check(G_named, exposure, outcome, trad_names)
    results["traditional"] = {
        "name": "Traditional Confounders",
        "description": "Variables that cause both exposure and outcome",
        "variables": trad_names,
        "valid": trad_check["valid"],
        "steps": trad_check["steps"]
    }

    # Disjunctive cause criterion
    disj_names = idx_to_names(disjunctive)
    disj_check = shrier_platt_check(G_named, exposure, outcome, disj_names)
    results["disjunctive"] = {
        "name": "Disjunctive Cause Criterion",
        "description": "Causes of exposure OR outcome (VanderWeele & Shpitser, 2011)",
        "citation": "VanderWeele TJ, Shpitser I. A new criterion for confounder selection. Biometrics. 2011;67(4):1406-1413.",
        "variables": disj_names,
        "valid": disj_check["valid"],
        "steps": disj_check["steps"]
    }

    # Minimal sufficient
    min_names = idx_to_names(minimal)
    min_check = shrier_platt_check(G_named, exposure, outcome, min_names)
    results["minimal"] = {
        "name": "Minimal Sufficient Set",
        "description": "Smallest set that blocks all backdoor paths",
        "variables": min_names,
        "valid": min_check["valid"],
        "steps": min_check["steps"]
    }

    return results


def compare_graphs(discovered_edges, ground_truth_edges):
    """
    Compare discovered edges against ground truth.

    Args:
        discovered_edges: set of (from, to) tuples
        ground_truth_edges: set of (from, to) tuples

    Returns:
        dict with comparison metrics and edge classifications
    """
    # True positives: edges in both discovered and ground truth (correct direction)
    true_positives = discovered_edges & ground_truth_edges

    # Create reverse edge sets for detecting reversed edges
    ground_truth_reversed = {(t, f) for f, t in ground_truth_edges}
    discovered_reversed = {(t, f) for f, t in discovered_edges}

    # Reversed: discovered edge exists but direction is wrong
    reversed_edges = discovered_edges & ground_truth_reversed

    # Spurious: discovered edges not in ground truth (either direction)
    all_ground_truth = ground_truth_edges | ground_truth_reversed
    spurious_edges = discovered_edges - ground_truth_edges - ground_truth_reversed

    # Missed: ground truth edges not discovered (either direction)
    all_discovered = discovered_edges | discovered_reversed
    missed_edges = ground_truth_edges - discovered_edges - discovered_reversed

    # Metrics
    n_discovered = len(discovered_edges)
    n_ground_truth = len(ground_truth_edges)
    n_tp = len(true_positives)
    n_reversed = len(reversed_edges)
    n_spurious = len(spurious_edges)
    n_missed = len(missed_edges)

    # Precision: of discovered edges, how many are correct?
    precision = n_tp / n_discovered if n_discovered > 0 else 0

    # Recall: of ground truth edges, how many were found (correct direction)?
    recall = n_tp / n_ground_truth if n_ground_truth > 0 else 0

    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Skeleton metrics (ignoring direction)
    skeleton_discovered = {tuple(sorted(e)) for e in discovered_edges}
    skeleton_ground_truth = {tuple(sorted(e)) for e in ground_truth_edges}
    skeleton_tp = skeleton_discovered & skeleton_ground_truth
    skeleton_precision = len(skeleton_tp) / len(skeleton_discovered) if skeleton_discovered else 0
    skeleton_recall = len(skeleton_tp) / len(skeleton_ground_truth) if skeleton_ground_truth else 0
    skeleton_f1 = 2 * skeleton_precision * skeleton_recall / (skeleton_precision + skeleton_recall) if (skeleton_precision + skeleton_recall) > 0 else 0

    return {
        "true_positives": true_positives,
        "reversed": reversed_edges,
        "spurious": spurious_edges,
        "missed": missed_edges,
        "n_discovered": n_discovered,
        "n_ground_truth": n_ground_truth,
        "n_tp": n_tp,
        "n_reversed": n_reversed,
        "n_spurious": n_spurious,
        "n_missed": n_missed,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "skeleton_precision": skeleton_precision,
        "skeleton_recall": skeleton_recall,
        "skeleton_f1": skeleton_f1
    }


def load_ground_truth_edges(filepath):
    """Load ground truth edges from CSV file."""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if 'from' in df.columns and 'to' in df.columns:
            return set(zip(df['from'], df['to']))
        return None
    except Exception:
        return None


# ============================================================================
# UI DEFINITION
# ============================================================================
app_ui = ui.page_fluid(
    ui.HTML(MODERN_CSS),
    ui.HTML(MATHJAX_HEAD),

    # Header
    ui.HTML('''
        <div class="app-header">
            <div style="max-width: 1200px; margin: 0 auto; padding: 0 24px;">
                <h1 class="app-title">Causal SHAP Workbench</h1>
                <p class="app-subtitle">from attribution to intervention &mdash; discovery, evaluation, and do()-attribution on one bench</p>
            </div>
        </div>
    '''),

    # Main content wrapper
    ui.div(
        ui.navset_tab(
            schematic.map_panel(),
            # =================================================================
            # TAB 1: DATA
            # =================================================================
            ui.nav_panel(
                "Data",
                ui.row(
                    ui.column(5,
                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Import Data</div>'),
                        ui.input_radio_buttons(
                            "dataset_choice", "Dataset",
                            choices={
                                "simcausal": "Simcausal (12 vars, known ground truth)",
                                "upload": "Upload custom CSV"
                            },
                            selected="simcausal"
                        ),
                        ui.panel_conditional(
                            "input.dataset_choice === 'upload'",
                            ui.input_file("file_upload", "Upload CSV", accept=[".csv"], multiple=False),
                        ),
                        ui.output_ui("dataset_description"),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Select Variables</div>'),
                        ui.output_ui("variable_selector"),
                        ui.HTML('</div>'),
                    ),
                    ui.column(7,
                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Data Preview</div>'),
                        ui.output_ui("data_status"),
                        ui.HTML('<div class="table-container">'),
                        ui.output_table("data_preview"),
                        ui.HTML('</div>'),
                        ui.HTML('</div>'),
                    )
                ),
            ),

            # =================================================================
            # TAB 2: DISCOVER
            # =================================================================
            ui.nav_panel(
                "Discover",
                ui.row(
                    ui.column(6,
                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Algorithm</div>'),
                        ui.input_radio_buttons(
                            "algorithm", "",
                            choices={
                                "PC": "PC Algorithm — Constraint-based, uses conditional independence tests",
                                "DirectLiNGAM": "DirectLiNGAM — Assumes linear non-Gaussian relationships",
                                "GES": "GES — Greedy Equivalence Search (score-based, BIC)"
                            },
                            selected="PC"
                        ),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Parameters</div>'),
                        ui.output_ui("algorithm_params"),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Equivalence-Class Handling</div>'),
                        ui.HTML('''<div class="info-box">
                            PC and GES return a CPDAG when some directions are not
                            identifiable. The graph shown and used by DAG-only tools is
                            one deterministic representative consistent extension: it
                            preserves the skeleton and compelled arrows and introduces no
                            new unshielded colliders. The original undirected pairs remain
                            available to the M5 ambiguity analysis.
                        </div>'''),
                        ui.HTML('</div>'),
                    ),
                    ui.column(6,
                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Constraints</div>'),
                        ui.HTML('<p style="font-size: 0.85rem; color: var(--gray-500); margin-bottom: 12px;">Incorporate domain knowledge. Format: <code>Source,Target</code> (one per line)</p>'),
                        ui.input_text_area("forbidden", "Forbidden edges", placeholder="pri,age\nBMI,age", rows=3),
                        ui.input_text_area("required", "Required edges", placeholder="age,min_Alb", rows=3),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Execute</div>'),
                        ui.output_ui("data_info_for_run"),
                        ui.input_action_button("run_btn", "Run Discovery", class_="btn-primary", width="100%"),
                        ui.output_ui("run_status"),
                        ui.HTML('</div>'),
                    )
                ),
                ui.HTML('''<div class="warning-box" style="margin-top: 16px;">
                    <strong>Note:</strong> Different algorithms can produce very different graphs from the same data.
                    Our experiments showed only 19.6% consistency between PC and GES algorithms.
                    Always validate results with domain expertise.
                </div>'''),
            ),

            # =================================================================
            # TAB 3: GRAPH
            # =================================================================
            ui.nav_panel(
                "Graph",
                ui.row(
                    ui.column(8,
                        ui.HTML('<div class="card" style="min-height: 550px;">'),
                        ui.HTML('<div class="card-title">Causal Graph</div>'),
                        ui.output_ui("dag_stats"),
                        ui.output_ui("network_plot"),
                        ui.HTML('</div>'),
                    ),
                    ui.column(4,
                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Node Inspector</div>'),
                        ui.input_select("selected_node", "Select variable", choices=[]),
                        ui.output_ui("node_details"),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Edge List</div>'),
                        ui.output_ui("edge_list"),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Quick Constraint</div>'),
                        ui.input_select("constraint_from", "From", choices=[]),
                        ui.input_select("constraint_to", "To", choices=[]),
                        ui.input_radio_buttons("constraint_type", "",
                                              choices={"forbid": "Forbid", "require": "Require"},
                                              inline=True),
                        ui.input_action_button("add_constraint_btn", "Add & Copy", class_="btn-secondary"),
                        ui.output_ui("constraint_feedback"),
                        ui.HTML('</div>'),
                    )
                ),
            ),

            # =================================================================
            # TAB 4: EVALUATE (Known Truth / Reference Comparison)
            # =================================================================
            ui.nav_panel(
                "Evaluate",
                ui.row(
                    ui.column(12,
                        ui.HTML('<div class="card" style="border-width:2px;">'),
                        ui.HTML('<div class="card-title">Structural-Importance Battery &mdash; M1&ndash;M5</div>'),
                        ui.HTML('''<p style="font-size:0.85rem;color:var(--gray-500);margin-bottom:12px;">
                            Dual-axis verdict on the discovered graph: does the picture <em>match</em> the reference
                            (M1&ndash;M2, ink), and does it <em>work</em> for the target relationship
                            (M3&ndash;M5, amber)? Set the target pair, then run. Requires a discovered graph
                            and a known simulation truth or uploaded reference graph.</p>'''),
                        ui.row(
                            ui.column(4, ui.input_select("eval_exposure", "Exposure D", choices=[])),
                            ui.column(4, ui.input_select("eval_outcome", "Outcome Y", choices=[])),
                            ui.column(4, ui.HTML('<div style="margin-top:26px;">'),
                                      ui.input_action_button("run_battery_btn", "Run M1\u2013M5",
                                                             class_="btn-primary", width="100%"),
                                      ui.HTML('</div>')),
                        ),
                        ui.output_ui("battery_results"),
                        ui.HTML('</div>'),
                    ),
                ),
                ui.row(
                    ui.column(6,
                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Reference Graph</div>'),
                        ui.input_file("gt_upload", "Upload reference edges CSV (from, to columns)", accept=[".csv"], multiple=False),
                        ui.output_ui("gt_status"),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Performance Metrics</div>'),
                        ui.output_ui("eval_metrics"),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Treatment Pathway Analysis</div>'),
                        ui.HTML('''<p style="font-size: 0.85rem; color: var(--gray-500); margin-bottom: 12px;">
                            For TMLE, the Treatment → Outcome pathway is critical. How did the algorithm perform?
                        </p>'''),
                        ui.output_ui("treatment_pathway_eval"),
                        ui.HTML('</div>'),
                    ),
                    ui.column(6,
                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Edge Classification</div>'),
                        ui.output_ui("edge_classification"),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Comparison Summary</div>'),
                        ui.output_ui("comparison_summary"),
                        ui.HTML('</div>'),
                    )
                ),
                ui.HTML('''<div class="warning-box" style="margin-top: 16px;">
                    <strong>Interpretation:</strong> High skeleton F1 with low directed F1 indicates the algorithm
                    found the right associations but struggled with edge orientation. This is common for
                    constraint-based methods like PC. Score-based methods (GES) may find different
                    equivalence classes but often have better orientation on causal pathways.
                </div>'''),
            ),

            # =================================================================
            # TAB 5: EXPORT
            # =================================================================
            ui.nav_panel(
                "Export",
                ui.row(
                    ui.column(6,
                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Identify Adjustment Sets</div>'),
                        ui.HTML('''<p style="font-size: 0.85rem; color: var(--gray-500); margin-bottom: 12px;">
                            Specify treatment and outcome. Uses Shrier-Platt algorithm to validate adjustment sets.
                        </p>'''),
                        ui.input_select("exposure_var", "Treatment / Exposure", choices=[]),
                        ui.input_select("outcome_var", "Outcome", choices=[]),
                        ui.input_action_button("identify_btn", "Run Shrier-Platt Check", class_="btn-secondary"),
                        ui.output_ui("adjustment_sets"),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Graph Statistics</div>'),
                        ui.output_ui("graph_statistics"),
                        ui.HTML('</div>'),
                    ),
                    ui.column(6,
                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">DAGitty Export</div>'),
                        ui.HTML('<p style="font-size: 0.85rem; color: var(--gray-500); margin-bottom: 12px;">Copy this code to <a href="http://dagitty.net" target="_blank">dagitty.net</a> for further analysis.</p>'),
                        ui.output_ui("dagitty_code"),
                        ui.HTML('</div>'),
                    )
                ),
                ui.HTML('''<div class="info-box" style="margin-top: 16px;">
                    <strong>Next steps:</strong> Use the discovered graph to inform your causal DAG,
                    then proceed to identification and estimation (e.g., TMLE, IPW) in Module 4.
                    Remember: <em>"The causes are not in the data"</em> — Judea Pearl
                </div>'''),
            ),

            # =================================================================
            # TAB 6: CAUSAL SHAP
            # =================================================================
            ui.nav_panel(
                "Causal SHAP",
                ui.row(
                    ui.column(4,
                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Configuration</div>'),
                        ui.input_select("shap_outcome", "Outcome variable", choices=[]),
                        ui.input_select("shap_treatment", "Treatment / Exposure", choices=[]),
                        ui.input_radio_buttons(
                            "shap_model_type", "Prediction model",
                            choices={
                                "gbm": "Gradient Boosting",
                                "rf": "Random Forest",
                                "linear": "Linear / Logistic",
                            },
                            selected="gbm"
                        ),
                        ui.input_radio_buttons(
                            "shap_method", "SHAP method",
                            choices={
                                "compare": "Compare All (recommended)",
                                "standard": "Standard SHAP only",
                                "causal": "DAG-Constrained (Causal Shapley)",
                                "adjustment": "Adjustment-Set SHAP",
                            },
                            selected="compare"
                        ),
                        ui.input_slider("shap_n_perms", "Permutations (causal)", min=20, max=200, value=50, step=10),
                        ui.input_action_button("shap_compute_btn", "Compute SHAP", class_="btn-primary", width="100%"),
                        ui.output_ui("shap_status"),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">DAG Source</div>'),
                        ui.input_radio_buttons(
                            "shap_dag_source", "",
                            choices={
                                "discovered": "Use displayed DAG (representative extension for PC/GES)",
                                "ground_truth": "Use known/uploaded reference DAG",
                            },
                            selected="discovered"
                        ),
                        ui.HTML('''<div class="info-box">
                            <strong>Standard SHAP</strong> ignores causal structure — permutes all features independently.<br><br>
                            <strong>Causal SHAP</strong> restricts permutations to valid topological orderings of the DAG
                            (Heskes et al. 2020).<br><br>
                            <strong>Adjustment-Set SHAP</strong> uses the treatment plus
                            DAG-identified adjustment covariates as features; a valid empty
                            adjustment set therefore means a treatment-only model.
                        </div>'''),
                        ui.HTML('</div>'),
                    ),
                    ui.column(8,
                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">SHAP Comparison</div>'),
                        ui.output_ui("shap_comparison_plot"),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Rank Changes</div>'),
                        ui.output_ui("shap_rank_table"),
                        ui.HTML('</div>'),

                        ui.HTML('<div class="card">'),
                        ui.HTML('<div class="card-title">Metrics</div>'),
                        ui.output_ui("shap_metrics"),
                        ui.HTML('</div>'),
                    )
                ),
                ui.HTML('''<div class="info-box" style="margin-top: 16px;">
                    <strong>Why Causal SHAP?</strong> Standard SHAP can inflate mediator importance and
                    miss true causal drivers. By constraining permutations to respect the DAG, Causal SHAP
                    aims to align feature attributions with an explicitly supplied structural model; the
                    result is only as credible as that model and its assumptions.
                    <em>The causes are not in the data — but a DAG tells SHAP where to look.</em>
                </div>'''),
            ),
            guide.guide_panel(),
            id="main_nav",
        ),
        style="max-width: 1200px; margin: 0 auto; padding: 0 24px;"
    ),

    # Footer
    ui.HTML('''
        <div class="footer">
            CAUSAL SHAP WORKBENCH &middot; experimental teaching software &middot;
            chassis: instats SSC Workshop M3 / ACIC 2026 (after Molak ch. 13)
        </div>
    '''),
)

# ============================================================================
# CAUSAL-LEARN HELPER FUNCTIONS
# ============================================================================
def _make_lingam_prior_knowledge(n, forbidden, required):
    """Create prior_knowledge matrix for causal-learn DirectLiNGAM.

    Values: -1 = unknown, 0 = no directed path, 1 = has directed path.
    Like DirectLiNGAM's learned matrix, prior knowledge uses [child, parent]
    indexing.
    """
    pk = -1 * np.ones((n, n), dtype=int)
    for source, target in forbidden:
        pk[target, source] = 0
    for source, target in required:
        pk[target, source] = 1
    return pk


# ============================================================================
# SERVER
# ============================================================================
def server(input, output, session):

    data_store = reactive.Value(pd.DataFrame())
    adj_matrix_store = reactive.Value(None)
    col_names_store = reactive.Value([])
    discovery_complete = reactive.Value(False)
    undirected_pairs_store = reactive.Value([])
    ground_truth_store = reactive.Value(None)

    # SHAP reactive stores
    shap_result_store = reactive.Value(None)
    shap_computing = reactive.Value(False)

    # -------------------------------------------------------------------------
    # DATA TAB
    # -------------------------------------------------------------------------
    @reactive.Effect
    @reactive.event(input.dataset_choice, input.file_upload)
    def _load_data():
        df = pd.DataFrame()
        choice = input.dataset_choice()

        if choice == "simcausal":
            if os.path.exists(SAMPLE_FILE):
                df = pd.read_csv(SAMPLE_FILE)
        elif choice == "upload":
            if input.file_upload() is not None:
                file_info = input.file_upload()[0]
                try:
                    df = pd.read_csv(file_info["datapath"])
                except Exception as e:
                    print(f"[ERROR] {e}")

        data_store.set(df)
        adj_matrix_store.set(None)
        col_names_store.set([])
        discovery_complete.set(False)
        undirected_pairs_store.set([])
        shap_result_store.set(None)
        ground_truth_store.set(
            load_ground_truth_edges(GROUND_TRUTH_FILE)
            if choice == "simcausal"
            else None
        )

    @output
    @render.ui
    def dataset_description():
        choice = input.dataset_choice()
        if choice == "simcausal":
            return ui.HTML('''<div class="info-box">
                <strong>Simcausal data:</strong> 500 obs, 12 variables with known causal structure.
                Treatment → Outcome direct effect = +5. Ground truth DAG available.
                Ideal for validating causal SHAP against known effects.
            </div>''')
        return ui.HTML('''<div class="info-box">
            Uploaded data stay in the current local Shiny session. If a known
            reference graph exists, upload a separate from/to edge list on the
            Evaluate tab.
        </div>''')

    @output
    @render.ui
    def data_status():
        df = data_store.get()
        if df.empty:
            return ui.HTML('<div class="warning-box">No data loaded</div>')

        n_numeric = df.select_dtypes(include=[np.number]).shape[1]
        return ui.HTML(f'''
            <div style="margin-bottom: 16px;">
                <span class="stat-pill"><span class="value">{df.shape[0]}</span> rows</span>
                <span class="stat-pill"><span class="value">{df.shape[1]}</span> columns</span>
                <span class="stat-pill primary"><span class="value">{n_numeric}</span> numeric</span>
            </div>
        ''')

    @output
    @render.table
    def data_preview():
        df = data_store.get()
        if df.empty:
            return pd.DataFrame({"Status": ["Load data to preview"]})
        return df.head(8)

    @output
    @render.ui
    def variable_selector():
        df = data_store.get()
        if df.empty:
            return ui.HTML('<p style="color: var(--gray-400);">Load data first</p>')

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        default = numeric_cols[:14] if len(numeric_cols) > 14 else numeric_cols
        return ui.input_checkbox_group("selected_vars", "", choices=numeric_cols, selected=default)

    # -------------------------------------------------------------------------
    # DISCOVER TAB
    # -------------------------------------------------------------------------
    @output
    @render.ui
    def data_info_for_run():
        df = data_store.get()
        selected = input.selected_vars() if input.selected_vars() else []

        if df.empty:
            return ui.HTML('<div class="warning-box">No data loaded</div>')

        n_vars = len(selected) if selected else 0
        df_subset = df[list(selected)].dropna() if selected else df

        return ui.HTML(f'''
            <div style="background: var(--gray-50); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <p style="margin: 0; font-size: 0.85rem;">
                    <strong>Sample size:</strong> {len(df_subset):,} observations<br>
                    <strong>Variables:</strong> {n_vars} selected
                </p>
            </div>
        ''')

    @output
    @render.ui
    def algorithm_params():
        algo = input.algorithm()
        if algo == "PC":
            return ui.div(
                ui.input_slider("pc_alpha", "Significance level (α)", min=0.01, max=0.2, value=0.05, step=0.01),
                ui.HTML('<p style="font-size: 0.8rem; color: var(--gray-400);">Lower = more conservative (fewer edges)</p>')
            )
        elif algo == "GES":
            return ui.div(
                ui.HTML('<p style="font-size: 0.8rem; color: var(--gray-400);">Uses BIC score to greedily search equivalence classes. No tuning parameters needed.</p>')
            )
        return ui.HTML('<p style="color: var(--gray-400);">No additional parameters</p>')

    @output
    @render.ui
    @reactive.event(input.run_btn)
    def run_status():
        if not CASTLE_AVAILABLE:
            return ui.HTML('<div class="error-box">causal-learn not installed. Run: pip install causal-learn</div>')

        df = data_store.get()
        if df.empty:
            return ui.HTML('<div class="error-box">No data loaded</div>')

        try:
            selected = list(input.selected_vars()) if input.selected_vars() else []
        except Exception:
            # selector not rendered yet (arrived straight from the Map)
            selected = []
        if not selected:
            # Default to all numeric columns (2026-08-12), matching the
            # SHAP tab's behavior
            selected = df.select_dtypes(include=[np.number]).columns.tolist()[:14]
        if len(selected) < 2:
            return ui.HTML('<div class="error-box">Select at least 2 variables</div>')

        df_subset = df[list(selected)].select_dtypes(include=[np.number]).dropna()
        cols = df_subset.columns.tolist()
        col_names_store.set(cols)

        def parse_edges(text, cols):
            edges = []
            if not text:
                return edges
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 2 and parts[0] in cols and parts[1] in cols:
                    edges.append((cols.index(parts[0]), cols.index(parts[1])))
            return edges

        forbidden = parse_edges(input.forbidden(), cols)
        required = parse_edges(input.required(), cols)
        forbidden_named = [(cols[source], cols[target]) for source, target in forbidden]
        required_named = [(cols[source], cols[target]) for source, target in required]

        # Build causal-learn background knowledge
        nodes_cl = [GraphNode(name) for name in cols]
        bk = BackgroundKnowledge()
        for (i, j) in forbidden:
            bk.add_forbidden_by_node(nodes_cl[i], nodes_cl[j])
        for (i, j) in required:
            bk.add_required_by_node(nodes_cl[i], nodes_cl[j])

        algo_name = input.algorithm()
        data_matrix = df_subset.values
        n_obs, n_vars = data_matrix.shape

        start_time = time.time()

        try:
            if algo_name == "PC":
                alpha = input.pc_alpha() if input.pc_alpha() else 0.05
                cg = pc_algorithm(data_matrix, alpha=alpha, indep_test='fisherz',
                                  stable=True, background_knowledge=bk,
                                  node_names=cols, show_progress=False)
                pdag = causal_learn_graph_to_pdag(cg.G, cols)
            elif algo_name == "DirectLiNGAM":
                pk = _make_lingam_prior_knowledge(len(cols), forbidden, required)
                model = lingam.DirectLiNGAM(random_state=42, prior_knowledge=pk)
                model.fit(data_matrix)
                pdag = direct_lingam_adjacency_to_pdag(
                    model.adjacency_matrix_, cols, threshold=0.01
                )
            elif algo_name == "GES":
                record = ges_algorithm(
                    data_matrix, score_func='local_score_BIC', node_names=cols
                )
                pdag = causal_learn_graph_to_pdag(record['G'], cols)

            elapsed_time = time.time() - start_time

            # Constraints are enforced on the PDAG itself.  Retain its
            # equivalence-class ambiguity for M5, then derive one deterministic
            # consistent extension for graph display and DAG-only analyses.
            pdag = apply_constraints(pdag, forbidden_named, required_named)
            und_pairs = sorted(pdag.undirected_edges)
            undirected_pairs_store.set(und_pairs)

            representative = deterministic_consistent_extension(pdag)
            adj_matrix = nx.to_numpy_array(
                representative, nodelist=cols, dtype=int
            )

            adj_matrix_store.set(adj_matrix)
            discovery_complete.set(True)

            n_edges = representative.number_of_edges()

            result_html = f'''<div class="success-box">
                <strong>Discovery complete</strong><br>
                <span style="font-size: 0.9rem;">
                    Algorithm: {algo_name}<br>
                    Sample: {n_obs:,} observations × {n_vars} variables<br>
                    Edges found: {n_edges}<br>
                    Time: {elapsed_time:.2f} seconds
                </span>
            '''
            if required:
                result_html += f'<br><span style="color: #065f46;">✓ Enforced {len(required)} required edge(s)</span>'
            if forbidden:
                result_html += f'<br><span style="color: #065f46;">✓ Removed {len(forbidden)} forbidden edge(s)</span>'
            if algo_name in ("PC", "GES"):
                result_html += (
                    '<br><span style="color: #1d4ed8;">Graph shown: one '
                    'deterministic representative consistent DAG extension; '
                    f'{len(und_pairs)} unresolved CPDAG pair(s) retained for M5.</span>'
                )
            else:
                result_html += (
                    '<br><span style="color: #1d4ed8;">Graph shown: directed '
                    'DirectLiNGAM result (matrix interpreted as child, parent).</span>'
                )
            result_html += '</div>'

            return ui.HTML(result_html)

        except Exception as e:
            return ui.HTML(
                f'<div class="error-box">Error: {html_escape(str(e))}</div>'
            )

    # -------------------------------------------------------------------------
    # GRAPH TAB
    # -------------------------------------------------------------------------
    @reactive.Effect
    def _update_selects():
        cols = col_names_store.get()
        if cols:
            ui.update_select("selected_node", choices=cols)
            ui.update_select("constraint_from", choices=cols)
            ui.update_select("constraint_to", choices=cols)
            ui.update_select("exposure_var", choices=cols)
            ui.update_select("outcome_var", choices=cols)
            exp_default = "Treatment" if "Treatment" in cols else cols[0]
            out_default = "Outcome" if "Outcome" in cols else cols[-1]
            ui.update_select("eval_exposure", choices=cols, selected=exp_default)
            ui.update_select("eval_outcome", choices=cols, selected=out_default)

    @output
    @render.ui
    def dag_stats():
        if not discovery_complete.get():
            return ui.HTML('')

        adj = adj_matrix_store.get()
        cols = col_names_store.get()
        if adj is None:
            return ui.HTML('')

        n_edges = int(np.sum(adj > 0))
        n_nodes = len(cols)
        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

        G = nx.DiGraph(adj)
        is_dag = nx.is_directed_acyclic_graph(G)

        return ui.HTML(f'''
            <div style="margin-bottom: 16px;">
                <span class="stat-pill"><span class="value">{n_nodes}</span> nodes</span>
                <span class="stat-pill"><span class="value">{n_edges}</span> edges</span>
                <span class="stat-pill"><span class="value">{density:.1%}</span> density</span>
                <span class="stat-pill {'primary' if is_dag else ''}" style="{'background: #d1fae5; color: #065f46;' if is_dag else 'background: #fee2e2; color: #991b1b;'}">
                    {'✓ DAG used downstream' if is_dag else '✗ Invalid representative'}
                </span>
            </div>
        ''')

    @output
    @render.ui
    def network_plot():
        if not discovery_complete.get():
            return ui.HTML('''
                <div style="text-align: center; padding: 80px 20px; color: var(--gray-400);">
                    <p style="font-size: 1.1rem;">Run discovery to generate graph</p>
                    <p style="font-size: 0.85rem;">Go to the Discover tab and click "Run Discovery"</p>
                </div>
            ''')

        adj = adj_matrix_store.get()
        cols = col_names_store.get()

        if adj is None or not cols:
            return ui.HTML('<div class="error-box">No graph available</div>')

        if not PYVIS_AVAILABLE:
            return ui.HTML('''<div class="warning-box">
                Graph rendering requires the Workbench dependency extra.
                The discovered edge list remains available below.
            </div>''')

        G = nx.DiGraph()
        for col in cols:
            G.add_node(col, label=col)

        rows_idx, cols_idx = np.where(adj > 0)
        for r, c in zip(rows_idx, cols_idx):
            G.add_edge(cols[r], cols[c])

        net = Network(height="450px", width="100%", notebook=False, directed=True,
                     cdn_resources='in_line', bgcolor="#ffffff")

        net.set_options('''
        {
            "nodes": {
                "color": {"background": "#ffffff", "border": "#111111", "highlight": {"background": "#fff7ed", "border": "#b45309"}},
                "font": {"color": "#111111", "size": 13, "face": "Georgia, serif"},
                "borderWidth": 2,
                "shadow": {"enabled": false}
            },
            "edges": {
                "color": {"color": "#111111", "highlight": "#b45309"},
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
                "smooth": {"type": "curvedCW", "roundness": 0.15},
                "width": 1.5
            },
            "physics": {
                "forceAtlas2Based": {"gravitationalConstant": -40, "springLength": 120},
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 80}
            },
            "interaction": {"hover": true, "tooltipDelay": 100}
        }
        ''')

        net.from_nx(G)
        html_content = net.generate_html("graph.html")
        b64 = base64.b64encode(html_content.encode()).decode()

        return ui.HTML(f'<iframe src="data:text/html;base64,{b64}" style="width:100%; height:450px; border:1px solid var(--gray-200); border-radius: 8px;"></iframe>')

    @output
    @render.ui
    def node_details():
        if not discovery_complete.get():
            return ui.HTML('<p style="color: var(--gray-400);">Run discovery first</p>')

        adj = adj_matrix_store.get()
        cols = col_names_store.get()
        selected = input.selected_node()

        if adj is None or not cols or not selected or selected not in cols:
            return ui.HTML('<p style="color: var(--gray-400);">Select a variable</p>')

        idx = cols.index(selected)
        parents = [cols[i] for i in range(len(cols)) if adj[i, idx] > 0]
        children = [cols[i] for i in range(len(cols)) if adj[idx, i] > 0]

        return ui.HTML(f'''
            <div class="node-inspector">
                <h5>{safe_text(selected)}</h5>
                <p><strong class="parents">← Inputs (parents):</strong><br>
                   {safe_join(parents) if parents else '<em style="color: var(--gray-400);">None</em>'}</p>
                <p><strong class="children">→ Outputs (children):</strong><br>
                   {safe_join(children) if children else '<em style="color: var(--gray-400);">None</em>'}</p>
                <p style="font-size: 0.8rem; color: var(--gray-400); margin-top: 8px;">
                    In-degree: {len(parents)} · Out-degree: {len(children)}
                </p>
            </div>
        ''')

    @output
    @render.ui
    def edge_list():
        if not discovery_complete.get():
            return ui.HTML('<p style="color: var(--gray-400);">Run discovery first</p>')

        adj = adj_matrix_store.get()
        cols = col_names_store.get()

        if adj is None:
            return ui.HTML('')

        edges = []
        rows_idx, cols_idx = np.where(adj > 0)
        for r, c in zip(rows_idx, cols_idx):
            edges.append(
                f'<div class="edge">{safe_text(cols[r])} → '
                f'{safe_text(cols[c])}</div>'
            )

        if not edges:
            return ui.HTML('<p style="color: var(--gray-400);">No edges</p>')

        html = '<div class="edge-list">' + ''.join(edges[:40])
        if len(edges) > 40:
            html += f'<div style="color: var(--gray-400); margin-top: 8px;">+ {len(edges)-40} more</div>'
        html += '</div>'

        return ui.HTML(html)

    @output
    @render.ui
    @reactive.event(input.add_constraint_btn)
    def constraint_feedback():
        f, t = input.constraint_from(), input.constraint_to()
        if not f or not t or f == t:
            return ui.HTML('')

        c_type = "Forbidden" if input.constraint_type() == "forbid" else "Required"
        return ui.HTML(
            f'<div class="success-box">Add to {c_type}: '
            f'<code>{safe_text(f)},{safe_text(t)}</code></div>'
        )

    # -------------------------------------------------------------------------
    # EXPORT TAB
    # -------------------------------------------------------------------------
    @output
    @render.ui
    @reactive.event(input.identify_btn)
    def adjustment_sets():
        if not discovery_complete.get():
            return ui.HTML('<div class="warning-box">Run discovery first</div>')

        adj = adj_matrix_store.get()
        cols = col_names_store.get()
        exp, out = input.exposure_var(), input.outcome_var()

        if not exp or not out or exp == out:
            return ui.HTML('<div class="warning-box">Select different exposure and outcome</div>')

        # Build graph and run identification
        G = nx.DiGraph(adj)
        results = identify_adjustment_sets(G, exp, out, cols)

        if not results:
            return ui.HTML('<div class="error-box">Could not identify adjustment sets</div>')

        # Check direct path
        exp_idx, out_idx = cols.index(exp), cols.index(out)
        has_path = nx.has_path(G, exp_idx, out_idx) if exp_idx in G and out_idx in G else False

        html = f'''
            <div class="node-inspector" style="margin-top: 12px;">
                <h5>{safe_text(exp)} → {safe_text(out)}</h5>
                <p><strong>Direct causal path:</strong> {'Yes' if has_path else 'No'}</p>
            </div>
        '''

        # Display each adjustment set criterion
        for key, result in results.items():
            valid_icon = '✓' if result['valid'] else '✗'
            valid_color = 'var(--success)' if result['valid'] else 'var(--danger)'
            valid_bg = '#d1fae5' if result['valid'] else '#fee2e2'

            vars_str = safe_join(result['variables']) if result['variables'] else '<em>None</em>'

            html += f'''
                <div style="background: {valid_bg}; border-radius: 8px; padding: 12px; margin: 12px 0;">
                    <p style="margin: 0 0 8px 0;">
                        <strong style="color: {valid_color};">{valid_icon} {result['name']}</strong>
                    </p>
                    <p style="font-size: 0.85rem; color: var(--gray-600); margin: 0 0 8px 0;">
                        {result['description']}
                    </p>
                    <p style="margin: 0 0 8px 0;">
                        <strong>Variables:</strong> {vars_str}
                    </p>
            '''

            # Add citation if present
            if 'citation' in result:
                html += f'''
                    <p style="font-size: 0.75rem; color: var(--gray-500); margin: 0 0 8px 0; font-style: italic;">
                        {result['citation']}
                    </p>
                '''

            # Add Shrier-Platt steps ONLY for minimal sufficient set
            if key == 'minimal' and result['steps']:
                steps_html = '<br>'.join(result['steps'])
                html += f'''
                    <details style="font-size: 0.8rem; margin-top: 8px;">
                        <summary style="cursor: pointer; color: var(--gray-500);">Shrier-Platt 6-step validation</summary>
                        <div style="margin-top: 8px; padding: 8px; background: rgba(255,255,255,0.5); border-radius: 4px;">
                            {steps_html}
                        </div>
                    </details>
                '''

            html += '</div>'

        html += '''
            <div class="info-box" style="margin-top: 12px;">
                <strong>Shrier-Platt Algorithm</strong> (Shrier & Platt, 2008) validates whether an adjustment set
                blocks all backdoor paths. Based on discovered graph only — unmeasured confounders cannot be detected.
            </div>
        '''

        return ui.HTML(html)

    @output
    @render.ui
    def graph_statistics():
        if not discovery_complete.get():
            return ui.HTML('<p style="color: var(--gray-400);">Run discovery first</p>')

        adj = adj_matrix_store.get()
        cols = col_names_store.get()

        if adj is None:
            return ui.HTML('')

        G = nx.DiGraph(adj)
        roots = [cols[n] for n in G.nodes() if G.in_degree(n) == 0]
        leaves = [cols[n] for n in G.nodes() if G.out_degree(n) == 0]
        is_dag = nx.is_directed_acyclic_graph(G)

        unresolved = undirected_pairs_store.get()

        html = f'''
            <p><strong>Graph used downstream:</strong> {'DAG ✓' if is_dag else 'Invalid'}</p>
            <p><strong>CPDAG ambiguity retained for M5:</strong> {len(unresolved)} undirected pair(s)</p>
            <p><strong>Roots</strong> (no parents): {safe_join(roots[:5])}{' ...' if len(roots) > 5 else '' if roots else 'None'}</p>
            <p><strong>Leaves</strong> (no children): {safe_join(leaves[:5])}{' ...' if len(leaves) > 5 else '' if leaves else 'None'}</p>
        '''

        return ui.HTML(html)

    @output
    @render.ui
    def dagitty_code():
        if not discovery_complete.get():
            return ui.HTML('<div class="code-block" style="color: #6b7280;">Run discovery first</div>')

        adj = adj_matrix_store.get()
        cols = col_names_store.get()

        if adj is None:
            return ui.HTML('')

        lines = ["dag {"]
        rows_idx, cols_idx = np.where(adj > 0)
        for r, c in zip(rows_idx, cols_idx):
            lines.append(f"  {cols[r]} -> {cols[c]}")
        lines.append("}")

        return ui.HTML(
            f'<div class="code-block">{html_escape(chr(10).join(lines))}</div>'
        )

    # -------------------------------------------------------------------------
    # EVALUATE TAB (Known Truth / Uploaded Reference Comparison)
    # -------------------------------------------------------------------------
    @reactive.Effect
    @reactive.event(input.gt_upload)
    def _load_ground_truth():
        if input.dataset_choice() == "upload" and input.gt_upload() is not None:
            file_info = input.gt_upload()[0]
            ground_truth_store.set(load_ground_truth_edges(file_info["datapath"]))

    @output
    @render.ui
    def gt_status():
        gt = ground_truth_store.get()
        if gt is None:
            return ui.HTML('''<div class="warning-box">
                No reference graph loaded. The known simulation truth loads
                automatically; custom data require a separate edge-list upload.
            </div>''')

        label = (
            "Known simulation truth"
            if input.dataset_choice() == "simcausal"
            else "Uploaded reference graph"
        )
        return ui.HTML(f'''
            <div class="success-box">
                <strong>{label} loaded:</strong> {len(gt)} edges
            </div>
        ''')

    @output
    @render.ui
    def eval_metrics():
        if not discovery_complete.get():
            return ui.HTML('<p style="color: var(--gray-400);">Run discovery first</p>')

        gt = ground_truth_store.get()
        if gt is None:
            return ui.HTML('<p style="color: var(--gray-400);">Load a reference graph first</p>')

        adj = adj_matrix_store.get()
        cols = col_names_store.get()

        if adj is None or not cols:
            return ui.HTML('<p style="color: var(--gray-400);">No discovered graph</p>')

        # Build discovered edges set
        discovered = set()
        rows_idx, cols_idx = np.where(adj > 0)
        for r, c in zip(rows_idx, cols_idx):
            discovered.add((cols[r], cols[c]))

        # Filter ground truth to only include variables in discovered graph
        gt_filtered = {(f, t) for f, t in gt if f in cols and t in cols}

        if not gt_filtered:
            return ui.HTML('<div class="warning-box">No overlapping variables between the discovered and reference graphs</div>')

        # Compare
        results = compare_graphs(discovered, gt_filtered)

        # SHD via the named-graph battery module
        G_learned = nx.DiGraph(); G_learned.add_nodes_from(cols)
        G_learned.add_edges_from(discovered)
        G_true = nx.DiGraph(); G_true.add_nodes_from(cols)
        G_true.add_edges_from(gt_filtered)
        shd = evaluation.m1_concordance(G_learned, G_true)["shd"]

        return ui.HTML(f'''
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;">
                <div style="background: var(--gray-50); padding: 12px; border-radius: 2px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--ink); font-family: 'Courier New', monospace;">{shd}</div>
                    <div style="font-size: 0.75rem; color: var(--gray-500);">SHD (edge edits)</div>
                </div>
                <div style="background: var(--gray-50); padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary);">{results['precision']:.1%}</div>
                    <div style="font-size: 0.75rem; color: var(--gray-500);">Precision</div>
                </div>
                <div style="background: var(--gray-50); padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary);">{results['recall']:.1%}</div>
                    <div style="font-size: 0.75rem; color: var(--gray-500);">Recall</div>
                </div>
                <div style="background: var(--gray-50); padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--success);">{results['f1']:.1%}</div>
                    <div style="font-size: 0.75rem; color: var(--gray-500);">F1 (Directed)</div>
                </div>
                <div style="background: var(--gray-50); padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--success);">{results['skeleton_f1']:.1%}</div>
                    <div style="font-size: 0.75rem; color: var(--gray-500);">F1 (Skeleton)</div>
                </div>
            </div>
            <div style="margin-top: 16px; font-size: 0.85rem; color: var(--gray-600);">
                <p><strong>Discovered:</strong> {results['n_discovered']} edges | <strong>Reference:</strong> {results['n_ground_truth']} edges</p>
                <p style="margin-top: 4px;">
                    <span style="color: var(--success);">✓ Correct: {results['n_tp']}</span> ·
                    <span style="color: var(--warning);">↔ Reversed: {results['n_reversed']}</span> ·
                    <span style="color: var(--danger);">+ Spurious: {results['n_spurious']}</span> ·
                    <span style="color: var(--gray-400);">− Missed: {results['n_missed']}</span>
                </p>
            </div>
        ''')

    @output
    @render.ui
    def edge_classification():
        if not discovery_complete.get():
            return ui.HTML('<p style="color: var(--gray-400);">Run discovery first</p>')

        gt = ground_truth_store.get()
        if gt is None:
            return ui.HTML('<p style="color: var(--gray-400);">Load a reference graph first</p>')

        adj = adj_matrix_store.get()
        cols = col_names_store.get()

        if adj is None or not cols:
            return ui.HTML('')

        # Build discovered edges set
        discovered = set()
        rows_idx, cols_idx = np.where(adj > 0)
        for r, c in zip(rows_idx, cols_idx):
            discovered.add((cols[r], cols[c]))

        gt_filtered = {(f, t) for f, t in gt if f in cols and t in cols}
        results = compare_graphs(discovered, gt_filtered)

        html = '<div class="edge-list" style="max-height: 350px;">'

        # True positives
        if results['true_positives']:
            html += '<div style="margin-bottom: 8px;"><strong style="color: var(--success);">✓ Correct</strong></div>'
            for f, t in sorted(results['true_positives']):
                html += f'<div class="edge" style="color: var(--success);">{safe_text(f)} → {safe_text(t)}</div>'

        # Reversed
        if results['reversed']:
            html += '<div style="margin: 12px 0 8px 0;"><strong style="color: var(--warning);">↔ Reversed</strong></div>'
            for f, t in sorted(results['reversed']):
                html += f'<div class="edge" style="color: var(--warning);">{safe_text(f)} → {safe_text(t)} <span style="font-size: 0.7rem;">(should be {safe_text(t)} → {safe_text(f)})</span></div>'

        # Spurious
        if results['spurious']:
            html += '<div style="margin: 12px 0 8px 0;"><strong style="color: var(--danger);">+ Spurious</strong></div>'
            for f, t in sorted(results['spurious']):
                html += f'<div class="edge" style="color: var(--danger);">{safe_text(f)} → {safe_text(t)}</div>'

        # Missed
        if results['missed']:
            html += '<div style="margin: 12px 0 8px 0;"><strong style="color: var(--gray-400);">− Missed</strong></div>'
            for f, t in sorted(results['missed']):
                html += f'<div class="edge" style="color: var(--gray-400);">{safe_text(f)} → {safe_text(t)}</div>'

        html += '</div>'
        return ui.HTML(html)

    @output
    @render.ui
    def treatment_pathway_eval():
        if not discovery_complete.get():
            return ui.HTML('<p style="color: var(--gray-400);">Run discovery first</p>')

        gt = ground_truth_store.get()
        adj = adj_matrix_store.get()
        cols = col_names_store.get()

        if gt is None or adj is None or not cols:
            return ui.HTML('<p style="color: var(--gray-400);">Load data first</p>')

        exp = input.eval_exposure() or ("Treatment" if "Treatment" in cols else cols[0])
        out = input.eval_outcome() or ("Outcome" if "Outcome" in cols else cols[-1])

        # Path-derived pathway edges from the loaded ground truth (2026-08-12:
        # replaces the hardcoded simcausal edge list)
        G_true = nx.DiGraph(); G_true.add_nodes_from(cols)
        G_true.add_edges_from((f, t) for f, t in gt if f in cols and t in cols)
        pathway = sorted(evaluation.target_pathway_edges(G_true, exp, out))

        if not pathway:
            return ui.HTML(f'<div class="info-box">No directed {safe_text(exp)} &rarr; {safe_text(out)} pathway in the reference graph. Pick the target pair in the battery card above.</div>')

        discovered = set()
        rows_idx, cols_idx = np.where(adj > 0)
        for r, c in zip(rows_idx, cols_idx):
            discovered.add((cols[r], cols[c]))

        html = '<table style="width: 100%;"><thead><tr><th>Edge</th><th>Status</th></tr></thead><tbody>'
        correct = 0
        for f, t in pathway:
            if (f, t) in discovered:
                html += f'<tr><td>{safe_text(f)} \u2192 {safe_text(t)}</td><td style="color: var(--success);">\u2713 Correct</td></tr>'
                correct += 1
            elif (t, f) in discovered:
                html += f'<tr><td>{safe_text(f)} \u2192 {safe_text(t)}</td><td style="color: var(--warning);">\u2194 Reversed</td></tr>'
            else:
                html += f'<tr><td>{safe_text(f)} \u2192 {safe_text(t)}</td><td style="color: var(--danger);">\u2717 Missing</td></tr>'
        html += '</tbody></table>'

        pct = correct / len(pathway) * 100
        summary_color = "var(--success)" if pct >= 80 else "var(--warning)" if pct >= 50 else "var(--danger)"
        html += f'''
            <div style="margin-top: 12px; padding: 8px; background: var(--gray-50); border-radius: 2px; text-align: center;">
                <span style="color: {summary_color}; font-weight: 600;">{correct}/{len(pathway)} pathway edges correct ({pct:.0f}%) &mdash; this is M2\u2019s scorecard</span>
            </div>
        '''
        return ui.HTML(html)

    @output
    @render.ui
    def comparison_summary():
        if not discovery_complete.get():
            return ui.HTML('<p style="color: var(--gray-400);">Run discovery first</p>')

        gt = ground_truth_store.get()
        if gt is None:
            return ui.HTML('<p style="color: var(--gray-400);">Load a reference graph first</p>')

        adj = adj_matrix_store.get()
        cols = col_names_store.get()

        if adj is None or not cols:
            return ui.HTML('')

        # Build discovered edges set
        discovered = set()
        rows_idx, cols_idx = np.where(adj > 0)
        for r, c in zip(rows_idx, cols_idx):
            discovered.add((cols[r], cols[c]))

        gt_filtered = {(f, t) for f, t in gt if f in cols and t in cols}
        results = compare_graphs(discovered, gt_filtered)

        # Determine overall assessment
        if results['f1'] >= 0.6:
            assessment = ("Good", "var(--success)", "The algorithm performed well on this dataset.")
        elif results['f1'] >= 0.4:
            assessment = ("Moderate", "var(--warning)", "Some edges correct, but significant errors remain.")
        else:
            assessment = ("Poor", "var(--danger)", "Low accuracy - consider different algorithm or parameters.")

        # Check skeleton vs directed discrepancy
        skeleton_gap = results['skeleton_f1'] - results['f1']

        html = f'''
            <div style="text-align: center; padding: 16px; background: var(--gray-50); border-radius: 8px; margin-bottom: 16px;">
                <div style="font-size: 1.25rem; font-weight: 700; color: {assessment[1]};">{assessment[0]}</div>
                <div style="font-size: 0.85rem; color: var(--gray-600); margin-top: 4px;">{assessment[2]}</div>
            </div>
        '''

        if skeleton_gap > 0.15:
            html += f'''
                <div class="info-box">
                    <strong>Orientation problem detected:</strong> Skeleton F1 ({results['skeleton_f1']:.1%}) is much higher
                    than directed F1 ({results['f1']:.1%}). The algorithm found the right associations but
                    got {results['n_reversed']} edge directions wrong.
                </div>
            '''

        if results['n_spurious'] > results['n_tp']:
            html += f'''
                <div class="warning-box">
                    <strong>High false positive rate:</strong> {results['n_spurious']} spurious edges vs
                    {results['n_tp']} correct. Consider increasing the threshold or alpha parameter.
                </div>
            '''

        return ui.HTML(html)


    # -------------------------------------------------------------------------
    # MAP NAVIGATION + M1-M5 BATTERY (2026-08-12)
    # -------------------------------------------------------------------------
    @reactive.Effect
    @reactive.event(input.goto_tab)
    def _goto_tab():
        ui.update_navs("main_nav", selected=input.goto_tab())

    def _teach(num, body):
        return (f'<details><summary>teaching card {num}</summary>'
                f'<div class="teach">{body}</div></details>')

    @output
    @render.ui
    @reactive.event(input.run_battery_btn)
    def battery_results():
        if not discovery_complete.get():
            return ui.HTML('<div class="warning-box">Run discovery first (station 2)</div>')
        gt = ground_truth_store.get()
        if gt is None:
            return ui.HTML('<div class="warning-box">Load a reference graph below first</div>')
        adj = adj_matrix_store.get()
        cols = col_names_store.get()
        exp, out = input.eval_exposure(), input.eval_outcome()
        if adj is None or not cols:
            return ui.HTML('<div class="warning-box">No discovered graph</div>')
        if not exp or not out or exp == out:
            return ui.HTML('<div class="warning-box">Pick distinct exposure and outcome</div>')

        G_learned = nx.DiGraph(); G_learned.add_nodes_from(cols)
        rows_idx, cols_idx = np.where(adj > 0)
        for r, c in zip(rows_idx, cols_idx):
            G_learned.add_edge(cols[r], cols[c])
        G_true = nx.DiGraph(); G_true.add_nodes_from(cols)
        G_true.add_edges_from((f, t) for f, t in gt if f in cols and t in cols)

        df = data_store.get()
        try:
            selected = list(input.selected_vars()) if input.selected_vars() else cols
        except Exception:
            selected = cols
        use_cols = [c for c in set(selected + [exp, out]) if c in df.columns]
        analysis_df = df[use_cols].dropna() if not df.empty else None

        true_effect = None
        if input.dataset_choice() == "simcausal" and out == "Outcome":
            try:
                with open(TRUE_EFFECTS_FILE) as fh:
                    te = json.load(fh)
                true_effect = te.get(exp)
            except Exception:
                true_effect = None

        try:
            res = evaluation.evaluate_battery(
                G_learned, G_true, exp, out,
                data=analysis_df if analysis_df is not None and len(analysis_df) else None,
                true_effect=true_effect,
                undirected_pairs=undirected_pairs_store.get())
        except Exception as e:
            return ui.HTML(
                f'<div class="error-box">Battery error: '
                f'{html_escape(str(e))}</div>'
            )

        m1, m2, m3 = res["m1"], res["m2"], res["m3"]
        m4, m5 = res.get("m4"), res["m5"]

        def tile(val, lab, color="var(--ink)"):
            return (f'<div class="stat-card"><div class="stat-value" '
                    f'style="color:{color}">{val}</div>'
                    f'<div class="stat-label">{lab}</div></div>')

        html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px;">'

        # ---- concordance axis (ink)
        html += '<div><div class="card-title">concordance axis</div>'
        html += '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">'
        html += tile(f"{m1['f1']:.0%}", "M1 F1 directed")
        html += tile(f"{m1['skeleton_f1']:.0%}", "M1 F1 skeleton")
        html += tile(m1["shd"], "M1 SHD")
        html += '</div>'
        html += _teach("M1", "Edge concordance: precision/recall/F1 over directed "
                       "edges and over the undirected skeleton, plus structural "
                       "Hamming distance. The field-standard anchor; it treats "
                       "every edge as equally important.")
        sp = safe_join(m2["spurious"]) if m2["spurious"] else "none"
        html += (f'<div style="margin-top:12px;"><strong>M2 target pathway '
                 f'({safe_text(exp)} &rarr; {safe_text(out)}):</strong> {m2["n_correct"]}/'
                 f'{m2["n_true_pathway"]} correct &middot; {m2["n_reversed"]} reversed '
                 f'&middot; {m2["n_missed"]} missed &middot; spurious: {sp}</div>')
        html += '<div class="edge-list" style="margin-top:8px;max-height:140px;">'
        for e_str, status in m2["edges"].items():
            col = {"correct": "var(--success)", "reversed": "var(--warning)",
                   "missed": "var(--danger)"}[status]
            html += f'<div class="edge" style="color:{col}">{safe_text(e_str)} &mdash; {status}</div>'
        html += '</div>'
        html += _teach("M2", "The same correct/reversed/spurious/missed grammar, "
                       "restricted to edges on directed D&rarr;Y paths in the true "
                       "graph. An error on the exposure pathway and an error on a "
                       "peripheral edge stop counting the same.")
        html += '</div>'

        # ---- structural-importance axis (amber)
        html += '<div><div class="card-title" style="color:var(--primary);border-color:var(--primary);">structural-importance axis</div>'
        v = m3["valid_in_true"]
        vcol = "var(--success)" if v else "var(--danger)"
        vtxt = "VALID" if v else "INVALID"
        zl = safe_join(m3["z_learned"]) if m3["z_learned"] else "&empty;"
        zt = safe_join(m3["z_true"]) if m3["z_true"] else "&empty;"
        html += (f'<div style="margin-bottom:6px;"><strong>M3 sufficiency transfer:</strong> '
                 f"Z&prime; = {{{zl}}} &rarr; <span style='color:{vcol};font-weight:700;'>{vtxt}</span> in the reference graph</div>")
        if m3["descendant_offenders"]:
            html += (f'<div class="warning-box">Z&prime; contains descendant(s) of {safe_text(exp)} '
                     f'in the reference graph: {safe_join(m3["descendant_offenders"])} '
                     f'&mdash; backdoor test alone would have missed this.</div>')
        html += (f'<div style="font-size:0.85rem;color:var(--gray-600);">true minimal set '
                 f'{{{zt}}} &middot; Jaccard {m3["jaccard"]:.2f} &middot; excess '
                 f'{m3["excess_size"]:+d}</div>')
        steps = "<br>".join(m3["verdict_steps"])
        html += (f'<details style="margin-top:4px;"><summary>Shrier&ndash;Platt trace</summary>'
                 f'<div class="teach">{steps}</div></details>')
        html += _teach("M3", "Derive the minimal adjustment set Z&prime; from the "
                       "learned graph, then test it inside the separate reference graph: "
                       "backdoor blocked AND no descendant of D. The graph is judged "
                       "by what an analyst would do with it, not what it looks like.")

        if m4:
            btxt = (f' &middot; bias {m4["bias"]:+.2f} vs frozen truth {m4["true_effect"]:.2f}'
                    if "bias" in m4 else " (no frozen truth for this pair)")
            html += (f'<div style="margin-top:12px;"><strong>M4 parameter fidelity:</strong> '
                     f'effect of {exp} under Z&prime; = {m4["estimate"]:.2f} '
                     f'(se {m4["se"]:.2f}){btxt}</div>')
            if "estimate_under_z_true" in m4:
                html += (f'<div style="font-size:0.85rem;color:var(--gray-600);">reference '
                         f'under the true minimal set: {m4["estimate_under_z_true"]:.2f} '
                         f'(se {m4["se_under_z_true"]:.2f})</div>')
            html += _teach("M4", "The cash value of M3: estimate the D&rarr;Y effect "
                           "adjusting for Z&prime; (OLS here) and price the structure "
                           "error in outcome units against the frozen do()-truth.")
        else:
            html += '<div style="margin-top:12px;"><strong>M4:</strong> needs loaded data &mdash; skipped</div>'

        if m5["mode"] == "exhaustive":
            extension_scope = (
                f'all {m5["n_extensions"]} unique consistent DAG extensions'
            )
        else:
            extension_scope = (
                f'{m5["n_extensions"]} unique consistent extensions found in '
                f'{m5["n_orientations_evaluated"]} sampled orientations'
            )
        fraction = (
            f'{m5["fraction_valid"]:.0%}'
            if np.isfinite(m5["fraction_valid"])
            else "not estimable"
        )
        html += (f'<div style="margin-top:12px;"><strong>M5 identification honesty:</strong> '
                 f'Z&prime; valid in {m5["n_valid"]} of {extension_scope} &rarr; {fraction}'
                 f' &middot; {m5["n_undirected"]} undirected pair(s) at discovery</div>')
        html += _teach("M5", "Discovery often honestly ends at a CPDAG. Count only "
                       "unique acyclic orientations that preserve the fixed arrows and "
                       "introduce no new unshielded colliders. Full adjustment validity "
                       "also excludes descendants of D. Large orientation spaces use an "
                       "explicitly labelled capped Monte Carlo diagnostic.")
        html += '</div></div>'

        html += ('<div class="info-box" style="margin-top:14px;">The two axes can '
                 'dissociate &mdash; a topologically wrong graph can still be '
                 'sufficient for the target parameter, and a better-looking graph '
                 'can fail sufficiency. Measuring that dissociation is the point. '
                 'See the Guide tab for definitions and interpretation.</div>')
        return ui.HTML(html)

    # -------------------------------------------------------------------------
    # CAUSAL SHAP TAB
    # -------------------------------------------------------------------------

    @reactive.Effect
    @reactive.event(input.dataset_choice, input.selected_vars)
    def _update_shap_selectors():
        df = data_store.get()
        if df.empty:
            return
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Set defaults based on dataset
        choice = input.dataset_choice()
        if choice == "simcausal":
            outcome_default = "Outcome" if "Outcome" in cols else cols[-1]
            treatment_default = "Treatment" if "Treatment" in cols else cols[0]
        else:
            outcome_default = cols[-1]
            treatment_default = cols[0]
        ui.update_select("shap_outcome", choices=cols, selected=outcome_default)
        ui.update_select("shap_treatment", choices=cols, selected=treatment_default)

    @output
    @render.ui
    @reactive.event(input.shap_compute_btn)
    def shap_status():
        if not SHAP_AVAILABLE:
            return ui.HTML('<div class="error-box">Causal SHAP module not available. Check causal_shap.py and dependencies (shap, scikit-learn, matplotlib, scipy).</div>')
        if not SKLEARN_AVAILABLE:
            return ui.HTML('<div class="error-box">scikit-learn not installed. Run: pip install scikit-learn</div>')

        df = data_store.get()
        if df.empty:
            return ui.HTML('<div class="error-box">No data loaded</div>')

        outcome = input.shap_outcome()
        treatment = input.shap_treatment()
        method = input.shap_method()
        model_type = input.shap_model_type()
        n_perms = input.shap_n_perms()
        dag_source = input.shap_dag_source()

        # Get feature columns
        selected = list(input.selected_vars()) if input.selected_vars() else []
        if not selected:
            selected = df.select_dtypes(include=[np.number]).columns.tolist()

        feature_cols = [c for c in selected if c != outcome]
        if not feature_cols:
            return ui.HTML('<div class="error-box">No feature columns selected</div>')

        # Prepare data
        analysis_df = df[selected + ([outcome] if outcome not in selected else [])].dropna()
        X = analysis_df[feature_cols]
        y = analysis_df[outcome]

        is_binary = len(y.unique()) <= 2

        # Fit prediction model
        if model_type == 'gbm':
            ModelClass = GradientBoostingClassifier if is_binary else GradientBoostingRegressor
            model = ModelClass(n_estimators=100, max_depth=4, random_state=42)
        elif model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            ModelClass = RandomForestClassifier if is_binary else RandomForestRegressor
            model = ModelClass(n_estimators=100, max_depth=6, random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression, LinearRegression
            ModelClass = LogisticRegression if is_binary else LinearRegression
            model = ModelClass(max_iter=1000) if is_binary else ModelClass()

        model.fit(X, y)

        # Build DAG
        dag = None
        if dag_source == "discovered" and adj_matrix_store.get() is not None:
            adj = adj_matrix_store.get()
            cols_names = col_names_store.get()
            dag = dag_from_adjacency(adj, cols_names)
        elif dag_source == "ground_truth":
            reference_edges = ground_truth_store.get()
            if reference_edges:
                dag = nx.DiGraph()
                dag.add_nodes_from(feature_cols)
                dag.add_edges_from(
                    (source, target)
                    for source, target in reference_edges
                    if source in feature_cols and target in feature_cols
                )

        if dag is None and method in ("causal", "compare"):
            return ui.HTML('<div class="warning-box">No DAG available. Run causal discovery first or load a reference DAG.</div>')

        result = {"method": method, "outcome": outcome, "treatment": treatment}

        try:
            # Standard SHAP
            if method in ("standard", "compare"):
                std_shap_df = compute_standard_shap(model, analysis_df, feature_cols)
                result["standard_shap"] = std_shap_df
                result["standard_importance"] = mean_abs_shap(std_shap_df).to_dict()

            # Causal SHAP
            if method in ("causal", "compare") and dag is not None:
                causal_shap_df = compute_causal_shap_fast(
                    model, analysis_df, dag, feature_cols, outcome, n_perms=n_perms
                )
                result["causal_shap"] = causal_shap_df
                result["causal_importance"] = mean_abs_shap(causal_shap_df).to_dict()

            # Adjustment-set SHAP
            if method in ("adjustment", "compare") and dag is not None:
                # Use the named graph directly; the earlier integer-adjacency
                # detour could mix variable names with matrix indices.
                mas = evaluation.minimal_adjustment_set(dag, treatment, outcome)
                adj_vars = ([v for v in mas["set"] if v in feature_cols]
                            if mas["valid"] else [])

                if mas["valid"]:
                    adj_result = compute_adjustment_set_shap(
                        analysis_df, outcome, treatment, adj_vars,
                        feature_cols, model_class=model_type
                    )
                    result["adj_shap"] = adj_result["adj_shap"]
                    result["adj_importance"] = mean_abs_shap(adj_result["adj_shap"]).to_dict()
                    result["adj_vars"] = adj_vars
                    if not adj_vars:
                        result["adj_note"] = (
                            "Valid empty adjustment set: the restricted model "
                            "includes the treatment only."
                        )
                else:
                    result["adj_note"] = "No valid adjustment set identified from DAG"

            # Comparison metrics
            if "standard_importance" in result and "causal_importance" in result:
                # Check if simcausal for ground truth comparison
                true_effects = None
                if input.dataset_choice() == "simcausal":
                    true_effects = SIMCAUSAL_TRUE_TOTAL_EFFECTS
                std_df = result.get("standard_shap", pd.DataFrame())
                csl_df = result.get("causal_shap", pd.DataFrame())
                if not std_df.empty and not csl_df.empty:
                    comparison = compare_shap_rankings(std_df, csl_df, true_effects)
                    result["comparison"] = comparison

            # Mediator inflation (simcausal only)
            if input.dataset_choice() == "simcausal" and "standard_shap" in result and "causal_shap" in result:
                inflation = mediator_inflation_ratio(
                    result["standard_shap"], result["causal_shap"],
                    mediator_vars=["Inflammation", "Oxygenation"],
                    root_cause_vars=["Treatment", "Comorbidity", "Age"]
                )
                result["mediator_inflation"] = inflation

            shap_result_store.set(result)
            return ui.HTML('<div class="success-box">SHAP computation complete.</div>')

        except Exception as e:
            return ui.HTML(
                f'<div class="error-box">Error: {html_escape(str(e))}</div>'
            )

    @output
    @render.ui
    def shap_comparison_plot():
        result = shap_result_store.get()
        if result is None:
            return ui.HTML('<div style="padding:40px;text-align:center;color:var(--gray-400)">Run SHAP computation to see results</div>')

        html_parts = []

        # If we have both standard and causal, show comparison
        if "standard_importance" in result and "causal_importance" in result:
            img_b64 = comparison_bar_plot_to_base64(
                result["standard_importance"], result["causal_importance"],
                title="Standard SHAP vs Causal SHAP"
            )
            if img_b64:
                html_parts.append(f'<img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:8px">')
        elif "standard_importance" in result:
            img_b64 = shap_bar_plot_to_base64(result["standard_importance"], title="Standard SHAP")
            if img_b64:
                html_parts.append(f'<img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:8px">')
        elif "causal_importance" in result:
            img_b64 = shap_bar_plot_to_base64(result["causal_importance"], title="Causal SHAP", color='#2563eb')
            if img_b64:
                html_parts.append(f'<img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:8px">')

        # Adjustment-set SHAP plot if available
        if "adj_importance" in result:
            img_b64 = shap_bar_plot_to_base64(result["adj_importance"], title="Adjustment-Set SHAP", color='#10b981')
            if img_b64:
                html_parts.append(f'<div style="margin-top:16px"><img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:8px"></div>')
            if "adj_vars" in result:
                adjustment_label = (
                    safe_join(result["adj_vars"])
                    if result["adj_vars"]
                    else "<em>None (treatment only)</em>"
                )
                html_parts.append(
                    '<div class="info-box" style="margin-top:8px">'
                    '<strong>Adjustment set:</strong> '
                    f'{adjustment_label}</div>'
                )

        if "adj_note" in result:
            html_parts.append(f'<div class="warning-box" style="margin-top:8px">{result["adj_note"]}</div>')

        if not html_parts:
            return ui.HTML('<div class="warning-box">No plots generated</div>')

        return ui.HTML("".join(html_parts))

    @output
    @render.ui
    def shap_rank_table():
        result = shap_result_store.get()
        if result is None or "comparison" not in result:
            return ui.HTML('')

        comparison = result["comparison"]
        if "rank_changes" in comparison:
            return ui.HTML(rank_change_table_html(comparison["rank_changes"]))
        return ui.HTML('')

    @output
    @render.ui
    def shap_metrics():
        result = shap_result_store.get()
        if result is None:
            return ui.HTML('')

        html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px">'

        comparison = result.get("comparison", {})

        if "kendall_tau" in comparison:
            tau = comparison["kendall_tau"]
            html += f'''<div class="stat-card">
                <div class="stat-label">Kendall's τ (Std vs Causal)</div>
                <div class="stat-value">{tau:.3f}</div>
                <div style="font-size:0.8rem;color:var(--gray-500)">{"High agreement" if abs(tau) > 0.7 else "Rankings differ substantially"}</div>
            </div>'''

        if "tau_vs_truth_standard" in comparison:
            html += f'''<div class="stat-card">
                <div class="stat-label">τ vs Truth (Standard)</div>
                <div class="stat-value" style="color:var(--danger)">{comparison["tau_vs_truth_standard"]:.3f}</div>
            </div>'''

        if "tau_vs_truth_causal" in comparison:
            html += f'''<div class="stat-card">
                <div class="stat-label">τ vs Truth (Causal)</div>
                <div class="stat-value" style="color:var(--success)">{comparison["tau_vs_truth_causal"]:.3f}</div>
            </div>'''

        inflation = result.get("mediator_inflation", {})
        if "mediator_inflation" in inflation:
            html += f'''<div class="stat-card">
                <div class="stat-label">Mediator Inflation Ratio</div>
                <div class="stat-value">{inflation["mediator_inflation"]:.2f}×</div>
                <div style="font-size:0.8rem;color:var(--gray-500)">Standard SHAP inflates mediators by this factor</div>
            </div>'''

        if "root_cause_boost" in inflation:
            html += f'''<div class="stat-card">
                <div class="stat-label">Root Cause Boost</div>
                <div class="stat-value" style="color:var(--success)">{inflation["root_cause_boost"]:.2f}×</div>
                <div style="font-size:0.8rem;color:var(--gray-500)">Causal SHAP boosts true causes by this factor</div>
            </div>'''

        html += '</div>'

        # Add stat-card CSS if not already present
        css = '''<style>
            .stat-card { background:var(--white); border:1px solid var(--gray-200); border-radius:8px; padding:16px; text-align:center; }
            .stat-label { font-size:0.8rem; color:var(--gray-500); margin-bottom:4px; }
            .stat-value { font-size:1.5rem; font-weight:700; color:var(--gray-900); }
        </style>'''

        return ui.HTML(css + html)


app = App(app_ui, server)
