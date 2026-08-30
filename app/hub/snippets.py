"""Faithful code cards: the library calls each stage actually makes.

Each snippet is generated at launch time from the same values handed to the
worker, so what the card shows cannot drift from what ran. The snippets name
the real tested functions (`causal_shap.*`, `workbench.attribution.*`) rather
than the hub's wrappers, because the audience question they answer is "what
method is this, exactly?" — and the honest answer is the library call.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def _tuple_lines(names: Sequence[str], indent: str = "    ") -> str:
    if len(names) <= 6:
        return repr(tuple(names))
    body = ",\n".join(f"{indent}{name!r}" for name in names)
    return f"(\n{body},\n)"


def naive_snippet(features: Sequence[str], outcome: str, model_type: str,
                  n_background: int, seed: int) -> str:
    return f"""# Stage 2 - naive SHAP benchmark (executed by hub.stages.run_naive_shap)
from hub.stages import fit_outcome_model
from workbench.attribution import compute_standard_shap, mean_abs_shap

features = {_tuple_lines(features)}
fit = fit_outcome_model(data, features, {outcome!r},
                        model_type={model_type!r}, seed={seed})
shap_df = compute_standard_shap(fit.model, data.dropna(subset=list(features)),
                                list(features), n_background={n_background}, seed={seed})
importance = mean_abs_shap(shap_df)   # what the model listened to"""


def discover_snippet(features: Sequence[str], outcome: str, algorithm: str,
                     alpha: float) -> str:
    call = (
        f"run_pc(frame, alpha={alpha})" if algorithm == "pc" else "run_ges(frame)"
    )
    return f"""# Stage 3 - structure discovery (executed by hub.stages.run_discovery)
from causal_shap.discovery import run_pc, run_ges
from causal_shap.graph_state import GraphProvenance, GraphState

columns = {_tuple_lines(list(features) + [outcome])}
frame = data[list(columns)].dropna()
result = {call}                      # returns a CPDAG, not the truth
graph = GraphState.from_pdag(         # ONE deterministic representative;
    result.pdag,                      # undirected pairs travel along for honesty
    GraphProvenance(source="discovered", algorithm=result.algorithm,
                    params=dict(result.params), n_rows=result.n_rows),
)"""


def flags_snippet(flags_id: str, outcome: str, preference: str | None) -> str:
    pref = repr(preference) if preference else "None  # auto-select"
    return f"""# Stage 4 - per-node depth flags (executed by hub.stages.run_flags)
from causal_shap.node_flags import NodeFlagRequest, select_flag_provider

provider = select_flag_provider({pref}, block_root=BLOCK_ROOT)  # env-configured
result = provider.flags(NodeFlagRequest(
    dataset_id={flags_id!r}, outcome={outcome!r}, feature_names=features,
))
# result.status is load-bearing: "unavailable" is an honest answer,
# never an empty table of flag=False. Channels are read separately."""


def surgery_snippet() -> str:
    return """# Stage 5 - graph surgery (executed by hub.theater.apply_surgery per action)
from causal_shap.graph_state import ConstraintEntry

# Flip / Require / Forbid / Remove act on a clicked edge; Add asserts a
# missing one. Every operation rebuilds the edge sets and calls:
revised = graph.with_constraints(
    directed_edges, undirected_pairs,
    ledger=(ConstraintEntry(edge, kind, applied="post_hoc", rationale=...),),
)
# GraphState refuses cycles by construction and the ledger records every
# judgement; source flips "discovered" -> "recovered" on the first edit."""


def shap_snippet(arm: str, features: Sequence[str], outcome: str, model_type: str,
                 n_perms: int, n_background: int, n_instances: int, seed: int,
                 *, graph_source: str = "?", graph_fingerprint: str = "?",
                 n_edges: int = 0, n_ledger: int = 0) -> str:
    surgery_line = (
        f"  {n_ledger} surgical judgement(s) in its ledger" if n_ledger
        else "  no surgeries: the graph is as discovered"
    )
    head = f"""# Stage 6 - causal SHAP, {arm} arm (executed by hub.stages.run_causal_shap)
# INPUT GRAPH: {graph_source}, {n_edges} edges, fingerprint {graph_fingerprint[:12]}
#{surgery_line}
graph = current_graph          # exactly the object the surgery stage produced
attributed = [f for f in features        # the graph GOVERNS eligibility:
              if f in nx.ancestors(graph.digraph(), {outcome!r})]
features = {_tuple_lines(features)}"""
    if arm == "structural":
        return head + f"""
from causal_shap.calibrate import fit_linear_logistic_scm
from causal_shap.structural_value import compute_structural_asymmetric_shap
from workbench.attribution import prediction_callable

fitted = fit_linear_logistic_scm(data[graph_nodes].dropna(), digraph, seed={seed})
result = compute_structural_asymmetric_shap(
    prediction_callable(model, list(features)),   # model refit on THIS feature tuple
    fitted.scm,
    evaluation_rows,                              # {n_instances} rows, seeded sample
    background_exogenous,                         # {n_background} abducted draws
    list(features),
    feature_edges,                                # graph edges induced on features
    n_permutations={n_perms}, seed={seed},
)   # value function: E[f(X) | do(X_S = x_S)] - the same engine as the frozen record"""
    return head + f"""
from workbench.attribution import _causal_shap_engine, fit_conditional_models

conditionals = fit_conditional_models(digraph, data, {outcome!r})
causal_df = _causal_shap_engine(model, data, digraph, list(features), {outcome!r},
                                n_perms={n_perms}, n_background={n_background},
                                n_instances={n_instances})   # seeded globals, see stages.py"""


def policy_snippet(outcome: str, budget: float, direction: str, alpha: float,
                   seed: int, *, arm: str = "scm", learner: str = "gbm") -> str:
    head = f"""# Stage 7 - price and dice, {arm} arm (executed by hub.stages.run_policy)
from causal_shap.calibrate import fit_linear_logistic_scm
from causal_shap.policy import InterventionProblem, abduct, rank_actions
from causal_shap.action_costs import CostModel

fitted = fit_linear_logistic_scm(data[graph_nodes].dropna(), digraph, seed={seed})
exogenous = abduct(fitted.scm, data[graph_nodes].dropna(), seed={seed})
ranking = rank_actions(
    InterventionProblem(
        scm=fitted.scm, outcome={outcome!r},
        cost_model=CostModel(specs=cost_sheet, budget={budget}),
        graph=digraph, direction={direction!r}, alpha={alpha},
    ),
    exogenous, seed={seed},
)   # benefit = paired do() contrast on shared exogenous draws;
    # every screened node carries the rule that refused it"""
    if arm != "semiparametric":
        return head
    return head + f"""

from causal_shap.shift_estimation import estimate_shift_effect

for action in ranking.feasible():        # Marschak's Maxim: the SCM surveys,
    (lever, delta), = action.action.items()   # one functional per survivor
    estimate = estimate_shift_effect(
        data[graph_nodes].dropna(), digraph, lever, delta, {outcome!r},
        direction={direction!r}, learner={learner!r}, seed={seed},
    )   # cross-fitted AIPW for the modified treatment policy d(a) = a + delta,
        # adjustment set = the lever's parents under the current graph;
        # feasibility (support of A + delta, weight diagnostics) rides along"""
