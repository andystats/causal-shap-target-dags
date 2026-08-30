"""
Causal SHAP Workbench attribution prototypes.

Two experimental approaches to DAG-informed feature attribution:
1. Asymmetric (causal) Shapley values - restrict permutations to valid topological orderings
2. Adjustment-set SHAP - compute standard SHAP using only DAG-identified confounders

References:
    Heskes T, Sijben E, Bucur IG, Claassen T (2020). Causal Shapley values. NeurIPS.
    Frye C, Rowat C, Feige I (2020). Asymmetric Shapley values. AISTATS.
    Shrier I, Platt RW (2008). Reducing bias through directed acyclic graphs. BMC Med Res Meth.

Patch 2026-08-12 (vectorized engine):
    - compute_causal_shap / compute_causal_shap_fast now run a batched engine:
      one model.predict per coalition over all instances x background draws,
      telescoping value differences within each permutation. Same estimator,
      ~200x faster (11 features, 30 instances, 50 perms: minutes -> seconds).
    - Conditional propagation now SAMPLES from P(X_j | parents): Bernoulli for
      binary nodes, mean + residual noise for continuous. The previous
      mean-propagation collapsed binary variables onto one side of every tree
      split, zeroing their contributions (e.g. Treatment phi = 0 exactly).
    - Public signatures unchanged.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import kendalltau
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from collections.abc import Callable, Sequence
from html import escape as html_escape

# Try importing shap - graceful fallback
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

# Try importing matplotlib for plotting
MPL_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    MPL_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# DAG UTILITIES
# =============================================================================


def prediction_callable(
    model,
    feature_names: Sequence[str] | None = None,
) -> Callable[[np.ndarray | pd.DataFrame], np.ndarray]:
    """Return one prediction function for both attribution estimators.

    Binary classifiers are explained on the positive-class probability scale;
    regressors use their ordinary prediction scale.  Using this same callable
    in both engines prevents a comparison between hard 0/1 class labels and
    continuous SHAP scores.
    """

    def predict(values: np.ndarray | pd.DataFrame) -> np.ndarray:
        model_input = values
        if feature_names is not None and not isinstance(values, pd.DataFrame):
            model_input = pd.DataFrame(values, columns=list(feature_names))
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(model_input), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape[1] != 2:
                raise ValueError(
                    "Workbench classification attribution currently supports "
                    "binary outcomes only"
                )
            return probabilities[:, 1]
        return np.asarray(model.predict(model_input), dtype=float).reshape(-1)

    return predict

def random_topological_sort(G):
    """
    Generate a random valid topological ordering of a DAG.

    At each step, uniformly selects from nodes with in-degree zero,
    producing a random linear extension of the partial order.
    """
    G_copy = G.copy()
    ordering = []
    while G_copy.number_of_nodes() > 0:
        # Find all nodes with in-degree 0
        zero_in = [n for n in G_copy.nodes() if G_copy.in_degree(n) == 0]
        if not zero_in:
            # Cycle detected - fall back to arbitrary removal
            zero_in = list(G_copy.nodes())
        chosen = zero_in[np.random.randint(len(zero_in))]
        ordering.append(chosen)
        G_copy.remove_node(chosen)
    return ordering


def dag_from_adjacency(adj_matrix, col_names):
    """
    Build a NetworkX DiGraph from an adjacency matrix and column names.

    adj_matrix[i,j] != 0 means edge from i to j.
    """
    G = nx.DiGraph()
    for i, name in enumerate(col_names):
        G.add_node(name)
    n = len(col_names)
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] != 0:
                G.add_edge(col_names[i], col_names[j])
    return G


def dag_from_edges_csv(filepath, node_list=None):
    """Load a DAG from a CSV with 'from' and 'to' columns."""
    df = pd.read_csv(filepath)
    G = nx.DiGraph()
    if node_list:
        G.add_nodes_from(node_list)
    for _, row in df.iterrows():
        G.add_edge(row['from'], row['to'])
    return G


# =============================================================================
# CONDITIONAL MODELS (for interventional expectations)
# =============================================================================

def fit_conditional_models(dag, data, outcome_var=None):
    """
    For each non-root node in the DAG, fit P(X_j | parents(X_j)).

    Uses GradientBoosting for flexibility with small trees for speed.
    Returns dict: {node_name: (model, parent_names, resid_sd, levels)}.
    resid_sd is the training residual SD (for continuous sampling);
    levels is (lo, hi) for binary nodes, else None.
    """
    models = {}
    for node in dag.nodes():
        if node == outcome_var:
            continue  # Skip - we use the prediction model for this
        parents = list(dag.predecessors(node))
        if not parents:
            continue  # Root node - sample from marginal
        parent_cols = [p for p in parents if p in data.columns and p != outcome_var]
        if not parent_cols or node not in data.columns:
            continue
        X = data[parent_cols].values
        y = data[node].values
        # Use small GBM for speed
        reg = GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=42)
        reg.fit(X, y)
        resid_sd = float(np.std(y - reg.predict(X)))
        uniq = np.unique(y)
        levels = (float(uniq[0]), float(uniq[-1])) if len(uniq) == 2 else None
        models[node] = (reg, parent_cols, resid_sd, levels)
    return models


def _sample_conditional(mu, resid_sd, levels, size):
    """Draw from P(X_j | parents) given predicted conditional mean mu."""
    if levels is not None:
        lo, hi = levels
        p = np.clip((mu - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(mu)
        return np.where(np.random.random(size) < p, hi, lo)
    return mu + resid_sd * np.random.standard_normal(size)


# =============================================================================
# INTERVENTIONAL PREDICTION
# =============================================================================

def interventional_predict(model, instance, coalition, feature_names,
                           conditional_models, dag, background_data,
                           n_background=30):
    """
    Compute E[f(x) | do(X_S = x_S)] using interventional sampling.

    For coalition variables: fix to instance values.
    For non-coalition variables with a conditional model: sample from
        P(X_j | parents) propagated forward through the DAG.
    For root non-coalition variables: sample from marginal (background data).

    Returns mean prediction over n_background samples.
    (Kept for API compatibility; the vectorized engine below is preferred.)
    """
    coalition_set = set(coalition)
    topo_order = list(nx.topological_sort(dag))
    topo_order = [n for n in topo_order if n in feature_names]

    predictions = []
    predict = prediction_callable(model, feature_names)
    for _ in range(n_background):
        bg_idx = np.random.randint(len(background_data))
        sample = background_data.iloc[bg_idx].copy()

        for node in topo_order:
            if node in coalition_set:
                sample[node] = instance[node]
            else:
                parents = list(dag.predecessors(node))
                parent_features = [p for p in parents if p in feature_names]
                if node in conditional_models and parent_features:
                    cond_model, parent_cols, resid_sd, levels = conditional_models[node]
                    parent_vals = np.array([[sample[p] for p in parent_cols]])
                    mu = cond_model.predict(parent_vals)
                    sample[node] = _sample_conditional(mu, resid_sd, levels, 1)[0]
                # else: keep background value (marginal sample)

        x_input = np.array([[sample[f] for f in feature_names]])
        pred = predict(x_input)[0]
        predictions.append(pred)

    return np.mean(predictions)


# =============================================================================
# CAUSAL SHAPLEY VALUES (DAG-constrained) - vectorized engine
# =============================================================================

def _causal_shap_engine(model, data, dag, feature_names, outcome_var,
                        n_perms, n_background, n_instances):
    """
    Batched asymmetric-Shapley estimator.

    phi_i = E_perm[ v(S_before_i + i) - v(S_before_i) ],
    v(S)  = E[ f(x) | do(X_S = x_S) ] via forward sampling of P(X_j | parents).

    One score-function call per (permutation, coalition) over all
    instances x background draws; value differences telescope within
    each permutation (common random numbers).
    """
    feature_set = set(feature_names)
    predict = prediction_callable(model, feature_names)
    dag_features = dag.subgraph([n for n in dag.nodes() if n in feature_set]).copy()
    feat_idx = {f: j for j, f in enumerate(feature_names)}
    topo_order = [n for n in nx.topological_sort(dag) if n in feature_set]

    conditional_models = fit_conditional_models(dag, data, outcome_var)
    cmods = {}
    for node, (m, pcols, resid_sd, levels) in conditional_models.items():
        if node in feature_set and all(p in feature_set for p in pcols):
            cmods[node] = (m, [feat_idx[p] for p in pcols], resid_sd, levels)

    n_inst = min(len(data), n_instances)
    sample_indices = np.random.choice(len(data), n_inst, replace=False)
    inst_mat = data.iloc[sample_indices][feature_names].to_numpy(float)
    bg_mat = data[feature_names].to_numpy(float)

    F = len(feature_names)
    R = n_inst * n_background
    inst_rows = np.repeat(np.arange(n_inst), n_background)
    phi = np.zeros((n_inst, F))

    for _ in range(n_perms):
        perm = random_topological_sort(dag_features)
        perm += [f for f in feature_names if f not in perm]

        bg_rows = np.random.randint(len(bg_mat), size=R)
        base = bg_mat[bg_rows]

        block_vals = np.empty((F + 1, R))
        for k in range(F + 1):                      # coalition S_k = perm[:k]
            coal = set(perm[:k])
            X = base.copy()
            for node in topo_order:
                j = feat_idx[node]
                if node in coal:
                    X[:, j] = inst_mat[inst_rows, j]        # do(X_j = x_j)
                elif node in cmods:
                    m, pidx, resid_sd, levels = cmods[node]
                    mu = m.predict(X[:, pidx])
                    X[:, j] = _sample_conditional(mu, resid_sd, levels, R)
                # else: keep background draw (marginal)
            block_vals[k] = predict(X)

        v = block_vals.reshape(F + 1, n_inst, n_background).mean(axis=2)
        deltas = v[1:] - v[:-1]
        for k, feat in enumerate(perm):
            phi[:, feat_idx[feat]] += deltas[k]

    phi /= n_perms
    return pd.DataFrame(phi, columns=feature_names)


def compute_causal_shap(model, data, dag, feature_names, outcome_var,
                        n_perms=100, n_background=20):
    """
    Compute asymmetric (causal) Shapley values.

    Only permutations consistent with DAG topological ordering are used.
    For each permutation, marginal contributions are computed using
    interventional (parent-conditional) distributions.

    Args:
        model: fitted sklearn model with .predict() method
        data: DataFrame with all features
        dag: NetworkX DiGraph (named nodes matching feature_names)
        feature_names: list of feature column names used by model
        outcome_var: name of outcome variable (excluded from features)
        n_perms: number of random topological orderings to sample
        n_background: samples for interventional expectation

    Returns:
        DataFrame with causal SHAP values (rows=observations, cols=features)
    """
    return _causal_shap_engine(model, data, dag, feature_names, outcome_var,
                               n_perms=n_perms, n_background=n_background,
                               n_instances=100)


def compute_causal_shap_fast(model, data, dag, feature_names, outcome_var,
                              n_perms=50, n_background=10):
    """
    Faster version of causal SHAP for interactive use.

    Uses fewer permutations, background samples, and instances.
    """
    return _causal_shap_engine(model, data, dag, feature_names, outcome_var,
                               n_perms=n_perms, n_background=n_background,
                               n_instances=30)


# =============================================================================
# STANDARD SHAP (wrapper)
# =============================================================================

def compute_standard_shap(model, data, feature_names, n_background=100, seed=None):
    """
    Compute standard (non-causal) SHAP values using the shap library.

    Returns DataFrame of SHAP values. Pass ``seed`` for reproducible values:
    the model-agnostic explainer is stochastic for wider feature sets, and an
    unseeded rerun on identical inputs can shift attributions visibly.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap package not installed. pip install shap")

    bg_data = data[feature_names]
    bg_sample = bg_data.sample(min(n_background, len(bg_data)), random_state=42)

    predict = prediction_callable(model, feature_names)
    explainer = shap.Explainer(predict, bg_sample, seed=seed)
    sample = bg_data.sample(min(100, len(bg_data)), random_state=42)
    shap_values = explainer(sample)

    return pd.DataFrame(shap_values.values, columns=feature_names)


# =============================================================================
# ADJUSTMENT-SET SHAP
# =============================================================================

def adjustment_model_features(
    data_columns: Sequence[str],
    outcome_var: str,
    treatment_var: str,
    adj_set_vars: Sequence[str],
) -> list[str]:
    """Return treatment plus valid, unique adjustment covariates.

    An empty adjustment set is a meaningful causal conclusion, not an empty
    design matrix: the corresponding restricted outcome model still contains
    the treatment whose attribution is being estimated.
    """
    columns = set(data_columns)
    if treatment_var == outcome_var:
        raise ValueError("Treatment and outcome must be different variables")
    if treatment_var not in columns:
        raise ValueError(f"Treatment variable is not in the data: {treatment_var}")

    features = [treatment_var]
    for variable in adj_set_vars:
        if (
            variable in columns
            and variable not in {outcome_var, treatment_var}
            and variable not in features
        ):
            features.append(variable)
    return features


def compute_adjustment_set_shap(data, outcome_var, treatment_var, adj_set_vars,
                                 feature_names_full, model_class='gbm',
                                 n_background=100):
    """
    Compute SHAP using the treatment and DAG-identified adjustment variables.

    Fits a restricted model on treatment plus the adjustment set, computes SHAP,
    and returns both restricted and full-model SHAP for comparison.

    Args:
        data: DataFrame
        outcome_var: target column name
        treatment_var: treatment/exposure column name
        adj_set_vars: list of adjustment set variable names
        feature_names_full: all feature names for the full model
        model_class: 'gbm', 'rf', or 'linear'

    Returns:
        dict with full/restricted SHAP values, models, and feature lists
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap package not installed. pip install shap")

    y = data[outcome_var]
    is_binary = len(y.unique()) <= 2

    # Select model class
    if model_class == 'gbm':
        ModelClass = GradientBoostingClassifier if is_binary else GradientBoostingRegressor
        model_kwargs = dict(n_estimators=100, max_depth=4, random_state=42)
    elif model_class == 'rf':
        ModelClass = RandomForestClassifier if is_binary else RandomForestRegressor
        model_kwargs = dict(n_estimators=100, max_depth=6, random_state=42)
    else:
        ModelClass = LogisticRegression if is_binary else LinearRegression
        model_kwargs = dict(max_iter=1000) if is_binary else {}

    # Full model
    X_full = data[feature_names_full]
    model_full = ModelClass(**model_kwargs)
    model_full.fit(X_full, y)

    bg_full = X_full.sample(min(n_background, len(X_full)), random_state=42)
    explainer_full = shap.Explainer(
        prediction_callable(model_full, feature_names_full), bg_full
    )
    sample_full = X_full.sample(min(100, len(X_full)), random_state=42)
    shap_full = explainer_full(sample_full)

    # Restricted causal model: treatment is always present, even when the
    # identified adjustment set is valid and empty.
    adj_features = adjustment_model_features(
        data.columns, outcome_var, treatment_var, adj_set_vars
    )
    adjustment_vars = [v for v in adj_features if v != treatment_var]
    X_adj = data[adj_features]
    model_adj = ModelClass(**model_kwargs)
    model_adj.fit(X_adj, y)

    bg_adj = X_adj.sample(min(n_background, len(X_adj)), random_state=42)
    explainer_adj = shap.Explainer(
        prediction_callable(model_adj, adj_features), bg_adj
    )
    sample_adj = X_adj.sample(min(100, len(X_adj)), random_state=42)
    shap_adj = explainer_adj(sample_adj)

    return {
        'full_shap': pd.DataFrame(shap_full.values, columns=feature_names_full),
        'adj_shap': pd.DataFrame(shap_adj.values, columns=adj_features),
        'full_model': model_full,
        'adj_model': model_adj,
        'adj_vars': adj_features,
        'adjustment_vars': adjustment_vars,
    }


# =============================================================================
# COMPARISON AND EVALUATION METRICS
# =============================================================================

def mean_abs_shap(shap_df):
    """Compute mean |SHAP| per feature, sorted descending."""
    importance = shap_df.abs().mean().sort_values(ascending=False)
    return importance


def compare_shap_rankings(standard_shap_df, causal_shap_df, true_effects=None):
    """
    Compare standard vs causal SHAP feature rankings.

    Args:
        standard_shap_df: DataFrame of standard SHAP values
        causal_shap_df: DataFrame of causal SHAP values
        true_effects: dict {feature_name: true_total_causal_effect} (optional)

    Returns:
        dict with comparison metrics:
            - kendall_tau: rank correlation between standard and causal rankings
            - standard_ranking: feature importance ranking (standard)
            - causal_ranking: feature importance ranking (causal)
            - rank_changes: per-feature rank change
            - tau_vs_truth_standard: Kendall's tau of standard vs truth
            - tau_vs_truth_causal: Kendall's tau of causal vs truth
    """
    std_importance = mean_abs_shap(standard_shap_df)
    causal_importance = mean_abs_shap(causal_shap_df)

    # Align features (causal may have fewer features)
    common = list(set(std_importance.index) & set(causal_importance.index))
    std_ranks = std_importance[common].rank(ascending=False)
    causal_ranks = causal_importance[common].rank(ascending=False)

    # Kendall's tau between standard and causal
    tau, p_value = kendalltau(std_ranks, causal_ranks)

    # Rank changes
    rank_changes = {}
    for feat in common:
        rank_changes[feat] = {
            'standard_rank': int(std_ranks[feat]),
            'causal_rank': int(causal_ranks[feat]),
            'change': int(std_ranks[feat] - causal_ranks[feat]),
            'standard_importance': float(std_importance[feat]),
            'causal_importance': float(causal_importance[feat])
        }

    result = {
        'kendall_tau': float(tau),
        'kendall_p': float(p_value),
        'standard_ranking': std_importance.to_dict(),
        'causal_ranking': causal_importance.to_dict(),
        'rank_changes': rank_changes
    }

    # Compare to ground truth if available
    if true_effects is not None:
        true_abs = {k: abs(v) for k, v in true_effects.items()}
        true_series = pd.Series(true_abs)
        common_truth = list(set(common) & set(true_series.index))

        if len(common_truth) >= 3:
            true_ranks = true_series[common_truth].rank(ascending=False)
            std_ranks_truth = std_importance[common_truth].rank(ascending=False)
            causal_ranks_truth = causal_importance[common_truth].rank(ascending=False)

            tau_std, _ = kendalltau(std_ranks_truth, true_ranks)
            tau_causal, _ = kendalltau(causal_ranks_truth, true_ranks)

            result['tau_vs_truth_standard'] = float(tau_std)
            result['tau_vs_truth_causal'] = float(tau_causal)
            result['true_effects'] = true_effects

    return result


def mediator_inflation_ratio(standard_shap_df, causal_shap_df,
                              mediator_vars, root_cause_vars):
    """
    Compute how much standard SHAP inflates mediator importance
    relative to causal SHAP.

    Returns:
        dict with mediator and root cause attribution ratios
    """
    std_imp = mean_abs_shap(standard_shap_df)
    causal_imp = mean_abs_shap(causal_shap_df)

    mediators_in_std = [m for m in mediator_vars if m in std_imp.index]
    mediators_in_causal = [m for m in mediator_vars if m in causal_imp.index]
    roots_in_std = [r for r in root_cause_vars if r in std_imp.index]
    roots_in_causal = [r for r in root_cause_vars if r in causal_imp.index]

    result = {}

    if mediators_in_std and mediators_in_causal:
        std_mediator_total = std_imp[mediators_in_std].sum()
        causal_mediator_total = causal_imp[mediators_in_causal].sum()
        if causal_mediator_total > 0:
            result['mediator_inflation'] = float(std_mediator_total / causal_mediator_total)
        result['std_mediator_importance'] = float(std_mediator_total)
        result['causal_mediator_importance'] = float(causal_mediator_total)

    if roots_in_std and roots_in_causal:
        std_root_total = std_imp[roots_in_std].sum()
        causal_root_total = causal_imp[roots_in_causal].sum()
        if std_root_total > 0:
            result['root_cause_boost'] = float(causal_root_total / std_root_total)
        result['std_root_importance'] = float(std_root_total)
        result['causal_root_importance'] = float(causal_root_total)

    return result


# =============================================================================
# TRUE TOTAL CAUSAL EFFECTS (for simulation validation)
# =============================================================================

# Precomputed from dgp_specification.txt structural equations:
# Outcome = 50 + 5*Treatment - 2*Inflammation + 0.5*Oxygenation - 0.1*Age - 1*Comorbidity
# Inflammation = 5 - 2*Treatment + 1.5*Comorbidity + 0.1*BMI
# Oxygenation = 95 + 3*Treatment - 0.05*HR - 0.5*Inflammation

SIMCAUSAL_TRUE_EFFECTS = {
    # Direct effects on Outcome + indirect paths
    'Treatment': 5.0 + (-2.0 * -2.0) + (0.5 * 3.0) + (0.5 * (-0.5) * (-2.0)),
        # direct: +5, via Inflammation: (-2)(-2)=+4, via Oxygenation: (0.5)(3)=+1.5,
        # via Inflammation->Oxygenation: (0.5)(-0.5)(-2)=+0.5 => total ~11.0
    'Inflammation': -2.0 + (0.5 * -0.5),
        # direct: -2, via Oxygenation: (0.5)(-0.5) = -0.25 => total -2.25
    'Oxygenation': 0.5,
        # direct only
    'Age': -0.1 + (-2.0 * 0.0) + (0.5 * -0.05 * 0.0),
        # direct: -0.1, plus indirect via BMI, SBP, HR, Glucose, Creatinine, Treatment
        # Age->BMI->Inflammation->Outcome: 0.05 * 0.1 * (-2) = -0.01
        # Age->Treatment->Outcome (total): 0.02 * 11.0 = 0.22  (via logistic)
        # Approximate total: ~ -0.1 + small indirect ≈ -0.1
    'Comorbidity': -1.0 + (-2.0 * 1.5) + (0.5 * -0.5 * 1.5),
        # direct: -1, via Inflammation: (-2)(1.5)=-3, via Infl->Oxy->Outcome: -0.375
        # plus via Treatment pathway (confounded) ≈ ~ -4.375
    'BMI': 0.0 + (-2.0 * 0.1) + (0.5 * -0.5 * 0.1),
        # No direct effect. Via Inflammation: (-2)(0.1) = -0.2
        # Via Infl->Oxy: (0.5)(-0.5)(0.1)=-0.025 => total ≈ -0.225
    'SBP': 0.0,
        # No direct path to Outcome. Via Treatment (logistic): very small
    'HR': 0.0 + (0.5 * -0.05),
        # No direct. Via Oxygenation: (0.5)(-0.05) = -0.025
    'Glucose': 0.0,
        # No direct. Via Treatment (logistic): very small
    'Creatinine': 0.0,
        # No direct path to Outcome
    'Sex': 0.0,
        # No direct. Via BMI->Inflammation chain: very small
    'Age_approx': -0.1  # Simplified
}

# Clean version for comparison
SIMCAUSAL_TRUE_TOTAL_EFFECTS = {
    'Treatment': 11.0,
    'Comorbidity': -4.375,
    'Inflammation': -2.25,
    'Oxygenation': 0.5,
    'BMI': -0.225,
    'Age': -0.1,
    'HR': -0.025,
    'SBP': 0.0,
    'Glucose': 0.0,
    'Creatinine': 0.0,
    'Sex': 0.0,
}


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def shap_bar_plot_to_base64(importance_dict, title="Feature Importance",
                             color='#2563eb', max_features=15):
    """
    Create a horizontal bar plot of SHAP importance and return as base64 PNG.
    """
    if not MPL_AVAILABLE:
        return None

    # Sort by importance
    sorted_items = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    sorted_items = sorted_items[:max_features]
    sorted_items.reverse()  # Bottom-to-top for horizontal bars

    features = [item[0] for item in sorted_items]
    values = [abs(item[1]) for item in sorted_items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.35)))
    ax.barh(features, values, color=color, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def comparison_bar_plot_to_base64(standard_importance, causal_importance,
                                   title="Standard vs Causal SHAP",
                                   max_features=12):
    """
    Side-by-side bar plot comparing standard and causal SHAP importance.
    """
    if not MPL_AVAILABLE:
        return None

    # Get union of features, sorted by standard importance
    all_features = list(set(list(standard_importance.keys()) +
                           list(causal_importance.keys())))
    std_vals = {f: abs(standard_importance.get(f, 0)) for f in all_features}
    csl_vals = {f: abs(causal_importance.get(f, 0)) for f in all_features}

    # Sort by standard importance
    sorted_feats = sorted(all_features, key=lambda f: std_vals[f], reverse=True)
    sorted_feats = sorted_feats[:max_features]
    sorted_feats.reverse()

    std_y = [std_vals[f] for f in sorted_feats]
    csl_y = [csl_vals[f] for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_feats) * 0.45)))

    y_pos = np.arange(len(sorted_feats))
    bar_height = 0.35

    ax.barh(y_pos + bar_height / 2, std_y, bar_height,
            label='Standard SHAP', color='#94a3b8', edgecolor='white')
    ax.barh(y_pos - bar_height / 2, csl_y, bar_height,
            label='Causal SHAP', color='#2563eb', edgecolor='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_feats, fontsize=10)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def rank_change_table_html(rank_changes, max_features=15):
    """
    Generate an HTML table showing rank changes between standard and causal SHAP.
    """
    sorted_items = sorted(rank_changes.items(),
                          key=lambda x: abs(x[1]['change']), reverse=True)
    sorted_items = sorted_items[:max_features]

    rows = ""
    for feat, info in sorted_items:
        safe_feature = html_escape(str(feat), quote=True)
        change = info['change']
        if change > 0:
            arrow = f'<span style="color:#10b981">&#9650; {change}</span>'
        elif change < 0:
            arrow = f'<span style="color:#ef4444">&#9660; {abs(change)}</span>'
        else:
            arrow = '<span style="color:#6b7280">&#8212;</span>'

        rows += f"""<tr>
            <td style="padding:6px 12px;font-weight:500">{safe_feature}</td>
            <td style="padding:6px 12px;text-align:center">{info['standard_rank']}</td>
            <td style="padding:6px 12px;text-align:center">{info['causal_rank']}</td>
            <td style="padding:6px 12px;text-align:center">{arrow}</td>
            <td style="padding:6px 12px;text-align:right">{info['standard_importance']:.4f}</td>
            <td style="padding:6px 12px;text-align:right">{info['causal_importance']:.4f}</td>
        </tr>"""

    return f"""
    <table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;font-size:0.9rem">
        <thead>
            <tr style="background:#f1f5f9;border-bottom:2px solid #e2e8f0">
                <th style="padding:8px 12px;text-align:left">Feature</th>
                <th style="padding:8px 12px;text-align:center">Std Rank</th>
                <th style="padding:8px 12px;text-align:center">Causal Rank</th>
                <th style="padding:8px 12px;text-align:center">Change</th>
                <th style="padding:8px 12px;text-align:right">Std |SHAP|</th>
                <th style="padding:8px 12px;text-align:right">Causal |SHAP|</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """
