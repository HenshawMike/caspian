"""Phase 15: Causal Discovery and Intervention Learning (experiments/phase15_experiments.py).
Registered in docs/phase15/EXPERIMENT_REGISTRY.md (CSP-P15-E001).
"""
import concurrent.futures
import copy
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

import experiments.phase7_experiments as p7
from experiments.phase11_experiments import (
    LR,
    STEP_OFFSET,
    bootstrap_ci,
    interaction_rows,
    paired_summary,
    rmse,
    train_model,
)
from models.causal import CausalStructuralModel
from models.mlp import PredictiveMLP

SEEDS15 = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
EXPLORATORY15 = [9001, 9002, 9003]
N_VARS15 = 4
EPOCHS15 = 120
BATCH_SIZE15 = 64
STEPS15 = (800, 2400)
EDGE_DETECT_ATE_THRESHOLD = 0.75

ARCHS15 = [
    "causal_intervention",
    "mlp_intervention",
    "random_intervention",
    "observational_mlp",
    "observational_linear",
    "oracle_causal",
]


@dataclass
class CausalWorldSpec:
    name: str
    true_parents: Set[int]
    true_weights: Tuple[float, ...]
    value_range: float
    n_vars: int = N_VARS15

    def true_outcome(self, X: np.ndarray, U: Optional[np.ndarray] = None) -> np.ndarray:
        """Ground-truth structural causal equation E[Y | do(X), U] (analysis-only)."""
        X = np.asarray(X, dtype=float)
        w = np.array(self.true_weights, dtype=float)
        y = X @ w + STEP_OFFSET
        if U is not None and self.name == "W15_hidden_confounder":
            y = y + 3.5 * np.asarray(U, dtype=float).reshape(-1)
        return y

    def true_ate(self) -> np.ndarray:
        """True Average Treatment Effect of changing X_k from 0.2 to 0.8 (delta = 0.6 * w_k)."""
        return 0.6 * np.array(self.true_weights, dtype=float)

    def sample_observational(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        U = rng.normal(0.0, 1.0, size=n)
        X = np.zeros((n, self.n_vars), dtype=float)
        if self.name == "W15_corr_vs_cause":
            x1 = rng.choice([0.2, 0.8], size=n)
            x0 = x1 + rng.normal(0.0, 0.05, size=n)
            x2 = rng.choice([0.2, 0.8], size=n)
            x3 = rng.choice([0.2, 0.8], size=n)
            X = np.column_stack([x0, x1, x2, x3])
        elif self.name == "W15_obs_confounder":
            x0 = rng.choice([0.2, 0.8], size=n)
            x1 = x0 + rng.normal(0.0, 0.06, size=n)
            x2 = rng.choice([0.2, 0.8], size=n)
            x3 = 1.0 - x0 + rng.normal(0.0, 0.06, size=n)
            X = np.column_stack([x0, x1, x2, x3])
        elif self.name == "W15_hidden_confounder":
            x0 = 0.5 + 0.28 * U + rng.normal(0.0, 0.05, size=n)
            x1 = rng.choice([0.2, 0.8], size=n)
            x2 = rng.choice([0.2, 0.8], size=n)
            x3 = rng.choice([0.2, 0.8], size=n)
            X = np.column_stack([x0, x1, x2, x3])
        elif self.name == "W15_multi_cause":
            x0 = rng.choice([0.2, 0.8], size=n)
            x1 = 0.9 * x0 + rng.normal(0.0, 0.07, size=n)
            x2 = rng.choice([0.2, 0.8], size=n)
            x3 = 0.9 * x2 + rng.normal(0.0, 0.07, size=n)
            X = np.column_stack([x0, x1, x2, x3])
        else:
            raise ValueError(self.name)

        eps_y = rng.normal(0.0, 0.25, size=n)
        Y = self.true_outcome(X, U=U) + eps_y
        return X, Y, U

    def sample_interventional(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Controlled active interventions: breaks all incoming arrows to X so variables vary independently."""
        rng = np.random.default_rng(seed + 5000)
        U = rng.normal(0.0, 1.0, size=n)
        X = rng.choice([0.2, 0.8], size=(n, self.n_vars)).astype(float)
        eps_y = rng.normal(0.0, 0.25, size=n)
        Y = self.true_outcome(X, U=U) + eps_y
        return X, Y, U


W15_CORR_VS_CAUSE = CausalWorldSpec(
    name="W15_corr_vs_cause",
    true_parents={1, 2},
    true_weights=(0.0, 6.0, -4.0, 0.0),
    value_range=6.0,
)
W15_OBS_CONFOUNDER = CausalWorldSpec(
    name="W15_obs_confounder",
    true_parents={0, 2},
    true_weights=(6.0, 0.0, 4.0, 0.0),
    value_range=6.0,
)
W15_HIDDEN_CONFOUNDER = CausalWorldSpec(
    name="W15_hidden_confounder",
    true_parents={1, 2},
    true_weights=(0.0, 6.0, -4.0, 0.0),
    value_range=6.0,
)
W15_MULTI_CAUSE = CausalWorldSpec(
    name="W15_multi_cause",
    true_parents={0, 2, 3},
    true_weights=(5.0, 0.0, -3.0, 4.0),
    value_range=7.2,
)

WORLDS15 = (W15_CORR_VS_CAUSE, W15_OBS_CONFOUNDER, W15_HIDDEN_CONFOUNDER, W15_MULTI_CAUSE)
WORLD_MAP_15 = {w.name: w for w in WORLDS15}


def make_p15_world(spec: CausalWorldSpec, X_pool: np.ndarray, Y_pool: np.ndarray):
    """Wrap sampled SCM entity realizations into a deterministic GridWorld config for Semantic Firewall & interaction collection."""
    from environment.entities import Entity
    cfg = copy.deepcopy(p7._load_world("phase7_world_base.json"))
    pool = []
    # Cap pool size at 64 representative entities for fast GridWorld instantiation
    step_idx = np.linspace(0, len(X_pool) - 1, num=min(64, len(X_pool)), dtype=int)
    for k, idx in enumerate(step_idx):
        pool.append({
            "entity_type": 400 + k,
            "hidden_state_delta": float(Y_pool[idx] - STEP_OFFSET),
            "is_interactive": True,
            "is_blocking": False,
            "properties": [float(v) for v in X_pool[idx]],
        })
    first = pool[0]
    cfg.initial_entities = [
        Entity(
            entity_id=1,
            entity_type=first["entity_type"],
            position=(2, 2),
            hidden_state_delta=first["hidden_state_delta"],
            properties=tuple(first["properties"]),
        )
    ]
    cfg.entity_type_pool = pool
    cfg.randomize_entity_positions = True
    cfg.allowed_entity_positions = None
    cfg.initial_agent_pos = (0, 0)
    cfg.initial_internal_state = 0.0
    cfg.min_internal_state = -1000.0
    cfg.max_internal_state = 1000.0
    return cfg


def collect_p15_datasets(spec: CausalWorldSpec, seed: int, steps: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Collect observational, random-perturbation, and controlled interventional datasets.
    Verifies GridWorld Semantic Firewall purity on a deterministic rollout slice.
    """
    n_events = max(120, steps // 5)
    X_obs, Y_obs, U_obs = spec.sample_observational(n_events, seed)
    X_int, Y_int, U_int = spec.sample_interventional(n_events, seed)

    # Random perturbation baseline: observational data + small untargeted Gaussian jitter (0.08)
    rng = np.random.default_rng(seed + 8000)
    X_rand = X_obs + rng.normal(0.0, 0.08, size=X_obs.shape)
    Y_rand = spec.true_outcome(X_rand, U=U_obs) + rng.normal(0.0, 0.25, size=n_events)

    # Verify GridWorld observation purity and transition consistency
    cfg = make_p15_world(spec, X_obs, Y_obs)
    ds = p7._collect_passive(cfg, seed=seed, num_steps=200)
    for t in ds.transitions[:5]:
        t.obs.assert_semantic_purity()

    return {
        "obs": (X_obs, Y_obs, U_obs),
        "int": (X_int, Y_int, U_int),
        "rand": (X_rand, Y_rand, U_obs),
    }


def evaluate_causal_metrics(spec: CausalWorldSpec, model, seed: int, is_oracle: bool = False) -> Dict[str, Any]:
    """Evaluate single-variable interventions, multi-variable interventions, ATE error,
    counterfactual abduction-action-prediction error, and causal parent graph F1."""
    def _pred(X_in: np.ndarray) -> np.ndarray:
        if is_oracle:
            return spec.true_outcome(X_in, U=None)
        return np.asarray(model.forward(X_in)).reshape(-1)

    # 1. All 2^4 = 16 deterministic interventional grid points {0.2, 0.8}^4
    grid = np.array(list(itertools.product((0.2, 0.8), repeat=spec.n_vars)), dtype=float)
    y_true_grid = spec.true_outcome(grid, U=None)
    y_pred_grid = _pred(grid)
    single_int_nrmse = rmse(y_pred_grid, y_true_grid) / spec.value_range

    # 2. Simultaneous multi-variable interventions (do(X_i, X_j) across all pairs with others fixed at 0.5)
    multi_X, multi_Y = [], []
    for i, j in itertools.combinations(range(spec.n_vars), 2):
        for vi, vj in itertools.product((0.2, 0.8), repeat=2):
            row = np.full(spec.n_vars, 0.5, dtype=float)
            row[i] = vi
            row[j] = vj
            multi_X.append(row)
    multi_X_arr = np.array(multi_X, dtype=float)
    multi_Y_true = spec.true_outcome(multi_X_arr, U=None)
    multi_Y_pred = _pred(multi_X_arr)
    multi_int_nrmse = rmse(multi_Y_pred, multi_Y_true) / spec.value_range

    # 3. Average Treatment Effect (ATE_k) estimation & causal parent recovery Pa(Y)
    ate_est = np.zeros(spec.n_vars, dtype=float)
    for k in range(spec.n_vars):
        g_lo = grid[grid[:, k] == 0.2].copy()
        g_hi = g_lo.copy()
        g_hi[:, k] = 0.8
        ate_est[k] = float(np.mean(_pred(g_hi) - _pred(g_lo)))
    ate_true = spec.true_ate()
    ate_nmae = float(np.mean(np.abs(ate_est - ate_true))) / spec.value_range

    if isinstance(model, CausalStructuralModel):
        detected_parents = model.discovered_parents()
    elif is_oracle:
        detected_parents = set(spec.true_parents)
    else:
        detected_parents = {int(k) for k in range(spec.n_vars) if abs(ate_est[k]) >= EDGE_DETECT_ATE_THRESHOLD}

    tp = len(detected_parents & spec.true_parents)
    fp = len(detected_parents - spec.true_parents)
    fn = len(spec.true_parents - detected_parents)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # 4. Counterfactual prediction (Exp 15.6): factual observation (X_f, Y_f) with hidden U -> counterfactual do(X_k <- x')
    X_f, Y_f, U_f = spec.sample_observational(128, seed + 25000)
    rng_cf = np.random.default_rng(seed + 35000)
    X_cf = X_f.copy()
    for idx_row in range(len(X_cf)):
        k_int = int(rng_cf.integers(0, spec.n_vars))
        X_cf[idx_row, k_int] = 0.8 if X_cf[idx_row, k_int] < 0.5 else 0.2
    Y_cf_true = spec.true_outcome(X_cf, U=U_f)
    if is_oracle:
        Y_cf_pred = Y_cf_true
    elif isinstance(model, CausalStructuralModel):
        Y_cf_pred = model.predict_counterfactual(X_f, Y_f, X_cf)
    else:
        Y_cf_pred = _pred(X_cf)
    cf_nrmse = rmse(Y_cf_pred, Y_cf_true) / spec.value_range

    return {
        "single_int_nrmse": round(float(single_int_nrmse), 5),
        "multi_int_nrmse": round(float(multi_int_nrmse), 5),
        "ate_nmae": round(float(ate_nmae), 5),
        "counterfactual_nrmse": round(float(cf_nrmse), 5),
        "causal_precision": round(float(prec), 4),
        "causal_recall": round(float(rec), 4),
        "causal_f1": round(float(f1), 4),
        "false_causal_edges": int(fp),
    }


def run_cell_p15(
    spec: CausalWorldSpec,
    arch: str,
    seed: int,
    steps: int,
    cache: Dict[Any, Any],
    epochs: int = EPOCHS15,
    lr: float = LR,
    batch_size: int = BATCH_SIZE15,
) -> Dict[str, Any]:
    key = (spec.name, seed, steps)
    if key not in cache:
        cache[key] = collect_p15_datasets(spec, seed, steps)
    ds_map = cache[key]

    is_oracle = (arch == "oracle_causal")
    if is_oracle:
        model = None
        n_params = len(spec.true_parents)
        n_train = len(ds_map["int"][0])
        final_loss = 0.0
    else:
        if arch == "causal_intervention":
            Xtr, Ytr, _ = ds_map["int"]
            model = CausalStructuralModel(spec.n_vars, seed=seed)
            model.fit_causal_structure(Xtr, Ytr)
            losses = [float(np.mean((np.asarray(model.forward(Xtr)).reshape(-1) - Ytr) ** 2))]
        elif arch == "mlp_intervention":
            Xtr, Ytr, _ = ds_map["int"]
            model = PredictiveMLP(spec.n_vars, (32, 16), 1, seed=seed)
            _, losses = train_model(model, Xtr, Ytr, seed, epochs=epochs, lr=lr, batch_size=batch_size)
        elif arch == "random_intervention":
            Xtr, Ytr, _ = ds_map["rand"]
            model = PredictiveMLP(spec.n_vars, (32, 16), 1, seed=seed)
            _, losses = train_model(model, Xtr, Ytr, seed, epochs=epochs, lr=lr, batch_size=batch_size)
        elif arch == "observational_mlp":
            Xtr, Ytr, _ = ds_map["obs"]
            model = PredictiveMLP(spec.n_vars, (32, 16), 1, seed=seed)
            _, losses = train_model(model, Xtr, Ytr, seed, epochs=epochs, lr=lr, batch_size=batch_size)
        elif arch == "observational_linear":
            Xtr, Ytr, _ = ds_map["obs"]
            model = CausalStructuralModel(spec.n_vars, seed=seed)
            model.fit_causal_structure(Xtr, Ytr)
            losses = [float(np.mean((np.asarray(model.forward(Xtr)).reshape(-1) - Ytr) ** 2))]
        else:
            raise ValueError(arch)
        n_params = int(model.num_parameters)
        n_train = int(len(Xtr))
        final_loss = round(float(losses[-1]), 5)

    metrics = evaluate_causal_metrics(spec, model, seed, is_oracle=is_oracle)
    return {
        "world": spec.name,
        "arch": arch,
        "seed": seed,
        "steps": steps,
        "n_params": n_params,
        "n_train_events": n_train,
        "leak_rows": 0,
        "finite": bool(all(np.isfinite(v) for v in metrics.values())),
        "final_train_loss": final_loss,
        **metrics,
    }


def _worker_p15_seed(args: Tuple[str, int, Tuple[str, ...], Tuple[str, ...]]):
    wname, s, levels, archs = args
    spec = WORLD_MAP_15[wname]
    cache: Dict[Any, Any] = {}
    out_rows = []
    for lv in levels:
        steps = STEPS15[0 if lv == "lo" else 1]
        for a in archs:
            out_rows.append(run_cell_p15(spec, a, s, steps, cache))
    return out_rows


def run_phase15_suite(
    worlds=WORLDS15,
    archs=ARCHS15,
    seeds=SEEDS15,
    levels=("lo", "hi"),
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    workers = max_workers or min(len(seeds), os.cpu_count() or 1)
    for spec in worlds:
        tasks = [(spec.name, s, tuple(levels), tuple(archs)) for s in seeds]
        if workers > 1 and len(tasks) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                for s, s_rows in zip(seeds, ex.map(_worker_p15_seed, tasks)):
                    rows.extend(s_rows)
                    if verbose:
                        hi_st = STEPS15[1]
                        summary = {r["arch"]: r["single_int_nrmse"] for r in s_rows if r["steps"] == hi_st}
                        print(spec.name, "seed", s, "done (hi):", summary, flush=True)
        else:
            for t in tasks:
                s_rows = _worker_p15_seed(t)
                rows.extend(s_rows)
                if verbose:
                    print(spec.name, "seed", t[1], "done", flush=True)
    return rows
