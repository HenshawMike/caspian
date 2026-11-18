"""Phase 14: Interaction Discovery and Structured Relationships (experiments/phase14_experiments.py).
Registered in docs/phase14/EXPERIMENT_REGISTRY.md (CSP-P14-E001).
"""
import concurrent.futures
import copy
import itertools
import json
import os
from dataclasses import dataclass, field
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
    r2,
    rmse,
    train_model,
)
from models.interactions import SparseInteractionDiscoveryModel, build_p14_model

SEEDS14 = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
EXPLORATORY14 = [9001, 9002, 9003]
N_PROPS14 = 5
EPOCHS14 = 120
BATCH_SIZE14 = 64
STEPS_PER_COMBO_14 = (100, 300)  # 22 train combos -> 2,200 steps (lo) and 6,600 steps (hi)
DETECT_THRESHOLD = 0.45
MAIN_EFFECTS_TRUE = (5.0, -4.0, 3.0, -2.0, 1.5)

ARCHS14 = [
    "sparse_interaction",
    "additive_residual",
    "additive",
    "factorised",
    "mlp_small",
    "mlp_large",
]


def _main_effect(c: Tuple[int, ...]) -> float:
    return (
        5.0 * c[0]
        - 4.0 * c[1]
        + 3.0 * c[2]
        - 2.0 * c[3]
        + 1.5 * c[4]
    )


def _delta_indep(c: Tuple[int, ...]) -> float:
    return _main_effect(c)


def _delta_sparse_pair(c: Tuple[int, ...]) -> float:
    return _main_effect(c) + 6.0 * c[0] * c[1]


def _delta_multi_pair(c: Tuple[int, ...]) -> float:
    return _main_effect(c) + 6.0 * c[0] * c[1] - 5.0 * c[1] * c[2]


def _delta_higher_order(c: Tuple[int, ...]) -> float:
    return _main_effect(c) + 4.0 * c[0] * c[1] + 6.0 * c[0] * c[1] * c[2]


ALL_COMBOS_14: List[Tuple[int, ...]] = list(itertools.product((0, 1), repeat=N_PROPS14))
HELDOUT_COMBOS_14: List[Tuple[int, ...]] = [
    c for c in ALL_COMBOS_14 if (c[0] == 1 and c[1] == 1 and c[3] == 1) or (c[3] == 1 and c[4] == 1)
]
TRAIN_COMBOS_14: List[Tuple[int, ...]] = [c for c in ALL_COMBOS_14 if c not in HELDOUT_COMBOS_14]


@dataclass
class InteractionWorldSpec:
    name: str
    delta_fn: Callable[[Tuple[int, ...]], float]
    true_interactions: Set[Tuple[int, ...]]
    value_range: float
    obs_noise_std: float = 0.0
    out_noise_std: float = 0.0
    n_props: int = N_PROPS14
    train_combos: List[Tuple[int, ...]] = field(default_factory=lambda: list(TRAIN_COMBOS_14))
    heldout_combos: List[Tuple[int, ...]] = field(default_factory=lambda: list(HELDOUT_COMBOS_14))
    steps_per_combo: Tuple[int, int] = STEPS_PER_COMBO_14

    def all_combos(self) -> List[Tuple[int, ...]]:
        return list(ALL_COMBOS_14)

    def combo_to_channel_vec(self, combo: Tuple[int, ...]) -> Tuple[float, ...]:
        return tuple(0.2 if b == 0 else 0.8 for b in combo)

    def decode(self, P: np.ndarray) -> np.ndarray:
        """Analysis-only: map continuous channel vectors P (N, 5) back to binary tuples {0, 1}^5."""
        return (np.asarray(P, dtype=float) >= 0.5).astype(int)

    def truth(self, P: np.ndarray) -> np.ndarray:
        L = self.decode(P)
        return np.array([self.delta_fn(tuple(int(v) for v in row)) + STEP_OFFSET for row in L], dtype=float)


W14_INDEP = InteractionWorldSpec(
    name="W14_indep",
    delta_fn=_delta_indep,
    true_interactions=set(),
    value_range=15.5,
)

W14_SPARSE_PAIR = InteractionWorldSpec(
    name="W14_sparse_pair",
    delta_fn=_delta_sparse_pair,
    true_interactions={(0, 1)},
    value_range=21.5,
)

W14_MULTI_PAIR = InteractionWorldSpec(
    name="W14_multi_pair",
    delta_fn=_delta_multi_pair,
    true_interactions={(0, 1), (1, 2)},
    value_range=21.5,
)

W14_HIGHER_ORDER = InteractionWorldSpec(
    name="W14_higher_order",
    delta_fn=_delta_higher_order,
    true_interactions={(0, 1), (0, 1, 2)},
    value_range=25.5,
)

W14_NOISY_PAIR = InteractionWorldSpec(
    name="W14_noisy_pair",
    delta_fn=_delta_sparse_pair,
    true_interactions={(0, 1)},
    value_range=21.5,
    obs_noise_std=0.03,
    out_noise_std=0.45,
)

WORLDS14 = (W14_INDEP, W14_SPARSE_PAIR, W14_MULTI_PAIR, W14_HIGHER_ORDER, W14_NOISY_PAIR)
WORLD_MAP_14 = {w.name: w for w in WORLDS14}


def make_p14_world(spec: InteractionWorldSpec, combos: Sequence[Tuple[int, ...]]):
    from environment.entities import Entity
    cfg = copy.deepcopy(p7._load_world("phase7_world_base.json"))
    pool = []
    for k, c in enumerate(combos):
        pool.append({
            "entity_type": 300 + k,
            "hidden_state_delta": spec.delta_fn(c),
            "is_interactive": True,
            "is_blocking": False,
            "properties": list(spec.combo_to_channel_vec(c)),
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


def collect_train_p14(spec: InteractionWorldSpec, seed: int, steps: int):
    cfg = make_p14_world(spec, spec.train_combos)
    ds = p7._collect_passive(cfg, seed=seed, num_steps=steps)
    P, Y = interaction_rows(ds, spec.n_props)
    if spec.obs_noise_std > 0.0 or spec.out_noise_std > 0.0:
        rng = np.random.default_rng(seed + 14000)
        if spec.obs_noise_std > 0.0:
            P = P + rng.normal(0.0, spec.obs_noise_std, size=P.shape)
        if spec.out_noise_std > 0.0:
            Y = Y + rng.normal(0.0, spec.out_noise_std, size=Y.shape)
    return ds, P, Y


def collect_eval_p14(spec: InteractionWorldSpec, which: str) -> np.ndarray:
    combos = spec.heldout_combos if which == "heldout" else spec.train_combos
    rows = [spec.combo_to_channel_vec(c) for c in combos]
    base_grid = np.array(rows, dtype=float)
    reps = max(1, 96 // len(base_grid))
    return np.tile(base_grid, (reps, 1))


def probe_model_interactions_and_main_effects(
    spec: InteractionWorldSpec,
    model,
    threshold: float = DETECT_THRESHOLD,
) -> Dict[str, Any]:
    """Model-agnostic discrete mixed partial derivative (Mobius) probe across all contexts on {0.2, 0.8}^5:
    - Evaluates single-property main-effect recovery across unseen/held-out contexts and baseline origin.
    - Evaluates all 10 3-way interactions across background contexts.
    - Evaluates all 10 pairwise interactions across background contexts (excluding contexts where an active 3-way term fires).
    """
    all_c = spec.all_combos()
    X_all = np.array([spec.combo_to_channel_vec(c) for c in all_c], dtype=float)
    preds = np.asarray(model.forward(X_all)).reshape(-1)
    pred_map = {c: float(v) for c, v in zip(all_c, preds)}

    # 1. Main-effect recovery across baseline + held-out context flips of additive properties
    c0 = (0, 0, 0, 0, 0)
    f0 = pred_map[c0]
    main_errs = []
    for i in range(spec.n_props):
        ci = [0] * spec.n_props
        ci[i] = 1
        main_errs.append(abs((pred_map[tuple(ci)] - f0) - MAIN_EFFECTS_TRUE[i]))
    # Also evaluate single-property flips of non-interacting properties (p3, p4) across all contexts touching held-out
    held_set = set(spec.heldout_combos)
    for c in all_c:
        for i in (3, 4):
            if c[i] == 0:
                c2 = list(c)
                c2[i] = 1
                c2_t = tuple(c2)
                if c in held_set or c2_t in held_set:
                    est_flip = pred_map[c2_t] - pred_map[c]
                    main_errs.append(abs(est_flip - MAIN_EFFECTS_TRUE[i]))
    main_effect_nmae = float(np.mean(main_errs)) / spec.value_range

    # 2. 3-way discrete mixed partial derivatives across background contexts
    detected: Set[Tuple[int, ...]] = set()
    for i, j, k in itertools.combinations(range(spec.n_props), 3):
        others = [m for m in range(spec.n_props) if m not in (i, j, k)]
        max_m3 = 0.0
        for bg in itertools.product((0, 1), repeat=len(others)):
            val = 0.0
            for a, b, cc in itertools.product((0, 1), repeat=3):
                c = [0] * spec.n_props
                c[i], c[j], c[k] = a, b, cc
                for om, ov in zip(others, bg):
                    c[om] = ov
                sign = -1.0 if ((3 - (a + b + cc)) % 2 == 1) else 1.0
                val += sign * pred_map[tuple(c)]
            if abs(val) > max_m3:
                max_m3 = abs(val)
        if max_m3 >= threshold:
            detected.add((i, j, k))

    # 3. Pairwise discrete mixed partial derivatives across contexts
    for i, j in itertools.combinations(range(spec.n_props), 2):
        others = [m for m in range(spec.n_props) if m not in (i, j)]
        max_m2 = 0.0
        for bg in itertools.product((0, 1), repeat=len(others)):
            if any(tuple(sorted((i, j, om))) in detected and ov == 1 for om, ov in zip(others, bg)):
                continue
            c_base = [0] * spec.n_props
            for om, ov in zip(others, bg):
                c_base[om] = ov
            val = 0.0
            for a, b in itertools.product((0, 1), repeat=2):
                c = list(c_base)
                c[i], c[j] = a, b
                sign = -1.0 if ((2 - (a + b)) % 2 == 1) else 1.0
                val += sign * pred_map[tuple(c)]
            if abs(val) > max_m2:
                max_m2 = abs(val)
        if max_m2 >= threshold:
            detected.add((i, j))

    true_set = spec.true_interactions
    tp = len(detected & true_set)
    fp = len(detected - true_set)
    fn = len(true_set - detected)

    if len(true_set) == 0:
        precision = 1.0 if fp == 0 else 0.0
        recall = 1.0
        f1 = 1.0 if fp == 0 else 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "interaction_precision": round(float(precision), 4),
        "interaction_recall": round(float(recall), 4),
        "interaction_f1": round(float(f1), 4),
        "false_interaction_count": int(fp),
        "detected_count": int(len(detected)),
        "main_effect_nmae": round(float(main_effect_nmae), 5),
    }


def _ensure_p14_data_cached(spec: InteractionWorldSpec, seed: int, steps: int, cache: Dict[Any, Any]):
    key = (spec.name, seed, steps)
    if key in cache:
        return cache[key]

    lo_steps = spec.steps_per_combo[0] * len(spec.train_combos)
    hi_steps = spec.steps_per_combo[1] * len(spec.train_combos)
    eval_key = (spec.name, "eval")
    if eval_key not in cache:
        Ph = collect_eval_p14(spec, "heldout")
        Pi = collect_eval_p14(spec, "indist")
        cache[eval_key] = (Ph, Pi)
    Ph, Pi = cache[eval_key]

    if steps in (lo_steps, hi_steps):
        hi_key = (spec.name, seed, hi_steps)
        if hi_key not in cache:
            ds_hi, Ptr_hi, Ytr_hi = collect_train_p14(spec, seed, hi_steps)
            cache[hi_key] = (Ptr_hi, Ytr_hi, Ph, Pi)
            n_lo = sum(1 for t in ds_hi.transitions[:lo_steps] if t.info.get("interaction", False) and t.obs.entities)
            cache[(spec.name, seed, lo_steps)] = (Ptr_hi[:n_lo], Ytr_hi[:n_lo], Ph, Pi)
        return cache[key]

    _, Ptr, Ytr = collect_train_p14(spec, seed, steps)
    cache[key] = (Ptr, Ytr, Ph, Pi)
    return cache[key]


def run_cell_p14(
    spec: InteractionWorldSpec,
    arch: str,
    seed: int,
    steps: int,
    cache: Dict[Any, Any],
    epochs: int = EPOCHS14,
    lr: float = LR,
    batch_size: int = BATCH_SIZE14,
) -> Dict[str, Any]:
    Ptr, Ytr, Ph, Pi = _ensure_p14_data_cached(spec, seed, steps, cache)

    L_tr = spec.decode(Ptr)
    held = {tuple(c) for c in spec.heldout_combos}
    leak = int(sum(tuple(int(v) for v in row) in held for row in L_tr))

    Yh = spec.truth(Ph)
    Yi = spec.truth(Pi)

    model = build_p14_model(arch, spec.n_props, seed=seed)
    if isinstance(model, SparseInteractionDiscoveryModel):
        y_main_target = model.fit_sparse_hierarchy(Ptr, Ytr)
        Ptr_clean = model.p_min + model._normalized_coords(Ptr) * model.p_span
        _, losses = train_model(
            model.main_trunk, Ptr_clean, y_main_target, seed, epochs=epochs, lr=lr, batch_size=batch_size, refit_normalizer=False
        )
    else:
        _, losses = train_model(model, Ptr, Ytr, seed, epochs=epochs, lr=lr, batch_size=batch_size)

    ph = np.asarray(model.forward(Ph)).reshape(-1)
    pi = np.asarray(model.forward(Pi)).reshape(-1)
    probe = probe_model_interactions_and_main_effects(spec, model)

    nr = spec.value_range
    return {
        "world": spec.name,
        "arch": arch,
        "seed": seed,
        "steps": steps,
        "epochs": epochs,
        "lr": lr,
        "n_params": int(model.num_parameters),
        "n_train_events": int(len(Ptr)),
        "leak_rows": leak,
        "finite": bool(np.all(np.isfinite(ph)) and np.all(np.isfinite(pi))),
        "final_train_loss": round(float(losses[-1]), 5),
        "heldout_nrmse": round(rmse(ph, Yh) / nr, 5),
        "indist_nrmse": round(rmse(pi, Yi) / nr, 5),
        "heldout_r2": round(r2(ph, Yh), 4),
        **probe,
    }


def _worker_p14_seed(args: Tuple[str, int, Tuple[str, ...], Tuple[str, ...]]):
    wname, s, levels, archs = args
    spec = WORLD_MAP_14[wname]
    cache: Dict[Any, Any] = {}
    out_rows = []
    ordered_levels = sorted(levels, key=lambda x: 0 if x == "hi" else 1)
    for lv in ordered_levels:
        steps = spec.steps_per_combo[0 if lv == "lo" else 1] * len(spec.train_combos)
        for a in archs:
            out_rows.append(run_cell_p14(spec, a, s, steps, cache))
    return out_rows


def run_phase14_suite(
    worlds=WORLDS14,
    archs=ARCHS14,
    seeds=SEEDS14,
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
                for s, s_rows in zip(seeds, ex.map(_worker_p14_seed, tasks)):
                    rows.extend(s_rows)
                    if verbose:
                        hi_steps = spec.steps_per_combo[1] * len(spec.train_combos)
                        summary = {r["arch"]: r["heldout_nrmse"] for r in s_rows if r["steps"] == hi_steps}
                        print(spec.name, "seed", s, "done (hi):", summary, flush=True)
        else:
            for t in tasks:
                s_rows = _worker_p14_seed(t)
                rows.extend(s_rows)
                if verbose:
                    print(spec.name, "seed", t[1], "done", flush=True)
    return rows
