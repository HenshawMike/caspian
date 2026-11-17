"""Phase 13: Autonomous Property Discovery and Representation Learning from Raw Observations.
Registered in docs/phase13/EXPERIMENT_REGISTRY.md.
"""
import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import experiments.phase7_experiments as p7
from experiments.phase11_experiments import (
    EPOCHS,
    LR,
    STEP_OFFSET,
    W1,
    W2,
    WorldSpec,
    bootstrap_ci,
    interaction_rows,
    linear_cka,
    paired_summary,
    r2,
    ridge_probe_r2,
    rmse,
    train_model,
)
from experiments.phase12_experiments import W3
from models.compositional import build_model
from models.discovery import LatentDiscoveryModel, build_discovery_model

SEEDS13 = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
EXPLORATORY13 = [9001, 9002, 9003]
RAW_DIM = 12
N_SLOTS = 6
NUISANCE_GAIN = 2.0

ARCHS13 = [
    "latent_additive",
    "latent_residual",
    "raw_naive_additive",
    "pca_statistical",
    "raw_mlp_small",
    "raw_mlp_large",
    "untrained_encoder",
    "random_repr",
    "oracle_channels_additive",
]


@dataclass
class RawWorldSpec:
    """Wraps a base WorldSpec (W1, W2, W3) with a D=12 raw entangled sensory observation generator."""
    base: WorldSpec
    raw_dim: int = RAW_DIM
    n_nuisance: int = 3
    nuisance_gain: float = NUISANCE_GAIN
    mix_seed: int = 1300

    def __post_init__(self):
        rng = np.random.default_rng(self.mix_seed)
        n_total = len(self.base.n_levels) + self.n_nuisance
        Q, _ = np.linalg.qr(rng.standard_normal((self.raw_dim, n_total)))
        self.M = Q[:, :n_total].T.copy()  # (6, 12) row-orthonormal mixing matrix: M @ M.T = I_6

    @property
    def name(self) -> str:
        return f"{self.base.name}_raw"

    @property
    def n_props(self) -> int:
        return len(self.base.n_levels)

    @property
    def train_combos(self) -> List[Tuple[int, ...]]:
        return self.base.train_combos

    @property
    def heldout_combos(self) -> List[Tuple[int, ...]]:
        return self.base.heldout_combos

    @property
    def value_range(self) -> float:
        return self.base.value_range

    @property
    def steps_per_combo(self) -> Tuple[int, int]:
        return self.base.steps_per_combo

    def all_combos(self):
        return self.base.all_combos()

    def nuisance_tuples(self) -> List[Tuple[float, ...]]:
        return list(itertools.product(*[(-1.0, 1.0) for _ in range(self.n_nuisance)]))

    def level_to_coord(self, prop_idx: int, level: int) -> float:
        n_lv = self.base.n_levels[prop_idx]
        if n_lv == 2:
            return -1.0 if level == 0 else 1.0
        # 3 levels: {0, 1, 2} -> {-1.0, 0.0, +1.0}
        return float(level) - 1.0

    def combo_to_coords(self, combo: Tuple[int, ...]) -> np.ndarray:
        return np.array([self.level_to_coord(k, combo[k]) for k in range(self.n_props)], dtype=float)

    def raw_sensor_vector(self, combo: Tuple[int, ...], nuis: Tuple[float, ...]) -> Tuple[float, ...]:
        p_coords = self.combo_to_coords(combo)
        u_coords = np.array(nuis, dtype=float) * self.nuisance_gain
        z = np.concatenate([p_coords, u_coords])
        x = z @ self.M
        return tuple(float(v) for v in x)

    def unmix_ground_truth(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Analysis-only helper (never called by any model): recover true (P_coords, U_coords) via M.T."""
        Z = np.asarray(X, dtype=float) @ self.M.T
        P_coords = Z[:, :self.n_props]
        U_coords = Z[:, self.n_props:] / self.nuisance_gain
        return P_coords, U_coords

    def decode(self, X: np.ndarray) -> np.ndarray:
        """Analysis-only: map raw sensor rows X (N, 12) to integer level tuples (N, 3)."""
        P_coords, _ = self.unmix_ground_truth(X)
        cols = []
        for k in range(self.n_props):
            grid = np.array([self.level_to_coord(k, lv) for lv in range(self.base.n_levels[k])])
            cols.append(np.argmin(np.abs(P_coords[:, k:k + 1] - grid[None, :]), axis=1))
        return np.stack(cols, axis=1)

    def clean_property_channels(self, X: np.ndarray) -> np.ndarray:
        """Upper-bound oracle helper: convert decoded levels to the clean 3-channel property vectors of Phase 11."""
        L = self.decode(X)
        out = []
        for row in L:
            combo = tuple(int(v) for v in row)
            # Extract only the non-nuisance property channels from base WorldSpec
            ch = [float(vals[combo[p]]) for p, vals in self.base.channels if p is not None]
            out.append(ch)
        return np.array(out, dtype=float)

    def truth(self, X: np.ndarray, variant: str = "additive") -> np.ndarray:
        L = self.decode(X)
        return np.array([self.base.delta(tuple(int(v) for v in row), variant) + STEP_OFFSET for row in L], dtype=float)


W1_RAW = RawWorldSpec(base=W1, mix_seed=1301)
W2_RAW = RawWorldSpec(base=W2, mix_seed=1302)
W3_RAW = RawWorldSpec(base=W3, mix_seed=1303)


def make_raw_world(spec: RawWorldSpec, combos: Sequence[Tuple[int, ...]], variant: str = "additive"):
    from environment.entities import Entity
    cfg = copy.deepcopy(p7._load_world("phase7_world_base.json"))
    pool = []
    idx = 0
    for c in combos:
        delta_val = spec.base.delta(c, variant)
        for nuis in spec.nuisance_tuples():
            pool.append({
                "entity_type": 200 + idx,
                "hidden_state_delta": delta_val,
                "is_interactive": True,
                "is_blocking": False,
                "properties": list(spec.raw_sensor_vector(c, nuis)),
            })
            idx += 1
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


def collect_train_raw(spec: RawWorldSpec, seed: int, steps: int):
    cfg = make_raw_world(spec, spec.train_combos, "additive")
    ds = p7._collect_passive(cfg, seed=seed, num_steps=steps)
    X, Y = interaction_rows(ds, spec.raw_dim)
    return ds, X, Y


def collect_eval_raw(spec: RawWorldSpec, seed: int, which: str, steps_per_combo: int = 1200) -> np.ndarray:
    combos = spec.heldout_combos if which == "heldout" else spec.train_combos
    base = 9000 if which == "heldout" else 8000
    Xs = []
    for i, c in enumerate(combos):
        ds = p7._collect_passive(make_raw_world(spec, [c], "additive"), seed=seed + base + i, num_steps=steps_per_combo)
        X, _ = interaction_rows(ds, spec.raw_dim)
        Xs.append(X)
    return np.concatenate(Xs, axis=0)


def canonical_eval_grid(spec: RawWorldSpec) -> np.ndarray:
    """Deterministic grid of all (combo x nuisance) raw sensor vectors for cross-seed CKA comparison."""
    rows = []
    for c in spec.all_combos():
        for nuis in spec.nuisance_tuples():
            rows.append(spec.raw_sensor_vector(c, nuis))
    return np.array(rows, dtype=float)


def slot_match_r2(Z: np.ndarray, P_true: np.ndarray) -> float:
    """Compute best injective assignment of 3 latent columns of Z (N, K) to the 3 ground-truth properties P_true (N, 3)
    using univariate squared Pearson correlation R^2."""
    n_props = P_true.shape[1]
    K = Z.shape[1]
    r2_mat = np.zeros((n_props, K), dtype=float)
    for p in range(n_props):
        yp = P_true[:, p]
        vp = np.std(yp)
        if vp < 1e-9:
            continue
        for m in range(K):
            zm = Z[:, m]
            vm = np.std(zm)
            if vm < 1e-9:
                r2_mat[p, m] = 0.0
            else:
                c = float(np.corrcoef(yp, zm)[0, 1])
                r2_mat[p, m] = c * c if np.isfinite(c) else 0.0

    best = 0.0
    if K >= n_props and K <= 12:
        for cols in itertools.permutations(range(K), n_props):
            score = sum(r2_mat[p, cols[p]] for p in range(n_props)) / n_props
            if score > best:
                best = score
    else:
        # Greedy bipartite matching for larger K (e.g. K=16)
        used = set()
        tot = 0.0
        for p in range(n_props):
            avail = [(r2_mat[p, m], m) for m in range(K) if m not in used]
            sc, m_best = max(avail)
            used.add(m_best)
            tot += sc
        best = tot / n_props
    return float(best)


def extract_discovery_slots(model, X: np.ndarray, spec: RawWorldSpec) -> np.ndarray:
    if hasattr(model, "get_discovered_slots"):
        return np.asarray(model.get_discovered_slots(X))
    if hasattr(model, "get_latent_state"):
        return np.asarray(model.get_latent_state(X))
    raise ValueError("Model does not expose latent state")


def evaluate_interventions_and_distractors(spec: RawWorldSpec, model, is_oracle: bool = False) -> Dict[str, float]:
    """Evaluate single-property causal interventions (do(p_k <- p_k + 1) with p_{-k} and u fixed)
    and nuisance interventions (do(u <- -u) with p fixed)."""
    held = {tuple(c) for c in spec.heldout_combos}
    nuis_list = spec.nuisance_tuples()

    def _pred(X_raw: np.ndarray) -> np.ndarray:
        inp = spec.clean_property_channels(X_raw) if is_oracle else X_raw
        return np.asarray(model.forward(inp)).reshape(-1)

    def _slots(X_raw: np.ndarray) -> np.ndarray:
        inp = spec.clean_property_channels(X_raw) if is_oracle else X_raw
        return extract_discovery_slots(model, inp, spec)

    X_grid = canonical_eval_grid(spec)
    Z_grid = _slots(X_grid)
    slot_std = np.std(Z_grid, axis=0, keepdims=True)
    slot_std = np.where(slot_std < 1e-6, 1.0, slot_std)

    seen_err, unseen_err, selectivities = [], [], []
    for c in spec.all_combos():
        for p in range(spec.n_props):
            if c[p] + 1 >= spec.base.n_levels[p]:
                continue
            c2 = list(c)
            c2[p] += 1
            c2 = tuple(c2)
            d_true = spec.base.delta(c2, "additive") - spec.base.delta(c, "additive")
            X1 = np.array([spec.raw_sensor_vector(c, u) for u in nuis_list], dtype=float)
            X2 = np.array([spec.raw_sensor_vector(c2, u) for u in nuis_list], dtype=float)
            d_pred = _pred(X2) - _pred(X1)
            err = float(np.mean(np.abs(d_pred - d_true))) / spec.value_range
            if c in held or c2 in held:
                unseen_err.append(err)
            else:
                seen_err.append(err)

            dZ = np.mean(np.abs((_slots(X2) - _slots(X1)) / slot_std), axis=0)
            tot_dz = float(np.sum(dZ))
            if tot_dz > 1e-9:
                selectivities.append(float(np.max(dZ) / tot_dz))

    # Distractor sensitivity: negate nuisance tuple u -> -u for all combinations
    X_pos, X_neg = [], []
    for c in spec.all_combos():
        for u in nuis_list:
            X_pos.append(spec.raw_sensor_vector(c, u))
            X_neg.append(spec.raw_sensor_vector(c, tuple(-v for v in u)))
    X_pos_arr = np.array(X_pos, dtype=float)
    X_neg_arr = np.array(X_neg, dtype=float)
    dist_sens = float(np.mean(np.abs(_pred(X_pos_arr) - _pred(X_neg_arr)))) / spec.value_range

    return {
        "flip_seen_nmae": round(float(np.mean(seen_err)), 5) if seen_err else float("nan"),
        "flip_unseen_nmae": round(float(np.mean(unseen_err)), 5) if unseen_err else float("nan"),
        "intervention_selectivity": round(float(np.mean(selectivities)), 4) if selectivities else 0.0,
        "distractor_sensitivity": round(dist_sens, 5),
    }


def pairwise_ordering_raw(spec: RawWorldSpec, X: np.ndarray, pred: np.ndarray) -> float:
    L = spec.decode(X)
    held = {tuple(c) for c in spec.heldout_combos}
    combos = sorted({tuple(int(v) for v in row) for row in L})
    mp = {c: float(np.mean(pred[np.all(L == np.array(c), axis=1)])) for c in combos}
    ok = tot = 0
    for i, a in enumerate(combos):
        for b in combos[i + 1:]:
            if a not in held and b not in held:
                continue
            ta = spec.base.delta(a, "additive")
            tb = spec.base.delta(b, "additive")
            if abs(ta - tb) < 1e-9:
                continue
            tot += 1
            ok += int((mp[a] - mp[b]) * (ta - tb) > 0)
    return round(ok / tot, 4) if tot else float("nan")


def run_cell_p13(
    spec: RawWorldSpec,
    arch: str,
    seed: int,
    steps: int,
    cache: Dict[Any, Any],
    epochs: int = EPOCHS,
    lr: float = LR,
    lat_store: Optional[Dict[Any, np.ndarray]] = None,
) -> Dict[str, Any]:
    key = (spec.name, seed, steps)
    if key not in cache:
        _, Xtr, Ytr = collect_train_raw(spec, seed, steps)
        Xh = collect_eval_raw(spec, seed, "heldout", steps_per_combo=1200)
        Xi = collect_eval_raw(spec, seed, "indist", steps_per_combo=600)
        cache[key] = (Xtr, Ytr, Xh, Xi)
    Xtr, Ytr, Xh, Xi = cache[key]

    L_tr = spec.decode(Xtr)
    held = {tuple(c) for c in spec.heldout_combos}
    leak = int(sum(tuple(int(v) for v in row) in held for row in L_tr))

    Yh = spec.truth(Xh, "additive")
    Yi = spec.truth(Xi, "additive")
    X_all = np.vstack([Xi, Xh])
    Y_all = np.concatenate([Yi, Yh])

    is_oracle = (arch == "oracle_channels_additive")
    if is_oracle:
        Ptr = spec.clean_property_channels(Xtr)
        Ph = spec.clean_property_channels(Xh)
        Pi = spec.clean_property_channels(Xi)
        P_all = spec.clean_property_channels(X_all)
        model = build_model("additive", spec.n_props, seed)
        _, losses = train_model(model, Ptr, Ytr, seed, epochs=epochs, lr=lr)
        ph = np.asarray(model.forward(Ph)).reshape(-1)
        pi = np.asarray(model.forward(Pi)).reshape(-1)
        Z_slots_all = P_all
        Htr = np.asarray(model.get_latent_state(Ptr))
        Hte = np.asarray(model.get_latent_state(Ph))
    else:
        model = build_discovery_model(arch, spec.raw_dim, seed=seed, n_slots=N_SLOTS)
        _, losses = train_model(model, Xtr, Ytr, seed, epochs=epochs, lr=lr)
        ph = np.asarray(model.forward(Xh)).reshape(-1)
        pi = np.asarray(model.forward(Xi)).reshape(-1)
        Z_slots_all = extract_discovery_slots(model, X_all, spec)
        Htr = np.asarray(model.get_latent_state(Xtr))
        Hte = np.asarray(model.get_latent_state(Xh))

    P_true_all, _ = spec.unmix_ground_truth(X_all)
    sm_r2 = round(slot_match_r2(Z_slots_all, P_true_all), 4)
    probe_r2 = round(ridge_probe_r2(Htr, Ytr, Hte, Yh), 4)
    probe_shuf = round(ridge_probe_r2(Htr, Ytr, Hte, Yh, shuffle_seed=seed), 4)
    interv = evaluate_interventions_and_distractors(spec, model, is_oracle=is_oracle)
    pw_acc = pairwise_ordering_raw(spec, X_all, np.concatenate([pi, ph]))

    if lat_store is not None and not is_oracle:
        X_grid = canonical_eval_grid(spec)
        lat_store[(spec.name, arch, seed, steps)] = np.asarray(model.get_latent_state(X_grid))

    nr = spec.value_range
    return {
        "world": spec.name,
        "arch": arch,
        "seed": seed,
        "steps": steps,
        "epochs": epochs,
        "lr": lr,
        "n_params": int(model.num_parameters),
        "n_train_events": int(len(Xtr)),
        "leak_rows": leak,
        "finite": bool(np.all(np.isfinite(ph)) and np.all(np.isfinite(pi))),
        "final_train_loss": round(float(losses[-1]), 5),
        "heldout_nrmse": round(rmse(ph, Yh) / nr, 5),
        "indist_nrmse": round(rmse(pi, Yi) / nr, 5),
        "heldout_r2": round(r2(ph, Yh), 4),
        "slot_match_r2": sm_r2,
        "probe_cross_combo_r2": probe_r2,
        "probe_shuffled_r2": probe_shuf,
        "pairwise_acc": pw_acc,
        **interv,
    }


def run_phase13_suite(
    worlds=(W1_RAW, W2_RAW, W3_RAW),
    archs=ARCHS13,
    seeds=SEEDS13,
    levels=("lo", "hi"),
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    cache: Dict[Any, Any] = {}
    lat: Dict[Any, np.ndarray] = {}

    for spec in worlds:
        for s in seeds:
            for lv in levels:
                steps = spec.steps_per_combo[0 if lv == "lo" else 1] * len(spec.train_combos)
                for a in archs:
                    rows.append(run_cell_p13(spec, a, s, steps, cache, lat_store=lat))
                if verbose:
                    summary = {
                        r["arch"]: r["heldout_nrmse"]
                        for r in rows
                        if r["world"] == spec.name and r["seed"] == s and r["steps"] == steps
                    }
                    print(spec.name, "seed", s, lv, summary, flush=True)

    # Compute inter-seed representation consistency (Linear CKA across all seed pairs per world, arch, steps)
    cka_rows: List[Dict[str, Any]] = []
    for spec in worlds:
        for lv in levels:
            steps = spec.steps_per_combo[0 if lv == "lo" else 1] * len(spec.train_combos)
            for a in archs:
                if a == "oracle_channels_additive":
                    continue
                for i, s1 in enumerate(seeds):
                    for s2 in seeds[i + 1:]:
                        H1 = lat.get((spec.name, a, s1, steps))
                        H2 = lat.get((spec.name, a, s2, steps))
                        if H1 is not None and H2 is not None:
                            cka_val = linear_cka(H1, H2)
                            cka_rows.append({
                                "world": spec.name,
                                "arch": a,
                                "steps": steps,
                                "level": lv,
                                "seed_pair": f"{s1}|{s2}",
                                "cka": round(float(cka_val), 4) if np.isfinite(cka_val) else 0.0,
                            })
    return rows, cka_rows
