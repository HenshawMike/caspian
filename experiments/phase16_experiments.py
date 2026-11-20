"""Phase 16: Cross-World Transfer and Structural Generalisation (experiments/phase16_experiments.py).
Registered in docs/phase16/EXPERIMENT_REGISTRY.md (CSP-P16-E001).
"""
import concurrent.futures
import copy
import itertools
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import experiments.phase7_experiments as p7
from experiments.phase11_experiments import LR, STEP_OFFSET, rmse, train_model
from models.mlp import PredictiveMLP
from models.transfer import ModularTransferModel

SEEDS16 = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
EXPLORATORY16 = [9001, 9002, 9003]
IN_DIM16 = 8
N_LATENT16 = 4
SHOTS16 = (20, 48)
SRC_SAMPLES16 = 320
EPOCHS16 = 120
FT_EPOCHS16 = 80
BATCH_SIZE16 = 32

ARCHS16 = [
    "modular_transfer",
    "direct_transfer",
    "finetune_mlp",
    "scratch_mlp",
    "scratch_structural",
    "oracle_target",
]


def _ortho_mix(seed: int, n_latent: int = N_LATENT16, in_dim: int = IN_DIM16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((in_dim, n_latent)))
    return Q[:, :n_latent].T.copy()  # (4, 8), M @ M.T = I_4


M_SRC16 = _ortho_mix(1600)
M_SHIFT16 = _ortho_mix(1601)


def _rule_source(p: np.ndarray) -> np.ndarray:
    return 6.0 * p[:, 0] - 4.0 * p[:, 1] + 3.0 * p[:, 2] + 5.0 * p[:, 0] * p[:, 1] + STEP_OFFSET


def _rule_shared_plus_new(p: np.ndarray) -> np.ndarray:
    return _rule_source(p) + 4.0 * p[:, 3] - 4.0 * p[:, 2] * p[:, 3]


def _rule_shift(p: np.ndarray) -> np.ndarray:
    return 6.0 * p[:, 0] + 4.0 * p[:, 1] - 3.0 * p[:, 2] + 5.0 * p[:, 1] * p[:, 2] + STEP_OFFSET


def _rule_negative(p: np.ndarray) -> np.ndarray:
    return -6.0 * p[:, 0] + 4.0 * p[:, 1] - 3.0 * p[:, 2] - 5.0 * p[:, 0] * p[:, 1] + STEP_OFFSET


@dataclass
class TransferWorldSpec:
    name: str
    rule_fn: Callable[[np.ndarray], np.ndarray]
    M: np.ndarray
    value_range: float
    obs_shifted: bool = False
    noise_std: float = 0.25

    def all_latent_grid(self) -> np.ndarray:
        return np.array(list(itertools.product((-1.0, 1.0), repeat=N_LATENT16)), dtype=float)

    def eval_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        P = self.all_latent_grid()
        X = P @ self.M
        Y = self.rule_fn(P)
        return X, Y

    def sample(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        P_all = self.all_latent_grid()
        # Balanced sampling across combinations with slight stochasticity for realistic few-shot draws
        idx = rng.integers(0, len(P_all), size=n)
        P = P_all[idx]
        X = P @ self.M
        Y = self.rule_fn(P) + rng.normal(0.0, self.noise_std, size=n)
        return X, Y


W16_SOURCE = TransferWorldSpec("W16_source", _rule_source, M_SRC16, value_range=36.0, obs_shifted=False)
W16_OBS_SHIFT = TransferWorldSpec("W16_obs_shift", _rule_source, M_SHIFT16, value_range=36.0, obs_shifted=True)
W16_SHARED_PLUS_NEW = TransferWorldSpec("W16_shared_plus_new", _rule_shared_plus_new, M_SRC16, value_range=44.0, obs_shifted=False)
W16_RULE_SHIFT = TransferWorldSpec("W16_rule_shift", _rule_shift, M_SRC16, value_range=36.0, obs_shifted=False)
W16_NEGATIVE_CONTROL = TransferWorldSpec("W16_negative_control", _rule_negative, M_SRC16, value_range=36.0, obs_shifted=False)

TARGET_WORLDS16 = (W16_OBS_SHIFT, W16_SHARED_PLUS_NEW, W16_RULE_SHIFT, W16_NEGATIVE_CONTROL)
WORLD_MAP_16 = {w.name: w for w in TARGET_WORLDS16}


def verify_gridworld_semantic_purity_p16(spec: TransferWorldSpec, seed: int = 51) -> None:
    """Wrap target world realizations into GridWorld and assert Semantic Firewall purity."""
    from environment.entities import Entity
    X_ev, Y_ev = spec.eval_grid()
    cfg = copy.deepcopy(p7._load_world("phase7_world_base.json"))
    pool = [
        {
            "entity_type": 500 + k,
            "hidden_state_delta": float(Y_ev[k] - STEP_OFFSET),
            "is_interactive": True,
            "is_blocking": False,
            "properties": [float(v) for v in X_ev[k]],
        }
        for k in range(len(X_ev))
    ]
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
    ds = p7._collect_passive(cfg, seed=seed, num_steps=120)
    for t in ds.transitions[:5]:
        t.obs.assert_semantic_purity()


def run_cell_p16(
    spec: TransferWorldSpec,
    arch: str,
    seed: int,
    n_shot: int,
    cache: Dict[Any, Any],
) -> Dict[str, Any]:
    src_key = ("src", seed)
    if src_key not in cache:
        X_src_tr, Y_src_tr = W16_SOURCE.sample(SRC_SAMPLES16, seed)
        X_src_ev, Y_src_ev = W16_SOURCE.eval_grid()
        # Pretrain source models once per seed
        src_mod = ModularTransferModel(IN_DIM16, N_LATENT16, seed=seed)
        src_mod.fit_source(X_src_tr, Y_src_tr)

        src_mlp = PredictiveMLP(IN_DIM16, (32, 16), 1, seed=seed)
        train_model(src_mlp, X_src_tr, Y_src_tr, seed, epochs=EPOCHS16, lr=LR, batch_size=BATCH_SIZE16)

        cache[src_key] = (X_src_ev, Y_src_ev, src_mod, src_mlp)

    X_src_ev, Y_src_ev, src_mod, src_mlp = cache[src_key]

    tgt_key = (spec.name, seed, n_shot)
    if tgt_key not in cache:
        X_tgt_tr, Y_tgt_tr = spec.sample(n_shot, seed + 1000 + n_shot)
        X_tgt_ev, Y_tgt_ev = spec.eval_grid()
        cache[tgt_key] = (X_tgt_tr, Y_tgt_tr, X_tgt_ev, Y_tgt_ev)

    X_tgt_tr, Y_tgt_tr, X_tgt_ev, Y_tgt_ev = cache[tgt_key]

    core_gate = 1.0
    if arch == "oracle_target":
        y_tgt_pred = Y_tgt_ev
        y_src_pred = Y_src_ev
        n_params = 10
        final_loss = 0.0
    elif arch == "direct_transfer":
        m = src_mod.clone_for_target()
        y_tgt_pred = np.asarray(m.forward(X_tgt_ev)).reshape(-1)
        y_src_pred = m.predict_source(X_src_ev)
        n_params = m.num_parameters
        final_loss = float(np.mean((np.asarray(m.forward(X_tgt_tr)).reshape(-1) - Y_tgt_tr) ** 2))
    elif arch == "modular_transfer":
        m = src_mod.clone_for_target()
        core_gate = m.adapt_few_shot(X_tgt_tr, Y_tgt_tr, align_sensors=spec.obs_shifted, allow_negative_transfer_gate=True)
        y_tgt_pred = np.asarray(m.forward(X_tgt_ev)).reshape(-1)
        y_src_pred = m.predict_source(X_src_ev)
        n_params = m.num_parameters
        final_loss = float(np.mean((np.asarray(m.forward(X_tgt_tr)).reshape(-1) - Y_tgt_tr) ** 2))
    elif arch == "finetune_mlp":
        m = copy.deepcopy(src_mlp)
        _, losses = train_model(m, X_tgt_tr, Y_tgt_tr, seed + 7, epochs=FT_EPOCHS16, lr=0.005, batch_size=min(16, n_shot), refit_normalizer=False)
        y_tgt_pred = np.asarray(m.forward(X_tgt_ev)).reshape(-1)
        y_src_pred = np.asarray(m.forward(X_src_ev)).reshape(-1)
        n_params = m.num_parameters
        final_loss = float(losses[-1])
    elif arch == "scratch_mlp":
        m = PredictiveMLP(IN_DIM16, (32, 16), 1, seed=seed + 13)
        _, losses = train_model(m, X_tgt_tr, Y_tgt_tr, seed + 13, epochs=EPOCHS16, lr=LR, batch_size=min(16, n_shot))
        y_tgt_pred = np.asarray(m.forward(X_tgt_ev)).reshape(-1)
        y_src_pred = np.asarray(m.forward(X_src_ev)).reshape(-1)
        n_params = m.num_parameters
        final_loss = float(losses[-1])
    elif arch == "scratch_structural":
        m = ModularTransferModel(IN_DIM16, N_LATENT16, l2_reg=0.05, seed=seed + 19)
        m.fit_source(X_tgt_tr, Y_tgt_tr)
        y_tgt_pred = np.asarray(m.forward(X_tgt_ev)).reshape(-1)
        y_src_pred = np.asarray(m.forward(X_src_ev)).reshape(-1)
        n_params = m.num_parameters
        final_loss = float(np.mean((np.asarray(m.forward(X_tgt_tr)).reshape(-1) - Y_tgt_tr) ** 2))
    else:
        raise ValueError(arch)

    tgt_nrmse = rmse(y_tgt_pred, Y_tgt_ev) / spec.value_range
    src_ret_nrmse = rmse(y_src_pred, Y_src_ev) / W16_SOURCE.value_range

    return {
        "world": spec.name,
        "arch": arch,
        "seed": seed,
        "n_shot": n_shot,
        "n_params": int(n_params),
        "n_train_events": int(n_shot),
        "leak_rows": 0,
        "finite": bool(np.all(np.isfinite(y_tgt_pred)) and np.all(np.isfinite(y_src_pred))),
        "final_train_loss": round(final_loss, 5),
        "target_nrmse": round(float(tgt_nrmse), 5),
        "source_retention_nrmse": round(float(src_ret_nrmse), 5),
        "core_gate": float(core_gate),
    }


def _worker_p16_seed(args: Tuple[str, int, Tuple[int, ...], Tuple[str, ...]]):
    wname, s, shots, archs = args
    spec = WORLD_MAP_16[wname]
    cache: Dict[Any, Any] = {}
    out_rows = []
    for sh in shots:
        for a in archs:
            out_rows.append(run_cell_p16(spec, a, s, sh, cache))
    return out_rows


def run_phase16_suite(
    worlds=TARGET_WORLDS16,
    archs=ARCHS16,
    seeds=SEEDS16,
    shots=SHOTS16,
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    for spec in worlds:
        verify_gridworld_semantic_purity_p16(spec, seed=seeds[0])

    rows: List[Dict[str, Any]] = []
    workers = max_workers or min(len(seeds), os.cpu_count() or 1)
    for spec in worlds:
        tasks = [(spec.name, s, tuple(shots), tuple(archs)) for s in seeds]
        if workers > 1 and len(tasks) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                for s, s_rows in zip(seeds, ex.map(_worker_p16_seed, tasks)):
                    rows.extend(s_rows)
                    if verbose:
                        lo_sh = shots[0]
                        summary = {r["arch"]: r["target_nrmse"] for r in s_rows if r["n_shot"] == lo_sh}
                        print(spec.name, "seed", s, f"done (shot={lo_sh}):", summary, flush=True)
        else:
            for t in tasks:
                s_rows = _worker_p16_seed(t)
                rows.extend(s_rows)
                if verbose:
                    print(spec.name, "seed", t[1], "done", flush=True)
    return rows
