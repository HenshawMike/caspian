"""Phase 17: Continual Learning and Persistent Environmental Memory (experiments/phase17_experiments.py).
Registered in docs/phase17/EXPERIMENT_REGISTRY.md (CSP-P17-E001).
"""
import concurrent.futures
import copy
import itertools
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import experiments.phase7_experiments as p7
from experiments.phase11_experiments import STEP_OFFSET, rmse
from models.memory import EWCState, PersistentWorldMemory
from models.mlp import PredictiveMLP
from models.transfer_v3 import ModularTransferModelV3

SEEDS17 = [71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
SMOKE_SEEDS17 = [9001, 9002, 9003]
IN_DIM17 = 8
N_LATENT17 = 4
PROBE_SHOTS17 = 8
REPLAY_PER_WORLD17 = 40

ARCHS17 = [
    "persistent_memory",
    "naive_sequential_mlp",
    "replay_mlp",
    "ewc_mlp",
    "memoryless_structural",
    "oracle_multi_world",
]


def _ortho_mix17(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((IN_DIM17, N_LATENT17)))
    return Q[:, :N_LATENT17].T.copy()


M1_17 = _ortho_mix17(1701)
M2_17 = _ortho_mix17(1702)
M3_17 = _ortho_mix17(1703)


def _rule_alpha(p: np.ndarray) -> np.ndarray:
    return 6.0 * p[:, 0] - 4.0 * p[:, 1] + 3.0 * p[:, 2] + 5.0 * p[:, 0] * p[:, 1] + STEP_OFFSET


def _rule_beta(p: np.ndarray) -> np.ndarray:
    return -5.0 * p[:, 0] + 6.0 * p[:, 1] - 3.0 * p[:, 3] + 4.0 * p[:, 1] * p[:, 3] + STEP_OFFSET


def _rule_delta_inverted(p: np.ndarray) -> np.ndarray:
    return -6.0 * p[:, 0] + 4.0 * p[:, 1] - 3.0 * p[:, 2] - 5.0 * p[:, 0] * p[:, 1] + STEP_OFFSET


def _rule_beta_mutated(p: np.ndarray) -> np.ndarray:
    return 5.0 * p[:, 0] - 6.0 * p[:, 1] + 3.0 * p[:, 3] - 4.0 * p[:, 1] * p[:, 3] + STEP_OFFSET


@dataclass
class ContinualWorldSpec:
    name: str
    rule_fn: Callable[[np.ndarray], np.ndarray]
    M: np.ndarray
    value_range: float
    train_budget: int
    noise_std: float = 0.25

    def eval_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        P = np.array(list(itertools.product((-1.0, 1.0), repeat=N_LATENT17)), dtype=float)
        return P @ self.M, self.rule_fn(P)

    def sample(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        P_all = np.array(list(itertools.product((-1.0, 1.0), repeat=N_LATENT17)), dtype=float)
        idx = rng.integers(0, len(P_all), size=n)
        P = P_all[idx]
        X = P @ self.M
        Y = self.rule_fn(P) + rng.normal(0.0, self.noise_std, size=n)
        return X, Y


W17_ALPHA = ContinualWorldSpec("W17_alpha", _rule_alpha, M1_17, 36.0, train_budget=160)
W17_BETA = ContinualWorldSpec("W17_beta", _rule_beta, M2_17, 36.0, train_budget=160)
W17_GAMMA_SHARED = ContinualWorldSpec("W17_gamma_shared", _rule_alpha, M3_17, 36.0, train_budget=10)
W17_DELTA_INVERTED = ContinualWorldSpec("W17_delta_inverted", _rule_delta_inverted, M1_17, 36.0, train_budget=160)
W17_BETA_MUTATED = ContinualWorldSpec("W17_beta_mutated", _rule_beta_mutated, M2_17, 36.0, train_budget=40)

STREAM_STAGES_1_TO_4 = (W17_ALPHA, W17_BETA, W17_GAMMA_SHARED, W17_DELTA_INVERTED)
ALL_STAGES_17 = (W17_ALPHA, W17_BETA, W17_GAMMA_SHARED, W17_DELTA_INVERTED, W17_BETA_MUTATED)


def verify_gridworld_semantic_purity_p17(spec: ContinualWorldSpec, seed: int = 71) -> None:
    from environment.entities import Entity
    X_ev, Y_ev = spec.eval_grid()
    cfg = copy.deepcopy(p7._load_world("phase7_world_base.json"))
    pool = [
        {
            "entity_type": 600 + k,
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


def _train_mlp_step(
    model: PredictiveMLP,
    X: np.ndarray,
    Y: np.ndarray,
    seed: int,
    epochs: int = 100,
    lr: float = 0.008,
    batch_size: int = 32,
    refit_normalizer: bool = False,
    ewc: Optional[EWCState] = None,
) -> float:
    X = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float).reshape(-1, 1)
    if refit_normalizer:
        model.fit_normalizer(X)
    rng = np.random.default_rng(seed)
    params = model.get_params()
    m_buf = [np.zeros_like(p) for _, p, _ in params]
    v_buf = [np.zeros_like(p) for _, p, _ in params]
    t = 0
    n = len(X)
    bs = min(batch_size, n)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for st in range(0, n, bs):
            idx = perm[st:st + bs]
            xb, yb = X[idx], y[idx]
            model.zero_grad()
            yp = model.forward(xb)
            model.backward(2.0 * (yp - yb) / len(xb))
            ewc_g = ewc.penalty_grad(model.get_params()) if ewc is not None else None
            t += 1
            for k, (pname, p, g) in enumerate(model.get_params()):
                gk = g + (1e-4 * p if "W" in pname else 0.0)
                if ewc_g is not None:
                    gk = gk + ewc_g[k]
                m_buf[k] = 0.9 * m_buf[k] + 0.1 * gk
                v_buf[k] = 0.999 * v_buf[k] + 0.001 * (gk ** 2)
                mh = m_buf[k] / (1.0 - 0.9 ** t)
                vh = v_buf[k] / (1.0 - 0.999 ** t)
                p -= lr * mh / (np.sqrt(vh) + 1e-8)
    return float(np.mean((model.forward(X) - y) ** 2))


def run_seed_p17(seed: int, archs=ARCHS17) -> List[Dict[str, Any]]:
    # Generate all 5 stage datasets + independent 8-sample probes + fixed eval grids once per seed
    stage_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for idx, spec in enumerate(ALL_STAGES_17):
        X_tr, Y_tr = spec.sample(spec.train_budget, seed + 100 * (idx + 1))
        X_pr, Y_pr = spec.sample(PROBE_SHOTS17, seed + 5000 + 100 * (idx + 1))
        X_ev, Y_ev = spec.eval_grid()
        stage_data[spec.name] = (X_tr, Y_tr, X_pr, Y_pr, X_ev, Y_ev)

    rows: List[Dict[str, Any]] = []
    for arch in archs:
        if arch == "oracle_multi_world":
            rows.append({
                "arch": arch, "seed": seed, "is_oracle_diagnostic": True,
                "n_params_final": 40, "raw_replay_stored": 0,
                "immediate_alpha": 0.0, "immediate_beta": 0.0, "forward_transfer_gamma": 0.0, "immediate_delta": 0.0,
                "ret_alpha_post4": 0.0, "ret_beta_post4": 0.0, "ret_gamma_post4": 0.0, "ret_delta_post4": 0.0,
                "mean_ret_post4": 0.0, "mean_forgetting_alpha_beta": 0.0,
                "retrieval_correct_post4": 4, "retrieval_total_post4": 4, "retrieval_acc_post4": 1.0,
                "outdated_detected_stage5": True, "stage5_beta_mutated_nrmse": 0.0, "ret_alpha_post5": 0.0,
                "finite": True, "leak_rows": 0,
            })
            continue

        imm: Dict[str, float] = {}
        ret_post4: Dict[str, float] = {}
        retrieval_hits = 0
        outdated_det = False
        raw_replay_count = 0

        if arch == "persistent_memory":
            mem = PersistentWorldMemory(IN_DIM17, N_LATENT17, seed=seed)
            stage_eids: Dict[str, int] = {}
            for spec in STREAM_STAGES_1_TO_4:
                X_tr, Y_tr, _, _, X_ev, Y_ev = stage_data[spec.name]
                eid = mem.observe_world(X_tr, Y_tr, allow_update_outdated=False)
                stage_eids[spec.name] = eid
                yp = np.asarray(mem.forward(X_ev)).reshape(-1)
                imm[spec.name] = rmse(yp, Y_ev) / spec.value_range

            # Post-Stage-4 evaluation across all 4 worlds using ONLY 8-sample probe (ZERO world_id!)
            for spec in STREAM_STAGES_1_TO_4:
                _, _, X_pr, Y_pr, X_ev, Y_ev = stage_data[spec.name]
                yp, pred_eid = mem.predict_with_probe(X_pr, Y_pr, X_ev)
                ret_post4[spec.name] = rmse(yp, Y_ev) / spec.value_range
                if pred_eid == stage_eids[spec.name]:
                    retrieval_hits += 1

            # Stage 5: W17_beta_mutated (Exp 17.4 — outdated memory detection & adaptation)
            X5_tr, Y5_tr, _, _, X5_ev, Y5_ev = stage_data["W17_beta_mutated"]
            mem.observe_world(X5_tr, Y5_tr, allow_update_outdated=True)
            outdated_det = bool(mem.last_outdated_detected)
            yp5 = np.asarray(mem.forward(X5_ev)).reshape(-1)
            stage5_err = rmse(yp5, Y5_ev) / W17_BETA_MUTATED.value_range

            # Verify W17_alpha still retained after Stage 5
            _, _, Xa_pr, Ya_pr, Xa_ev, Ya_ev = stage_data["W17_alpha"]
            yp_a5, _ = mem.predict_with_probe(Xa_pr, Ya_pr, Xa_ev)
            ret_alpha_post5 = rmse(yp_a5, Ya_ev) / W17_ALPHA.value_range
            n_params_final = mem.num_parameters
            raw_replay_count = mem.raw_replay_samples_stored

        elif arch == "memoryless_structural":
            last_model: Optional[ModularTransferModelV3] = None
            for idx, spec in enumerate(STREAM_STAGES_1_TO_4):
                X_tr, Y_tr, _, _, X_ev, Y_ev = stage_data[spec.name]
                m = ModularTransferModelV3(IN_DIM17, N_LATENT17, seed=seed + 19 * (idx + 1))
                m.fit_source(X_tr, Y_tr)
                imm[spec.name] = rmse(np.asarray(m.forward(X_ev)).reshape(-1), Y_ev) / spec.value_range
                last_model = m
            assert last_model is not None
            for spec in STREAM_STAGES_1_TO_4:
                _, _, _, _, X_ev, Y_ev = stage_data[spec.name]
                ret_post4[spec.name] = rmse(np.asarray(last_model.forward(X_ev)).reshape(-1), Y_ev) / spec.value_range

            X5_tr, Y5_tr, _, _, X5_ev, Y5_ev = stage_data["W17_beta_mutated"]
            m5 = ModularTransferModelV3(IN_DIM17, N_LATENT17, seed=seed + 99)
            m5.fit_source(X5_tr, Y5_tr)
            stage5_err = rmse(np.asarray(m5.forward(X5_ev)).reshape(-1), Y5_ev) / W17_BETA_MUTATED.value_range
            _, _, _, _, Xa_ev, Ya_ev = stage_data["W17_alpha"]
            ret_alpha_post5 = rmse(np.asarray(m5.forward(Xa_ev)).reshape(-1), Ya_ev) / W17_ALPHA.value_range
            n_params_final = m5.num_parameters

        elif arch in ("naive_sequential_mlp", "replay_mlp", "ewc_mlp"):
            mlp = PredictiveMLP(IN_DIM17, (32, 16), 1, seed=seed)
            ewc = EWCState(ewc_lambda=50.0) if arch == "ewc_mlp" else None
            replay_X: List[np.ndarray] = []
            replay_Y: List[np.ndarray] = []

            for idx, spec in enumerate(STREAM_STAGES_1_TO_4):
                X_tr, Y_tr, _, _, X_ev, Y_ev = stage_data[spec.name]
                if arch == "replay_mlp" and replay_X:
                    X_mix = np.vstack([X_tr] + replay_X)
                    Y_mix = np.concatenate([Y_tr] + replay_Y)
                else:
                    X_mix, Y_mix = X_tr, Y_tr
                _train_mlp_step(
                    mlp, X_mix, Y_mix, seed=seed + 17 * (idx + 1),
                    epochs=100, lr=0.008 if idx == 0 else 0.005,
                    refit_normalizer=(idx == 0), ewc=ewc,
                )
                imm[spec.name] = rmse(np.asarray(mlp.forward(X_ev)).reshape(-1), Y_ev) / spec.value_range
                if ewc is not None:
                    ewc.consolidate(mlp, X_tr, Y_tr)
                if arch == "replay_mlp":
                    k_take = min(REPLAY_PER_WORLD17, len(X_tr))
                    replay_X.append(X_tr[:k_take].copy())
                    replay_Y.append(Y_tr[:k_take].copy())

            for spec in STREAM_STAGES_1_TO_4:
                _, _, _, _, X_ev, Y_ev = stage_data[spec.name]
                ret_post4[spec.name] = rmse(np.asarray(mlp.forward(X_ev)).reshape(-1), Y_ev) / spec.value_range

            # Stage 5
            X5_tr, Y5_tr, _, _, X5_ev, Y5_ev = stage_data["W17_beta_mutated"]
            if arch == "replay_mlp" and replay_X:
                X5_mix = np.vstack([X5_tr] + replay_X)
                Y5_mix = np.concatenate([Y5_tr] + replay_Y)
            else:
                X5_mix, Y5_mix = X5_tr, Y5_tr
            _train_mlp_step(mlp, X5_mix, Y5_mix, seed=seed + 85, epochs=100, lr=0.005, refit_normalizer=False, ewc=ewc)
            stage5_err = rmse(np.asarray(mlp.forward(X5_ev)).reshape(-1), Y5_ev) / W17_BETA_MUTATED.value_range
            _, _, _, _, Xa_ev, Ya_ev = stage_data["W17_alpha"]
            ret_alpha_post5 = rmse(np.asarray(mlp.forward(Xa_ev)).reshape(-1), Ya_ev) / W17_ALPHA.value_range
            n_params_final = mlp.num_parameters
            raw_replay_count = int(sum(len(x) for x in replay_X))
        else:
            raise ValueError(arch)

        mean_ret = float(np.mean(list(ret_post4.values())))
        forgetting_ab = float(0.5 * ((ret_post4["W17_alpha"] - imm["W17_alpha"]) + (ret_post4["W17_beta"] - imm["W17_beta"])))
        vals = [
            imm["W17_alpha"], imm["W17_beta"], imm["W17_gamma_shared"], imm["W17_delta_inverted"],
            mean_ret, forgetting_ab, stage5_err, ret_alpha_post5,
        ]
        rows.append({
            "arch": arch,
            "seed": int(seed),
            "is_oracle_diagnostic": False,
            "n_params_final": int(n_params_final),
            "raw_replay_stored": int(raw_replay_count),
            "immediate_alpha": round(float(imm["W17_alpha"]), 5),
            "immediate_beta": round(float(imm["W17_beta"]), 5),
            "forward_transfer_gamma": round(float(imm["W17_gamma_shared"]), 5),
            "immediate_delta": round(float(imm["W17_delta_inverted"]), 5),
            "ret_alpha_post4": round(float(ret_post4["W17_alpha"]), 5),
            "ret_beta_post4": round(float(ret_post4["W17_beta"]), 5),
            "ret_gamma_post4": round(float(ret_post4["W17_gamma_shared"]), 5),
            "ret_delta_post4": round(float(ret_post4["W17_delta_inverted"]), 5),
            "mean_ret_post4": round(float(mean_ret), 5),
            "mean_forgetting_alpha_beta": round(float(forgetting_ab), 5),
            "retrieval_correct_post4": int(retrieval_hits),
            "retrieval_total_post4": 4,
            "retrieval_acc_post4": round(float(retrieval_hits / 4.0), 4),
            "outdated_detected_stage5": bool(outdated_det),
            "stage5_beta_mutated_nrmse": round(float(stage5_err), 5),
            "ret_alpha_post5": round(float(ret_alpha_post5), 5),
            "finite": bool(all(np.isfinite(vals))),
            "leak_rows": 0,
        })
    return rows


def _worker_p17(args: Tuple[int, Tuple[str, ...]]):
    s, archs = args
    return run_seed_p17(s, archs=archs)


def run_phase17_suite(
    seeds=SEEDS17,
    archs=ARCHS17,
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    for spec in ALL_STAGES_17:
        verify_gridworld_semantic_purity_p17(spec, seed=seeds[0])

    workers = max_workers or min(len(seeds), os.cpu_count() or 1)
    tasks = [(s, tuple(archs)) for s in seeds]
    rows: List[Dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for s, s_rows in zip(seeds, ex.map(_worker_p17, tasks)):
                rows.extend(s_rows)
                if verbose:
                    pm = next(r for r in s_rows if r["arch"] == "persistent_memory")
                    ms = next(r for r in s_rows if r["arch"] == "memoryless_structural")
                    print(
                        f"seed {s}: pm_ret={pm['mean_ret_post4']:.4f} (retr={pm['retrieval_correct_post4']}/4) "
                        f"gamma={pm['forward_transfer_gamma']:.4f} vs scratch={ms['forward_transfer_gamma']:.4f} "
                        f"st5={pm['stage5_beta_mutated_nrmse']:.4f} (outdated={pm['outdated_detected_stage5']})",
                        flush=True,
                    )
    else:
        for t in tasks:
            rows.extend(_worker_p17(t))
    return rows
