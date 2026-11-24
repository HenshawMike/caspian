"""Phase 18: Integrated Predictive World Model & Autonomous Planning (CSP-P18-E001).

Implements:
1. Phase18World: controlled neutral dynamical environment with 8D sensor observations,
   4D hidden latent state, 5 discrete actions (NOOP passive drift + 4 sparse causal
   interventions), non-additive multi-step payoff basins, and Semantic Firewall verification.
2. Fair baselines: Persistence, Random, Reactive (H=1), Memory-No-Planner (H=1),
   Unstructured MLP World Model + MPC, Observational-Only World Model, Risk-Neutral
   Caspian (beta=0), and Diagnostic Oracle Planner (strictly isolated in evaluator).
3. Full 7-hypothesis experimental suite (P18-H1 through P18-H7) with paired bootstrap CI,
   exact/MC permutation tests, Holm-Bonferroni correction, AST leakage audit, and
   exact determinism verification.
"""
import ast
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from environment.observations import EntityObservation, Observation
from models.mlp import PredictiveMLP
from models.optimizers import Adam
from models.planner import AutonomousPlanner
from models.world_model import CaspianWorldModel

IN_DIM18 = 8
N_LATENT18 = 4
N_ACTIONS18 = 5
SMOKE_SEEDS18 = [9201, 9202, 9203]
CONFIRMATORY_SEEDS18 = list(range(171, 186))  # 15 disjoint seeds (171..185)


def _ortho_mix18(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((IN_DIM18, N_LATENT18)))
    return Q[:, :N_LATENT18].T.copy()  # (4, 8)


M_ALPHA = _ortho_mix18(1801)
M_BETA = _ortho_mix18(1802)
M_GAMMA_SHIFTED = _ortho_mix18(1803)
OFFSET_GAMMA = np.full((1, IN_DIM18), 0.5, dtype=float)

A_DRIFT_ALPHA = np.array([
    [0.85, 0.00, 0.00, 0.00],
    [0.15, 0.85, 0.00, 0.00],
    [0.00, 0.00, 0.85, 0.00],
    [0.00, 0.00, 0.15, 0.85],
], dtype=float)

B_ACT_ALPHA = np.array([
    [ 0.00,  0.00,  0.00,  0.00],  # a=0: NOOP (passive drift)
    [ 0.85,  0.00,  0.00,  0.00],  # a=1: +s0
    [ 0.00,  0.85,  0.00,  0.00],  # a=2: +s1
    [-0.85,  0.00,  0.80,  0.00],  # a=3: -s0, +s2 (greedy trap from s0 < 0!)
    [ 0.00,  0.00,  0.00,  0.80],  # a=4: +s3
], dtype=float)

B_ACT_BETA = np.array([
    [ 0.00,  0.00,  0.00,  0.00],
    [ 0.00,  0.00,  0.85,  0.00],
    [ 0.00,  0.00,  0.00,  0.85],
    [ 0.80,  0.00, -0.85,  0.00],
    [ 0.00,  0.80,  0.00,  0.00],
], dtype=float)

ACTION_COSTS_DEFAULT = np.array([0.0, 0.25, 0.25, 0.25, 0.25], dtype=float)


def _rule_alpha(s: np.ndarray) -> np.ndarray:
    """Non-additive saddle-crossing payoff: 1-step greedy goes negative (a=3),
    while 3-step planning crosses to s0 > 0, s1 > 0 for +5.0 * s0 * s1 synergy.
    """
    return (
        -1.5 * s[:, 0]
        + 2.0 * s[:, 1]
        + 5.0 * s[:, 0] * s[:, 1]
        + 1.5 * s[:, 2]
        + 1.0 * s[:, 3]
        - 0.5 * (s[:, 0] ** 2 + s[:, 1] ** 2)
    )


def _rule_beta(s: np.ndarray) -> np.ndarray:
    return (
        1.5 * s[:, 0]
        + 1.0 * s[:, 1]
        - 1.5 * s[:, 2]
        + 2.0 * s[:, 3]
        + 5.0 * s[:, 2] * s[:, 3]
        - 0.5 * (s[:, 2] ** 2 + s[:, 3] ** 2)
    )


def _rule_alpha_mutated(s: np.ndarray) -> np.ndarray:
    return -_rule_alpha(s)


def _rule_trap(s: np.ndarray) -> np.ndarray:
    """High-uncertainty trap world for H4:
    Smooth quadratic increase in s0 <= 0.5, but sharp hidden cliff for s0 > 0.75.
    A risk-neutral extrapolator walks into s0 > 0.75; risk-aware Bayesian MPC avoids it.
    """
    base = 2.5 * s[:, 0] + 2.0 * s[:, 1] + 1.5 * s[:, 0] ** 2 - 0.6 * s[:, 1] ** 2 + 1.0 * s[:, 2]
    cliff = np.where(s[:, 0] > 0.75, -18.0 * (s[:, 0] - 0.75), 0.0)
    return base + cliff


@dataclass
class Phase18WorldSpec:
    name: str
    M: np.ndarray
    offset: np.ndarray
    A_drift: np.ndarray
    B_act: np.ndarray
    rule_fn: Callable[[np.ndarray], np.ndarray]
    action_costs: np.ndarray
    noise_s: float = 0.05
    noise_e: float = 0.15
    state_bound: float = 1.8

    def true_causal_action_mask(self) -> np.ndarray:
        return (np.abs(self.B_act) > 0.1).astype(float)

    def observe(self, s: np.ndarray) -> np.ndarray:
        s_arr = np.asarray(s, dtype=float)
        if s_arr.ndim == 1:
            return (s_arr.reshape(1, -1) @ self.M + self.offset).reshape(-1)
        return s_arr @ self.M + self.offset

    def step_true(
        self, s: np.ndarray, a: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, float]:
        s_vec = np.asarray(s, dtype=float).reshape(-1)
        noise_s = rng.normal(0.0, self.noise_s, size=N_LATENT18) if rng is not None else np.zeros(N_LATENT18)
        noise_e = float(rng.normal(0.0, self.noise_e)) if rng is not None else 0.0
        s_next = np.clip(self.A_drift @ s_vec + self.B_act[int(a)] + noise_s, -self.state_bound, self.state_bound)
        de = float(self.rule_fn(s_next.reshape(1, -1))[0]) - float(self.action_costs[int(a)]) + noise_e
        return s_next, de

    def make_neutral_observation(self, s: np.ndarray, energy: float, t: int, prev_a: Optional[int]) -> Observation:
        x = self.observe(s)
        obs = Observation(
            agent_position=(0, 0),
            entities=(EntityObservation(entity_id=1, entity_type=0, relative_x=0, relative_y=0, distance=0.0, properties=tuple(float(v) for v in x)),),
            collision=False,
            internal_state=float(energy),
            timestep=int(t),
            previous_action=prev_a,
        )
        obs.assert_semantic_purity()
        return obs

    def collect_transitions(
        self,
        n_steps: int,
        seed: int,
        s_clip_train: Optional[Tuple[float, float]] = None,
        action_probs: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        X_t = np.zeros((n_steps, IN_DIM18), dtype=float)
        A_t = np.zeros(n_steps, dtype=int)
        X_n = np.zeros((n_steps, IN_DIM18), dtype=float)
        dE = np.zeros(n_steps, dtype=float)
        S_t = np.zeros((n_steps, N_LATENT18), dtype=float)
        S_n = np.zeros((n_steps, N_LATENT18), dtype=float)

        s = rng.uniform(-1.2, 1.2, size=N_LATENT18)
        if s_clip_train is not None:
            s[0] = np.clip(s[0], s_clip_train[0], s_clip_train[1])

        for i in range(n_steps):
            if i % 12 == 0:
                s = rng.uniform(-1.3, 1.3, size=N_LATENT18)
                if s_clip_train is not None:
                    s[0] = rng.uniform(s_clip_train[0], s_clip_train[1])
            if action_probs is not None:
                a = int(rng.choice(N_ACTIONS18, p=action_probs))
            else:
                a = int(rng.integers(0, N_ACTIONS18))
            if s_clip_train is not None and a == 1 and s[0] > s_clip_train[1] - 0.5:
                a = int(rng.choice([0, 2, 3, 4]))
            s_next, de = self.step_true(s, a, rng=rng)
            if s_clip_train is not None and s_next[0] > s_clip_train[1]:
                s_next[0] = s_clip_train[1]
                de = float(self.rule_fn(s_next.reshape(1, -1))[0]) - float(self.action_costs[a]) + float(rng.normal(0.0, self.noise_e))
            S_t[i] = s
            S_n[i] = s_next
            X_t[i] = self.observe(s)
            X_n[i] = self.observe(s_next)
            A_t[i] = a
            dE[i] = de
            s = s_next

        return {"X_t": X_t, "A_t": A_t, "X_next": X_n, "dE": dE, "S_t": S_t, "S_next": S_n}


W18_ALPHA = Phase18WorldSpec(
    name="W18_alpha", M=M_ALPHA, offset=np.zeros((1, IN_DIM18)),
    A_drift=A_DRIFT_ALPHA, B_act=B_ACT_ALPHA, rule_fn=_rule_alpha, action_costs=ACTION_COSTS_DEFAULT,
)
W18_BETA = Phase18WorldSpec(
    name="W18_beta", M=M_BETA, offset=np.zeros((1, IN_DIM18)),
    A_drift=A_DRIFT_ALPHA, B_act=B_ACT_BETA, rule_fn=_rule_beta, action_costs=ACTION_COSTS_DEFAULT,
)
W18_GAMMA_SHIFTED = Phase18WorldSpec(
    name="W18_gamma_shifted", M=M_GAMMA_SHIFTED, offset=OFFSET_GAMMA,
    A_drift=A_DRIFT_ALPHA, B_act=B_ACT_ALPHA, rule_fn=_rule_alpha, action_costs=ACTION_COSTS_DEFAULT,
)
W18_ALPHA_MUTATED = Phase18WorldSpec(
    name="W18_alpha_mutated", M=M_ALPHA, offset=np.zeros((1, IN_DIM18)),
    A_drift=A_DRIFT_ALPHA, B_act=B_ACT_ALPHA, rule_fn=_rule_alpha_mutated, action_costs=ACTION_COSTS_DEFAULT,
)
W18_TRAP = Phase18WorldSpec(
    name="W18_trap", M=M_ALPHA, offset=np.zeros((1, IN_DIM18)),
    A_drift=A_DRIFT_ALPHA, B_act=B_ACT_ALPHA, rule_fn=_rule_trap, action_costs=ACTION_COSTS_DEFAULT,
)


# ---------------------------------------------------------------------------
# Fair Baselines (Unstructured MLP World Model, Reactive, Oracle Upper Bound)
# ---------------------------------------------------------------------------

class MLPWorldModelBaseline:
    """Parameter- and data-budget-matched unstructured PredictiveMLP world model + MPC."""

    def __init__(self, in_dim: int = IN_DIM18, n_actions: int = N_ACTIONS18, seed: int = 0):
        self.in_dim = in_dim
        self.n_actions = n_actions
        self.model = PredictiveMLP(
            input_dim=in_dim + n_actions,
            hidden_dims=(24, 16),
            output_dim=in_dim + 1,  # [dx (8D), dE (1D)]
            hidden_activation="tanh",
            seed=seed,
        )
        self.y_mean = np.zeros((1, in_dim + 1), dtype=float)
        self.y_std = np.ones((1, in_dim + 1), dtype=float)

    def _encode_xa(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        A_arr = np.asarray(A, dtype=int).reshape(-1)
        oh = np.zeros((len(X_arr), self.n_actions), dtype=float)
        oh[np.arange(len(X_arr)), A_arr] = 1.0
        return np.hstack([X_arr, oh])

    def fit(self, X_t: np.ndarray, A_t: np.ndarray, X_next: np.ndarray, dE: np.ndarray, epochs: int = 120, lr: float = 0.015) -> None:
        X_in = self._encode_xa(X_t, A_t)
        Y_raw = np.hstack([X_next - X_t, dE.reshape(-1, 1)])
        self.model.fit_normalizer(X_in)
        self.y_mean = np.mean(Y_raw, axis=0, keepdims=True)
        self.y_std = np.maximum(np.std(Y_raw, axis=0, keepdims=True), 1e-4)
        Y_n = (Y_raw - self.y_mean) / self.y_std

        opt = Adam(self.model.get_params(), lr=lr)
        N = len(X_in)
        for _ in range(epochs):
            self.model.zero_grad()
            pred = self.model.forward(X_in)
            grad = 2.0 * (pred - Y_n) / max(N, 1)
            self.model.backward(grad)
            opt.step()

    def predict_step(self, X_t: np.ndarray, A_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X_t, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_in = self._encode_xa(X_arr, A_t)
        pred_n = self.model.forward(X_in)
        pred = pred_n * self.y_std + self.y_mean
        return X_arr + pred[:, :self.in_dim], pred[:, self.in_dim]

    def rollout(self, X_0: np.ndarray, seqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seqs = np.asarray(seqs, dtype=int)
        if seqs.ndim == 1:
            seqs = seqs.reshape(1, -1)
        N, H = seqs.shape
        X_cur = np.asarray(X_0, dtype=float)
        if X_cur.ndim == 1:
            X_cur = np.tile(X_cur.reshape(1, -1), (N, 1))
        x_traj = np.zeros((N, H + 1, self.in_dim), dtype=float)
        de_traj = np.zeros((N, H), dtype=float)
        x_traj[:, 0, :] = X_cur
        for h in range(H):
            X_cur, de_h = self.predict_step(X_cur, seqs[:, h])
            x_traj[:, h + 1, :] = X_cur
            de_traj[:, h] = de_h
        return x_traj, de_traj

    def plan_action(self, x_t: np.ndarray, horizon: int = 3, gamma: float = 0.95) -> int:
        seqs = np.array(list(itertools.product(range(self.n_actions), repeat=horizon)), dtype=int)
        _, de_traj = self.rollout(x_t, seqs)
        disc = np.array([gamma ** h for h in range(horizon)], dtype=float)
        scores = de_traj @ disc
        return int(seqs[int(np.argmax(scores)), 0])


def oracle_plan_action(spec: Phase18WorldSpec, s_t: np.ndarray, horizon: int = 3, gamma: float = 0.95) -> int:
    """DIAGNOSTIC UPPER BOUND ONLY — uses true simulator dynamics; never accessible to Caspian."""
    seqs = np.array(list(itertools.product(range(N_ACTIONS18), repeat=horizon)), dtype=int)
    N = len(seqs)
    S_cur = np.tile(np.asarray(s_t, dtype=float).reshape(1, -1), (N, 1))
    scores = np.zeros(N, dtype=float)
    for h in range(horizon):
        acts = seqs[:, h]
        S_cur = np.clip(S_cur @ spec.A_drift.T + spec.B_act[acts], -spec.state_bound, spec.state_bound)
        de_h = spec.rule_fn(S_cur) - spec.action_costs[acts]
        scores += (gamma ** h) * de_h
    return int(seqs[int(np.argmax(scores)), 0])


# ---------------------------------------------------------------------------
# Statistical Helpers (Paired Bootstrap CI, Exact/MC Permutation Test, Holm, AUROC)
# ---------------------------------------------------------------------------

def paired_stats(a: np.ndarray, b: np.ndarray, higher_is_better: bool = True, seed: int = 0) -> Dict[str, Any]:
    """Paired comparison of candidate `a` vs baseline `b`. Positive delta = candidate better."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = (a - b) if higher_is_better else (b - a)
    n = len(diff)
    mean_d = float(np.mean(diff))
    sd_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    dz = float(mean_d / sd_d) if sd_d > 1e-12 else 0.0

    rng = np.random.default_rng(seed + 1800)
    # Paired bootstrap 95% CI
    idx = rng.integers(0, n, size=(5000, n))
    boot_means = np.mean(diff[idx], axis=1)
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    # Two-sided paired sign-flip permutation test (10,000 draws)
    signs = rng.choice([-1.0, 1.0], size=(10000, n))
    perm_means = np.mean(signs * diff.reshape(1, -1), axis=1)
    p_val = float((np.sum(np.abs(perm_means) >= abs(mean_d) - 1e-12) + 1) / (10000 + 1))

    return {
        "n": n,
        "mean_delta": round(mean_d, 5),
        "sd_delta": round(sd_d, 5),
        "ci95_low": round(ci_lo, 5),
        "ci95_high": round(ci_hi, 5),
        "cohens_dz": round(dz, 4),
        "perm_p": round(p_val, 5),
        "wins": int(np.sum(diff > 1e-12)),
        "ci_strictly_positive": bool(ci_lo > 0.0),
    }


def holm_bonferroni(p_map: Dict[str, float]) -> Dict[str, float]:
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {}
    running_max = 0.0
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        out[k] = round(running_max, 5)
    return out


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos.reshape(-1, 1) - neg.reshape(1, -1)
    return float(np.mean((diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    sx, sy = np.std(rx), np.std(ry)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def ast_leakage_audit_phase18() -> Dict[str, Any]:
    """Static AST audit across models/ ensuring zero oracle / world_id / ground-truth leakage."""
    forbidden = {
        "world_id", "s_true", "M_true", "A_drift", "B_act", "oracle",
        "step_true", "hidden_state_delta", "rule_fn", "ground_truth",
    }
    files = sorted(Path("models").glob("*.py"))
    violations: List[str] = []
    for fpath in files:
        tree = ast.parse(fpath.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden:
                violations.append(f"{fpath}:{node.lineno}: Name '{node.id}'")
            elif isinstance(node, ast.Attribute) and node.attr in forbidden:
                violations.append(f"{fpath}:{node.lineno}: Attr '{node.attr}'")
    return {
        "files_scanned": [str(f) for f in files],
        "violations": violations,
        "passed": len(violations) == 0,
    }


# ---------------------------------------------------------------------------
# Episode Rollout Runner
# ---------------------------------------------------------------------------

def run_episode(
    spec: Phase18WorldSpec,
    policy_fn: Callable[[np.ndarray, np.ndarray, int], int],
    s_init: np.ndarray,
    steps: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    s = np.asarray(s_init, dtype=float).copy()
    total_de = 0.0
    traj_de: List[float] = []
    for t in range(steps):
        x = spec.observe(s)
        a = int(policy_fn(x, s, t))
        s, de = spec.step_true(s, a, rng=rng)
        total_de += de
        traj_de.append(de)
    return {"total_return": float(total_de), "traj_de": traj_de, "final_s": s.tolist()}


# ---------------------------------------------------------------------------
# Per-Seed Complete Evaluation (H1 through H7)
# ---------------------------------------------------------------------------

def run_single_seed(seed: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # Verify Semantic Firewall on neutral Observation
    W18_ALPHA.make_neutral_observation(np.zeros(N_LATENT18), 10.0, 0, None)

    # Collect training transitions on W18_alpha (250 transitions)
    tr_alpha = W18_ALPHA.collect_transitions(n_steps=250, seed=seed * 100 + 1)

    # Fit Integrated Caspian World Model + Baselines on exact same data
    cwm = CaspianWorldModel(in_dim=IN_DIM18, n_latent=N_LATENT18, n_actions=N_ACTIONS18, seed=seed)
    eid_alpha = cwm.observe_transitions(tr_alpha["X_t"], tr_alpha["A_t"], tr_alpha["X_next"], tr_alpha["dE"])

    mlp_wm = MLPWorldModelBaseline(seed=seed)
    mlp_wm.fit(tr_alpha["X_t"], tr_alpha["A_t"], tr_alpha["X_next"], tr_alpha["dE"])

    obs_only = CaspianWorldModel(in_dim=IN_DIM18, n_latent=N_LATENT18, n_actions=N_ACTIONS18, seed=seed)
    obs_only.observe_transitions(tr_alpha["X_t"], tr_alpha["A_t"], tr_alpha["X_next"], tr_alpha["dE"], only_noop_obs=True)

    # --- H1: 1-step & multi-step (H=1,2,3,5) open-loop prediction ---
    n_roll = 30
    H_max = 5
    x_scale = float(np.std(tr_alpha["X_t"])) + 1e-8
    de_scale = float(np.std(tr_alpha["dE"])) + 1e-8
    err_cwm_x = {h: [] for h in (1, 2, 3, 5)}
    err_mlp_x = {h: [] for h in (1, 2, 3, 5)}
    err_pers_x = {h: [] for h in (1, 2, 3, 5)}
    err_cwm_de = {h: [] for h in (1, 2, 3, 5)}
    err_mlp_de = {h: [] for h in (1, 2, 3, 5)}

    for r_i in range(n_roll):
        s_cur = rng.uniform(-1.1, 1.1, size=N_LATENT18)
        x0 = W18_ALPHA.observe(s_cur)
        acts = rng.integers(0, N_ACTIONS18, size=H_max)
        # Ground truth open-loop rollout (noise-free deterministic expectation for clean structural eval)
        true_x = [x0]
        true_de = []
        s_tmp = s_cur.copy()
        for h in range(H_max):
            s_tmp, de_h = W18_ALPHA.step_true(s_tmp, int(acts[h]), rng=None)
            true_x.append(W18_ALPHA.observe(s_tmp))
            true_de.append(de_h)

        cwm_roll = cwm.rollout_latent(x0, acts.reshape(1, -1), eid=eid_alpha)
        mlp_x, mlp_de = mlp_wm.rollout(x0, acts.reshape(1, -1))
        for h in (1, 2, 3, 5):
            err_cwm_x[h].append(float(np.mean((cwm_roll["x_traj"][0, h] - true_x[h]) ** 2)))
            err_mlp_x[h].append(float(np.mean((mlp_x[0, h] - true_x[h]) ** 2)))
            err_pers_x[h].append(float(np.mean((x0 - true_x[h]) ** 2)))
            err_cwm_de[h].append(float((cwm_roll["de_traj"][0, h - 1] - true_de[h - 1]) ** 2))
            err_mlp_de[h].append(float((mlp_de[0, h - 1] - true_de[h - 1]) ** 2))

    h1_metrics = {
        f"cwm_x_nrmse_h{h}": float(np.sqrt(np.mean(err_cwm_x[h])) / x_scale) for h in (1, 2, 3, 5)
    }
    h1_metrics.update({
        f"mlp_x_nrmse_h{h}": float(np.sqrt(np.mean(err_mlp_x[h])) / x_scale) for h in (1, 2, 3, 5)
    })
    h1_metrics.update({
        f"pers_x_nrmse_h{h}": float(np.sqrt(np.mean(err_pers_x[h])) / x_scale) for h in (1, 2, 3, 5)
    })
    h1_metrics["cwm_de_nrmse_h1"] = float(np.sqrt(np.mean(err_cwm_de[1])) / de_scale)
    h1_metrics["mlp_de_nrmse_h1"] = float(np.sqrt(np.mean(err_mlp_de[1])) / de_scale)

    # --- H2: Causal Action Prediction (Interventional do(a) vs Observational-Only) ---
    # Compare discovered causal action mask against true causal action effects (aligned to latent basis!)
    # Note: W_adapter aligns latent coordinates up to signed permutation (Phase 13 kurtosis unmixing).
    # We evaluate causal F1 by projecting discovered action-delta vectors into observation space vs true B_act @ M!
    dyn_a = cwm.dynamics[eid_alpha]
    e_a = cwm._get_engram(eid_alpha)
    W_a = e_a.model.W_adapter  # (8, 4)
    # Map true B_act into the model's canonical latent basis: B_true_z = (B_act @ M) @ W_adapter
    B_true_z = (W18_ALPHA.B_act @ W18_ALPHA.M) @ W_a  # (5, 4)
    true_mask_z = (np.abs(B_true_z[1:]) > 0.20).astype(int)
    pred_mask_z = dyn_a.causal_action_mask[1:].astype(int)
    tp = int(np.sum((true_mask_z == 1) & (pred_mask_z == 1)))
    fp = int(np.sum((true_mask_z == 0) & (pred_mask_z == 1)))
    fn = int(np.sum((true_mask_z == 1) & (pred_mask_z == 0)))
    causal_f1 = float((2 * tp) / max(2 * tp + fp + fn, 1))

    # Interventional transition nRMSE on active interventions a in {1..4}
    te_int = W18_ALPHA.collect_transitions(n_steps=80, seed=seed * 100 + 2, action_probs=np.array([0.0, 0.25, 0.25, 0.25, 0.25]))
    pred_cwm_int = cwm.predict_step(te_int["X_t"], te_int["A_t"], eid=eid_alpha)["x_next"]
    pred_obs_int = obs_only.predict_step(te_int["X_t"], te_int["A_t"])["x_next"]
    h2_metrics = {
        "causal_action_f1": causal_f1,
        "cwm_interventional_nrmse": float(np.sqrt(np.mean((pred_cwm_int - te_int["X_next"]) ** 2)) / x_scale),
        "obs_only_interventional_nrmse": float(np.sqrt(np.mean((pred_obs_int - te_int["X_next"]) ** 2)) / x_scale),
    }

    # --- H3: Counterfactual Action Evaluation (Zero Environment Steps) ---
    cf_rhos: List[float] = []
    cf_top1: List[float] = []
    cf_sq_err: List[float] = []
    for _ in range(40):
        s_q = rng.uniform(-1.2, 1.2, size=N_LATENT18)
        x_q = W18_ALPHA.observe(s_q)
        # Observed factual step with random action (for Pearl Abduction-Action-Prediction)
        a_obs = int(rng.integers(0, N_ACTIONS18))
        noise_s_fixed = rng.normal(0.0, W18_ALPHA.noise_s, size=N_LATENT18)
        noise_e_fixed = float(rng.normal(0.0, W18_ALPHA.noise_e))
        s_obs_n = np.clip(W18_ALPHA.A_drift @ s_q + W18_ALPHA.B_act[a_obs] + noise_s_fixed, -W18_ALPHA.state_bound, W18_ALPHA.state_bound)
        de_obs = float(W18_ALPHA.rule_fn(s_obs_n.reshape(1, -1))[0]) - float(W18_ALPHA.action_costs[a_obs]) + noise_e_fixed

        # Ground-truth counterfactuals under SAME exogenous noise (computed strictly in evaluator!)
        true_cf_de = np.zeros(N_ACTIONS18, dtype=float)
        for a_alt in range(N_ACTIONS18):
            s_alt = np.clip(W18_ALPHA.A_drift @ s_q + W18_ALPHA.B_act[a_alt] + noise_s_fixed, -W18_ALPHA.state_bound, W18_ALPHA.state_bound)
            true_cf_de[a_alt] = float(W18_ALPHA.rule_fn(s_alt.reshape(1, -1))[0]) - float(W18_ALPHA.action_costs[a_alt]) + noise_e_fixed

        pred_cf = cwm.evaluate_counterfactuals(x_q, observed_action=a_obs, X_next_obs=W18_ALPHA.observe(s_obs_n), dE_obs=de_obs, eid=eid_alpha)["de_mean"]
        cf_rhos.append(spearman_corr(pred_cf, true_cf_de))
        cf_top1.append(1.0 if int(np.argmax(pred_cf)) == int(np.argmax(true_cf_de)) else 0.0)
        cf_sq_err.append(float(np.mean((pred_cf - true_cf_de) ** 2)))

    h3_metrics = {
        "cf_spearman_rho": float(np.mean(cf_rhos)),
        "cf_top1_acc": float(np.mean(cf_top1)),
        "cf_nrmse": float(np.sqrt(np.mean(cf_sq_err)) / de_scale),
    }

    # --- H4: Uncertainty Calibration, OOD Detection & Safe Planning in Trap World ---
    # 4a. OOD AUROC: in-distribution W18_alpha vs out-of-subspace W18_beta observations
    te_id = W18_ALPHA.collect_transitions(n_steps=60, seed=seed * 100 + 3)
    te_ood = W18_BETA.collect_transitions(n_steps=60, seed=seed * 100 + 4)
    ood_scores = np.concatenate([
        cwm.subspace_ood_score(te_id["X_t"], eid=eid_alpha),
        cwm.subspace_ood_score(te_ood["X_t"], eid=eid_alpha),
    ])
    ood_labels = np.concatenate([np.zeros(60, dtype=int), np.ones(60, dtype=int)])
    ood_auroc = compute_auroc(ood_scores, ood_labels)

    # 4b. Calibration & Trap-World Risk-Aware Planning (beta > 0 vs beta = 0)
    tr_trap = W18_TRAP.collect_transitions(n_steps=200, seed=seed * 100 + 5, s_clip_train=(-1.3, 0.45))
    cwm_trap = CaspianWorldModel(in_dim=IN_DIM18, n_latent=N_LATENT18, n_actions=N_ACTIONS18, seed=seed)
    eid_trap = cwm_trap.observe_transitions(tr_trap["X_t"], tr_trap["A_t"], tr_trap["X_next"], tr_trap["dE"])

    # Evaluate error-uncertainty correlation across both in-domain (s0 <= 0.45) and extrapolated (s0 in [0.5, 1.5])
    te_cal = W18_TRAP.collect_transitions(n_steps=100, seed=seed * 100 + 6, s_clip_train=None)
    pred_cal = cwm_trap.predict_step(te_cal["X_t"], te_cal["A_t"], eid=eid_trap)
    abs_err_cal = np.abs(pred_cal["de_mean"] - te_cal["dE"])
    pred_std_cal = pred_cal["de_std"]
    err_std_corr = float(np.corrcoef(pred_std_cal, abs_err_cal)[0, 1]) if np.std(pred_std_cal) > 1e-9 else 0.0
    # 95% PI coverage on in-distribution transitions
    te_id_trap = W18_TRAP.collect_transitions(n_steps=80, seed=seed * 100 + 7, s_clip_train=(-1.2, 0.45))
    pred_id_trap = cwm_trap.predict_step(te_id_trap["X_t"], te_id_trap["A_t"], eid=eid_trap)
    pi95_cov = float(np.mean(np.abs(pred_id_trap["de_mean"] - te_id_trap["dE"]) <= 1.96 * pred_id_trap["de_std"]))

    planner_safe = AutonomousPlanner(cwm_trap, horizon=3, gamma=0.95, beta_uncertainty=2.0)
    planner_risk_neutral = AutonomousPlanner(cwm_trap, horizon=3, gamma=0.95, beta_uncertainty=0.0)
    s_trap_init = np.array([0.2, 0.2, 0.0, 0.0], dtype=float)
    ret_safe = np.mean([
        run_episode(W18_TRAP, lambda x, s, t: planner_safe.plan_action(x, eid=eid_trap)[0], s_trap_init, steps=15, seed=seed * 1000 + ep)["total_return"]
        for ep in range(5)
    ])
    ret_neutral = np.mean([
        run_episode(W18_TRAP, lambda x, s, t: planner_risk_neutral.plan_action(x, eid=eid_trap)[0], s_trap_init, steps=15, seed=seed * 1000 + ep)["total_return"]
        for ep in range(5)
    ])

    h4_metrics = {
        "ood_auroc": float(ood_auroc),
        "err_uncertainty_corr": float(err_std_corr),
        "pi95_coverage": float(pi95_cov),
        "trap_return_risk_aware": float(ret_safe),
        "trap_return_risk_neutral": float(ret_neutral),
    }

    # --- H5: Autonomous Multi-Step Planning vs Baselines & Ablations ---
    planner_caspian = AutonomousPlanner(cwm, horizon=3, gamma=0.95, beta_uncertainty=0.5)
    planner_mem_greedy = AutonomousPlanner(cwm, horizon=1, gamma=0.95, beta_uncertainty=0.0)

    # Start in saddle-crossing initial states where 1-step greedy gets trapped in a=3!
    init_states = [
        np.array([-1.0, -0.2, 0.0, 0.0], dtype=float),
        np.array([-0.9, -0.1, -0.2, 0.1], dtype=float),
        np.array([-1.1, -0.3, 0.1, -0.1], dtype=float),
        np.array([-0.8, -0.2, 0.0, 0.0], dtype=float),
        np.array([-1.0, -0.1, -0.1, 0.0], dtype=float),
    ]

    rets = {k: [] for k in ["integrated_caspian", "oracle_planner", "mlp_world_model", "memory_no_planner", "reactive_policy", "random_policy"]}
    for ep_i, s0 in enumerate(init_states):
        ep_seed = seed * 2000 + ep_i
        rng_rand = np.random.default_rng(ep_seed + 99)
        rets["integrated_caspian"].append(run_episode(W18_ALPHA, lambda x, s, t: planner_caspian.plan_action(x, eid=eid_alpha)[0], s0, steps=20, seed=ep_seed)["total_return"])
        rets["oracle_planner"].append(run_episode(W18_ALPHA, lambda x, s, t: oracle_plan_action(W18_ALPHA, s, horizon=3), s0, steps=20, seed=ep_seed)["total_return"])
        rets["mlp_world_model"].append(run_episode(W18_ALPHA, lambda x, s, t: mlp_wm.plan_action(x, horizon=3), s0, steps=20, seed=ep_seed)["total_return"])
        rets["memory_no_planner"].append(run_episode(W18_ALPHA, lambda x, s, t: planner_mem_greedy.plan_action(x, eid=eid_alpha)[0], s0, steps=20, seed=ep_seed)["total_return"])
        rets["reactive_policy"].append(run_episode(W18_ALPHA, lambda x, s, t: mlp_wm.plan_action(x, horizon=1), s0, steps=20, seed=ep_seed)["total_return"])
        rets["random_policy"].append(run_episode(W18_ALPHA, lambda x, s, t: int(rng_rand.integers(0, N_ACTIONS18)), s0, steps=20, seed=ep_seed)["total_return"])

    casp_mean = float(np.mean(rets["integrated_caspian"]))
    orac_mean = float(np.mean(rets["oracle_planner"]))
    h5_metrics = {f"return_{k}": float(np.mean(v)) for k, v in rets.items()}
    h5_metrics["oracle_ratio"] = float(casp_mean / max(orac_mean, 1e-6))
    h5_metrics["planning_success_rate"] = float(np.mean([
        1.0 if r >= 0.75 * r_o else 0.0 for r, r_o in zip(rets["integrated_caspian"], rets["oracle_planner"])
    ]))

    # --- H6: Persistent Knowledge Integration, Continual Learning & Mid-Episode Mutation Recovery ---
    # Measure W18_alpha prediction error BEFORE learning W18_beta
    te_alpha_ret = W18_ALPHA.collect_transitions(n_steps=60, seed=seed * 100 + 8)
    pred_alpha_pre = cwm.predict_step(te_alpha_ret["X_t"], te_alpha_ret["A_t"], eid=eid_alpha)["de_mean"]
    err_alpha_pre = float(np.sqrt(np.mean((pred_alpha_pre - te_alpha_ret["dE"]) ** 2)) / de_scale)

    # Continually learn W18_beta
    tr_beta = W18_BETA.collect_transitions(n_steps=250, seed=seed * 100 + 9)
    eid_beta = cwm.observe_transitions(tr_beta["X_t"], tr_beta["A_t"], tr_beta["X_next"], tr_beta["dE"])

    # Zero-world_id autonomous retrieval test on W18_alpha and W18_beta
    probe_a = W18_ALPHA.collect_transitions(n_steps=8, seed=seed * 100 + 10)
    probe_b = W18_BETA.collect_transitions(n_steps=8, seed=seed * 100 + 11)
    ret_eid_a, _, _ = cwm.retrieve_context(probe_a["X_t"], probe_a["A_t"], probe_a["X_next"], probe_a["dE"])
    ret_eid_b, _, _ = cwm.retrieve_context(probe_b["X_t"], probe_b["A_t"], probe_b["X_next"], probe_b["dE"])
    retr_correct = int(ret_eid_a == eid_alpha) + int(ret_eid_b == eid_beta)

    # Measure W18_alpha retention AFTER learning W18_beta
    pred_alpha_post = cwm.predict_step(te_alpha_ret["X_t"], te_alpha_ret["A_t"], eid=ret_eid_a)["de_mean"]
    err_alpha_post = float(np.sqrt(np.mean((pred_alpha_post - te_alpha_ret["dE"]) ** 2)) / de_scale)
    forgetting_alpha = max(0.0, err_alpha_post - err_alpha_pre)

    # Mid-episode rule mutation recovery on W18_alpha_mutated (autonomous surprise monitor)
    cwm.memory.active_engram_id = eid_alpha
    rec_planner = AutonomousPlanner(cwm, horizon=3, gamma=0.95, beta_uncertainty=0.5, surprise_threshold=3.0, surprise_window=3)
    rec_planner.reset_monitor()
    mut_stream = W18_ALPHA_MUTATED.collect_transitions(n_steps=60, seed=seed * 100 + 12)
    pre_rec_errs: List[float] = []
    post_rec_errs: List[float] = []
    recovered = False
    buf_x, buf_a, buf_xn, buf_de = [], [], [], []
    for t_m in range(60):
        xt, at, xnt, det = mut_stream["X_t"][t_m], int(mut_stream["A_t"][t_m]), mut_stream["X_next"][t_m], float(mut_stream["dE"][t_m])
        surp, de_hat, _ = rec_planner.compute_surprise(xt, at, det)
        abs_e = abs(det - de_hat)
        buf_x.append(xt); buf_a.append(at); buf_xn.append(xnt); buf_de.append(det)
        if not recovered:
            pre_rec_errs.append(abs_e)
            if rec_planner.should_recover() and len(buf_x) >= 30:
                rec_planner.trigger_recovery(np.array(buf_x), np.array(buf_a), np.array(buf_xn), np.array(buf_de))
                recovered = True
        else:
            post_rec_errs.append(abs_e)

    # Also verify W18_beta was NOT corrupted by W18_alpha_mutated recovery!
    te_beta_ret = W18_BETA.collect_transitions(n_steps=60, seed=seed * 100 + 13)
    pred_beta_post_mut = cwm.predict_step(te_beta_ret["X_t"], te_beta_ret["A_t"], eid=eid_beta)["de_mean"]
    err_beta_post_mut = float(np.sqrt(np.mean((pred_beta_post_mut - te_beta_ret["dE"]) ** 2)) / de_scale)

    pre_mut_mae = float(np.mean(pre_rec_errs[:10])) if pre_rec_errs else 1.0
    post_mut_mae = float(np.mean(post_rec_errs[-10:])) if post_rec_errs else pre_mut_mae
    recovery_reduction = float(1.0 - post_mut_mae / max(pre_mut_mae, 1e-6))

    footprint = cwm.memory_footprint_bytes()
    h6_metrics = {
        "forgetting_alpha": float(forgetting_alpha),
        "retrieval_acc": float(retr_correct / 2.0),
        "mutation_detected_and_recovered": bool(recovered),
        "pre_mutation_mae": pre_mut_mae,
        "post_recovery_mae": post_mut_mae,
        "recovery_error_reduction": recovery_reduction,
        "beta_retention_post_mutation": err_beta_post_mut,
        "memory_footprint": footprint,
    }

    # --- H7: Generalization to Unseen Initial States & Novel Sensor-Shifted World (W18_gamma_shifted) ---
    # 50-transition few-shot adaptation to W18_gamma_shifted (new orthonormal subspace + c=0.5 offset)
    cwm_gen = CaspianWorldModel(in_dim=IN_DIM18, n_latent=N_LATENT18, n_actions=N_ACTIONS18, seed=seed)
    cwm_gen.observe_transitions(tr_alpha["X_t"], tr_alpha["A_t"], tr_alpha["X_next"], tr_alpha["dE"])
    tr_gamma = W18_GAMMA_SHIFTED.collect_transitions(n_steps=60, seed=seed * 100 + 14)
    eid_gamma = cwm_gen.observe_transitions(tr_gamma["X_t"], tr_gamma["A_t"], tr_gamma["X_next"], tr_gamma["dE"])

    mlp_gamma = MLPWorldModelBaseline(seed=seed)
    mlp_gamma.fit(tr_gamma["X_t"], tr_gamma["A_t"], tr_gamma["X_next"], tr_gamma["dE"])

    te_gamma = W18_GAMMA_SHIFTED.collect_transitions(n_steps=60, seed=seed * 100 + 15)
    cwm_g_pred = cwm_gen.predict_step(te_gamma["X_t"], te_gamma["A_t"], eid=eid_gamma)
    mlp_g_x, _ = mlp_gamma.predict_step(te_gamma["X_t"], te_gamma["A_t"])
    g_x_scale = float(np.std(te_gamma["X_t"])) + 1e-8

    planner_gamma = AutonomousPlanner(cwm_gen, horizon=3, gamma=0.95, beta_uncertainty=0.5)
    unseen_inits = [
        np.array([-1.3, -0.4, 0.3, -0.3], dtype=float),
        np.array([-1.2, -0.2, -0.4, 0.4], dtype=float),
        np.array([-0.7, -0.5, 0.2, 0.2], dtype=float),
    ]
    g_ret_cwm, g_ret_mlp, g_ret_react = [], [], []
    for ui_i, s0_u in enumerate(unseen_inits):
        ep_s = seed * 3000 + ui_i
        g_ret_cwm.append(run_episode(W18_GAMMA_SHIFTED, lambda x, s, t: planner_gamma.plan_action(x, eid=eid_gamma)[0], s0_u, steps=20, seed=ep_s)["total_return"])
        g_ret_mlp.append(run_episode(W18_GAMMA_SHIFTED, lambda x, s, t: mlp_gamma.plan_action(x, horizon=3), s0_u, steps=20, seed=ep_s)["total_return"])
        g_ret_react.append(run_episode(W18_GAMMA_SHIFTED, lambda x, s, t: mlp_gamma.plan_action(x, horizon=1), s0_u, steps=20, seed=ep_s)["total_return"])

    h7_metrics = {
        "shifted_1step_x_nrmse_cwm": float(np.sqrt(np.mean((cwm_g_pred["x_next"] - te_gamma["X_next"]) ** 2)) / g_x_scale),
        "shifted_1step_x_nrmse_mlp": float(np.sqrt(np.mean((mlp_g_x - te_gamma["X_next"]) ** 2)) / g_x_scale),
        "shifted_return_cwm": float(np.mean(g_ret_cwm)),
        "shifted_return_mlp": float(np.mean(g_ret_mlp)),
        "shifted_return_reactive": float(np.mean(g_ret_react)),
    }

    return {
        "seed": seed,
        "wall_time_s": round(time.perf_counter() - t0, 3),
        "H1": h1_metrics,
        "H2": h2_metrics,
        "H3": h3_metrics,
        "H4": h4_metrics,
        "H5": h5_metrics,
        "H6": h6_metrics,
        "H7": h7_metrics,
    }


def aggregate_phase18(seed_results: List[Dict[str, Any]], label: str = "confirmatory") -> Dict[str, Any]:
    seeds = [r["seed"] for r in seed_results]
    n = len(seeds)

    def vec(sec: str, key: str) -> np.ndarray:
        return np.array([r[sec][key] for r in seed_results], dtype=float)

    # H1 stats
    h1_cwm_h1 = vec("H1", "cwm_x_nrmse_h1")
    h1_cwm_h3 = vec("H1", "cwm_x_nrmse_h3")
    h1_vs_pers_1 = paired_stats(h1_cwm_h1, vec("H1", "pers_x_nrmse_h1"), higher_is_better=False, seed=1)
    h1_vs_mlp_1 = paired_stats(h1_cwm_h1, vec("H1", "mlp_x_nrmse_h1"), higher_is_better=False, seed=2)
    h1_vs_pers_3 = paired_stats(h1_cwm_h3, vec("H1", "pers_x_nrmse_h3"), higher_is_better=False, seed=3)
    h1_vs_mlp_3 = paired_stats(h1_cwm_h3, vec("H1", "mlp_x_nrmse_h3"), higher_is_better=False, seed=4)
    h1_holm = holm_bonferroni({
        "pers_h1": h1_vs_pers_1["perm_p"], "mlp_h1": h1_vs_mlp_1["perm_p"],
        "pers_h3": h1_vs_pers_3["perm_p"], "mlp_h3": h1_vs_mlp_3["perm_p"],
    })
    h1_met = bool(
        float(np.mean(h1_cwm_h1)) < 0.05
        and all(c["ci_strictly_positive"] for c in (h1_vs_pers_1, h1_vs_mlp_1, h1_vs_pers_3, h1_vs_mlp_3))
        and all(p < 0.05 for p in h1_holm.values())
    )

    # H2 stats
    h2_f1 = vec("H2", "causal_action_f1")
    h2_vs_obs = paired_stats(vec("H2", "cwm_interventional_nrmse"), vec("H2", "obs_only_interventional_nrmse"), higher_is_better=False, seed=5)
    h2_met = bool(float(np.mean(h2_f1)) >= 0.90 and h2_vs_obs["ci_strictly_positive"] and h2_vs_obs["perm_p"] < 0.05)

    # H3 stats
    h3_rho = float(np.mean(vec("H3", "cf_spearman_rho")))
    h3_top1 = float(np.mean(vec("H3", "cf_top1_acc")))
    h3_met = bool(h3_rho >= 0.85 and h3_top1 >= 0.85)

    # H4 stats
    h4_auroc = float(np.mean(vec("H4", "ood_auroc")))
    h4_corr = float(np.mean(vec("H4", "err_uncertainty_corr")))
    h4_pi95 = float(np.mean(vec("H4", "pi95_coverage")))
    h4_vs_neutral = paired_stats(vec("H4", "trap_return_risk_aware"), vec("H4", "trap_return_risk_neutral"), higher_is_better=True, seed=6)
    h4_met = bool(h4_auroc >= 0.90 and h4_corr > 0.40 and h4_vs_neutral["ci_strictly_positive"] and h4_vs_neutral["perm_p"] < 0.05)

    # H5 stats
    casp_ret = vec("H5", "return_integrated_caspian")
    h5_comps = {
        b: paired_stats(casp_ret, vec("H5", f"return_{b}"), higher_is_better=True, seed=10 + i)
        for i, b in enumerate(["random_policy", "reactive_policy", "memory_no_planner", "mlp_world_model"])
    }
    h5_holm = holm_bonferroni({k: v["perm_p"] for k, v in h5_comps.items()})
    h5_oracle_ratio = float(np.mean(vec("H5", "oracle_ratio")))
    h5_met = bool(
        all(v["ci_strictly_positive"] for v in h5_comps.values())
        and all(p < 0.05 for p in h5_holm.values())
        and h5_oracle_ratio >= 0.75
    )

    # H6 stats
    h6_forg = float(np.mean(vec("H6", "forgetting_alpha")))
    h6_retr = float(np.mean(vec("H6", "retrieval_acc")))
    h6_red = float(np.mean(vec("H6", "recovery_error_reduction")))
    h6_met = bool(h6_forg < 1e-6 and h6_retr >= 0.95 and h6_red >= 0.50)

    # H7 stats
    h7_vs_mlp = paired_stats(vec("H7", "shifted_return_cwm"), vec("H7", "shifted_return_mlp"), higher_is_better=True, seed=20)
    h7_vs_react = paired_stats(vec("H7", "shifted_return_cwm"), vec("H7", "shifted_return_reactive"), higher_is_better=True, seed=21)
    h7_holm = holm_bonferroni({"mlp_world_model": h7_vs_mlp["perm_p"], "reactive_policy": h7_vs_react["perm_p"]})
    h7_met = bool(
        h7_vs_mlp["ci_strictly_positive"] and h7_vs_react["ci_strictly_positive"]
        and all(p < 0.05 for p in h7_holm.values())
    )

    # Determinism check on first seed
    det_r1 = run_single_seed(seeds[0])
    det_diff = abs(det_r1["H5"]["return_integrated_caspian"] - seed_results[0]["H5"]["return_integrated_caspian"])
    ast_audit = ast_leakage_audit_phase18()

    verdicts = {
        "P18-H1": h1_met, "P18-H2": h2_met, "P18-H3": h3_met,
        "P18-H4": h4_met, "P18-H5": h5_met, "P18-H6": h6_met, "P18-H7": h7_met,
    }
    n_met = sum(1 for v in verdicts.values() if v)

    return {
        "experiment_id": "CSP-P18-E001",
        "split": label,
        "seeds": seeds,
        "n_seeds": n,
        "hypotheses_met_count": f"{n_met}/7",
        "verdicts": verdicts,
        "H1": {
            "met": h1_met,
            "means": {
                "cwm_x_h1": round(float(np.mean(h1_cwm_h1)), 5),
                "cwm_x_h2": round(float(np.mean(vec("H1", "cwm_x_nrmse_h2"))), 5),
                "cwm_x_h3": round(float(np.mean(h1_cwm_h3)), 5),
                "cwm_x_h5": round(float(np.mean(vec("H1", "cwm_x_nrmse_h5"))), 5),
                "mlp_x_h1": round(float(np.mean(vec("H1", "mlp_x_nrmse_h1"))), 5),
                "mlp_x_h3": round(float(np.mean(vec("H1", "mlp_x_nrmse_h3"))), 5),
                "pers_x_h1": round(float(np.mean(vec("H1", "pers_x_nrmse_h1"))), 5),
                "pers_x_h3": round(float(np.mean(vec("H1", "pers_x_nrmse_h3"))), 5),
                "cwm_de_h1": round(float(np.mean(vec("H1", "cwm_de_nrmse_h1"))), 5),
                "mlp_de_h1": round(float(np.mean(vec("H1", "mlp_de_nrmse_h1"))), 5),
            },
            "comparisons": {
                "pers_h1": h1_vs_pers_1, "mlp_h1": h1_vs_mlp_1,
                "pers_h3": h1_vs_pers_3, "mlp_h3": h1_vs_mlp_3,
            },
            "holm_p": h1_holm,
        },
        "H2": {
            "met": h2_met,
            "mean_causal_action_f1": round(float(np.mean(h2_f1)), 5),
            "mean_cwm_int_nrmse": round(float(np.mean(vec("H2", "cwm_interventional_nrmse"))), 5),
            "mean_obs_only_int_nrmse": round(float(np.mean(vec("H2", "obs_only_interventional_nrmse"))), 5),
            "vs_obs_only": h2_vs_obs,
        },
        "H3": {
            "met": h3_met,
            "mean_cf_spearman_rho": round(h3_rho, 5),
            "mean_cf_top1_acc": round(h3_top1, 5),
            "mean_cf_nrmse": round(float(np.mean(vec("H3", "cf_nrmse"))), 5),
        },
        "H4": {
            "met": h4_met,
            "mean_ood_auroc": round(h4_auroc, 5),
            "mean_err_uncertainty_corr": round(h4_corr, 5),
            "mean_pi95_coverage": round(h4_pi95, 5),
            "mean_trap_return_risk_aware": round(float(np.mean(vec("H4", "trap_return_risk_aware"))), 4),
            "mean_trap_return_risk_neutral": round(float(np.mean(vec("H4", "trap_return_risk_neutral"))), 4),
            "vs_risk_neutral": h4_vs_neutral,
        },
        "H5": {
            "met": h5_met,
            "means": {
                k: round(float(np.mean(vec("H5", f"return_{k}"))), 4)
                for k in ["integrated_caspian", "oracle_planner", "mlp_world_model", "memory_no_planner", "reactive_policy", "random_policy"]
            },
            "mean_oracle_ratio": round(h5_oracle_ratio, 4),
            "mean_planning_success_rate": round(float(np.mean(vec("H5", "planning_success_rate"))), 4),
            "comparisons": h5_comps,
            "holm_p": h5_holm,
        },
        "H6": {
            "met": h6_met,
            "mean_forgetting_alpha": round(h6_forg, 6),
            "mean_retrieval_acc": round(h6_retr, 5),
            "mutation_recovery_rate": round(float(np.mean([1.0 if r["H6"]["mutation_detected_and_recovered"] else 0.0 for r in seed_results])), 4),
            "mean_pre_mutation_mae": round(float(np.mean(vec("H6", "pre_mutation_mae"))), 5),
            "mean_post_recovery_mae": round(float(np.mean(vec("H6", "post_recovery_mae"))), 5),
            "mean_recovery_error_reduction": round(h6_red, 5),
            "mean_beta_retention_post_mutation": round(float(np.mean(vec("H6", "beta_retention_post_mutation"))), 5),
            "memory_footprint": seed_results[0]["H6"]["memory_footprint"],
        },
        "H7": {
            "met": h7_met,
            "mean_shifted_1step_x_nrmse_cwm": round(float(np.mean(vec("H7", "shifted_1step_x_nrmse_cwm"))), 5),
            "mean_shifted_1step_x_nrmse_mlp": round(float(np.mean(vec("H7", "shifted_1step_x_nrmse_mlp"))), 5),
            "mean_shifted_return_cwm": round(float(np.mean(vec("H7", "shifted_return_cwm"))), 4),
            "mean_shifted_return_mlp": round(float(np.mean(vec("H7", "shifted_return_mlp"))), 4),
            "mean_shifted_return_reactive": round(float(np.mean(vec("H7", "shifted_return_reactive"))), 4),
            "vs_mlp": h7_vs_mlp,
            "vs_reactive": h7_vs_react,
            "holm_p": h7_holm,
        },
        "integrity_audit": {
            "determinism_diff": float(det_diff),
            "ast_leakage_passed": ast_audit["passed"],
            "ast_violations": ast_audit["violations"],
            "mean_wall_time_s": round(float(np.mean([r["wall_time_s"] for r in seed_results])), 3),
        },
        "per_seed": seed_results,
    }


def run_phase18(seeds: List[int], label: str, out_path: str) -> Dict[str, Any]:
    results = [run_single_seed(s) for s in seeds]
    summary = aggregate_phase18(results, label=label)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        s = run_phase18(SMOKE_SEEDS18, "exploratory_smoke", "logs/phase18/p18_smoke_results.json")
    elif mode == "confirmatory":
        s = run_phase18(CONFIRMATORY_SEEDS18, "confirmatory", "logs/phase18/p18_confirmatory_results.json")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    print(f"=== Phase 18 ({s['split']}, n={s['n_seeds']}) ===")
    print("Hypotheses met:", s["hypotheses_met_count"], s["verdicts"])
    print("H1 cwm_x_h1/h3:", s["H1"]["means"]["cwm_x_h1"], s["H1"]["means"]["cwm_x_h3"], "vs mlp:", s["H1"]["means"]["mlp_x_h1"], s["H1"]["means"]["mlp_x_h3"])
    print("H2 causal_f1:", s["H2"]["mean_causal_action_f1"], "int_nrmse:", s["H2"]["mean_cwm_int_nrmse"], "vs obs:", s["H2"]["mean_obs_only_int_nrmse"])
    print("H3 cf_rho/top1:", s["H3"]["mean_cf_spearman_rho"], s["H3"]["mean_cf_top1_acc"])
    print("H4 auroc/corr/trap:", s["H4"]["mean_ood_auroc"], s["H4"]["mean_err_uncertainty_corr"], s["H4"]["mean_trap_return_risk_aware"], "vs", s["H4"]["mean_trap_return_risk_neutral"])
    print("H5 returns:", s["H5"]["means"], "oracle_ratio:", s["H5"]["mean_oracle_ratio"])
    print("H6 forg/retr/red:", s["H6"]["mean_forgetting_alpha"], s["H6"]["mean_retrieval_acc"], s["H6"]["mean_recovery_error_reduction"])
    print("H7 shifted:", s["H7"]["mean_shifted_return_cwm"], "vs mlp:", s["H7"]["mean_shifted_return_mlp"], "vs react:", s["H7"]["mean_shifted_return_reactive"])
    print("Integrity:", s["integrity_audit"])
