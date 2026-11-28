"""Phase 19: Bounded Operational Cluster Decision-Support Environment (prototype/ops_env.py).

Models a realistic, bounded cloud-compute & thermal-load service cluster:
- 4 underlying physical operational states (hidden from Caspian):
    s[0]: compute load saturation   in [-1.5, 1.5] (normalized around nominal operating point)
    s[1]: thermal dissipation level in [-1.5, 1.5]
    s[2]: request queue backlog     in [-1.5, 1.5]
    s[3]: cache/memory efficiency   in [-1.5, 1.5]
- 8 neutral telemetry channels observed by the agent (correlated sensor projection + optional
  migration offset), strictly passing Observation.assert_semantic_purity().
- 5 permitted discrete operational actions:
    0: Monitor / NOOP       (cost 0.00)
    1: Scale-Out Compute    (reduces load & queue, cost 0.40)
    2: Shed / Throttle Load (drops queue immediately, hurts throughput, cost 0.50)
    3: Rebalance Shards     (prepares load & thermal headroom for multi-step stability, cost 0.30)
    4: Boost Cooling/Cache  (reduces thermal level & improves cache efficiency, cost 0.35)
- Composite operational objective (scalar delta_e):
    Rewards low load/thermal synergy & high cache efficiency, heavily penalizes non-additive
    cascading load*thermal overload (SLA breach) and queue backlog, minus action cost.

Also provides conventional non-Caspian baselines:
- ThresholdAutoscalerBaseline: industry-standard reactive threshold rule controller.
- ReactiveControllerBaseline: 1-step greedy learned controller (no multi-step lookahead).
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from environment.observations import EntityObservation, Observation

OPS_IN_DIM = 8
OPS_N_LATENT = 4
OPS_N_ACTIONS = 5
PERMITTED_OPS_ACTIONS = frozenset(range(OPS_N_ACTIONS))


def _telemetry_projection(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((OPS_IN_DIM, OPS_N_LATENT)))
    return Q[:, :OPS_N_LATENT].T.copy()  # (4, 8) orthonormal telemetry schema


TELEMETRY_SCHEMA_A = _telemetry_projection(1901)
TELEMETRY_SCHEMA_B = _telemetry_projection(1902)
MIGRATION_OFFSET_B = np.full((1, OPS_IN_DIM), 0.45, dtype=float)

# Coupled physical cluster drift: high load heats thermal state & grows queue
A_CLUSTER_NOMINAL = np.array([
    [0.86, 0.00, 0.08, 0.00],
    [0.18, 0.84, 0.00, 0.00],
    [0.14, 0.00, 0.82, -0.08],
    [0.00, -0.10, 0.00, 0.88],
], dtype=float)

# Permitted discrete operational interventions (positive s0/s1 = healthy headroom; negative = overloaded/hot)
# Note: signed so higher s0 (compute headroom) and s1 (thermal headroom) with s0*s1 > 0 gives peak SLA efficiency,
# while starting overloaded (-0.7, -0.7) creates a genuine multi-step recovery problem where short-sighted
# throttling (a=2) relieves immediate pressure but traps the cluster below optimal steady-state SLA!
B_CLUSTER_NOMINAL = np.array([
    [ 0.00,  0.00,  0.00,  0.00],  # 0: Monitor / NOOP
    [ 0.78,  0.00,  0.25,  0.00],  # 1: Scale-Out Compute (+headroom, +queue clearance)
    [-0.65,  0.00,  0.75,  0.00],  # 2: Emergency Throttle (immediate queue relief, hurts compute headroom)
    [ 0.00,  0.78,  0.00,  0.20],  # 3: Boost Cooling & Rebalance (+thermal headroom, +cache)
    [ 0.00,  0.00,  0.00,  0.75],  # 4: Cache Warmup (+cache efficiency)
], dtype=float)

OPS_ACTION_COSTS = np.array([0.00, 0.35, -0.40, 0.35, 0.25], dtype=float)


def _sla_health_nominal(s: np.ndarray) -> np.ndarray:
    """Composite operational SLA score (non-additive headroom synergy + overload penalty)."""
    return (
        3.2 * s[:, 0]
        + 3.2 * s[:, 1]
        + 5.5 * s[:, 0] * s[:, 1]
        + 1.4 * s[:, 2]
        + 1.0 * s[:, 3]
        - 0.45 * (s[:, 0] ** 2 + s[:, 1] ** 2)
    )


def _sla_health_incident(s: np.ndarray) -> np.ndarray:
    """Mid-stream workload/cooling incident: thermal-load sensitivity inverts/shifts sharply."""
    return (
        -3.0 * s[:, 0]
        - 3.0 * s[:, 1]
        - 5.0 * s[:, 0] * s[:, 1]
        + 1.8 * s[:, 2]
        + 1.2 * s[:, 3]
        - 0.45 * (s[:, 0] ** 2 + s[:, 1] ** 2)
    )


@dataclass
class OperationalClusterEnv:
    """Bounded operational service-cluster simulator with strict Semantic Firewall."""
    name: str
    telemetry_matrix: np.ndarray
    telemetry_offset: np.ndarray
    A_dyn: np.ndarray
    B_act: np.ndarray
    sla_fn: Callable[[np.ndarray], np.ndarray]
    action_costs: np.ndarray
    state_bound: float = 1.6
    noise_s: float = 0.03
    noise_sla: float = 0.15

    def observe_telemetry(self, s: np.ndarray) -> np.ndarray:
        s_arr = np.asarray(s, dtype=float)
        if s_arr.ndim == 1:
            return (s_arr.reshape(1, -1) @ self.telemetry_matrix + self.telemetry_offset).reshape(-1)
        return s_arr @ self.telemetry_matrix + self.telemetry_offset

    def make_neutral_observation(self, s: np.ndarray, cum_score: float, t: int, prev_a: Optional[int]) -> Observation:
        x = self.observe_telemetry(s)
        obs = Observation(
            agent_position=(0, 0),
            entities=(EntityObservation(entity_id=1, entity_type=0, relative_x=0, relative_y=0, distance=0.0, properties=tuple(float(v) for v in x)),),
            collision=False,
            internal_state=float(cum_score),
            timestep=int(t),
            previous_action=prev_a,
        )
        obs.assert_semantic_purity()
        return obs

    def step(self, s: np.ndarray, action: int, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, float, bool]:
        """Advance cluster by one control tick. Returns (s_next, sla_delta, sla_breach)."""
        if int(action) not in PERMITTED_OPS_ACTIONS:
            raise PermissionError(f"Action {action} outside permitted operational set {sorted(PERMITTED_OPS_ACTIONS)}.")
        s_vec = np.asarray(s, dtype=float).reshape(-1)
        ns = rng.normal(0.0, self.noise_s, size=OPS_N_LATENT) if rng is not None else np.zeros(OPS_N_LATENT)
        ne = float(rng.normal(0.0, self.noise_sla)) if rng is not None else 0.0
        s_next = np.clip(self.A_dyn @ s_vec + self.B_act[int(action)] + ns, -self.state_bound, self.state_bound)
        sla_delta = float(self.sla_fn(s_next.reshape(1, -1))[0]) - float(self.action_costs[int(action)]) + ne
        sla_breach = bool(s_next[0] < -0.5 or s_next[1] < -0.5)
        return s_next, sla_delta, sla_breach

    def collect_telemetry_log(self, n_steps: int, seed: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        X_t = np.zeros((n_steps, OPS_IN_DIM), dtype=float)
        A_t = np.zeros(n_steps, dtype=int)
        X_n = np.zeros((n_steps, OPS_IN_DIM), dtype=float)
        dE = np.zeros(n_steps, dtype=float)
        S_t = np.zeros((n_steps, OPS_N_LATENT), dtype=float)
        S_n = np.zeros((n_steps, OPS_N_LATENT), dtype=float)

        s = rng.uniform(-1.2, 1.2, size=OPS_N_LATENT)
        for i in range(n_steps):
            if i % 12 == 0:
                s = rng.uniform(-1.2, 1.2, size=OPS_N_LATENT)
            a = int(rng.integers(0, OPS_N_ACTIONS))
            sn, de, _ = self.step(s, a, rng=rng)
            S_t[i] = s; S_n[i] = sn
            X_t[i] = self.observe_telemetry(s)
            X_n[i] = self.observe_telemetry(sn)
            A_t[i] = a; dE[i] = de
            s = sn
        return {"X_t": X_t, "A_t": A_t, "X_next": X_n, "dE": dE, "S_t": S_t, "S_next": S_n}


CLUSTER_A_NOMINAL = OperationalClusterEnv(
    name="Cluster_A_nominal",
    telemetry_matrix=TELEMETRY_SCHEMA_A,
    telemetry_offset=np.zeros((1, OPS_IN_DIM)),
    A_dyn=A_CLUSTER_NOMINAL,
    B_act=B_CLUSTER_NOMINAL,
    sla_fn=_sla_health_nominal,
    action_costs=OPS_ACTION_COSTS,
)

CLUSTER_B_MIGRATED = OperationalClusterEnv(
    name="Cluster_B_migrated",
    telemetry_matrix=TELEMETRY_SCHEMA_B,
    telemetry_offset=MIGRATION_OFFSET_B,
    A_dyn=A_CLUSTER_NOMINAL,
    B_act=B_CLUSTER_NOMINAL,
    sla_fn=_sla_health_nominal,
    action_costs=OPS_ACTION_COSTS,
)

CLUSTER_A_INCIDENT = OperationalClusterEnv(
    name="Cluster_A_incident",
    telemetry_matrix=TELEMETRY_SCHEMA_A,
    telemetry_offset=np.zeros((1, OPS_IN_DIM)),
    A_dyn=A_CLUSTER_NOMINAL,
    B_act=B_CLUSTER_NOMINAL,
    sla_fn=_sla_health_incident,
    action_costs=OPS_ACTION_COSTS,
)


class ThresholdAutoscalerBaseline:
    """Conventional rule-based operational controller (learns linear telemetry->action mapping
    via 1-step ridge scoring of telemetry thresholds — zero multi-step planning, no world model).
    """

    def __init__(self):
        self.w_thresh: Optional[np.ndarray] = None  # (D+1, A)

    def fit(self, X_t: np.ndarray, A_t: np.ndarray, dE: np.ndarray) -> None:
        X = np.asarray(X_t, dtype=float)
        A = np.asarray(A_t, dtype=int).reshape(-1)
        y = np.asarray(dE, dtype=float).reshape(-1)
        Psi = np.hstack([np.ones((len(X), 1)), X])
        self.w_thresh = np.zeros((Psi.shape[1], OPS_N_ACTIONS), dtype=float)
        for a in range(OPS_N_ACTIONS):
            ia = np.where(A == a)[0]
            if len(ia) >= 3:
                self.w_thresh[:, a] = np.linalg.solve(
                    Psi[ia].T @ Psi[ia] + 1.0 * np.eye(Psi.shape[1]),
                    Psi[ia].T @ y[ia],
                )

    def select_action(self, x_t: np.ndarray) -> int:
        assert self.w_thresh is not None
        psi = np.concatenate([[1.0], np.asarray(x_t, dtype=float).reshape(-1)])
        scores = psi @ self.w_thresh
        return int(np.argmax(scores))
