"""Phase 19: Bounded Operational Decision-Support Prototype Agent (prototype/caspian_ops_agent.py).

Wraps Phase 18D's verified CaspianWorldModel (transfer_mode="uncertainty_guided") and
AutonomousPlanner with production application constraints:
1. Explicit Action Permission Gate: raises PermissionError on any out-of-scope action.
2. Human-Oversight Escalation Gate: flags unfamiliar telemetry (ood_score > ood_threshold)
   or sustained prediction surprise (incident detected) for human operator review.
3. Privacy-Preserving Persistence: zero raw telemetry logs persisted (to_bytes / from_bytes).
4. Per-Decision Latency & Operational Telemetry Auditing.
"""
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from models.planner import AutonomousPlanner
from models.world_model import CaspianWorldModel
from prototype.ops_env import OPS_IN_DIM, OPS_N_ACTIONS, OPS_N_LATENT, PERMITTED_OPS_ACTIONS


class CaspianOpsAgent:
    """Bounded operational decision-support prototype powered by CaspianWorldModel + AutonomousPlanner."""

    def __init__(
        self,
        seed: int = 0,
        horizon: int = 3,
        beta_uncertainty: float = 0.5,
        ood_escalation_threshold: float = 0.25,
        surprise_threshold: float = 3.0,
        transfer_mode: str = "uncertainty_guided",
    ):
        self.seed = int(seed)
        self.ood_escalation_threshold = float(ood_escalation_threshold)
        self.world_model = CaspianWorldModel(
            in_dim=OPS_IN_DIM,
            n_latent=OPS_N_LATENT,
            n_actions=OPS_N_ACTIONS,
            seed=self.seed,
            boundary_aware=True,
            basis_sync=True,
            transfer_mode=transfer_mode,
        )
        self.planner = AutonomousPlanner(
            self.world_model,
            horizon=horizon,
            gamma=0.95,
            beta_uncertainty=beta_uncertainty,
            surprise_threshold=surprise_threshold,
            surprise_window=3,
        )
        self.decision_latencies_ms: List[float] = []
        self.escalation_events: int = 0

    def ingest_operational_log(
        self,
        X_t: np.ndarray,
        A_t: np.ndarray,
        X_next: np.ndarray,
        dE_t: np.ndarray,
        allow_update_outdated: bool = False,
    ) -> int:
        """Learn or selectively transfer operational cluster model from neutral telemetry (zero cluster_id)."""
        return self.world_model.observe_transitions(
            X_t, A_t, X_next, dE_t, allow_update_outdated=allow_update_outdated
        )

    def recommend_action(
        self, x_t: np.ndarray, eid: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Produce a permitted operational action + human-oversight advisory & latency measurement."""
        t0 = time.perf_counter()
        ood_score = float(self.world_model.subspace_ood_score(x_t, eid=eid)[0])
        action, info = self.planner.plan_action(x_t, eid=eid)
        if int(action) not in PERMITTED_OPS_ACTIONS:
            raise PermissionError(f"Recommended action {action} outside permitted set.")
        escalate = bool(ood_score > self.ood_escalation_threshold)
        if escalate:
            self.escalation_events += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.decision_latencies_ms.append(latency_ms)
        info["ood_score"] = ood_score
        info["escalation_flag"] = escalate
        info["latency_ms"] = latency_ms
        return int(action), info

    def monitor_tick(
        self,
        x_t: np.ndarray,
        a_t: int,
        de_obs: float,
        eid: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Monitor operational surprise after executing a_t and flag incident escalation if needed."""
        surp, de_hat, de_std = self.planner.compute_surprise(x_t, a_t, de_obs, eid=eid)
        incident = self.planner.should_recover()
        if incident:
            self.escalation_events += 1
        return {
            "surprise": surp,
            "de_hat": de_hat,
            "de_std": de_std,
            "incident_escalated": incident,
        }

    def adapt_to_incident(
        self,
        recent_X: np.ndarray,
        recent_A: np.ndarray,
        recent_Xn: np.ndarray,
        recent_dE: np.ndarray,
    ) -> Tuple[int, bool]:
        """Autonomously recover operational model from recent post-incident telemetry."""
        return self.planner.trigger_recovery(recent_X, recent_A, recent_Xn, recent_dE)

    def retrieve_cluster_context(
        self, X_p: np.ndarray, A_p: np.ndarray, Xn_p: np.ndarray, dE_p: np.ndarray
    ) -> Tuple[Optional[int], float, bool]:
        return self.world_model.retrieve_context(X_p, A_p, Xn_p, dE_p)

    def footprint_audit(self) -> Dict[str, int]:
        return self.world_model.memory_footprint_bytes()

    def mean_latency_ms(self) -> float:
        return float(np.mean(self.decision_latencies_ms)) if self.decision_latencies_ms else 0.0
