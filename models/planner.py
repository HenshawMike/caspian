"""Phase 18: Uncertainty-Aware Autonomous Multi-Step Planner & Surprise Monitor (models/planner.py).

Operates strictly on neutral observations and CaspianWorldModel internal rollouts:
- Zero access to ground-truth environment state, dynamics, future observations, or world_id.
- Evaluates all candidate action sequences a in A^H internally via vectorized rollout_latent().
- Applies risk-averse uncertainty penalization:
      J(a) = sum_{h=0}^{H-1} gamma^h * (de_hat_h(a) - beta * sigma_total_h(a))
- Monitors closed-loop standardized prediction surprise S_t = |de_obs - de_hat| / sigma_total
  to autonomously trigger engram re-retrieval or online outdated-memory adaptation.
"""
import itertools
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from models.world_model import CaspianWorldModel


class AutonomousPlanner:
    """Receding-horizon uncertainty-aware MPC planner with closed-loop surprise monitoring."""

    def __init__(
        self,
        world_model: CaspianWorldModel,
        horizon: int = 3,
        gamma: float = 0.95,
        beta_uncertainty: float = 1.0,
        surprise_threshold: float = 3.5,
        surprise_window: int = 3,
    ):
        self.world_model = world_model
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.beta_uncertainty = float(beta_uncertainty)
        self.surprise_threshold = float(surprise_threshold)
        self.surprise_window = int(surprise_window)

        self._sequences = np.array(
            list(itertools.product(range(self.world_model.n_actions), repeat=self.horizon)),
            dtype=int,
        )
        self._discounts = np.array([self.gamma ** h for h in range(self.horizon)], dtype=float)
        self.surprise_history: List[float] = []
        self.recovery_triggers: int = 0

    def reset_monitor(self) -> None:
        self.surprise_history.clear()
        self.recovery_triggers = 0

    def plan_action(
        self, x_t: np.ndarray, eid: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Select optimal first action via purely internal multi-step counterfactual rollout."""
        if self.world_model.basis_sync:
            self.world_model.assert_basis_synchronized(eid=eid)
        roll = self.world_model.rollout_latent(x_t, self._sequences, eid=eid)
        de_traj = roll["de_traj"]         # (N_seq, H)
        std_traj = roll["de_std_traj"]    # (N_seq, H)

        penalized = de_traj - self.beta_uncertainty * std_traj
        scores = penalized @ self._discounts  # (N_seq,)

        best_idx = int(np.argmax(scores))
        best_seq = self._sequences[best_idx]
        best_action = int(best_seq[0])

        # Also record best score grouped by first action (counterfactual plan ranking)
        first_action_scores = np.full(self.world_model.n_actions, -np.inf, dtype=float)
        for a in range(self.world_model.n_actions):
            mask_a = self._sequences[:, 0] == a
            first_action_scores[a] = float(np.max(scores[mask_a]))

        return best_action, {
            "best_sequence": best_seq.tolist(),
            "best_score": float(scores[best_idx]),
            "predicted_step0_de": float(de_traj[best_idx, 0]),
            "predicted_step0_std": float(std_traj[best_idx, 0]),
            "first_action_scores": first_action_scores.tolist(),
        }

    def compute_surprise(
        self,
        x_t: np.ndarray,
        a_t: int,
        de_obs: float,
        eid: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Compute standardized prediction surprise S_t = |de_obs - de_hat| / sigma_total."""
        pred = self.world_model.predict_step(x_t, np.array([int(a_t)]), eid=eid)
        de_hat = float(pred["de_mean"][0])
        de_std = max(float(pred["de_std"][0]), 1e-3)
        surprise = abs(float(de_obs) - de_hat) / de_std
        self.surprise_history.append(surprise)
        return surprise, de_hat, de_std

    def should_recover(self) -> bool:
        """Detect sustained prediction surprise requiring engram re-retrieval or online revision."""
        if len(self.surprise_history) < self.surprise_window:
            return False
        recent = self.surprise_history[-self.surprise_window:]
        return bool(np.mean(recent) > self.surprise_threshold)

    def trigger_recovery(
        self,
        recent_X: np.ndarray,
        recent_A: np.ndarray,
        recent_Xn: np.ndarray,
        recent_dE: np.ndarray,
    ) -> Tuple[int, bool]:
        """Autonomously re-retrieve matching engram or adapt/update outdated memory (zero world_id)."""
        self.recovery_triggers += 1
        eid, _, confident = self.world_model.retrieve_context(recent_X, recent_A, recent_Xn, recent_dE)
        if confident and eid is not None:
            self.surprise_history.clear()
            return eid, True
        # No existing engram explains recent transitions -> adapt / update outdated memory online
        new_eid = self.world_model.observe_transitions(
            recent_X, recent_A, recent_Xn, recent_dE, allow_update_outdated=True
        )
        self.surprise_history.clear()
        return new_eid, False
