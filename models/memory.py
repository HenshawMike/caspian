"""Phase 17: Continual Learning & Persistent Environmental Memory (models/memory.py).

Implements:
1. PersistentWorldMemory: pure-NumPy modular persistent environmental memory bank built on
   compact 66-parameter ModularTransferModelV3 engrams.
   - Zero world_id privilege: autonomously retrieves the matching engram from a small probe
     batch (X_probe, Y_probe) via predictive residual & noise-floor surprise detection.
   - Forward transfer: when encountering a novel world, tests Empirical Bayes marginal
     likelihood across stored structural cores to transfer reusable rules (e.g. underdetermined
     10-shot shifted worlds) while rejecting negative transfer via core_gate = 0.0.
   - Outdated memory detection (Exp 17.4): detects when a previously encountered sensor subspace
     now produces high predictive surprise and updates/branches memory without corrupting
     other stored worlds.
2. EWCState: diagonal empirical Fisher Information regularizer for the EWC-MLP baseline.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from models.transfer_v3 import ModularTransferModelV3


@dataclass
class WorldEngram:
    engram_id: int
    model: ModularTransferModelV3
    sigma2_ref: float
    source_engram_id: Optional[int] = None


class PersistentWorldMemory:
    """Persistent environmental memory bank with autonomous retrieval, forward transfer,
    and outdated-memory detection (zero world_id privilege).
    """

    def __init__(
        self,
        in_dim: int = 8,
        n_latent: int = 4,
        seed: int = 0,
        alignment_mode: str = "occam_hierarchy",
        occam_restarts: int = 30,
        occam_noise_mult: float = 4.0,
        occam_var_frac: float = 0.015,
    ):
        self.in_dim = int(in_dim)
        self.n_latent = int(n_latent)
        self.seed = int(seed)
        self.alignment_mode = str(alignment_mode)
        self.occam_restarts = int(occam_restarts)
        self.occam_noise_mult = float(occam_noise_mult)
        self.occam_var_frac = float(occam_var_frac)
        self.engrams: List[WorldEngram] = []
        self.active_engram_id: Optional[int] = None
        self.last_outdated_detected: bool = False
        self.last_retrieved_id: Optional[int] = None
        self.last_transfer_source_id: Optional[int] = None
        self.raw_replay_samples_stored: int = 0

    @property
    def num_parameters(self) -> int:
        return int(sum(e.model.num_parameters for e in self.engrams))

    @property
    def num_engrams(self) -> int:
        return len(self.engrams)

    def _engram_mse(self, engram: WorldEngram, X: np.ndarray, Y: np.ndarray) -> float:
        yp = np.asarray(engram.model.forward(X), dtype=float).reshape(-1)
        yt = np.asarray(Y, dtype=float).reshape(-1)
        return float(np.mean((yp - yt) ** 2))

    def _subspace_residual(self, engram: WorldEngram, X: np.ndarray) -> float:
        Xc = np.asarray(X, dtype=float) - engram.model.x_mean
        W = engram.model.W_adapter
        proj = (Xc @ W) @ W.T
        denom = float(np.linalg.norm(Xc, ord="fro"))
        if denom < 1e-8:
            return 0.0
        return float(np.linalg.norm(Xc - proj, ord="fro") / denom)

    def retrieve(self, X_probe: np.ndarray, Y_probe: np.ndarray) -> Tuple[Optional[int], float, bool]:
        """Autonomously identify the best matching engram from a small probe batch (zero world_id).

        Returns:
            (best_engram_id, best_mse, is_confident_match)
        """
        if not self.engrams:
            self.last_retrieved_id = None
            return None, float("inf"), False

        X_p = np.asarray(X_probe, dtype=float)
        Y_p = np.asarray(Y_probe, dtype=float).reshape(-1)
        var_y = max(float(np.var(Y_p)), 1e-6)

        best_e: Optional[WorldEngram] = None
        best_mse = float("inf")
        for e in self.engrams:
            mse = self._engram_mse(e, X_p, Y_p)
            if mse < best_mse:
                best_mse = mse
                best_e = e

        assert best_e is not None
        thresh = max(8.0 * best_e.sigma2_ref, 0.04 * var_y)
        confident = bool(best_mse <= thresh)
        self.active_engram_id = best_e.engram_id
        self.last_retrieved_id = best_e.engram_id
        return best_e.engram_id, best_mse, confident

    def predict_with_probe(self, X_probe: np.ndarray, Y_probe: np.ndarray, X_eval: np.ndarray) -> Tuple[np.ndarray, Optional[int]]:
        """Autonomously retrieve context from (X_probe, Y_probe) and predict on X_eval."""
        eid, _, _ = self.retrieve(X_probe, Y_probe)
        assert eid is not None
        engram = next(e for e in self.engrams if e.engram_id == eid)
        return np.asarray(engram.model.forward(X_eval), dtype=float).reshape(-1), eid

    def observe_world(self, X_tr: np.ndarray, Y_tr: np.ndarray, allow_update_outdated: bool = False) -> int:
        """Continually learn a new or mutated world from (X_tr, Y_tr) with zero world_id privilege.

        1. Checks if an existing engram already explains (X_tr, Y_tr) within noise floor.
        2. Detects whether a previously stored engram shares the same sensor subspace yet fails
           badly on (X_tr, Y_tr) (outdated-memory / rule-change detection, Exp 17.4).
        3. Evaluates Empirical Bayes marginal likelihood across:
           (a) fresh scratch ModularTransferModelV3, and
           (b) transferring each stored engram's frozen structural core via adapt_few_shot().
        4. Commits the highest-evidence model as a new engram (or updates the outdated engram if
           allow_update_outdated=True and an outdated same-subspace engram was detected).
        """
        X_tr = np.asarray(X_tr, dtype=float)
        Y_tr = np.asarray(Y_tr, dtype=float).reshape(-1)
        Xu, yu, wu, sig2 = ModularTransferModelV3._deduplicate_canonical_with_noise(X_tr, Y_tr.reshape(-1, 1))
        var_y = max(float(np.var(Y_tr)), 1e-6)

        self.last_outdated_detected = False
        self.last_transfer_source_id = None

        # Check existing engrams for exact match or same-subspace outdated rule
        outdated_candidates: List[WorldEngram] = []
        if self.engrams:
            best_id, best_mse, confident = self.retrieve(X_tr, Y_tr)
            if confident and best_id is not None:
                return best_id
            for e in self.engrams:
                r_sub = self._subspace_residual(e, Xu)
                mse_e = self._engram_mse(e, X_tr, Y_tr)
                if r_sub < 0.12 and mse_e > max(10.0 * e.sigma2_ref, 0.10 * var_y):
                    outdated_candidates.append(e)
            if outdated_candidates:
                self.last_outdated_detected = True

        # Candidate 0: fresh structural engram (no transfer)
        cand_fresh = ModularTransferModelV3(
            self.in_dim, self.n_latent, seed=self.seed + 31 * len(self.engrams),
            alignment_mode=self.alignment_mode,
            occam_restarts=self.occam_restarts,
            occam_noise_mult=self.occam_noise_mult,
            occam_var_frac=self.occam_var_frac,
        )
        cand_fresh.fit_source(X_tr, Y_tr)
        Phi_f = cand_fresh._compute_phi((Xu - cand_fresh.x_mean) @ cand_fresh.W_adapter + cand_fresh.b_adapter)
        _, best_nll, _ = ModularTransferModelV3._empirical_bayes_ridge(Phi_f, yu, wu, sig2)
        best_model = cand_fresh
        best_src_id: Optional[int] = None

        # Candidates 1..M: transfer from each stored engram's frozen structural core
        for e in self.engrams:
            mc = e.model.clone_for_target()
            # Ensure core is the engram's learned rule (in case e was itself adapted with residual)
            if mc.core_gate == 0.0:
                mc.w_core = mc.w_world.copy()
                mc.b_core = mc.b_world.copy()
                mc.w_world.fill(0.0)
                mc.b_world.fill(0.0)
                mc.core_gate = 1.0
            else:
                mc.w_core = mc.w_core + mc.w_world
                mc.b_core = mc.b_core + mc.b_world
                mc.w_world.fill(0.0)
                mc.b_world.fill(0.0)
            mc._source_x_mean = mc.x_mean.copy()
            mc._source_W_adapter = mc.W_adapter.copy()
            mc._source_b_adapter = mc.b_adapter.copy()

            gate = mc.adapt_few_shot(X_tr, Y_tr, align_sensors="auto", allow_negative_transfer_gate=True)
            if gate == 1.0:
                Phi_c = mc._compute_phi((Xu - mc.x_mean) @ mc.W_adapter + mc.b_adapter)
                yc = Phi_c @ mc.w_core + mc.b_core
                _, nll_c, _ = ModularTransferModelV3._empirical_bayes_ridge(Phi_c, yu - yc, wu, sig2)
                if nll_c < best_nll - 1.5:
                    best_nll = nll_c
                    best_model = mc
                    best_src_id = e.engram_id

        self.last_transfer_source_id = best_src_id

        if allow_update_outdated and len(outdated_candidates) == 1:
            target_e = outdated_candidates[0]
            target_e.model = best_model
            target_e.sigma2_ref = sig2
            target_e.source_engram_id = best_src_id
            self.active_engram_id = target_e.engram_id
            return target_e.engram_id

        new_id = len(self.engrams)
        self.engrams.append(WorldEngram(engram_id=new_id, model=best_model, sigma2_ref=sig2, source_engram_id=best_src_id))
        self.active_engram_id = new_id
        return new_id

    # Standard forward/backward delegation to active engram (for gradient-check & interface compatibility)
    def forward(self, X: np.ndarray) -> np.ndarray:
        assert self.active_engram_id is not None and len(self.engrams) > 0
        engram = next(e for e in self.engrams if e.engram_id == self.active_engram_id)
        return engram.model.forward(X)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self.active_engram_id is not None and len(self.engrams) > 0
        engram = next(e for e in self.engrams if e.engram_id == self.active_engram_id)
        return engram.model.backward(grad_out)

    def zero_grad(self) -> None:
        if self.active_engram_id is not None and self.engrams:
            engram = next(e for e in self.engrams if e.engram_id == self.active_engram_id)
            engram.model.zero_grad()

    def get_params(self):
        assert self.active_engram_id is not None and len(self.engrams) > 0
        engram = next(e for e in self.engrams if e.engram_id == self.active_engram_id)
        return engram.model.get_params()


class EWCState:
    """Diagonal empirical Fisher Information regularizer for Elastic Weight Consolidation (EWC-MLP)."""

    def __init__(self, ewc_lambda: float = 50.0):
        self.ewc_lambda = float(ewc_lambda)
        self.anchors: List[Tuple[List[np.ndarray], List[np.ndarray]]] = []

    def consolidate(self, model: Any, X: np.ndarray, Y: np.ndarray) -> None:
        """Estimate diagonal empirical Fisher F_i = E[(dL/dtheta_i)^2] on (X, Y) and store anchor."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(Y, dtype=float).reshape(-1, 1)
        params = model.get_params()
        fisher = [np.zeros_like(p) for _, p, _ in params]
        n = min(len(X), 64)
        for i in range(n):
            xi = X[i:i + 1]
            yi = y[i:i + 1]
            model.zero_grad()
            yp = model.forward(xi)
            model.backward(2.0 * (yp - yi))
            for k, (_, _, g) in enumerate(model.get_params()):
                fisher[k] += (g ** 2) / float(n)
        theta_star = [p.copy() for _, p, _ in model.get_params()]
        self.anchors.append((theta_star, fisher))

    def penalty_grad(self, params: List[Tuple[str, np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        grads = [np.zeros_like(p) for _, p, _ in params]
        for theta_star, fisher in self.anchors:
            for k, (_, p, _) in enumerate(params):
                grads[k] += self.ewc_lambda * fisher[k] * (p - theta_star[k])
        return grads
