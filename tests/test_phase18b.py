"""Unit tests for Phase 18B: exact affine-orthogonal basis synchronization,
covariance invariance, desynchronization guard, serialization round-trip, and boundary awareness.
"""
import numpy as np
import pytest

from experiments.phase18_experiments import W18_ALPHA, ast_leakage_audit_phase18
from models.planner import AutonomousPlanner
from models.world_model import BasisDesynchronizationError, CaspianWorldModel


def test_p18b_basis_desynchronization_guard_blocks_planning():
    tr = W18_ALPHA.collect_transitions(n_steps=160, seed=9301)
    cwm = CaspianWorldModel(seed=9301, boundary_aware=True, basis_sync=True)
    eid = cwm.observe_transitions(tr["X_t"], tr["A_t"], tr["X_next"], tr["dE"])
    pl = AutonomousPlanner(cwm, horizon=3)

    # Corrupt W_adapter without synchronizing -> planner MUST raise BasisDesynchronizationError
    Q, _ = np.linalg.qr(np.random.default_rng(11).standard_normal((4, 4)))
    cwm._get_engram(eid).model.W_adapter = cwm._get_engram(eid).model.W_adapter @ Q
    with pytest.raises(BasisDesynchronizationError):
        pl.plan_action(tr["X_t"][0], eid=eid)

    # Synchronize atomically -> planner succeeds again
    synced, orth_err = cwm.sync_dynamics_to_basis(eid)
    assert synced and orth_err < 1e-12
    act, _ = pl.plan_action(tr["X_t"][0], eid=eid)
    assert 0 <= act < 5


def test_p18b_exact_mean_and_covariance_invariance_under_rotation():
    tr = W18_ALPHA.collect_transitions(n_steps=200, seed=9302)
    cwm1 = CaspianWorldModel(seed=9302, boundary_aware=True, basis_sync=True)
    eid = cwm1.observe_transitions(tr["X_t"], tr["A_t"], tr["X_next"], tr["dE"])
    cwm2 = CaspianWorldModel.from_bytes(cwm1.to_bytes())

    rng = np.random.default_rng(42)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    e1 = cwm1._get_engram(eid)
    e2 = cwm2._get_engram(eid)
    W_old, xm_old, b_old = e1.model.W_adapter.copy(), e1.model.x_mean.copy(), e1.model.b_adapter.copy()
    e2.model.W_adapter = W_old @ Q
    e2.model.b_adapter = b_old @ Q + rng.normal(0.0, 0.25, size=(1, 4))
    synced, orth_err = cwm2.sync_dynamics_to_basis(eid)
    assert synced and orth_err < 1e-12
    assert cwm2.dynamics[eid].basis_version == 1

    # Test interior points (away from clipping box corners) for exact linear+covariance equivalence
    Z_pts = rng.uniform(-0.7, 0.7, size=(12, 4))
    X_pts = (Z_pts - b_old) @ W_old.T + xm_old
    A_pts = rng.integers(0, 5, size=12)
    p1 = cwm1.predict_step(X_pts, A_pts, eid=eid)
    p2 = cwm2.predict_step(X_pts, A_pts, eid=eid)

    assert np.max(np.abs(p1["x_next"] - p2["x_next"])) < 1e-12
    P1_obs = np.einsum("ij,njk,lk->nil", W_old, p1["P_next"], W_old)
    P2_obs = np.einsum("ij,njk,lk->nil", e2.model.W_adapter, p2["P_next"], e2.model.W_adapter)
    assert np.max(np.abs(P1_obs - P2_obs)) < 1e-12


def test_p18b_serialization_roundtrip_and_ast_zero_leakage():
    tr = W18_ALPHA.collect_transitions(n_steps=160, seed=9303)
    cwm = CaspianWorldModel(seed=9303, boundary_aware=True, basis_sync=True)
    eid = cwm.observe_transitions(tr["X_t"], tr["A_t"], tr["X_next"], tr["dE"])
    raw = cwm.to_bytes()
    cwm_restored = CaspianWorldModel.from_bytes(raw)
    cwm_restored.assert_basis_synchronized(eid)

    p1 = cwm.predict_step(tr["X_t"][:10], tr["A_t"][:10], eid=eid)
    p2 = cwm_restored.predict_step(tr["X_t"][:10], tr["A_t"][:10], eid=eid)
    assert np.max(np.abs(p1["x_next"] - p2["x_next"])) == 0.0
    assert np.max(np.abs(p1["de_std"] - p2["de_std"])) == 0.0
    assert ast_leakage_audit_phase18()["passed"]
