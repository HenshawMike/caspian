"""Unit, analytical gradient, semantic firewall, determinism, and AST leakage tests for Phase 18."""
import numpy as np

from experiments.phase18_experiments import (
    W18_ALPHA,
    W18_BETA,
    ast_leakage_audit_phase18,
    run_single_seed,
)
from models.planner import AutonomousPlanner
from models.world_model import CaspianWorldModel


def test_p18_semantic_purity_and_ast_zero_leakage():
    obs = W18_ALPHA.make_neutral_observation(np.array([0.2, -0.3, 0.1, 0.4]), 10.0, 0, None)
    obs.assert_semantic_purity()
    audit = ast_leakage_audit_phase18()
    assert audit["passed"], f"AST leakage violations: {audit['violations']}"


def test_p18_analytical_transition_gradients_match_finite_diff():
    tr = W18_ALPHA.collect_transitions(n_steps=80, seed=9201)
    cwm = CaspianWorldModel(seed=9201)
    eid = cwm.observe_transitions(tr["X_t"], tr["A_t"], tr["X_next"], tr["dE"])
    Z = cwm.encode(tr["X_t"][:6], eid=eid)
    dyn = cwm.dynamics[eid]
    a = 1
    rng = np.random.default_rng(7)
    target = rng.normal(size=Z.shape)

    def loss_fn(Z_in, theta_mat):
        old = dyn.theta_trans[a].copy()
        dyn.theta_trans[a] = theta_mat
        pred = cwm.forward_transition(Z_in, a, eid=eid)
        dyn.theta_trans[a] = old
        return 0.5 * float(np.sum((pred - target) ** 2))

    pred0 = cwm.forward_transition(Z, a, eid=eid)
    grad_z_ana = cwm.backward_transition(pred0 - target, eid=eid)
    grad_th_ana = cwm.grad_theta_trans[a].copy()

    eps = 1e-6
    grad_z_num = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Zp, Zm = Z.copy(), Z.copy()
            Zp[i, j] += eps; Zm[i, j] -= eps
            grad_z_num[i, j] = (loss_fn(Zp, dyn.theta_trans[a]) - loss_fn(Zm, dyn.theta_trans[a])) / (2 * eps)

    grad_th_num = np.zeros_like(dyn.theta_trans[a])
    for i in range(grad_th_num.shape[0]):
        for j in range(grad_th_num.shape[1]):
            Tp, Tm = dyn.theta_trans[a].copy(), dyn.theta_trans[a].copy()
            Tp[i, j] += eps; Tm[i, j] -= eps
            grad_th_num[i, j] = (loss_fn(Z, Tp) - loss_fn(Z, Tm)) / (2 * eps)

    assert np.max(np.abs(grad_z_ana - grad_z_num)) < 1e-6
    assert np.max(np.abs(grad_th_ana - grad_th_num)) < 1e-6


def test_p18_internal_counterfactuals_and_ood_uncertainty():
    tr = W18_ALPHA.collect_transitions(n_steps=160, seed=9201)
    cwm = CaspianWorldModel(seed=9201)
    eid = cwm.observe_transitions(tr["X_t"], tr["A_t"], tr["X_next"], tr["dE"])

    # Internal counterfactuals produce finite distinct predictions across all 5 actions
    x0 = W18_ALPHA.observe(np.array([-0.8, -0.2, 0.0, 0.0]))
    cf = cwm.evaluate_counterfactuals(x0, eid=eid)
    assert cf["de_mean"].shape == (5,)
    assert np.all(np.isfinite(cf["de_mean"]))
    assert np.std(cf["de_mean"]) > 0.1

    # OOD score clearly separates W18_beta from W18_alpha
    id_ood = float(np.mean(cwm.subspace_ood_score(tr["X_t"][:20], eid=eid)))
    te_beta = W18_BETA.collect_transitions(n_steps=20, seed=9202)
    ood_ood = float(np.mean(cwm.subspace_ood_score(te_beta["X_t"], eid=eid)))
    assert id_ood < 0.05 and ood_ood > 0.30

    # Planner returns valid action without touching any environment
    planner = AutonomousPlanner(cwm, horizon=3, beta_uncertainty=0.5)
    act, info = planner.plan_action(x0, eid=eid)
    assert 0 <= act < 5
    assert len(info["best_sequence"]) == 3


def test_p18_deterministic_reproducibility():
    r1 = run_single_seed(9201)
    r2 = run_single_seed(9201)
    assert r1["H1"]["cwm_x_nrmse_h1"] == r2["H1"]["cwm_x_nrmse_h1"]
    assert r1["H5"]["return_integrated_caspian"] == r2["H5"]["return_integrated_caspian"]
    assert r1["H6"]["memory_footprint"]["raw_observations_stored_bytes"] == 0
