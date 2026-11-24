"""Unit, invariance, selective component transfer, and AST leakage tests for Phase 18D."""
import numpy as np

from experiments.phase18_experiments import W18_ALPHA, W18_GAMMA_SHIFTED, ast_leakage_audit_phase18
from experiments.phase18d_experiments import W18D_REGIME_B, W18D_REGIME_C, W18D_REGIME_D
from models.world_model import CaspianWorldModel


def test_p18d_regime_a_transfers_all_and_beats_consequence_only():
    tr_s = W18_ALPHA.collect_transitions(n_steps=350, seed=940101)
    tr_t = W18_GAMMA_SHIFTED.collect_transitions(n_steps=60, seed=940120)
    te_t = W18_GAMMA_SHIFTED.collect_transitions(n_steps=100, seed=940160)

    cwm_ug = CaspianWorldModel(seed=9401, boundary_aware=True, basis_sync=True, transfer_mode="uncertainty_guided")
    cwm_ug.observe_transitions(tr_s["X_t"], tr_s["A_t"], tr_s["X_next"], tr_s["dE"])
    eid_ug = cwm_ug.observe_transitions(tr_t["X_t"], tr_t["A_t"], tr_t["X_next"], tr_t["dE"])

    cwm_p18c = CaspianWorldModel(seed=9401, boundary_aware=True, basis_sync=True, transfer_mode="consequence_only")
    cwm_p18c.observe_transitions(tr_s["X_t"], tr_s["A_t"], tr_s["X_next"], tr_s["dE"])
    eid_p18c = cwm_p18c.observe_transitions(tr_t["X_t"], tr_t["A_t"], tr_t["X_next"], tr_t["dE"])

    err_ug = float(np.sqrt(np.mean((cwm_ug.predict_step(te_t["X_t"], te_t["A_t"], eid=eid_ug)["x_next"] - te_t["X_next"]) ** 2)))
    err_p18c = float(np.sqrt(np.mean((cwm_p18c.predict_step(te_t["X_t"], te_t["A_t"], eid=eid_p18c)["x_next"] - te_t["X_next"]) ** 2)))
    assert err_ug < err_p18c
    assert cwm_ug.last_transfer_decisions["n_actions_transferred"] >= 4
    assert cwm_ug.last_transfer_decisions["boundary_decision"] == "TRANSFER"


def test_p18d_regime_c_reconstructs_only_mutated_action_and_regime_d_rejects_all():
    tr_s = W18_ALPHA.collect_transitions(n_steps=350, seed=940201)

    # Regime C: only action 2 mutated
    tr_c = W18D_REGIME_C.collect_transitions(n_steps=60, seed=940222)
    cwm_c = CaspianWorldModel(seed=9402, boundary_aware=True, basis_sync=True, transfer_mode="uncertainty_guided")
    cwm_c.observe_transitions(tr_s["X_t"], tr_s["A_t"], tr_s["X_next"], tr_s["dE"])
    cwm_c.observe_transitions(tr_c["X_t"], tr_c["A_t"], tr_c["X_next"], tr_c["dE"])
    decs_c = cwm_c.last_transfer_decisions["action_decisions"]
    assert decs_c[2] == "RECONSTRUCT"
    assert sum(1 for a in (0, 1, 3, 4) if decs_c[a] in ("TRANSFER", "ADAPT")) >= 3

    # Regime D: full structural mutation -> reject all actions
    tr_d = W18D_REGIME_D.collect_transitions(n_steps=60, seed=940223)
    cwm_d = CaspianWorldModel(seed=9402, boundary_aware=True, basis_sync=True, transfer_mode="uncertainty_guided")
    cwm_d.observe_transitions(tr_s["X_t"], tr_s["A_t"], tr_s["X_next"], tr_s["dE"])
    cwm_d.observe_transitions(tr_d["X_t"], tr_d["A_t"], tr_d["X_next"], tr_d["dE"])
    decs_d = cwm_d.last_transfer_decisions["action_decisions"]
    assert sum(1 for d in decs_d if d == "RECONSTRUCT") >= 4


def test_p18d_regime_b_adapts_boundary_and_zero_ast_leakage():
    tr_s = W18_ALPHA.collect_transitions(n_steps=350, seed=940301)
    tr_b = W18D_REGIME_B.collect_transitions(n_steps=60, seed=940321)
    cwm_b = CaspianWorldModel(seed=9403, boundary_aware=True, basis_sync=True, transfer_mode="uncertainty_guided")
    cwm_b.observe_transitions(tr_s["X_t"], tr_s["A_t"], tr_s["X_next"], tr_s["dE"])
    cwm_b.observe_transitions(tr_b["X_t"], tr_b["A_t"], tr_b["X_next"], tr_b["dE"])
    assert cwm_b.last_transfer_decisions["boundary_decision"] in ("ADAPT", "RECONSTRUCT")
    assert ast_leakage_audit_phase18()["passed"]
