import json
import numpy as np

from experiments.phase17_experiments import W17_ALPHA, W17_BETA, W17_GAMMA_SHARED
from experiments.phase17b_verification import run_ast_leakage_audit, run_memory_accounting
from models.memory import PersistentWorldMemory
from models.transfer_v3 import ModularTransferModelV3


def test_original_v3_and_occam_modes_both_available():
    m_orig = ModularTransferModelV3(8, 4, seed=1, alignment_mode="original_v3")
    m_occ = ModularTransferModelV3(8, 4, seed=1, alignment_mode="occam_hierarchy")
    assert m_orig.num_parameters == 66 and m_occ.num_parameters == 66
    assert m_orig.alignment_mode == "original_v3" and m_occ.alignment_mode == "occam_hierarchy"


def test_p17b_determinism_and_permutation_invariance():
    Xs, Ys = W17_ALPHA.sample(160, 9201)
    Xb, Yb = W17_BETA.sample(160, 9301)
    Xg, Yg = W17_GAMMA_SHARED.sample(10, 9401)
    Xge, _ = W17_GAMMA_SHARED.eval_grid()

    m1 = PersistentWorldMemory(8, 4, seed=9101, alignment_mode="occam_hierarchy", occam_restarts=30)
    m1.observe_world(Xs, Ys); m1.observe_world(Xb, Yb); m1.observe_world(Xg, Yg)
    p1 = np.asarray(m1.forward(Xge)).reshape(-1)

    perm = np.random.default_rng(123).permutation(len(Xg))
    m2 = PersistentWorldMemory(8, 4, seed=9101, alignment_mode="occam_hierarchy", occam_restarts=30)
    m2.observe_world(Xs, Ys); m2.observe_world(Xb, Yb); m2.observe_world(Xg[perm], Yg[perm])
    p2 = np.asarray(m2.forward(Xge)).reshape(-1)

    assert float(np.max(np.abs(p1 - p2))) < 1e-12


def test_p17b_ast_leakage_audit_zero_violations():
    res = run_ast_leakage_audit()
    assert res["passed"] is True and len(res["violations"]) == 0


def test_p17b_memory_accounting_reports_all_nine_categories():
    acct = run_memory_accounting(seed=9101)
    assert acct["1_trainable_parameter_count"] == 264
    assert acct["6_stored_raw_observations_bytes"] == 0
    assert acct["9_total_bytes_required_to_save_and_restore"] > 0


def test_p17b_saved_confirmatory_results_integrity():
    res = json.load(open("logs/phase17b/p17b_full_results.json"))
    assert res["ast_leakage_audit"]["passed"] is True
    assert res["hypotheses"]["P17B-H3"]["met"] is True
    assert res["hypotheses"]["P17B-H5"]["met"] is True
    assert res["hypotheses"]["P17B-H5"]["determinism_diff"] == 0.0
    assert res["hypotheses"]["P17B-H5"]["perm_diff"] == 0.0
    assert res["memory_accounting"]["9_total_bytes_required_to_save_and_restore"] > 0

