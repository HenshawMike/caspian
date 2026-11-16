import numpy as np
from experiments import phase10_experiments as P


def test_eta_squared_recovers_planted_effects():
    rng = np.random.default_rng(0)
    T = np.repeat([0, 0, 1, 1], 200); R = np.repeat([0, 1, 0, 1], 200)
    H = np.stack([3.0 * T + rng.normal(0, 0.1, 800), 1.0 * R + rng.normal(0, 0.1, 800), rng.normal(0, 1.0, 800)], axis=1)
    e = P.eta_squared(H, T, R)
    assert e["type"] > 0.5 and e["type"] > 5 * e["region"] and e["region"] > e["interaction"]
    assert abs(sum(e.values()) - 1.0) < 1e-9


def test_eta_squared_constant_activations():
    e = P.eta_squared(np.ones((8, 3)), np.array([0, 0, 1, 1] * 2), np.array([0, 1] * 4))
    assert e["residual"] == 1.0


def test_compositional_world_rule_and_properties():
    for combo in P.TRAIN_COMBOS + P.HELDOUT_COMBOS:
        cfg = P.make_comp_world([(combo, 1)])
        X, Y, Pp = P._interaction_rows(P.p7._collect_passive(cfg, seed=3, num_steps=400))
        assert len(X) > 3
        assert np.allclose(Y, P.combo_delta(combo) - 1.0)            # step penalty 1
        assert np.allclose(Pp, [P.COMBO_VALUES[int(x)] for x in combo])


def test_interaction_variant_breaks_additivity():
    add = P.combo_delta((1, 1, 0), "additive"); inter = P.combo_delta((1, 1, 0), "interaction")
    assert inter - add == 5.0 and P.combo_delta((1, 0, 0), "interaction") == P.combo_delta((1, 0, 0), "additive")


def test_heldout_combos_never_in_training_for_any_arm_and_budgets_exact():
    for arm in ["passive_natural", "passive_uniform", "coverage_count", "one_factor"]:
        ds = P.collect_arm(arm, seed=5, budget=700)
        assert len(ds.transitions) == 700
        rep = P.leakage_report(ds)
        assert rep["heldout_property_vectors_in_training"] == [] and rep["heldout_codes_in_training"] == []
        codes = {int(e.entity_type) for t in ds.transitions for e in t.obs.entities}
        assert codes, "leakage test must be able to see codes"


def test_train_and_heldout_sets_are_disjoint_and_identifiable():
    assert not set(P.TRAIN_COMBOS) & set(P.HELDOUT_COMBOS)
    A = np.array([[*c, 1] for c in P.TRAIN_COMBOS], dtype=float)
    assert np.linalg.matrix_rank(A) == 4            # additive weights are identifiable from the training combos


def test_baselines_on_noise_free_additive_data():
    ds = P.collect_arm("coverage_count", seed=2, budget=1500)
    X, Y, Pt = P._interaction_rows(ds)
    (Xh, Yh, Ph), _ = P._eval_sets(2, "additive")
    b = P._baselines(Pt, Y, Ph, Yh)
    assert b["additive_linear_on_channels"] > 0.99      # structure oracle extrapolates perfectly
    assert b["nearest_combo_memorization"] < b["additive_linear_on_channels"]
