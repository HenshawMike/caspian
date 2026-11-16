import numpy as np
import pytest
from experiments.phase11_experiments import (W1, W2, ARCHS, build_model, train_model, predict, collect_train, collect_eval, make_world, interaction_rows,
                                             baselines, rmse, paired_summary, linear_cka, marginal_eta2, STEP_OFFSET)
import experiments.phase7_experiments as p7


@pytest.mark.parametrize("arch", ARCHS)
def test_gradients_match_numerical(arch):
    m = build_model(arch, 3, 0); X = np.random.default_rng(1).random((6, 3)); Y = np.random.default_rng(2).random((6, 1))
    m.zero_grad(); y = m.forward(X); m.backward(2 * (y - Y) / y.size)
    loss = lambda: float(np.mean((m.forward(X) - Y) ** 2))
    for name, p, g in m.get_params()[:: max(1, len(m.get_params()) // 4)]:
        i = tuple(0 for _ in p.shape); a = g[i].copy(); eps = 1e-6
        p[i] += eps; l1 = loss(); p[i] -= 2 * eps; l2 = loss(); p[i] += eps
        assert abs(a - (l1 - l2) / (2 * eps)) < 1e-5, (arch, name)


def test_additive_architecture_is_strictly_additive():
    m = build_model("additive", 3, 3); X = np.random.default_rng(0).random((20, 3)); train_model(m, X, X @ np.array([1., -2., 3.]) + 0.5, 0, 5)
    f = lambda a, b, c: float(predict(m, np.array([[a, b, c]]))[0])
    assert abs((f(.8, .8, .2) - f(.2, .8, .2)) - (f(.8, .2, .2) - f(.2, .2, .2))) < 1e-9


def test_param_counts_documented():
    counts = {a: build_model(a, 3, 0).num_parameters for a in ARCHS}
    assert counts == {"mlp_small": 673, "mlp_large": 2369, "factorised": 921, "additive": 513, "additive_mlp": 665}


@pytest.mark.parametrize("spec", [W1, W2])
def test_split_properties_and_negative_control(spec):
    held = set(spec.heldout_combos); train = set(spec.train_combos)
    assert not held & train and held | train == set(spec.all_combos())
    for c in train:  # interaction term must vanish on every training combination (unidentifiable)
        assert spec.delta(c, "additive") == spec.delta(c, "interaction")
    assert any(spec.delta(c, "additive") != spec.delta(c, "interaction") for c in held)


def test_w1_has_no_ab_in_training():
    assert all(not (c[0] == 1 and c[1] == 1) for c in W1.train_combos)
    assert W1.heldout_combos == [(1, 1, 0), (1, 1, 1)]


@pytest.mark.parametrize("spec", [W1, W2])
def test_truth_matches_simulator_and_no_leakage(spec):
    for variant in ("additive", "interaction"):
        for c in spec.heldout_combos[:2]:
            ds = p7._collect_passive(make_world(spec, [c], variant), seed=5, num_steps=1500)
            P, Y = interaction_rows(ds, spec.n_channels)
            assert len(P) > 10 and np.allclose(Y, spec.truth(P, variant))
    _, P, _ = collect_train(spec, 5, 2500)
    held = {tuple(c) for c in spec.heldout_combos}
    assert not any(tuple(int(x) for x in l) in held for l in spec.decode(P))


def test_training_data_identical_across_variants():
    spec = W1; P1, Y1 = interaction_rows(p7._collect_passive(make_world(spec, spec.train_combos, "additive"), 7, 2500), 3)
    P2, Y2 = interaction_rows(p7._collect_passive(make_world(spec, spec.train_combos, "interaction"), 7, 2500), 3)
    assert np.allclose(P1, P2) and np.allclose(Y1, Y2)


def test_baselines_additive_fit_identifiable_but_fails_on_interaction():
    for spec in (W1, W2):
        _, Ptr, Ytr = collect_train(spec, 3, 6000); Ph = collect_eval(spec, 3, "heldout", 1500)
        b = baselines(spec, Ptr, Ytr, Ph)
        assert rmse(b["additive_fit"], spec.truth(Ph, "additive")) < 1e-6
        assert rmse(b["additive_fit"], spec.truth(Ph, "interaction")) > 2.0
        assert rmse(b["oracle_additive_formula"], spec.truth(Ph, "additive")) == 0


def test_training_deterministic():
    spec = W1; _, P, Y = collect_train(spec, 11, 2000)
    a = build_model("factorised", 3, 11); train_model(a, P, Y, 11, 5)
    b = build_model("factorised", 3, 11); train_model(b, P, Y, 11, 5)
    assert np.array_equal(predict(a, P), predict(b, P))


def test_stats_helpers():
    s = paired_summary(np.zeros(10), np.ones(10) * 0.2, 0.1); assert s["met"] and s["seeds_meeting_threshold"] == 10
    s = paired_summary(np.zeros(10), np.random.default_rng(0).normal(0, 1, 10), 0.5); assert not s["met"]
    X = np.random.default_rng(0).normal(size=(50, 4)); assert abs(linear_cka(X, X) - 1) < 1e-9
    assert marginal_eta2(np.array([[0.], [0.], [1.], [1.]]), np.array([0, 0, 1, 1])) == 1.0
    assert STEP_OFFSET == -1.0
