import numpy as np
import pytest
import experiments.phase7_experiments as p7
from experiments.phase11_experiments import interaction_rows
from experiments.phase14_experiments import (
    ARCHS14,
    N_PROPS14,
    W14_HIGHER_ORDER,
    W14_INDEP,
    W14_MULTI_PAIR,
    W14_NOISY_PAIR,
    W14_SPARSE_PAIR,
    WORLDS14,
    collect_train_p14,
    make_p14_world,
    run_cell_p14,
)
from models.interactions import SparseInteractionDiscoveryModel, build_p14_model


@pytest.mark.parametrize("arch", ARCHS14)
def test_p14_gradients_match_numerical(arch):
    if arch == "sparse_interaction":
        m = SparseInteractionDiscoveryModel(n_props=N_PROPS14, max_order=3, l1_lambda=0.12, freeze_sparse_after_init=False, seed=3)
        m.w_int = np.random.default_rng(7).uniform(0.2, 0.9, size=m.w_int.shape) * np.random.default_rng(8).choice([-1.0, 1.0], size=m.w_int.shape)
    else:
        m = build_p14_model(arch, N_PROPS14, seed=3)

    X = np.random.default_rng(1).uniform(0.2, 0.8, size=(8, N_PROPS14))
    Y = np.random.default_rng(2).normal(size=(8, 1))
    m.zero_grad()
    y = m.forward(X)
    m.backward(2.0 * (y - Y) / y.size)

    def total_loss():
        yp = m.forward(X)
        mse = float(np.mean((yp - Y) ** 2))
        if isinstance(m, SparseInteractionDiscoveryModel):
            return mse + m.l1_lambda * float(np.sum(np.abs(m.w_int)))
        return mse

    for name, p, g in m.get_params()[::4]:
        idx = tuple(0 for _ in p.shape)
        ana = float(g[idx])
        eps = 1e-6
        p[idx] += eps
        l1 = total_loss()
        p[idx] -= 2 * eps
        l2 = total_loss()
        p[idx] += eps
        num = (l1 - l2) / (2 * eps)
        assert abs(ana - num) < 1e-5, (arch, name, ana, num)


@pytest.mark.parametrize("spec", [W14_INDEP, W14_SPARSE_PAIR, W14_MULTI_PAIR, W14_HIGHER_ORDER])
def test_p14_worlds_truth_semantic_purity_and_zero_leakage(spec):
    assert len(spec.train_combos) == 22 and len(spec.heldout_combos) == 10
    assert not (set(spec.train_combos) & set(spec.heldout_combos))
    for c in spec.heldout_combos[:2]:
        ds = p7._collect_passive(make_p14_world(spec, [c]), seed=5, num_steps=400)
        for t in ds.transitions[:5]:
            t.obs.assert_semantic_purity()
        P, Y = interaction_rows(ds, N_PROPS14)
        assert len(P) > 3 and np.allclose(Y, spec.truth(P))

    _, Ptr, _ = collect_train_p14(spec, seed=31, steps=1200)
    held = set(spec.heldout_combos)
    assert not any(tuple(int(v) for v in row) in held for row in spec.decode(Ptr))


@pytest.mark.parametrize("spec", WORLDS14)
def test_sparse_interaction_recovers_exact_ground_truth_support(spec):
    _, Ptr, Ytr = collect_train_p14(spec, seed=31, steps=3600)
    m = SparseInteractionDiscoveryModel(N_PROPS14, seed=31)
    m.fit_sparse_hierarchy(Ptr, Ytr)
    disc = set(m.discovered_interactions().keys())
    assert disc == spec.true_interactions, (spec.name, disc, spec.true_interactions)


def test_run_cell_p14_smoke():
    r = run_cell_p14(W14_HIGHER_ORDER, "sparse_interaction", seed=9001, steps=2400, cache={}, epochs=10)
    assert r["leak_rows"] == 0 and r["finite"] and r["interaction_f1"] == 1.0
