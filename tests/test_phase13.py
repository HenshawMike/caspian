import numpy as np
import pytest
import experiments.phase7_experiments as p7
from experiments.phase11_experiments import interaction_rows, train_model
from experiments.phase13_experiments import (
    ARCHS13,
    RAW_DIM,
    W1_RAW,
    W2_RAW,
    W3_RAW,
    collect_eval_raw,
    collect_train_raw,
    make_raw_world,
    run_cell_p13,
    slot_match_r2,
)
from models.discovery import (
    LatentDiscoveryModel,
    OrthogonalUnmixingLayer,
    _kurtosis_loss_and_grad_z,
    build_discovery_model,
)


def test_kurtosis_loss_and_grad_numerical():
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((16, 6))
    loss0, grad_z = _kurtosis_loss_and_grad_z(Z)
    eps = 1e-6
    for i in range(0, 16, 5):
        for j in range(6):
            Z[i, j] += eps
            l1, _ = _kurtosis_loss_and_grad_z(Z)
            Z[i, j] -= 2 * eps
            l2, _ = _kurtosis_loss_and_grad_z(Z)
            Z[i, j] += eps
            num = (l1 - l2) / (2 * eps)
            assert abs(grad_z[i, j] - num) < 1e-6, (i, j, grad_z[i, j], num)


@pytest.mark.parametrize("mode", ["additive", "residual"])
def test_latent_discovery_gradients_match_numerical(mode):
    m = LatentDiscoveryModel(raw_dim=12, n_slots=6, mode=mode, kurt_weight=0.05, orth_weight=0.1, freeze_unmix=False, seed=3)
    X = np.random.default_rng(1).standard_normal((8, 12))
    Y = np.random.default_rng(2).standard_normal((8, 1))
    m.zero_grad()
    y = m.forward(X)
    m.backward(2.0 * (y - Y) / y.size)

    def total_loss():
        yp = m.forward(X)
        mse = float(np.mean((yp - Y) ** 2))
        return mse + m.unmix.auxiliary_loss()

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
        assert abs(ana - num) < 1e-5, (mode, name, ana, num)


@pytest.mark.parametrize("spec", [W1_RAW, W2_RAW, W3_RAW])
def test_raw_world_entanglement_and_no_leakage(spec):
    # Verify M is row-orthonormal and dense (no raw channel aligned with a single factor)
    assert spec.M.shape == (6, RAW_DIM)
    assert np.allclose(spec.M @ spec.M.T, np.eye(6), atol=1e-7)
    assert np.max(np.abs(spec.M)) < 0.85

    # Verify simulator truth match and Semantic Firewall purity on held-out combinations
    for c in spec.heldout_combos[:2]:
        ds = p7._collect_passive(make_raw_world(spec, [c], "additive"), seed=5, num_steps=800)
        for t in ds.transitions[:10]:
            t.obs.assert_semantic_purity()
        X, Y = interaction_rows(ds, RAW_DIM)
        assert len(X) > 5 and np.allclose(Y, spec.truth(X, "additive"))

    # Verify 0 combination leakage in training collection
    _, Xtr, _ = collect_train_raw(spec, seed=21, steps=2000)
    held = {tuple(c) for c in spec.heldout_combos}
    assert not any(tuple(int(v) for v in row) in held for row in spec.decode(Xtr))


def test_orthogonal_unmixing_recovers_all_properties_and_beats_pca():
    _, Xtr, _ = collect_train_raw(W1_RAW, seed=21, steps=4000)
    Xh = collect_eval_raw(W1_RAW, seed=21, which="heldout", steps_per_combo=800)
    X_all = np.vstack([Xtr, Xh])
    P_true, _ = W1_RAW.unmix_ground_truth(X_all)

    unmix = OrthogonalUnmixingLayer(RAW_DIM, n_slots=6, seed=21)
    unmix.fit_subspace(Xtr)
    Z_disc = unmix.forward(X_all)
    assert slot_match_r2(Z_disc, P_true) > 0.98

    pca = build_discovery_model("pca_statistical", RAW_DIM, seed=21)
    pca.fit_normalizer(Xtr)
    Z_pca = pca.get_latent_state(X_all)
    assert slot_match_r2(Z_pca, P_true) < 0.20


def test_run_cell_p13_smoke():
    cache = {}
    r = run_cell_p13(W1_RAW, "latent_additive", seed=9001, steps=2000, cache=cache, epochs=5)
    assert r["leak_rows"] == 0 and r["finite"] and r["slot_match_r2"] > 0.90
