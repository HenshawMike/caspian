import numpy as np
import pytest
from experiments.phase16_experiments import (
    IN_DIM16,
    N_LATENT16,
    TARGET_WORLDS16,
    W16_NEGATIVE_CONTROL,
    W16_OBS_SHIFT,
    W16_SHARED_PLUS_NEW,
    W16_SOURCE,
    run_cell_p16,
    verify_gridworld_semantic_purity_p16,
)
from models.transfer import ModularTransferModel


def test_modular_transfer_gradients_match_numerical():
    m = ModularTransferModel(in_dim=IN_DIM16, n_latent=N_LATENT16, seed=3)
    rng = np.random.default_rng(7)
    m.W_adapter = rng.normal(0.0, 0.4, size=m.W_adapter.shape)
    m.w_core = rng.normal(0.0, 0.4, size=m.w_core.shape)
    m.w_world = rng.normal(0.0, 0.4, size=m.w_world.shape)
    X = rng.normal(0.0, 1.0, size=(6, IN_DIM16))
    Y = rng.normal(0.0, 1.0, size=(6, 1))

    m.zero_grad()
    y = m.forward(X)
    m.backward(2.0 * (y - Y) / y.size)

    def loss():
        return float(np.mean((m.forward(X) - Y) ** 2))

    for name, p, g in m.get_params():
        idx = tuple(0 for _ in p.shape)
        ana = float(g[idx])
        eps = 1e-6
        p[idx] += eps
        l1 = loss()
        p[idx] -= 2 * eps
        l2 = loss()
        p[idx] += eps
        num = (l1 - l2) / (2 * eps)
        assert abs(ana - num) < 1e-5, (name, ana, num)


@pytest.mark.parametrize("spec", TARGET_WORLDS16)
def test_p16_semantic_purity(spec):
    verify_gridworld_semantic_purity_p16(spec, seed=51)


def test_negative_transfer_gate_fires_on_inverted_world():
    X_src, Y_src = W16_SOURCE.sample(320, seed=51)
    X_neg, Y_neg = W16_NEGATIVE_CONTROL.sample(20, seed=51)
    m = ModularTransferModel(IN_DIM16, N_LATENT16, seed=51)
    m.fit_source(X_src, Y_src)
    gate = m.adapt_few_shot(X_neg, Y_neg, align_sensors=False, allow_negative_transfer_gate=True)
    assert gate == 0.0


def test_run_cell_p16_smoke():
    r = run_cell_p16(W16_SHARED_PLUS_NEW, "modular_transfer", seed=9001, n_shot=20, cache={})
    assert r["finite"] and r["leak_rows"] == 0 and r["source_retention_nrmse"] < 0.02
