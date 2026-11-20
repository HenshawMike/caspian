import numpy as np

from experiments.phase16b_independent_harness import (
    M_SHIFT_IND,
    W_NEG,
    W_OBS,
    W_SRC,
    IndWorldSpec,
    _w_source,
    ind_nrmse,
)
from models.transfer_v3 import ModularTransferModelV3


def test_v3_gradients_match_numerical_and_param_count():
    m = ModularTransferModelV3(in_dim=8, n_latent=4, seed=5)
    assert m.num_parameters == 66
    rng = np.random.default_rng(11)
    m.W_adapter = rng.normal(0.0, 0.4, size=m.W_adapter.shape)
    m.b_adapter = rng.normal(0.0, 0.2, size=m.b_adapter.shape)
    m.w_core = rng.normal(0.0, 0.4, size=m.w_core.shape)
    m.w_world = rng.normal(0.0, 0.4, size=m.w_world.shape)
    X = rng.normal(0.0, 1.0, size=(6, 8))
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


def test_v3_exact_permutation_invariance_and_offset_invariance():
    Xs, Ys = W_SRC.sample(320, 101)
    Xt, Yt = W_OBS.sample(20, 1121)
    Xte, Yte = W_OBS.eval_grid()

    m0 = ModularTransferModelV3(8, 4, seed=101)
    m0.fit_source(Xs, Ys)

    # Exact permutation invariance on W16_obs_shift
    m1 = m0.clone_for_target(); m1.adapt_few_shot(Xt, Yt, align_sensors="auto")
    perm = np.random.default_rng(42).permutation(len(Xt))
    m2 = m0.clone_for_target(); m2.adapt_few_shot(Xt[perm], Yt[perm], align_sensors="auto")
    assert float(np.max(np.abs(m1.forward(Xte) - m2.forward(Xte)))) < 1e-12

    # Exact invariance to additive sensor offset c = 0.5
    w_off = IndWorldSpec("W16_off", _w_source, M_SHIFT_IND, 36.0, additive_offset=0.5)
    Xt_off, Yt_off = w_off.sample(20, 1121)
    Xte_off, Yte_off = w_off.eval_grid()
    m3 = m0.clone_for_target(); g3 = m3.adapt_few_shot(Xt_off, Yt_off, align_sensors="auto")
    assert g3 == 1.0
    assert abs(ind_nrmse(m3.forward(Xte_off), Yte_off, 36.0) - ind_nrmse(m1.forward(Xte), Yte, 36.0)) < 1e-10


def test_v3_negative_control_gate_rejects_and_avoids_seed107_blowup():
    Xs, Ys = W_SRC.sample(320, 107)
    Xt, Yt = W_NEG.sample(20, 1127)
    Xte, Yte = W_NEG.eval_grid()
    m = ModularTransferModelV3(8, 4, seed=107)
    m.fit_source(Xs, Ys)
    mc = m.clone_for_target()
    g = mc.adapt_few_shot(Xt, Yt, align_sensors="auto")
    assert g == 0.0
    assert ind_nrmse(mc.forward(Xte), Yte, 36.0) < 0.02
