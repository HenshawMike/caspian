import numpy as np
import pytest
from experiments.phase15_experiments import (
    ARCHS15,
    N_VARS15,
    W15_CORR_VS_CAUSE,
    W15_HIDDEN_CONFOUNDER,
    W15_MULTI_CAUSE,
    W15_OBS_CONFOUNDER,
    WORLDS15,
    collect_p15_datasets,
    run_cell_p15,
)
from models.causal import CausalStructuralModel


def test_causal_structural_model_gradients_match_numerical():
    m = CausalStructuralModel(n_vars=N_VARS15, freeze_mask=False, seed=3)
    m.linear.weights = np.random.default_rng(7).normal(size=(N_VARS15, 1))
    X = np.random.default_rng(1).uniform(0.2, 0.8, size=(8, N_VARS15))
    Y = np.random.default_rng(2).normal(size=(8, 1))
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
        assert abs(ana - num) < 1e-6, (name, ana, num)


@pytest.mark.parametrize("spec", WORLDS15)
def test_causal_intervention_recovers_true_parents_while_observational_confounded(spec):
    ds_map = collect_p15_datasets(spec, seed=41, steps=2400)
    X_int, Y_int, _ = ds_map["int"]
    m_int = CausalStructuralModel(spec.n_vars, seed=41)
    m_int.fit_causal_structure(X_int, Y_int)
    assert m_int.discovered_parents() == spec.true_parents, (spec.name, m_int.discovered_parents(), spec.true_parents)

    if spec.name == "W15_hidden_confounder":
        X_obs, Y_obs, _ = ds_map["obs"]
        m_obs = CausalStructuralModel(spec.n_vars, seed=41)
        m_obs.fit_causal_structure(X_obs, Y_obs)
        # Observational model is fooled by hidden confounder U into picking X_0 as a false parent
        assert 0 in m_obs.discovered_parents()


def test_run_cell_p15_smoke():
    r = run_cell_p15(W15_HIDDEN_CONFOUNDER, "causal_intervention", seed=9001, steps=2400, cache={})
    assert r["leak_rows"] == 0 and r["finite"] and r["causal_f1"] == 1.0
