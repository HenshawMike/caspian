import numpy as np
from experiments.phase12_experiments import W3, collect_train, collect_eval, run_one, ARCHS12
from experiments.phase11_experiments import baselines, rmse, make_world, interaction_rows
import experiments.phase7_experiments as p7
from models.compositional import build_model
from experiments.phase11_experiments import train_model, predict


def test_w3_split_and_identifiability():
    train, held = set(W3.train_combos), set(W3.heldout_combos)
    assert not train & held and train | held == set(W3.all_combos())
    # AB term visible in training: some training combo has A=B=1; additive fit cannot match it
    assert any(c[0] == 1 and c[1] == 1 for c in train)
    assert all(not (c[0] == 1 and c[1] == 1) for c in held)
    _, P, Y = collect_train(W3, 3, 4000)
    Ph = collect_eval(W3, 3, "heldout", 1500)
    b = baselines(W3, P, Y, Ph)
    assert rmse(b["additive_fit"], W3.truth(Ph, "additive")) > 1.0  # an additive fit to training data is wrong in W3
    assert rmse(b["oracle_additive_formula"], W3.truth(Ph, "additive")) == 0  # (in W3 "oracle_additive_formula" is the true rule)


def test_w3_truth_matches_simulator_no_leakage():
    for c in W3.heldout_combos:
        ds = p7._collect_passive(make_world(W3, [c], "additive"), seed=5, num_steps=1500)
        P, Y = interaction_rows(ds, 3)
        assert len(P) > 10 and np.allclose(Y, W3.truth(P, "additive"))
    _, P, _ = collect_train(W3, 5, 2500)
    held = {tuple(c) for c in W3.heldout_combos}
    assert not any(tuple(int(x) for x in l) in held for l in W3.decode(P))


def test_additive_residual_gradients_and_params():
    m = build_model("additive_residual", 3, 0); assert m.num_parameters == 1434
    X = np.random.default_rng(1).random((6, 3)); Y = np.random.default_rng(2).random((6, 1))
    m.zero_grad(); y = m.forward(X); m.backward(2 * (y - Y) / y.size)
    loss = lambda: float(np.mean((m.forward(X) - Y) ** 2))
    for name, p, g in m.get_params()[::5]:
        i = tuple(0 for _ in p.shape); a = g[i].copy(); eps = 1e-6
        p[i] += eps; l1 = loss(); p[i] -= 2 * eps; l2 = loss(); p[i] += eps
        assert abs(a - (l1 - l2) / (2 * eps)) < 1e-5, name


def test_run_one_smoke():
    r = run_one(W3, "additive_residual", 9001, 4000, {}, epochs=5)
    assert r["leak_rows"] == 0 and r["finite"]
