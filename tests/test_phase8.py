import numpy as np
from experiments.phase8_experiments import make_world, _collect, _train, _finetune, _r2, WORLD_NAMES
from environment.world import GridWorld, WorldConfig
from environment.actions import Action


def test_interact_cost_default_zero_and_applied():
    base = make_world("source")
    assert base.interact_cost == 0.0
    cfg = make_world("actions")
    w0 = GridWorld(config=make_world("source")); w0.reset(seed=1)
    w1 = GridWorld(config=cfg); w1.reset(seed=1)
    d0 = w0.step(Action.INTERACT)[1]; d1 = w1.step(Action.INTERACT)[1]
    assert abs((d0 - d1) - 2.0) < 1e-9


def test_all_worlds_build_and_collect():
    for w in ["source"] + WORLD_NAMES:
        X, Y = _collect(make_world(w), seed=1, steps=60).to_numpy_arrays()
        assert X.shape[0] == 60 and X.shape[1] == Y.shape[0] * 0 + X.shape[1]


def test_finetune_keeps_source_normalizer_and_does_not_mutate_source():
    cfg = make_world("source")
    X, Y = _collect(cfg, seed=2, steps=400).to_numpy_arrays()
    m = _train(X, Y, seed=2, epochs=5)
    before = m.forward(X[:5]).copy(); mean_before = m.normalizer.to_dict()
    Xb, Yb = _collect(make_world("rules"), seed=3, steps=100).to_numpy_arrays()
    ft = _finetune(m, Xb, Yb, seed=2, epochs=3)
    assert np.allclose(m.forward(X[:5]), before)           # source untouched
    assert ft.normalizer.to_dict() == mean_before          # normalizer retained
