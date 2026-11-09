import numpy as np
from experiments.phase9_experiments import make_world, _collect, _interaction_xy, run_experiment_e001, HELD_OUT, COMPLEXITY


def test_worlds_respect_held_out_split():
    for lv in COMPLEXITY:
        assert not set(make_world(lv).allowed_entity_positions) & set(HELD_OUT)
        assert set(make_world(lv, held_out_only=True).allowed_entity_positions) == set(HELD_OUT)


def test_held_out_test_world_yields_enough_varied_interactions():
    for lv in COMPLEXITY:
        X, Y = _interaction_xy(_collect(make_world(lv, held_out_only=True), seed=1, steps=3000))
        assert len(X) >= 100 and np.unique(Y).size >= 2   # the degenerate-0.0 failure mode


def test_small_sweep_runs_and_reports_checks():
    r = run_experiment_e001(seeds=[1], sizes={"tiny": (4,), "small": (8,)}, data_steps=[300, 900], levels=["C1"], epochs=5, test_steps=1500, verbose=False)
    assert len(r["cells"]) == 4 and "H3_met" in r["preregistered_checks"]
    assert all("params" in c for c in r["cells"].values())
