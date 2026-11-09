import unittest
from environment.world import GridWorld
from environment.actions import Action
from experiments.phase7_experiments import (
    _step_towards, _step_away, _manhattan, _approach_and_interact,
    _confounded_type_configs, _swapped_type_configs, _collect_active_budgeted, _load_world,
)


class TestNavigationPrimitives(unittest.TestCase):
    def test_step_towards_moves_closer(self):
        for agent, target in [((0, 0), (3, 2)), ((4, 4), (1, 1)), ((2, 0), (2, 4)), ((3, 3), (0, 3))]:
            act = _step_towards(agent, target)
            dx, dy = act.get_delta()
            nxt = (agent[0] + dx, agent[1] + dy)
            self.assertLess(_manhattan(nxt, target), _manhattan(agent, target))

    def test_step_towards_at_target_is_noop(self):
        self.assertEqual(_step_towards((2, 2), (2, 2)), Action.NOOP)

    def test_step_away_increases_distance_and_stays_in_bounds(self):
        for agent, target in [((2, 2), (2, 2)), ((0, 0), (1, 0)), ((4, 4), (3, 4))]:
            act = _step_away(agent, target, 5, 5)
            dx, dy = act.get_delta()
            nxt = (agent[0] + dx, agent[1] + dy)
            self.assertTrue(0 <= nxt[0] < 5 and 0 <= nxt[1] < 5)
            self.assertGreaterEqual(_manhattan(nxt, target), _manhattan(agent, target))


class TestApproachAndInteract(unittest.TestCase):
    def test_reaches_entity_and_interacts(self):
        cfg = _load_world("phase7_world_single_type.json")
        for seed in range(10):
            world = GridWorld(config=cfg)
            world.reset(seed=seed)
            res = _approach_and_interact(world, world._entities[0].position)
            self.assertIsNotNone(res)
            self.assertTrue(res[2].get("interaction"))
            self.assertAlmostEqual(res[1], 9.0)  # +10 effect - 1 step penalty, no clamping at init=50


class TestConfoundConstruction(unittest.TestCase):
    def test_training_worlds_have_disjoint_regions(self):
        t1, t3 = _confounded_type_configs()
        self.assertTrue(set(t1.allowed_entity_positions).isdisjoint(set(t3.allowed_entity_positions)))

    def test_swapped_worlds_exchange_regions_but_keep_functions(self):
        t1, t3 = _confounded_type_configs()
        t1s, t3s = _swapped_type_configs()
        self.assertEqual(set(t1s.allowed_entity_positions), set(t3.allowed_entity_positions))
        self.assertEqual(set(t3s.allowed_entity_positions), set(t1.allowed_entity_positions))
        self.assertEqual(t1s.initial_entities[0].hidden_state_delta, t1.initial_entities[0].hidden_state_delta)
        self.assertEqual(t3s.initial_entities[0].hidden_state_delta, t3.initial_entities[0].hidden_state_delta)


class TestBudgetedActiveCollection(unittest.TestCase):
    def test_uses_exactly_the_step_budget(self):
        cfg = _load_world("phase6_world_base.json")
        for budget in (50, 137, 400):
            ds = _collect_active_budgeted(cfg, num_steps=budget, seed=1)
            self.assertEqual(len(ds), budget)

    def test_collects_real_interactions(self):
        cfg = _load_world("phase6_world_base.json")
        ds = _collect_active_budgeted(cfg, num_steps=300, seed=2)
        self.assertGreater(sum(1 for t in ds.transitions if t.info.get("interaction")), 10)


if __name__ == "__main__":
    unittest.main()


def test_e004_arm_respects_exact_budget_and_sites():
    from experiments.phase7_experiments import _run_e004_arm, _all_sites
    for arm in ["confounded", "random", "roundrobin", "count", "uncertainty"]:
        ds, chosen = _run_e004_arm(arm, seed=3, budget=900, seed_block=100, probe_steps=20, block=100)
        assert len(ds.transitions) == 900
        assert all(c in _all_sites() for c in chosen)
    _, chosen = _run_e004_arm("confounded", seed=3, budget=900, seed_block=100, block=100)
    assert set(chosen) <= {(1, (0, 0)), (3, (4, 4))}


def test_e005_arms_budget_and_consistent_broad_world():
    from experiments.phase7_experiments import _run_e005_arm, _load_world
    for arm in ["count", "count_mix", "uncertainty_v2", "uncertainty_disc", "uncertainty_disc_mix"]:
        ds, chosen = _run_e005_arm(arm, seed=3, budget=700, seed_block=100, probe_steps=20, block=100, members=2, epochs=5,
                                   broad_world="phase7_world_broad_consistent.json")
        assert len(ds.transitions) == 700
    pool = {e["entity_type"]: e["hidden_state_delta"] for e in [vars(x) if not isinstance(x, dict) else x for x in (_load_world("phase7_world_broad_consistent.json").entity_type_pool or [])]}
    assert pool.get(1) == 10.0 and pool.get(3) == -10.0


def test_outcome_noise_default_is_deterministic_and_noise_is_seeded():
    import copy
    from experiments.phase7_experiments import _load_world, _approach_and_interact
    from environment.world import GridWorld
    base = _load_world("phase7_world_single_type.json")
    assert base.outcome_noise_std == 0.0
    def outcomes(std, seed):
        cfg = copy.deepcopy(base); cfg.outcome_noise_std = std
        w = GridWorld(config=cfg); w.reset(seed=seed)
        r = _approach_and_interact(w, w._entities[0].position)
        return r[1]
    assert outcomes(0.0, 1) == outcomes(0.0, 2) == 9.0
    assert outcomes(3.0, 5) == outcomes(3.0, 5)      # seeded
    assert outcomes(3.0, 5) != 9.0                   # noise actually applied


def test_e007_arms_budget_and_pooled_r2_not_degenerate():
    from experiments.phase7_experiments import _e007_arm, _pooled_r2, _clean_site_config, _train_xy, _E007_TYPES, _E007_CORNERS
    for arm in ["confounded", "random", "count"]:
        ds = _e007_arm(arm, seed=2, budget=600, seed_block=100, block=100)
        assert len(ds.transitions) == 600
    m = _train_xy(_e007_arm("count", seed=2, budget=1500, seed_block=100, block=100), seed=2)
    cfgs = [_clean_site_config(t, d, _E007_CORNERS[i], 0.0) for i, (t, d) in enumerate(_E007_TYPES.items())]
    r2 = _pooled_r2(m, cfgs, seed=1, num_steps=300)
    assert r2 != 0.0 and r2 <= 1.0
