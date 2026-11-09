import unittest
from environment.world import GridWorld, WorldConfig
from environment.entities import Entity
from environment.actions import Action


def make_flicker_world(period=4, visible_steps=2, max_timesteps=40):
    config = WorldConfig(
        grid_width=5,
        grid_height=5,
        initial_agent_pos=(2, 2),
        initial_entities=[
            Entity(
                entity_id=1, entity_type=1, position=(2, 2),
                is_interactive=True, is_blocking=False, hidden_state_delta=10.0,
                flicker_period=period, flicker_visible_steps=visible_steps,
            )
        ],
        max_timesteps=max_timesteps,
        seed=42,
    )
    return GridWorld(config=config)


class TestWorldObservability(unittest.TestCase):
    def test_observation_hides_entity_on_schedule(self):
        world = make_flicker_world()
        world.reset(seed=1)
        expected = [True, True, False, False]  # period=4, visible_steps=2
        for t in range(12):
            obs = world.get_observation()
            self.assertEqual(len(obs.entities) == 1, expected[t % 4], msg=f"t={t}")
            world.step(Action.NOOP)

    def test_interaction_still_works_while_entity_invisible(self):
        # period=8, visible_steps=2 -> hidden for t in {2..7}; both the pre- and
        # post-interact timesteps below (3 and 4) fall in that hidden range.
        world = make_flicker_world(period=8, visible_steps=2)
        world.reset(seed=1)
        for _ in range(3):
            world.step(Action.NOOP)  # advance to t=3 -> entity hidden this step
        obs_before = world.get_observation()
        self.assertEqual(len(obs_before.entities), 0, "entity should be hidden at t=3")

        _, delta, _, info = world.step(Action.INTERACT)
        self.assertNotEqual(delta, 0.0, "interaction should still take effect on the ground-truth entity even while hidden")
        self.assertEqual(info["observable_entity_ids"], [], "entity should be reported as unobservable in info at this step")

    def test_observable_entity_ids_in_info(self):
        world = make_flicker_world()
        world.reset(seed=1)
        _, _, _, info = world.step(Action.NOOP)  # t=0 -> t=1 step; entity visible during period [0,2)
        self.assertIn("observable_entity_ids", info)

    def test_randomize_entity_positions_varies_across_episodes(self):
        config = WorldConfig(
            grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
            initial_entities=[Entity(entity_id=1, entity_type=1, position=(2, 2), flicker_period=8, flicker_visible_steps=3)],
            max_timesteps=20, seed=1, randomize_entity_positions=True,
        )
        world = GridWorld(config=config)
        positions = set()
        for seed in range(15):
            world.reset(seed=seed)
            positions.add(world._entities[0].position)
        self.assertGreater(len(positions), 1, "entity position should vary across episodes when randomize_entity_positions=True")
        self.assertNotIn((0, 0), positions, "entity should never spawn on the agent's start cell")

    def test_randomize_entity_positions_default_off_preserves_phase1_behavior(self):
        config = WorldConfig()  # default: randomize_entity_positions=False
        world = GridWorld(config=config)
        pos_before = world._entities[0].position
        world.reset(seed=999)
        self.assertEqual(world._entities[0].position, pos_before)

    def test_allowed_entity_positions_restricts_sampling(self):
        allowed = [(0, 2), (2, 0), (4, 4)]
        config = WorldConfig(
            grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
            initial_entities=[Entity(entity_id=1, entity_type=1, position=(2, 2))],
            max_timesteps=20, seed=1, randomize_entity_positions=True,
            allowed_entity_positions=allowed,
        )
        world = GridWorld(config=config)
        seen = set()
        for seed in range(30):
            world.reset(seed=seed)
            pos = world._entities[0].position
            self.assertIn(pos, allowed, f"entity spawned outside allowed_entity_positions: {pos}")
            seen.add(pos)
        self.assertGreater(len(seen), 1, "should sample more than one of the allowed cells across episodes")

    def test_entity_type_pool_resamples_functional_identity_each_episode(self):
        pool = [
            {"entity_type": 1, "hidden_state_delta": 10.0},
            {"entity_type": 2, "hidden_state_delta": -10.0},
            {"entity_type": 3, "hidden_state_delta": 10.0},
        ]
        config = WorldConfig(
            grid_width=5, grid_height=5, initial_agent_pos=(0, 0),
            initial_entities=[Entity(entity_id=1, entity_type=99, position=(2, 2))],
            max_timesteps=20, seed=1, entity_type_pool=pool,
        )
        world = GridWorld(config=config)
        seen_types = set()
        for seed in range(40):
            world.reset(seed=seed)
            e = world._entities[0]
            seen_types.add(e.entity_type)
            # hidden_state_delta must match whichever template this episode's type came from
            expected_delta = next(t["hidden_state_delta"] for t in pool if t["entity_type"] == e.entity_type)
            self.assertEqual(e.hidden_state_delta, expected_delta)
            # position should be untouched by the type pool (entity_type_pool alone, no randomize_entity_positions)
            self.assertEqual(e.position, (2, 2))
        self.assertEqual(seen_types, {1, 2, 3}, "should sample all three pool templates across enough episodes")

    def test_entity_type_pool_default_off_preserves_behavior(self):
        config = WorldConfig()  # entity_type_pool=None by default
        world = GridWorld(config=config)
        world.reset(seed=5)
        self.assertEqual(world._entities[0].entity_type, 1)
        self.assertEqual(world._entities[0].hidden_state_delta, 10.0)


if __name__ == "__main__":
    unittest.main()
