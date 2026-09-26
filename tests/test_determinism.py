"""Unit tests for exact mathematical determinism and reproducibility in Project Caspian."""

import unittest

from environment.world import GridWorld, WorldConfig
from environment.actions import Action
from environment.entities import Entity


class TestDeterminism(unittest.TestCase):
    """Test suite verifying bitwise determinism under fixed seeds across resets and instances."""

    def test_same_seed_produces_identical_initial_state(self):
        """Acceptance Test 1: Creating environment with same seed produces identical initial state."""
        # Unfixed initial pos relies on deterministic RNG
        config1 = WorldConfig(initial_agent_pos=None, seed=12345)
        config2 = WorldConfig(initial_agent_pos=None, seed=12345)

        world1 = GridWorld(config=config1)
        world2 = GridWorld(config=config2)

        state1 = world1.get_state()
        state2 = world2.get_state()

        self.assertEqual(state1.agent_position, state2.agent_position)
        self.assertEqual(state1.internal_state, state2.internal_state)
        self.assertEqual(state1.timestep, state2.timestep)
        self.assertEqual(world1.get_observation(), world2.get_observation())

    def test_differing_seeds_diverge(self):
        """Verify that distinct seeds produce different initial agent positions when randomized."""
        config1 = WorldConfig(initial_agent_pos=None, seed=12345)
        config2 = WorldConfig(initial_agent_pos=None, seed=99999)

        world1 = GridWorld(config=config1)
        world2 = GridWorld(config=config2)

        state1 = world1.get_state()
        state2 = world2.get_state()

        # With grid 5x5 and differing seeds, positions differ
        self.assertNotEqual(state1.agent_position, state2.agent_position)

    def test_same_action_sequence_produces_identical_trajectory(self):
        """Acceptance Test 2: Running the same sequence of actions with the same seed produces identical trajectory."""
        actions = [
            Action.RIGHT,
            Action.RIGHT,
            Action.UP,
            Action.UP,
            Action.INTERACT,
            Action.LEFT,
            Action.DOWN,
            Action.NOOP,
            Action.UP,
            Action.INTERACT,
        ]

        def run_trajectory(seed: int):
            world = GridWorld(WorldConfig(seed=seed))
            trajectory = []
            trajectory.append((world.get_observation().to_dict(), world.get_state()))
            for act in actions:
                obs, delta, done, info = world.step(act)
                trajectory.append((obs.to_dict(), world.get_state(), delta, done, info))
            return trajectory

        traj1 = run_trajectory(seed=42)
        traj2 = run_trajectory(seed=42)

        self.assertEqual(len(traj1), len(traj2))
        for step_idx, (step1, step2) in enumerate(zip(traj1, traj2)):
            self.assertEqual(
                step1,
                step2,
                f"Trajectory mismatch at step {step_idx} between identical seed runs."
            )

    def test_reset_reproducibility(self):
        """Acceptance Test 6: Reset returns environment to initial deterministic state and produces same trajectory."""
        world = GridWorld(WorldConfig(seed=777))
        actions = [Action.UP, Action.RIGHT, Action.INTERACT, Action.UP, Action.LEFT]

        # Run 1
        traj_run1 = []
        for a in actions:
            obs, delta, done, _ = world.step(a)
            traj_run1.append((obs.to_dict(), delta))

        # Reset
        world.reset(seed=777)

        # Run 2
        traj_run2 = []
        for a in actions:
            obs, delta, done, _ = world.step(a)
            traj_run2.append((obs.to_dict(), delta))

        self.assertEqual(traj_run1, traj_run2)

    def test_collision_and_interaction_determinism(self):
        """Acceptance Tests 7 & 8: Collision and interaction behavior are strictly deterministic."""
        config = WorldConfig(
            grid_width=3,
            grid_height=3,
            initial_agent_pos=(0, 0),
            initial_entities=[
                Entity(
                    entity_id=1,
                    entity_type=1,
                    position=(0, 1),
                    is_interactive=True,
                    is_blocking=False,
                    hidden_state_delta=25.0,
                )
            ],
            seed=42,
        )

        def execute_stress_sequence():
            world = GridWorld(config=config)
            steps_data = []
            # Intentional boundary collisions and interactions
            actions = [
                Action.DOWN,      # Boundary collision at bottom
                Action.LEFT,      # Boundary collision at left
                Action.UP,        # Move to (0, 1)
                Action.INTERACT,  # Interact with Entity 1 (in range)
                Action.UP,        # Move to (0, 2)
                Action.UP,        # Boundary collision at top
                Action.INTERACT,  # Interact out of range (dist=1 to (0,1) is still <= 1)
                Action.RIGHT,     # Move to (1, 2)
                Action.RIGHT,     # Move to (2, 2)
                Action.INTERACT,  # Interact out of range (dist=2+1=3 > 1)
            ]
            for a in actions:
                obs, delta, done, info = world.step(a)
                steps_data.append((obs.agent_position, obs.collision, obs.internal_state, delta, info))
            return steps_data

        run_a = execute_stress_sequence()
        run_b = execute_stress_sequence()
        self.assertEqual(run_a, run_b)


if __name__ == "__main__":
    unittest.main()
