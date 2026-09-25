"""Unit tests for delayed environment dynamics and temporal dependency tracking."""

import unittest
from environment.world import GridWorld, WorldConfig
from environment.entities import Entity
from environment.actions import Action


class TestDelayedDynamics(unittest.TestCase):
    """Test suite for delayed interaction dynamics in GridWorld."""

    def test_immediate_interaction_delay_zero(self):
        """Verify delay=0 delivers delta immediately (Phase 1 behavior)."""
        config = WorldConfig(
            grid_width=5,
            grid_height=5,
            initial_agent_pos=(2, 1),
            initial_entities=[
                Entity(
                    entity_id=1,
                    entity_type=1,
                    position=(2, 2),
                    is_interactive=True,
                    is_blocking=False,
                    hidden_state_delta=10.0,
                    interaction_delay=0,
                )
            ],
            initial_internal_state=50.0,
            step_penalty=1.0,
            seed=42,
        )
        world = GridWorld(config)

        # Agent interacts with entity at (2,2)
        obs, delta, done, info = world.step(Action.INTERACT)

        # Immediate delta: +10.0 - 1.0 (step penalty) = +9.0
        self.assertEqual(delta, 9.0)
        self.assertEqual(world.get_state().internal_state, 59.0)
        self.assertEqual(len(world.get_state().pending_events), 0)

    def test_delayed_interaction_consequence_timing(self):
        """Verify delay=d delivers consequence exactly at t + d steps."""
        delay = 2
        config = WorldConfig(
            grid_width=5,
            grid_height=5,
            initial_agent_pos=(2, 1),
            initial_entities=[
                Entity(
                    entity_id=1,
                    entity_type=1,
                    position=(2, 2),
                    is_interactive=True,
                    is_blocking=False,
                    hidden_state_delta=10.0,
                    interaction_delay=delay,
                )
            ],
            initial_internal_state=50.0,
            step_penalty=1.0,
            seed=42,
        )
        world = GridWorld(config)

        # Step 1: Agent interacts at t=0 -> target t=1. Interaction triggers, event scheduled for t=1+2=3.
        obs1, delta1, _, info1 = world.step(Action.INTERACT)
        self.assertEqual(delta1, -1.0, "Step 1 delta should only reflect step penalty, not delayed delta.")
        self.assertEqual(world.get_state().internal_state, 49.0)
        self.assertEqual(len(world.get_state().pending_events), 1)

        # Step 2: Agent moves NOOP at t=1 -> target t=2. Event still pending for t=3.
        obs2, delta2, _, _ = world.step(Action.NOOP)
        self.assertEqual(delta2, -1.0)
        self.assertEqual(world.get_state().internal_state, 48.0)
        self.assertEqual(len(world.get_state().pending_events), 1)

        # Step 3: Agent moves NOOP at t=2 -> target t=3. Event matures at t=3!
        obs3, delta3, _, _ = world.step(Action.NOOP)
        # Expected delta: -1.0 (step penalty) + 10.0 (delayed consequence) = +9.0
        self.assertEqual(delta3, 9.0, "Delayed consequence of +10.0 must arrive exactly at target timestep 3.")
        self.assertEqual(world.get_state().internal_state, 57.0)
        self.assertEqual(len(world.get_state().pending_events), 0)

    def test_environment_reset_clears_pending_events(self):
        """CRITICAL ISOLATION TEST: Resetting world clears any scheduled pending events."""
        config = WorldConfig(
            grid_width=5,
            grid_height=5,
            initial_agent_pos=(2, 1),
            initial_entities=[
                Entity(
                    entity_id=1,
                    entity_type=1,
                    position=(2, 2),
                    is_interactive=True,
                    is_blocking=False,
                    hidden_state_delta=10.0,
                    interaction_delay=5,
                )
            ],
            seed=42,
        )
        world = GridWorld(config)

        # Episode A: trigger interaction
        world.step(Action.INTERACT)
        self.assertEqual(len(world.get_state().pending_events), 1)

        # Reset world for Episode B
        world.reset(seed=101)
        self.assertEqual(
            len(world.get_state().pending_events),
            0,
            "Pending event queue must be strictly empty after reset."
        )


if __name__ == "__main__":
    unittest.main()
