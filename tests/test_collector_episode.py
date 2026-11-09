import unittest
import json
from pathlib import Path
from environment.world import WorldConfig
from learning.collector import TrajectoryCollector


class TestEpisodeCollection(unittest.TestCase):
    def setUp(self):
        config_path = Path(__file__).parent.parent / "configs" / "phase2_world.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.world_config = WorldConfig.from_dict(json.load(f))

    def test_collects_correct_number_and_length(self):
        collector = TrajectoryCollector(world_config=self.world_config)
        ds = collector.collect_episode_dataset(num_episodes=5, steps_per_episode=16, base_seed=1)
        self.assertEqual(len(ds), 5)
        for ep in ds.episodes:
            self.assertEqual(len(ep), 16)
            self.assertEqual(len(ep.entity_observable), 16)

    def test_observable_flags_follow_flicker_schedule(self):
        # phase2_world.json: period=8, visible_steps=3 -> visible at t%8 in {0,1,2} (t after increment)
        collector = TrajectoryCollector(world_config=self.world_config)
        ds = collector.collect_episode_dataset(num_episodes=1, steps_per_episode=16, base_seed=1)
        ep = ds.episodes[0]
        expected = [((t + 1) % 8) < 3 for t in range(16)]
        self.assertEqual(ep.entity_observable, expected)

    def test_seeded_reproducibility(self):
        collector = TrajectoryCollector(world_config=self.world_config)
        ds1 = collector.collect_episode_dataset(num_episodes=3, steps_per_episode=10, base_seed=99)
        ds2 = collector.collect_episode_dataset(num_episodes=3, steps_per_episode=10, base_seed=99)
        for ep1, ep2 in zip(ds1.episodes, ds2.episodes):
            for tr1, tr2 in zip(ep1.transitions, ep2.transitions):
                self.assertEqual(tr1.action, tr2.action)
                self.assertEqual(tr1.state_delta, tr2.state_delta)


if __name__ == "__main__":
    unittest.main()
