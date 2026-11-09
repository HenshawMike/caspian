import unittest
from environment.entities import Entity


class TestEntityFlicker(unittest.TestCase):
    def test_default_always_observable(self):
        e = Entity(entity_id=1, entity_type=1, position=(2, 2))
        for t in range(20):
            self.assertTrue(e.is_observable(t))

    def test_flicker_pattern(self):
        e = Entity(entity_id=1, entity_type=1, position=(2, 2), flicker_period=8, flicker_visible_steps=3)
        expected = [True, True, True, False, False, False, False, False]
        for t in range(24):
            self.assertEqual(e.is_observable(t), expected[t % 8], msg=f"t={t}")

    def test_zero_period_treated_as_always_visible(self):
        e = Entity(entity_id=1, entity_type=1, position=(0, 0), flicker_period=0, flicker_visible_steps=0)
        self.assertTrue(e.is_observable(5))

    def test_roundtrip_dict_with_flicker(self):
        e = Entity(entity_id=2, entity_type=1, position=(1, 1), flicker_period=5, flicker_visible_steps=2)
        e2 = Entity.from_dict(e.to_dict())
        self.assertEqual(e, e2)

    def test_roundtrip_dict_without_flicker_backward_compatible(self):
        legacy_dict = {
            "entity_id": 1, "entity_type": 1, "position": (2, 2),
            "is_interactive": True, "is_blocking": False, "hidden_state_delta": 10.0,
        }
        e = Entity.from_dict(legacy_dict)
        self.assertIsNone(e.flicker_period)
        self.assertTrue(e.is_observable(0))
        self.assertTrue(e.is_observable(999))


if __name__ == "__main__":
    unittest.main()
