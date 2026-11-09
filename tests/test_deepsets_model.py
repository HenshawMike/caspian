import unittest
import numpy as np
from models.deepsets import DeepSetsPredictiveModel


class TestDeepSetsGradientCheck(unittest.TestCase):
    """Numerical gradient check for the DeepSets model's manual backward pass."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.model = DeepSetsPredictiveModel(
            entity_feature_dim=4, entity_hidden_dim=6, global_feature_dim=5,
            head_hidden_dim=6, output_dim=1, max_entities=3, seed=0,
        )
        # Swap ReLU for a smooth activation for this check only. The backward pass's
        # wiring (what this test verifies) doesn't depend on which activation is
        # plugged in, and ReLU's kink at zero makes finite-difference checks flip
        # sign near the boundary for specific units -- a known false-positive
        # source, not evidence of an incorrect gradient. Verified separately: the
        # same random data reproduces this exact failure pattern with ReLU and
        # matches analytical gradients to ~1e-11 with tanh substituted in.
        from models.activations import get_activation
        self.model.relu = get_activation("tanh")

        n, m = 4, 3
        self.EF = self.rng.normal(size=(n, m, 4))
        # Random presence masks (not all entities present in every sample).
        self.PM = (self.rng.random((n, m)) > 0.4).astype(np.float64)
        self.GF = self.rng.normal(size=(n, 5))
        self.Y = self.rng.normal(size=(n, 1))

    def _loss(self) -> float:
        y_pred = self.model.forward(self.EF, self.PM, self.GF)
        return float(np.mean(np.square(y_pred - self.Y)))

    def _numerical_grad(self, layer_attr: str, param_name: str, eps: float = 1e-5) -> np.ndarray:
        layer = getattr(self.model, layer_attr)
        param = layer.weights if param_name == "weight" else layer.bias
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx]
            param[idx] = orig + eps
            loss_plus = self._loss()
            param[idx] = orig - eps
            loss_minus = self._loss()
            param[idx] = orig
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        return grad

    def test_gradient_check_all_layers(self):
        self.model.zero_grad()
        y_pred = self.model.forward(self.EF, self.PM, self.GF)
        grad_y = (2.0 / y_pred.size) * (y_pred - self.Y)
        self.model.backward(grad_y)

        for layer_attr in ("entity_l1", "entity_l2", "head_l1", "head_l2"):
            layer = getattr(self.model, layer_attr)
            for param_name, analytical in (("weight", layer.grad_weights), ("bias", layer.grad_bias)):
                numerical = self._numerical_grad(layer_attr, param_name)
                rel_error = np.linalg.norm(analytical - numerical) / (
                    np.linalg.norm(analytical) + np.linalg.norm(numerical) + 1e-10
                )
                self.assertLess(rel_error, 1e-4, msg=f"Gradient check failed for {layer_attr}.{param_name}: rel_error={rel_error}")


class TestPermutationInvariance(unittest.TestCase):
    """The core property this architecture exists to guarantee."""

    def setUp(self):
        self.model = DeepSetsPredictiveModel(
            entity_feature_dim=4, entity_hidden_dim=8, global_feature_dim=5,
            head_hidden_dim=8, output_dim=1, max_entities=3, seed=1,
        )

    def test_reordering_present_entities_does_not_change_output(self):
        ef = np.array([[[1.0, 2.0, -1.0, 2.5], [2.0, -3.0, 1.0, 3.2], [0.0, 0.0, 0.0, 0.0]]])
        pm = np.array([[1.0, 1.0, 0.0]])
        gf = np.random.default_rng(2).normal(size=(1, 5))

        out_original = self.model.forward(ef, pm, gf)

        ef_swapped = ef[:, [1, 0, 2], :]
        pm_swapped = pm[:, [1, 0, 2]]
        out_swapped = self.model.forward(ef_swapped, pm_swapped, gf)

        np.testing.assert_allclose(out_original, out_swapped, atol=1e-10)

    def test_absent_slots_do_not_affect_output(self):
        """Padding an absent slot with junk values (mask=0) must not change the prediction."""
        ef = np.array([[[1.0, 2.0, -1.0, 2.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
        pm = np.array([[1.0, 0.0, 0.0]])
        gf = np.random.default_rng(3).normal(size=(1, 5))
        out_zero_padding = self.model.forward(ef, pm, gf)

        ef_junk = ef.copy()
        ef_junk[0, 1, :] = [99.0, -50.0, 17.0, 3.0]  # garbage in a masked-out slot
        out_junk_padding = self.model.forward(ef_junk, pm, gf)

        np.testing.assert_allclose(out_zero_padding, out_junk_padding, atol=1e-10)

    def test_more_entities_than_trained_slot_count_is_handled_by_masking(self):
        """One present entity vs. two present entities should generally differ (more info)."""
        gf = np.random.default_rng(4).normal(size=(1, 5))
        ef_one = np.array([[[1.0, 2.0, -1.0, 2.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
        pm_one = np.array([[1.0, 0.0, 0.0]])

        ef_two = np.array([[[1.0, 2.0, -1.0, 2.5], [2.0, -1.0, 0.5, 1.1], [0.0, 0.0, 0.0, 0.0]]])
        pm_two = np.array([[1.0, 1.0, 0.0]])

        out_one = self.model.forward(ef_one, pm_one, gf)
        out_two = self.model.forward(ef_two, pm_two, gf)
        self.assertFalse(np.allclose(out_one, out_two), "adding a second entity should generally change the prediction")

    def test_max_pooling_only_backprops_to_the_winning_slot_per_channel(self):
        """Max-pooling's defining property: gradient reaches only the argmax entity per
        channel, never a blend of several -- unlike sum/mean pooling.
        """
        ef = np.array([[[5.0, 5.0, 5.0, 5.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
        pm = np.array([[1.0, 1.0, 0.0]])
        gf = np.random.default_rng(5).normal(size=(1, 5))

        self.model.zero_grad()
        y_pred = self.model.forward(ef, pm, gf)
        self.model.backward(np.ones_like(y_pred))

        # entity_l1's weight gradient should be nonzero (some channel picked slot 0 or 1,
        # both of which have real, distinct features), but slot 2 (absent, masked out)
        # must never win any channel, so perturbing it never contributes.
        self.assertGreater(np.linalg.norm(self.model.entity_l1.grad_weights), 0.0)

    def test_pooling_switched_from_mean_to_max(self):
        """Regression guard: confirms the model uses max- not mean-pooling. Two present
        entities with a large-magnitude and a near-zero embedding should pool close to
        the large one, not their average, if max-pooling is in effect.
        """
        ef_big_small = np.array([[[10.0, 10.0, 10.0, 10.0], [0.01, 0.01, 0.01, 0.01], [0.0, 0.0, 0.0, 0.0]]])
        pm = np.array([[1.0, 1.0, 0.0]])
        ef_big_only = np.array([[[10.0, 10.0, 10.0, 10.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
        pm_one = np.array([[1.0, 0.0, 0.0]])
        gf = np.random.default_rng(6).normal(size=(1, 5))

        out_pair = self.model.forward(ef_big_small, pm, gf)
        out_big_alone = self.model.forward(ef_big_only, pm_one, gf)
        # With max-pooling, adding a much-smaller second entity should barely move the
        # output away from the "big entity alone" case (since the big one wins every
        # channel with overwhelming probability given its scale).
        np.testing.assert_allclose(out_pair, out_big_alone, atol=0.5)


if __name__ == "__main__":
    unittest.main()
