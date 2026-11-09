import unittest
import numpy as np
from models.normalization import FeatureNormalizer
from models.mlp import PredictiveMLP
from learning.trainer import PredictiveTrainer


class TestFeatureNormalizer(unittest.TestCase):
    def test_fit_transform_gives_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=50.0, scale=12.0, size=(500, 5))
        norm = FeatureNormalizer()
        X_t = norm.fit_transform(X)
        np.testing.assert_allclose(X_t.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(X_t.std(axis=0), 1.0, atol=1e-8)

    def test_zero_variance_column_not_blown_up(self):
        X = np.column_stack([np.ones(100), np.random.default_rng(1).normal(size=100)])
        norm = FeatureNormalizer()
        X_t = norm.fit_transform(X)
        np.testing.assert_allclose(X_t[:, 0], 0.0)  # constant column -> stays 0 after centering, not NaN/inf

    def test_transform_before_fit_raises(self):
        norm = FeatureNormalizer()
        with self.assertRaises(RuntimeError):
            norm.transform(np.zeros((2, 2)))

    def test_roundtrip_dict(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(50, 4))
        norm = FeatureNormalizer()
        norm.fit(X)
        restored = FeatureNormalizer.from_dict(norm.to_dict())
        np.testing.assert_allclose(restored.mean, norm.mean)
        np.testing.assert_allclose(restored.std, norm.std)
        np.testing.assert_allclose(restored.transform(X), norm.transform(X))


class TestNormalizerIntegration(unittest.TestCase):
    def test_unfit_normalizer_leaves_forward_unchanged(self):
        """A model that never went through a trainer (normalizer unfit) must behave
        exactly as before this change -- backward compatibility for direct model use
        (gradient-check tests, mock-model tests, etc.).
        """
        model = PredictiveMLP(input_dim=5, hidden_dims=(4,), output_dim=1, seed=0)
        self.assertFalse(model.normalizer.is_fit)
        X = np.random.default_rng(0).normal(size=(3, 5))
        out1 = model.forward(X)
        out2 = model.forward(X)
        np.testing.assert_allclose(out1, out2)  # deterministic, no hidden normalization state

    def test_trainer_fits_normalizer_and_improves_dead_unit_rate(self):
        """The actual regression this fix targets: a feature with a huge unnormalized
        scale relative to the others should no longer collapse most ReLU units to dead.
        """
        rng = np.random.default_rng(0)
        n = 2000
        small_scale_features = rng.normal(scale=1.0, size=(n, 4))
        huge_scale_feature = rng.normal(loc=50.0, scale=12.0, size=(n, 1))  # mimics internal_state
        X = np.hstack([small_scale_features, huge_scale_feature])
        Y = (small_scale_features[:, 0:1] * 2.0 + rng.normal(scale=0.1, size=(n, 1)))

        model = PredictiveMLP(input_dim=5, hidden_dims=(32,), output_dim=1, hidden_activation="relu", seed=0)
        trainer = PredictiveTrainer(model=model, batch_size=32, epochs=30, early_stopping_patience=None, seed=0)
        trainer.fit((X, Y))

        self.assertTrue(model.normalizer.is_fit)
        z = model.layers[0].forward(model.normalizer.transform(X))
        a = model.hidden_act.forward(z)
        structurally_dead = np.mean(np.mean(a == 0.0, axis=0) > 0.95)
        self.assertLess(structurally_dead, 0.5, "normalization should leave well under half the units permanently dead")

    def test_get_latent_state_applies_same_normalization_as_forward(self):
        """Regression guard: get_latent_state() has its own independent forward loop
        (doesn't call self.forward()), so it's possible to fix normalization in one and
        silently miss the other -- exactly the bug this test catches. Once a normalizer
        is fit, get_latent_state() must apply it, consistent with forward().
        """
        rng = np.random.default_rng(0)
        X = rng.normal(loc=50.0, scale=12.0, size=(20, 5))
        Y = rng.normal(size=(20, 1))

        model = PredictiveMLP(input_dim=5, hidden_dims=(8, 4), output_dim=1, hidden_activation="relu", seed=0)
        trainer = PredictiveTrainer(model=model, batch_size=8, epochs=5, early_stopping_patience=None, seed=0)
        trainer.fit((X, Y))

        # Manually replicate forward()'s hidden-layer computation (normalized) and
        # compare directly against get_latent_state()'s output for the same input.
        normalized = model.normalizer.transform(X)
        curr = normalized
        for layer in model.layers[:-1]:
            curr = model.hidden_act.forward(layer.forward(curr))
        expected_latent = curr

        actual_latent = model.get_latent_state(X)
        np.testing.assert_allclose(actual_latent, expected_latent, atol=1e-10)
        # And it must NOT just be echoing the unnormalized computation (the bug's symptom).
        unnormalized_curr = X
        for layer in model.layers[:-1]:
            unnormalized_curr = model.hidden_act.forward(layer.forward(unnormalized_curr))
        self.assertFalse(np.allclose(actual_latent, unnormalized_curr))


if __name__ == "__main__":
    unittest.main()
