import unittest
import numpy as np
from models.optimizers import Adam, SGD


class TestWeightDecayAppliesToPrefixedNames(unittest.TestCase):
    """Regression test: weight_decay must apply to names like 'layer_0_weight',
    not just the bare literal 'weight' -- every MLP-family model in this project
    (PredictiveMLP, DeepSetsPredictiveModel) prefixes parameter names.
    """

    def _check(self, optimizer_cls, **kwargs):
        w = np.array([1.0, 1.0])
        g = np.zeros_like(w)  # zero gradient -> any movement is purely from weight_decay
        params = [("layer_0_weight", w, g)]
        opt = optimizer_cls(params=params, lr=0.1, weight_decay=0.5, **kwargs)
        opt.step()
        self.assertFalse(np.allclose(w, [1.0, 1.0]), f"{optimizer_cls.__name__}: weight_decay had no effect on a prefixed 'weight' name")

    def test_adam_applies_weight_decay_to_prefixed_names(self):
        self._check(Adam)

    def test_sgd_applies_weight_decay_to_prefixed_names(self):
        self._check(SGD)

    def test_bias_names_are_unaffected_by_weight_decay(self):
        b = np.array([1.0, 1.0])
        g = np.zeros_like(b)
        params = [("layer_0_bias", b, g)]
        opt = Adam(params=params, lr=0.1, weight_decay=0.5)
        opt.step()
        np.testing.assert_allclose(b, [1.0, 1.0], err_msg="weight_decay should never be applied to bias parameters")


if __name__ == "__main__":
    unittest.main()
