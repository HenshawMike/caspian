"""Training loop for DeepSetsPredictiveModel (permutation-invariant entity encoding).

Mirrors learning.trainer.PredictiveTrainer's interface (batching, Adam, early
stopping) but operates on the three-array (entity_features, presence_mask,
global_features) input structure instead of a single flat feature vector.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from models.deepsets import DeepSetsPredictiveModel
from models.optimizers import Optimizer, Adam, MSELoss, LossFunction


class DeepSetsTrainer:
    """Supervised trainer for DeepSetsPredictiveModel."""

    def __init__(
        self,
        model: DeepSetsPredictiveModel,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[LossFunction] = None,
        batch_size: int = 32,
        epochs: int = 150,
        early_stopping_patience: Optional[int] = 25,
        seed: Optional[int] = 42,
    ):
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else MSELoss()
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Default weight_decay=0.05 is deliberately high: with entity_type constant
        # across all training episodes (only one type ever seen), the entity
        # encoder's weight on that input dimension gets essentially no training
        # signal and can sit at an arbitrary post-init value; strong L2 pressure
        # keeps it near zero so a never-seen type value at test time doesn't
        # produce an unbounded extrapolation. Swept empirically: id-set MSE
        # degrades smoothly past ~0.05, while OOD MSE (novel type, multi-entity)
        # is minimized right around this value -- see CSP-P3-E003.
        self.optimizer = optimizer if optimizer is not None else Adam(params=self.model.get_params(), lr=0.005, weight_decay=0.05)

    def fit(
        self,
        train_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train via minibatch SGD/Adam.

        Args:
            train_data: (entity_features, presence_mask, global_features, Y) arrays.
            val_data: Optional held-out tuple of the same shape.
            verbose: If True, print progress periodically.

        Returns:
            Dict[str, Any]: Training history.
        """
        EF, PM, GF, Y = train_data
        n = EF.shape[0]
        if n == 0:
            raise ValueError("Training dataset is empty.")

        if hasattr(self.model, "fit_normalizer"):
            self.model.fit_normalizer(EF, PM, GF)

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_state: Optional[Dict[str, Any]] = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            indices = np.arange(n)
            self.rng.shuffle(indices)
            EF_s, PM_s, GF_s, Y_s = EF[indices], PM[indices], GF[indices], Y[indices]

            epoch_loss_sum = 0.0
            num_batches = int(np.ceil(n / self.batch_size))
            for b in range(num_batches):
                start, end = b * self.batch_size, min((b + 1) * self.batch_size, n)
                ef_b, pm_b, gf_b, y_b = EF_s[start:end], PM_s[start:end], GF_s[start:end], Y_s[start:end]

                self.model.zero_grad()
                y_pred = self.model.forward(ef_b, pm_b, gf_b)
                batch_loss = self.loss_fn.forward(y_pred, y_b)
                epoch_loss_sum += batch_loss * (end - start)

                grad = self.loss_fn.gradient(y_pred, y_b)
                self.model.backward(grad)
                self.optimizer.step()

            train_loss = epoch_loss_sum / n
            train_losses.append(float(train_loss))

            if val_data is not None and val_data[0].shape[0] > 0:
                EFv, PMv, GFv, Yv = val_data
                val_pred = self.model.forward(EFv, PMv, GFv)
                val_loss = float(self.loss_fn.forward(val_pred, Yv))
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss, best_epoch = val_loss, epoch
                    best_state = self.model.to_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if self.early_stopping_patience is not None and patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}.")
                        break
            else:
                if train_loss < best_val_loss:
                    best_val_loss, best_epoch = train_loss, epoch
                    best_state = self.model.to_dict()

            if verbose and epoch % 10 == 0:
                val_str = f", Val Loss: {val_losses[-1]:.6f}" if val_losses else ""
                print(f"Epoch {epoch:3d}/{self.epochs:3d} — Train Loss: {train_loss:.6f}{val_str}")

        if best_state is not None:
            restored = DeepSetsPredictiveModel.from_dict(best_state)
            self.model.entity_l1, self.model.entity_l2 = restored.entity_l1, restored.entity_l2
            self.model.head_l1, self.model.head_l2 = restored.head_l1, restored.head_l2

        return {
            "epochs_run": len(train_losses),
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else train_losses[-1],
        }
