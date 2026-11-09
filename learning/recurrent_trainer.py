"""Training loop for the Phase 2 recurrent (memory) predictive model.

Mirrors learning.trainer.PredictiveTrainer's interface and reuses the same
Optimizer / LossFunction primitives, but operates on full episode sequences
(batch, T, dim) and drives RecurrentPredictiveModel.forward_sequence /
backward_sequence (BPTT) instead of per-sample forward / backward.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from models.rnn import RecurrentPredictiveModel
from models.optimizers import Optimizer, Adam, MSELoss, LossFunction
from learning.sequence_dataset import EpisodeDataset


class RecurrentTrainer:
    """Supervised BPTT trainer for RecurrentPredictiveModel."""

    def __init__(
        self,
        model: RecurrentPredictiveModel,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[LossFunction] = None,
        batch_size: int = 8,
        epochs: int = 100,
        early_stopping_patience: Optional[int] = 15,
        seed: Optional[int] = 42,
    ):
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else MSELoss()
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = Adam(params=self.model.get_params(), lr=0.01, weight_decay=1e-5)

    def fit(
        self,
        train_data: Union[EpisodeDataset, Tuple[np.ndarray, np.ndarray]],
        val_data: Optional[Union[EpisodeDataset, Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train the recurrent model via BPTT, batching over episodes.

        Args:
            train_data: EpisodeDataset or (X, Y) arrays of shape (N, T, dim).
            val_data: Optional held-out EpisodeDataset or (X, Y) arrays.
            verbose: If True, prints progress periodically.

        Returns:
            Dict[str, Any]: Training history (mirrors PredictiveTrainer.fit()).
        """
        X_train, Y_train = train_data.to_numpy_arrays() if isinstance(train_data, EpisodeDataset) else train_data
        if val_data is not None:
            X_val, Y_val = val_data.to_numpy_arrays() if isinstance(val_data, EpisodeDataset) else val_data
        else:
            X_val, Y_val = None, None

        num_episodes = X_train.shape[0]
        if num_episodes == 0:
            raise ValueError("Training dataset is empty.")

        if hasattr(self.model, "fit_normalizer"):
            self.model.fit_normalizer(X_train)

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_state: Optional[Dict[str, Any]] = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            indices = np.arange(num_episodes)
            self.rng.shuffle(indices)
            X_shuffled, Y_shuffled = X_train[indices], Y_train[indices]

            epoch_loss_sum = 0.0
            num_batches = int(np.ceil(num_episodes / self.batch_size))

            for b in range(num_batches):
                start, end = b * self.batch_size, min((b + 1) * self.batch_size, num_episodes)
                X_b, Y_b = X_shuffled[start:end], Y_shuffled[start:end]

                self.model.zero_grad()
                Y_pred = self.model.forward_sequence(X_b)
                batch_loss = self.loss_fn.forward(Y_pred, Y_b)
                epoch_loss_sum += batch_loss * (end - start)

                grad_Y = self.loss_fn.gradient(Y_pred, Y_b)
                self.model.backward_sequence(grad_Y)
                self.optimizer.step()

            train_loss = epoch_loss_sum / num_episodes
            train_losses.append(float(train_loss))

            if X_val is not None and len(X_val) > 0:
                val_pred = self.model.forward_sequence(X_val)
                val_loss = float(self.loss_fn.forward(val_pred, Y_val))
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss, best_epoch = val_loss, epoch
                    best_state = self.model.to_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if self.early_stopping_patience is not None and patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch}.")
                        break
            else:
                if train_loss < best_val_loss:
                    best_val_loss, best_epoch = train_loss, epoch
                    best_state = self.model.to_dict()

            if verbose and epoch % 10 == 0:
                val_str = f", Val Loss: {val_losses[-1]:.6f}" if val_losses else ""
                print(f"Epoch {epoch:3d}/{self.epochs:3d} — Train Loss: {train_loss:.6f}{val_str}")

        if best_state is not None:
            restored = RecurrentPredictiveModel.from_dict(best_state)
            self.model.W_xh, self.model.W_hh = restored.W_xh, restored.W_hh
            self.model.b_h, self.model.W_hy, self.model.b_y = restored.b_h, restored.W_hy, restored.b_y

        return {
            "epochs_run": len(train_losses),
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else train_losses[-1],
        }
