"""Training loop and optimization driver for Project Caspian (Phase 1).

Trains PredictiveMLP neural models on experience datasets with validation tracking
and history recording.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from models.mlp import PredictiveMLP
from models.optimizers import Optimizer, Adam, MSELoss, LossFunction
from learning.dataset import ExperienceDataset


class PredictiveTrainer:
    """Supervised/Self-Supervised trainer for predictive interaction models."""

    def __init__(
        self,
        model: PredictiveMLP,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[LossFunction] = None,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: Optional[int] = 20,
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
            self.optimizer = Adam(
                params=self.model.get_params(),
                lr=0.01,
                weight_decay=1e-4,
            )

    def fit(
        self,
        train_data: Union[ExperienceDataset, Tuple[np.ndarray, np.ndarray]],
        val_data: Optional[Union[ExperienceDataset, Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train the model on the provided dataset.

        Args:
            train_data: Training dataset (ExperienceDataset or (X, Y) tuple).
            val_data: Optional validation dataset.
            verbose: If True, prints progress per epoch.

        Returns:
            Dict[str, Any]: Training history containing train_losses, val_losses,
                            best_epoch, best_val_loss, and convergence stats.
        """
        if isinstance(train_data, ExperienceDataset):
            X_train, Y_train = train_data.to_numpy_arrays()
        else:
            X_train, Y_train = train_data

        if val_data is not None:
            if isinstance(val_data, ExperienceDataset):
                X_val, Y_val = val_data.to_numpy_arrays()
            else:
                X_val, Y_val = val_data
        else:
            X_val, Y_val = None, None

        num_samples = X_train.shape[0]
        if num_samples == 0:
            raise ValueError("Training dataset is empty.")

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_params_state: Optional[Dict[str, Any]] = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # Shuffle training data
            indices = np.arange(num_samples)
            self.rng.shuffle(indices)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            epoch_loss_sum = 0.0
            num_batches = int(np.ceil(num_samples / self.batch_size))

            for b in range(num_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, num_samples)

                X_b = X_shuffled[start:end]
                Y_b = Y_shuffled[start:end]

                # Zero gradients
                self.model.zero_grad()

                # Forward pass
                y_pred = self.model.forward(X_b)

                # Compute loss
                batch_loss = self.loss_fn.forward(y_pred, Y_b)
                epoch_loss_sum += batch_loss * (end - start)

                # Compute output gradient
                grad_out = self.loss_fn.gradient(y_pred, Y_b)

                # Backward pass
                self.model.backward(grad_out)

                # Optimizer step
                self.optimizer.step()

            train_loss = epoch_loss_sum / num_samples
            train_losses.append(float(train_loss))

            # Evaluate on validation set
            if X_val is not None and len(X_val) > 0:
                y_val_pred = self.model.forward(X_val)
                val_loss = float(self.loss_fn.forward(y_val_pred, Y_val))
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_params_state = self.model.to_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if (
                        self.early_stopping_patience is not None
                        and patience_counter >= self.early_stopping_patience
                    ):
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch}.")
                        break
            else:
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_epoch = epoch
                    best_params_state = self.model.to_dict()

            if verbose and epoch % 10 == 0:
                val_str = f", Val Loss: {val_losses[-1]:.6f}" if val_losses else ""
                print(f"Epoch {epoch:3d}/{self.epochs:3d} — Train Loss: {train_loss:.6f}{val_str}")

        # Restore best weights if validation was performed
        if best_params_state is not None:
            restored = PredictiveMLP.from_dict(best_params_state)
            self.model.layers = restored.layers

        return {
            "epochs_run": len(train_losses),
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else train_losses[-1],
        }
