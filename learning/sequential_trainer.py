"""Sequential training loop and BPTT optimization driver for Project Caspian (Phase 2).

Trains RecurrentPredictor neural models on sequential trajectory datasets using
Backpropagation Through Time (BPTT), batch mini-batching, and validation monitoring.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from models.recurrent import RecurrentPredictor
from models.optimizers import Optimizer, Adam, MSELoss, LossFunction
from learning.sequential_dataset import SequentialExperienceDataset


class SequentialTrainer:
    """Supervised / Self-Supervised sequence trainer for recurrent memory models."""

    def __init__(
        self,
        model: RecurrentPredictor,
        optimizer: Optional[Optimizer] = None,
        loss_fn: Optional[LossFunction] = None,
        batch_size: int = 8,
        epochs: int = 150,
        early_stopping_patience: Optional[int] = 30,
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
                lr=0.008,
                weight_decay=1e-4,
            )

    def fit(
        self,
        train_data: Union[SequentialExperienceDataset, Tuple[np.ndarray, np.ndarray]],
        val_data: Optional[Union[SequentialExperienceDataset, Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train the RecurrentPredictor on sequence data via BPTT.

        Args:
            train_data: SequentialExperienceDataset or (X_train, Y_train) 3D numpy arrays.
            val_data: Optional validation sequence dataset or tuple.
            verbose: If True, prints periodic training logs.

        Returns:
            Dict[str, Any]: History dictionary with training and validation losses.
        """
        if isinstance(train_data, SequentialExperienceDataset):
            X_train, Y_train = train_data.to_numpy_sequences()
        else:
            X_train, Y_train = train_data

        if val_data is not None:
            if isinstance(val_data, SequentialExperienceDataset):
                X_val, Y_val = val_data.to_numpy_sequences()
            else:
                X_val, Y_val = val_data
        else:
            X_val, Y_val = None, None

        num_episodes = X_train.shape[0]
        if num_episodes == 0:
            raise ValueError("Training dataset contains no episodes.")

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_params_state: Optional[Dict[str, Any]] = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # Shuffle episodes
            indices = np.arange(num_episodes)
            self.rng.shuffle(indices)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            epoch_loss_sum = 0.0
            num_batches = int(np.ceil(num_episodes / self.batch_size))

            for b in range(num_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, num_episodes)

                X_b = X_shuffled[start:end]
                Y_b = Y_shuffled[start:end]

                # Zero gradients
                self.model.zero_grad()

                # Forward pass over sequence
                y_pred_b = self.model.forward(X_b)

                # Compute sequence loss
                batch_loss = self.loss_fn.forward(y_pred_b, Y_b)
                epoch_loss_sum += batch_loss * (end - start)

                # Compute upstream loss gradient
                grad_out = self.loss_fn.gradient(y_pred_b, Y_b)

                # Backpropagation Through Time
                self.model.backward(grad_out)

                # Parameter optimization step
                self.optimizer.step()

            train_loss = epoch_loss_sum / num_episodes
            train_losses.append(float(train_loss))

            # Evaluate on validation episodes
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

        # Restore best parameters
        if best_params_state is not None:
            restored = RecurrentPredictor.from_dict(best_params_state)
            self.model.gru = restored.gru
            self.model.head = restored.head

        return {
            "epochs_run": len(train_losses),
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else train_losses[-1],
        }
