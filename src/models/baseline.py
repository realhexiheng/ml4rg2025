import torch.nn as nn
import torch
import pytorch_lightning as pl
import csv
import numpy as np
from pathlib import Path

class LinearModel(pl.LightningModule):
    """
    Ridge regression model.
    Parameters
    ----------
    window_size : int
        Size of the embedding context window.
    """

    def __init__(
        self,
        window_size: int = 500,
        n_conditions: int = 18,
        learning_rate: float = 1e-3,
    ):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(window_size * 768, n_conditions)
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # Flatten x before linear
        x = torch.flatten(x, start_dim=1)

        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.float())
        loss = self.loss_fn(y_hat, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.float())
        loss = self.loss_fn(y_hat, y.float())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class ConvolutionalModel(pl.LightningModule):
    """
    Model that uses 1D convolution with kernel size 1 across the embedding dimension,
    followed by pooling across the window size.

    Parameters
    ----------
    n_conditions : int
        Number of output conditions to predict.
    learning_rate : float
        Learning rate for the optimizer.
    pooling_type : str
        Type of pooling to use, either 'mean' or 'max'.
    """

    def __init__(
        self,
        n_conditions: int = 18,
        learning_rate: float = 1e-3,
        pooling_type: str = "mean",
        average_window: bool = False,
        weight_decay: float = 1e-4,
    ):
        super().__init__()

        # Validate pooling type
        if pooling_type not in ["mean", "max"]:
            raise ValueError("pooling_type must be either 'mean' or 'max'")
        self.pooling_type = pooling_type
        self.average_window = average_window
        self.weight_decay = weight_decay

        # Conv1d expects input shape [batch_size, in_channels, sequence_length]
        # Here in_channels = 768 (embedding dimension)
        # and sequence_length = window_size
        # Output channels = n_conditions (one channel per condition)
        self.conv = nn.Conv1d(in_channels=768, out_channels=n_conditions, kernel_size=1)

        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

        # tracking setup
        self.csv_log_path = "max_positions_log.csv"
        self.logged_rows = []

        # For collecting validation predictions and targets
        self.validation_step_outputs = []

    def _init_csv_log(self):
        base_path = Path(self.csv_log_path)
        path = base_path
        counter = 1
        while path.exists():
            path = base_path.with_name(f"{base_path.stem}_{counter}{base_path.suffix}")
            counter += 1
        self.csv_log_path = path

        with open(self.csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "gene", "condition", "max_position", "max_value"])


    def forward(self, x):
        # Average across the window size dimension
        if self.average_window:
            x = torch.mean(x, dim=1)
            x = x.unsqueeze(1)

        # x shape: [batch_size, window_size, 768]
        # Transpose to get [batch_size, 768, window_size] for Conv1d
        x = x.transpose(1, 2)

        # Apply convolution
        x = self.conv(x)

        # Pooling across window_size dimension
        if self.pooling_type == "mean":
            x = torch.mean(x, dim=2)  # [batch_size, n_conditions]
        else:
            x = torch.max(x, dim=2)[0]  # [batch_size, n_conditions]
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.float())
        loss = self.loss_fn(y_hat, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)

        # Store predictions and targets for correlation/explained variance calculation
        self.validation_step_outputs.append({"y_hat": y_hat.detach(), "y": y.detach()})

        if self.pooling_type == "max":
            self._log_max_positions(x, batch_idx,test=False)

        return loss
    
    def on_fit_start(self):
        self._init_csv_log()

    def on_test_start(self):
        self.test_step_outputs = []
        self._init_csv_log()


    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss)

        # Store predictions and targets for later use
        self.test_step_outputs.append({"y_hat": y_hat.detach(), "y": y.detach()})

        # Also track max-pooled positions if applicable
        if self.pooling_type == "max":
            self._log_max_positions(x, batch_idx,test=True)

        return loss

    def _log_max_positions(self, x, batch_idx, test=False):
        if self.trainer.training and self.current_epoch == 0 and self.global_step == 0:
            return

        x_conv = self.conv(x.transpose(1, 2))  # shape: [B, C, L]
        max_vals, max_positions = torch.max(x_conv, dim=2)

        batch_size = x.shape[0]
        dataset = (
            self.trainer.test_dataloaders.dataset
            if test
            else self.trainer.val_dataloaders.dataset
        )

        for i in range(batch_size):
            dataset_idx = batch_idx * batch_size + i
            gene = dataset.genes[dataset_idx]

            for cond in range(max_vals.shape[1]):
                self.logged_rows.append(
                    [
                        self.current_epoch,
                        gene,
                        cond,
                        int(max_positions[i, cond].item()),
                        float(max_vals[i, cond].item()),
                    ]
                )

    def on_test_epoch_end(self):
        # Log max positions from test
        if self.logged_rows:
            with open(self.csv_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.logged_rows)
            self.logged_rows = []


    def on_validation_epoch_end(self):
        # Log max positions if available
        if self.logged_rows:
            with open(self.csv_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.logged_rows)
            self.logged_rows = []

        # Calculate and log correlation and explained variance
        if self.validation_step_outputs:
            # Concatenate all predictions and targets
            all_y_hat = torch.cat(
                [x["y_hat"] for x in self.validation_step_outputs], dim=0
            )
            all_y = torch.cat([x["y"] for x in self.validation_step_outputs], dim=0)

            # Calculate metrics for each condition
            correlations = []
            explained_variances = []

            for condition in range(all_y.shape[1]):
                y_true = all_y[:, condition]
                y_pred = all_y_hat[:, condition]

                # Calculate correlation
                if torch.var(y_true) > 0 and torch.var(y_pred) > 0:
                    correlation = torch.corrcoef(torch.stack([y_true, y_pred]))[0, 1]
                    if not torch.isnan(correlation):
                        correlations.append(correlation.item())

                # Calculate explained variance
                if torch.var(y_true) > 0:
                    explained_var = 1 - torch.var(y_true - y_pred) / torch.var(y_true)
                    if not torch.isnan(explained_var):
                        explained_variances.append(explained_var.item())

            # Log average metrics across all conditions
            if correlations:
                avg_correlation = np.mean(correlations)
                self.log("val_correlation", avg_correlation, prog_bar=True)

            if explained_variances:
                avg_explained_variance = np.mean(explained_variances)
                self.log("val_explained_variance", avg_explained_variance, prog_bar=True)

            # Clear the outputs for next epoch
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
