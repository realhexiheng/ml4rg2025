import torch.nn as nn
import torch
import pytorch_lightning as pl
import csv


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
    window_size : int
        Size of the embedding context window.
    n_conditions : int
        Number of output conditions to predict.
    learning_rate : float
        Learning rate for the optimizer.
    pooling_type : str
        Type of pooling to use, either 'mean' or 'max'.
    """

    def __init__(
        self,
        window_size: int = 500,
        n_conditions: int = 18,
        learning_rate: float = 1e-3,
        pooling_type: str = "mean",
    ):
        super().__init__()
        self.window_size = window_size

        # Validate pooling type
        if pooling_type not in ["mean", "max"]:
            raise ValueError("pooling_type must be either 'mean' or 'max'")
        self.pooling_type = pooling_type

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

    def on_fit_start(self):
        # create position tracking file
        with open(self.csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "gene", "condition", "max_position", "max_value"])

    def forward(self, x):
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
        y_hat = self(x.float())
        loss = self.loss_fn(y_hat, y.float())
        self.log("val_loss", loss)

        if self.pooling_type == "max":
            self._log_max_positions(x, batch_idx)

        return loss

    def _log_max_positions(self, x, batch_idx):
        if self.current_epoch == 0 and self.global_step == 0:
            return
        x_conv = self.conv(x.transpose(1, 2))  # shape: [B, C, L]
        max_vals, max_positions = torch.max(x_conv, dim=2)

        batch_size = x.shape[0]
        for i in range(batch_size):
            dataset_idx = batch_idx * batch_size + i
            gene = self.trainer.val_dataloaders.dataset.genes[dataset_idx]

            for cond in range(max_vals.shape[1]):
                self.logged_rows.append(
                    [
                        self.current_epoch,  # ← epoch tracking
                        gene,
                        cond,
                        int(max_positions[i, cond].item()),
                        float(max_vals[i, cond].item()),
                    ]
                )

    def on_validation_epoch_end(self):
        if self.logged_rows:
            with open(self.csv_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.logged_rows)
            self.logged_rows = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
