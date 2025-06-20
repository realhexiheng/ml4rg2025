import torch.nn as nn
import torch
import pytorch_lightning as pl
import csv


class ConvolutionalModel(pl.LightningModule):
    def __init__(
        self,
        window_size: int = 500,
        n_conditions: int = 18,
        learning_rate: float = 1e-3,
        pooling_type: str = "mean",
    ):
        super().__init__()
        self.window_size = window_size

        if pooling_type not in ["mean", "max"]:
            raise ValueError("pooling_type must be either 'mean' or 'max'")
        self.pooling_type = pooling_type

        self.conv = nn.Conv1d(in_channels=768, out_channels=n_conditions, kernel_size=1)

        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()

        # tracking setup
        self.csv_log_path = "max_positions_log.csv"
        self.logged_rows = []

        

    def on_fit_start(self):
        with open(self.csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch","gene", "condition", "max_position", "max_value"])

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        if self.pooling_type == "mean":
            x = torch.mean(x, dim=2)
        else:
            x = torch.max(x, dim=2)[0]
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
        y_hat = self(x)
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
                self.logged_rows.append([
                      self.current_epoch,  # ← epoch tracking
                      gene,
                      cond,
                      int(max_positions[i, cond].item()),
                      float(max_vals[i, cond].item())
                  ])

    def on_validation_epoch_end(self):
        if self.logged_rows:
            with open(self.csv_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.logged_rows)
            self.logged_rows = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
