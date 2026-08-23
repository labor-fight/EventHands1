#!/usr/bin/env python3
"""Minimal multi-GPU Lightning DDP smoke test (file-based, not stdin)."""
import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class M(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(8, 1)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, idx):
        x, y = batch
        loss = ((self(x) - y) ** 2).mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main():
    x = torch.randn(4096, 8)
    y = torch.randn(4096, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=256, num_workers=0)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        max_steps=10,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )
    print("fitting...", flush=True)
    trainer.fit(M(), loader)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
