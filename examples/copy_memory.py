"""
Copying Memory Task, according to https://github.com/philipperemy/keras-tcn
Torch lightning is a prerequisite.
"""

import logging
from typing import Tuple

import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deep_tcn.tcn import TcNetwork

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def data_generator(
    t: int, mem_length: int, b_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data for the copying memory task. https://github.com/philipperemy/keras-tcn

    Args:
        t (int): The total blank time length.
        mem_length (int): The length of the memory to be recalled.
        b_size (int): The batch size.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input and target data tensor.
    """
    seq = torch.randint(1, 9, size=(b_size, mem_length), dtype=torch.float32)
    zeros = torch.zeros((b_size, t), dtype=torch.float32)
    marker = 9 * torch.ones((b_size, mem_length + 1))
    placeholders = torch.zeros((b_size, mem_length))

    x = torch.cat((seq, zeros[:, :-1], marker), 1)
    y = torch.cat((placeholders, zeros, seq), 1)

    return x.unsqueeze(1), y.unsqueeze(1)


class MyLightningModule(pl.LightningModule):
    def __init__(self, model):
        super(MyLightningModule, self).__init__()
        self.model = model
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)["output"]

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(y, out).mean()
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss(y, out).mean()
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics["val_loss"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            },
        }


if __name__ == "__main__":
    blank_len, seq_len = 100, 10

    # Data
    loaders = {
        "train": DataLoader(
            TensorDataset(*data_generator(blank_len, seq_len, b_size=1024)),
            batch_size=16,
        ),
        "valid": DataLoader(
            TensorDataset(*data_generator(blank_len, seq_len, b_size=128)),
            batch_size=16,
        ),
    }

    # Architecture definition
    head = {
        "conv_kwargs": [
            {
                "out_channels": 1,
                "kernel_size": 1,
                "dilation": 1,
                "separable": True,
            }
        ],
    }
    model = TcNetwork(
        input_channels=1,
        blocks=1,
        dilations=7,
        channels=16,
        kernel_size=2,
        separable=True,
        activation=nn.ReLU,
        top_pooling=head,
    )
    logger.info(f"Receptive field size: {model.receptive_field_size}")

    # Fitting
    module = MyLightningModule(model)
    trainer = pl.Trainer(max_epochs=20, accelerator="auto")
    trainer.fit(module, loaders["train"], loaders["valid"])

    # Plotting predictions for a sample
    _x, _y = next(iter(loaders["valid"]))
    out = module(_x).detach().numpy().squeeze()

    idx = 10
    inp = _x.squeeze()
    truth = _y.squeeze()
    plt.plot(inp[idx, :], label="Input")
    plt.plot(out[idx, :], label="Output")
    plt.plot(truth[idx, :], label="Target")
    plt.legend()
    plt.show()
