import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from segnet import SegNet
from dataloader import FreiburgDataLoader

import numpy as np

class LitSegNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = SegNet(num_classes=6)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # print(x.shape)
        embedding = self.model(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        # x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.nll_loss(x_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        # REQUIRED
        dl = FreiburgDataLoader(train=True)
        return DataLoader(dl, batch_size=3)

    def val_dataloader(self):
        # OPTIONAL
        dl = FreiburgDataLoader(train=False)
        return DataLoader(dl, batch_size=3)


segnet_model = LitSegNet()

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=0, min_epochs=1, max_epochs=100, check_val_every_n_epoch=5)
# trainer.fit(segnet_model)

trained_model = LitSegNet.load_from_checkpoint(checkpoint_path="lightning_logs/version_97506/checkpoints/epoch=316-step=24408.ckpt")
# prints the learning_rate you used in this checkpoint

trained_model.eval()
ds = FreiburgDataLoader(train=False)
dl = DataLoader(ds, batch_size=1)
for i,batch in enumerate(dl):
    # ds.result_to_image(batch[1].squeeze(), i)
    y_hat = trained_model(batch[0])
    y_hat = torch.argmax(y_hat.squeeze(), dim=0)
    # print(y_hat.shape)
    result = y_hat
    ds.result_to_image(y_hat, i)
    if i > 10: break
