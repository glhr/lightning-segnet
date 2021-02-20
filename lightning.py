import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from segnet import SegNet
from dataloader import FreiburgDataLoader

import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser()



parser.add_argument('--train', action='store_true', default=False)

parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--test_samples', type=int, default=10)



from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='lightning_logs',
    filename='{epoch}-{val_loss:.2f}',
    verbose=True,
    monitor='val_loss',
    mode='min'
)

class LitSegNet(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--bs', type=int, default=16)
        parser.add_argument('--lr', type=int, default=0.1)
        parser.add_argument('--momentum', type=int, default=0.9)
        parser.add_argument('--optim', type=str, default="SGD")
        parser.add_argument('--num_classes', type=int, default=7)
        return parser

    def __init__(self, conf, **kwargs):
        super().__init__()
        self.save_hyperparameters(conf)
        self.model = SegNet(num_classes=self.hparams.num_classes)

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
        x_hat = torch.softmax(x_hat, dim=1)
        loss = F.cross_entropy(x_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        x_hat = torch.softmax(x_hat, dim=1)
        loss = F.cross_entropy(x_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        if self.hparams.optim == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        # REQUIRED
        dl = FreiburgDataLoader(train=True)
        return DataLoader(dl, batch_size=self.hparams.bs)

    def val_dataloader(self):
        # OPTIONAL
        dl = FreiburgDataLoader(train=False)
        return DataLoader(dl, batch_size=self.hparams.bs)

parser = LitSegNet.add_model_specific_args(parser)
args = parser.parse_args()

print(args)
segnet_model = LitSegNet(conf=args)

if args.train:
    wandb_logger = WandbLogger(project='segnet-freiburg', log_model = False)
    wandb_logger.log_hyperparams(segnet_model.hparams)

    trainer = pl.Trainer.from_argparse_args(args, check_val_every_n_epoch=5, log_every_n_steps=10, logger=wandb_logger, checkpoint_callback=checkpoint_callback)
    trainer.fit(segnet_model)

trained_model = LitSegNet.load_from_checkpoint(checkpoint_path="lightning_logs/epoch=219-val_loss=1.36.ckpt", conf=args)
# prints the learning_rate you used in this checkpoint

trained_model.eval()
ds = FreiburgDataLoader(train=False, modalities=["rgb"])
dl = DataLoader(ds, batch_size=1)
for i,batch in enumerate(dl):
    if i >= args.test_samples: break
    # ds.result_to_image(batch[1].squeeze(), i)
    y_hat = trained_model(batch[0])
    y_hat = torch.argmax(y_hat.squeeze(), dim=0)
    # print(y_hat.shape)
    result = y_hat
    ds.result_to_image(y_hat, i, orig=batch[0], gt=batch[1].squeeze())
