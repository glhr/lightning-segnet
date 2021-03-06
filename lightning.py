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
from losses import SORDLoss, flatten_tensors
from dataloader import MMDataLoader, FreiburgDataLoader, CityscapesDataLoader, KittiDataLoader, OwnDataLoader

import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser()

from datetime import datetime
from plotting import plot_confusion_matrix

timestamp = datetime.now().strftime('%Y-%m-%d %H-%M')


parser.add_argument('--train', action='store_true', default=False)

parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--test_samples', type=int, default=None)
parser.add_argument('--test_checkpoint', default="lightning_logs/test.ckpt")
parser.add_argument('--train_checkpoint', default="lightning_logs/last.ckpt")
parser.add_argument('--prefix', default=None)

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.classification import IoU, ConfusionMatrix

class LitSegNet(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--bs', type=int, default=16)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--momentum', type=int, default=0.9)
        parser.add_argument('--optim', type=str, default="SGD")
        parser.add_argument('--num_classes', type=int, default=4)
        parser.add_argument('--workers', type=int, default=8)
        parser.add_argument('--mode', default="affordances")
        parser.add_argument('--dataset', default="freiburg")
        parser.add_argument('--loss', default="ce")
        return parser

    def __init__(self, conf, test_checkpoint = None, test_max=None, **kwargs):
        super().__init__()

        self.save_hyperparameters(conf)
        self.metric = IoU(num_classes=self.hparams.num_classes, ignore_index=0)
        self.model = SegNet(num_classes=self.hparams.num_classes)

        self.datasets = {
            "freiburg": FreiburgDataLoader,
            "cityscapes": CityscapesDataLoader,
            "kitti": KittiDataLoader,
            "own": OwnDataLoader
        }
        self.sord = SORDLoss(n_classes = self.hparams.num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

        self.test_checkpoint = test_checkpoint
        self.ds = self.get_dataset(train=False)
        self.test_max = test_max

        self.num_cls = 4 if self.hparams.mode == "convert" else self.hparams.num_classes
        self.CM = ConfusionMatrix(num_classes=self.num_cls, normalize='none')
        self.IoU = IoU(num_classes=self.num_cls, ignore_index=0)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # print(x.shape)
        embedding = self.model(x)
        return embedding

    def compute_loss(self, x_hat, y, loss="ce"):
        if loss == "ce":
            return self.ce(x_hat, y)
        elif loss == "sord":
            return self.sord(x_hat, y)


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x_hat = self.model(x)

        loss = self.compute_loss(x_hat, y, loss=self.hparams.loss)

        x_hat = torch.softmax(x_hat, dim=1)
        iou = self.metric(x_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_iou', iou, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.compute_loss(x_hat, y, loss=self.hparams.loss)

        x_hat = torch.softmax(x_hat, dim=1)
        iou = self.metric(x_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_iou', iou, on_epoch=True)
        return loss

    def reduce_cm(self, cms):

        labels = self.ds.cls_labels

        cms = torch.reshape(cms, (-1,self.num_cls,self.num_cls))
        cm = torch.sum(cms,dim=0,keepdim=False)

        # ignore void class
        cm = cm[1:, 1:]
        labels.pop(0)

        print(cm)

        cm = cm / cm.sum(axis=1, keepdim=True) # normalize confusion matrix

        plot_confusion_matrix(cm.numpy(), labels=labels, filename=f"{self.hparams.mode}-{self.test_checkpoint}", folder=f"results/{self.hparams.dataset}")
        return 0

    def test_step(self, batch, batch_idx):
        sample, target = batch

        if self.test_max is None or batch_idx < self.test_max:
            pred = self.model(sample)
            pred = torch.softmax(pred, dim=1)

            if self.hparams.mode == "convert": pred = self.ds.labels_obj_to_aff(pred, proba=True)
            pred_cls = torch.argmax(pred, dim=1)

            if len(target) > 1:
                target = target.squeeze()
            if self.hparams.mode == "convert": target = self.ds.labels_obj_to_aff(target)

            # print("pred",pred_cls.shape,"target",target.shape)

            for i,(o,p,c,t) in enumerate(zip(sample,pred,pred_cls,target)):
                # print(p.shape)
                test = p.squeeze()[1] * 0 + p.squeeze()[2] * 1 + p.squeeze()[3] * 2
                iter = batch_idx*self.hparams.bs + i
                self.ds.result_to_image(iter=batch_idx+i, pred_proba=test, folder=f"{self.hparams.dataset}", filename_prefix=f"{self.test_checkpoint}")
                self.ds.result_to_image(iter=batch_idx+i, gt=t, folder=f"{self.hparams.dataset}", filename_prefix=f"ref")

            cm = self.CM(pred_cls, target)
            # print(cm.shape)
            iou = self.IoU(pred_cls, target)

            self.log('test_iou', iou, on_step=False, prog_bar=False, on_epoch=True)
            self.log('cm', cm, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_cm)
            return pred


    def configure_optimizers(self):
        if self.hparams.optim == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def get_dataset(self, train=False):
        train = True if self.hparams.dataset == "kitti" else train
        return self.datasets[self.hparams.dataset](train=train, mode=self.hparams.mode, modalities=["rgb"])

    def train_dataloader(self):
        # REQUIRED
        dl = self.get_dataset(train=True)
        return DataLoader(dl, batch_size=self.hparams.bs, num_workers=self.hparams.workers, shuffle=True)

    def val_dataloader(self):
        # OPTIONAL
        dl = self.get_dataset(train=False)
        return DataLoader(dl, batch_size=self.hparams.bs, num_workers=self.hparams.workers, shuffle=True)

    def test_dataloader(self):
        # OPTIONAL
        dl = self.get_dataset(train=False)
        return DataLoader(dl, batch_size=self.hparams.bs, num_workers=self.hparams.workers, shuffle=False)

parser = LitSegNet.add_model_specific_args(parser)
args = parser.parse_args()

print(args)
segnet_model = LitSegNet(conf=args)

if args.prefix is None:
    args.prefix = f"{timestamp}-{segnet_model.hparams.dataset}-c{segnet_model.hparams.num_classes}-{segnet_model.hparams.loss}"
print(args.prefix)

checkpoint_callback = ModelCheckpoint(
    dirpath='lightning_logs',
    filename= args.prefix + '-{epoch}-{val_loss:.4f}',
    verbose=True,
    monitor='val_loss',
    mode='min',
    save_last = True
)
checkpoint_callback.CHECKPOINT_NAME_LAST = f"{args.prefix}-last"

if args.train:
    wandb_logger = WandbLogger(project='segnet-freiburg', log_model = False)
    wandb_logger.log_hyperparams(segnet_model.hparams)

    trainer = pl.Trainer.from_argparse_args(args,
        check_val_every_n_epoch=1,
        # ~ log_every_n_steps=10,
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=args.train_checkpoint)
    trainer.fit(segnet_model)

else:
    trainer = pl.Trainer.from_argparse_args(args)
    trained_model = LitSegNet.load_from_checkpoint(checkpoint_path=args.test_checkpoint, test_max = args.test_samples, test_checkpoint=args.test_checkpoint.split("/")[-1].replace(".ckpt",""), conf=args)
    trainer.test(trained_model)
