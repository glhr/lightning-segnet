import os
import sys

import pandas as pd
import seaborn as sn
import torch
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
import matplotlib.pyplot as plt
import numpy as np
import itertools

from pathlib import Path
from PIL import Image, ImageFile
from argparse import ArgumentParser

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.fashion_test = FashionMNIST(self.data_dir, train=False, transform=self.transform, download=True)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        loaders = {
             "mnist":DataLoader(self.mnist_test, batch_size=1),
             "fashion":DataLoader(self.fashion_test, batch_size=1)
         }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=1)

class DummyDataModule(LightningDataModule):
    class singleimg(Dataset):
        def __init__(self, image_path, transform=None):
            self.img = Image.open(image_path).convert('RGB')
            if transform is None:
                self.transform = transforms.Compose([transforms.Grayscale(),transforms.Resize((28,28)), transforms.ToTensor()])
            else:
                self.transform = transform

        def __len__(self):
            return 100

        def __getitem__(self, x):
            # open image here as PIL / numpy
            image = self.img
            label =  1
            if self.transform is not None:
                image = self.transform(image)

            return image, label

    def __init__(self, data_dir: str = "./toy_datasets/dummy"):
        super().__init__()
        self.data_dir = data_dir
        self.img_path = Path(data_dir) / "train.jpg"
        #self.dataset = self.singleimg(image_path=self.img_path)

    def train_dataloader(self):
        train_split = self.singleimg(image_path=self.img_path)
        return DataLoader(train_split)
    def val_dataloader(self):
        val_split = self.singleimg(image_path=self.img_path)
        return DataLoader(val_split)
    def test_dataloader(self):

        loaders = {
             "dummy":DataLoader(self.singleimg(image_path=self.img_path), batch_size=1),
             "mnist": DataLoader(MNIST(self.data_dir, download=True, train=False, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                )), batch_size=1)
         }
        return CombinedLoader(loaders, mode="max_size_cycle")

class LitMNIST(LightningModule):
    def __init__(self, num_classes, dims, hidden_size=64, learning_rate=2e-3, ):

        super().__init__()

        # Set our init args as class attributes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = num_classes
        self.dims = dims
        channels, width, height = self.dims

        # Define PyTorch model
        self.feat_extract = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(100, hidden_size)
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(hidden_size, self.num_classes)
        )
        self.performance_predictor = nn.Sequential(
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(hidden_size, 1)
        )

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.training_transforms = nn.Sequential(
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomResizedCrop(size=(28,28))
            )

    def forward(self, x):
        feat = self.feat_extract(x)
        x = self.classifier(feat)
        perf = self.performance_predictor(feat)
        return feat, x, perf

    def on_test_epoch_start(self):
        def apply_dropout(m):
            if type(m) == nn.Dropout:
                print(m)
                m.train()
        self.apply(apply_dropout)

    def test_epoch_end(self, outputs):
        outputs = list(itertools.chain.from_iterable(outputs))

        df = pd.DataFrame(outputs)
        #print(df)

        df.cls = df.cls.astype(str)

        color_dict ={k:np.random.randint(0,100,size=3).astype(float)/100 for k in df.cls.unique()}


        sn.scatterplot(data=df,x='feat_1',y='feat_2',hue="dataset",size="feat_1_var",sizes=(20, 1000))
        plt.show()
        #fig, axes = plt.subplots(1, 2, figsize=(16,4))
        df.hist(column="output_entropy_across_classes",by='dataset',legend=True)
        plt.show()
        df.hist(column="output_entropy_across_samples",by='dataset',legend=True)
        plt.show()

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        if optimizer_idx == None: optimizer_idx = 0
        if optimizer_idx == 0: # classifier
            x, y = batch
            pred = self(x)
            logits = F.softmax(pred[1],dim=1)

            loss_ce = F.cross_entropy(logits, y)
            self.log("train_loss_ce", loss_ce, prog_bar=True)
            return loss_ce

        elif optimizer_idx == 1: # perf branch
            x, y = batch
            x = self.training_transforms(x)
            pred = self(x)
            logits = F.softmax(pred[1],dim=1)
            perf = pred[-1]

            ce_samples = F.cross_entropy(logits, y,reduction="none").unsqueeze(-1)
            loss_perf = F.mse_loss(perf,ce_samples)
            self.log("train_loss_perf", loss_perf, prog_bar=True)
            return loss_perf

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = F.softmax(self(x)[-2],dim=1)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        results = []
        for dataset, sample in batch.items():
            x, y = sample

            dropout_predictions = torch.tensor([]).to(self.device)
            dropout_feats = torch.tensor([]).to(self.device)
            if y > self.num_classes - 1:
                y =  y -y
            for i in range(2):
                feats, out, perf = self(x)
                logits = F.softmax(out,dim=1)
                dropout_predictions = torch.concat([dropout_predictions,logits.unsqueeze(0)])
                dropout_feats = torch.concat([dropout_feats,feats.unsqueeze(0)])
            assert dropout_predictions[0].shape == logits.shape

            mean = torch.mean(dropout_predictions,dim=0)

            mean_feat = torch.mean(dropout_feats,dim=0)
            var_feat = torch.var(dropout_feats,dim=0)
            #print(mean_feat)
                # Calculating variance across multiple MCD forward passes
            variance = torch.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)

            epsilon = 0.001
            assert not torch.sum(torch.isnan(torch.log(mean + epsilon)))
            # Calculating entropy across multiple MCD forward passes
            entropy_across_samples = -torch.sum(mean*torch.log(mean + epsilon), axis=-1) # shape (n_samples,)
            entropy_across_classes = torch.mean(-torch.sum(dropout_predictions*torch.log(dropout_predictions + epsilon),
                                                    axis=-1), axis=0)

            # Calculating mutual information across multiple MCD forward passes
            mutual_info = entropy_across_samples - entropy_across_classes # shape (n_samples,)

            #print("---")
            loss = F.cross_entropy(mean, y)
            pred_cls = torch.argmax(mean, dim=1)
            #self.test_accuracy.update(pred_cls, y)

            # Calling self.log will surface up scalars for you in TensorBoard
            self.log("test_loss", loss, prog_bar=True)
            #self.log("test_acc", self.test_accuracy, prog_bar=True)
            #self.log("test_variance", variance, prog_bar=True, on_step=True)

            #assert pred_cls.shape == y.shape
            results.append({
            #"correct": (pred_cls == y).item(),
            "output_entropy_across_samples": entropy_across_samples.item(),
            "output_entropy_across_classes": entropy_across_classes.item(),
            "output_mutual_info": mutual_info.item(),
            "cls": y.item(),
            "feat_1": mean_feat.squeeze()[0].item(),
            "feat_1_var": var_feat.squeeze()[0].item() + var_feat.squeeze()[1].item(),
            "feat_2": mean_feat.squeeze()[1].item(),
            "dataset": dataset
            })
        return results

    def configure_optimizers(self):
        optimizer_classifier = torch.optim.Adam([
                    {'params': self.classifier.parameters()},
                    {'params': self.feat_extract.parameters()}
                ], lr=self.learning_rate)
        optimizer_perf = torch.optim.Adam([
                    {'params': self.performance_predictor.parameters()},
                    {'params': self.feat_extract.parameters()}
                ], lr=self.learning_rate)
        return [optimizer_classifier,optimizer_perf]


parser = ArgumentParser()
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--test_samples', type=int, default=None)

args = parser.parse_args()

trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=args.max_epochs,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=[CSVLogger(save_dir="logs/"), WandbLogger(project="MNIST")],
    limit_test_batches=args.test_samples
)


if args.train:
    model = LitMNIST(num_classes=10, dims=(1, 28, 28))
    #dm = DummyDataModule()
    dm = MNISTDataModule()
    trainer.fit(model,dm)
    trainer.test(model,dm)
else:
    model = LitMNIST.load_from_checkpoint("logs/lightning_logs/version_62/checkpoints/epoch=99-step=21500.ckpt")
    trainer.test(model)


# metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
# del metrics["step"]
# metrics.set_index("epoch", inplace=True)
#
# sn.relplot(data=metrics, kind="line")
# plt.show()
