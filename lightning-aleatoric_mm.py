import os
import sys

import pandas as pd
import seaborn as sn
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class LitMNIST(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=128, learning_rate=2e-3):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.feat_extract = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 2)
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2, self.num_classes)
        )

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        feat = self.feat_extract(x)
        x = self.classifier(feat)
        return feat, x

    def on_test_epoch_start(self):
        def apply_dropout(m):
            if type(m) == nn.Dropout:
                print(m)
                m.train()
        self.apply(apply_dropout)

    def test_epoch_end(self, outputs):
        #print(outputs)
        df = pd.DataFrame(outputs)

        df.cls = df.cls.astype(str)

        color_dict ={k:np.random.randint(0,100,size=3).astype(float)/100 for k in df.cls.unique()}


        df.plot(x='feat_1',y='feat_2',c=[color_dict[k] for k in df.cls],kind="scatter")
        plt.show()
        #fig, axes = plt.subplots(1, 2, figsize=(16,4))
        df.hist(column="output_entropy_across_classes",by='correct',legend=True)
        plt.show()
        df.hist(column="output_entropy_across_samples",by='correct',legend=True)
        plt.show()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = F.softmax(self(x)[-1],dim=1)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = F.softmax(self(x)[-1],dim=1)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        dropout_predictions = torch.tensor([]).to(self.device)
        dropout_feats = torch.tensor([]).to(self.device)
        for i in range(3):
            feats, out = self(x)
            logits = F.softmax(out,dim=1)
            dropout_predictions = torch.concat([dropout_predictions,logits.unsqueeze(0)])
            dropout_feats = torch.concat([dropout_feats,feats.unsqueeze(0)])
        assert dropout_predictions[0].shape == logits.shape

        mean = torch.mean(dropout_predictions,dim=0)

        mean_feat = torch.mean(dropout_feats,dim=0)
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
        self.test_accuracy.update(pred_cls, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)
        #self.log("test_variance", variance, prog_bar=True, on_step=True)

        assert pred_cls.shape == y.shape
        return {
        "correct": (pred_cls == y).item(),
        "output_entropy_across_samples": entropy_across_samples.item(),
        "output_entropy_across_classes": entropy_across_classes.item(),
        "output_mutual_info": mutual_info.item(),
        "cls": y.item(),
        "feat_1": mean_feat.squeeze()[0].item(),
        "feat_2": mean_feat.squeeze()[1].item()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=1)



trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=100,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=[CSVLogger(save_dir="logs/"), WandbLogger(project="MNIST")],
    #limit_test_batches=40
)

#model = LitMNIST.load_from_checkpoint("logs/lightning_logs/version_53/checkpoints/epoch=51-step=11180.ckpt")
model = LitMNIST()
trainer.fit(model)



trainer.test(model)

# metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
# del metrics["step"]
# metrics.set_index("epoch", inplace=True)
#
# sn.relplot(data=metrics, kind="line")
# plt.show()
