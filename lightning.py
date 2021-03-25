import os
import numpy as np
import random

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from segnet import SegNet
from losses import SORDLoss, KLLoss, CompareLosses
from metrics import MaskedIoU, ConfusionMatrix
from dataloader import FreiburgDataLoader, CityscapesDataLoader, KittiDataLoader, OwnDataLoader
from plotting import plot_confusion_matrix
from utils import create_folder, logger, enable_debug, RANDOM_SEED

from argparse import ArgumentParser
from datetime import datetime

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_deterministic(True)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

timestamp = datetime.now().strftime('%Y-%m-%d %H-%M')

parser = ArgumentParser()
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--test_samples', type=int, default=None)
parser.add_argument('--test_checkpoint', default="lightning_logs/test.ckpt")
parser.add_argument('--train_checkpoint', default="lightning_logs/last.ckpt")
parser.add_argument('--prefix', default=None)
parser.add_argument('--debug', default=False, action="store_true")


class LitSegNet(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--bs', type=int, default=16)
        parser.add_argument('--lr', type=float, default=None)
        parser.add_argument('--momentum', type=int, default=None)
        parser.add_argument('--optim', type=str, default=None)
        parser.add_argument('--num_classes', type=int, default=3)
        parser.add_argument('--workers', type=int, default=0)
        parser.add_argument('--mode', default="affordances")
        parser.add_argument('--dataset', default="freiburg")
        parser.add_argument('--augment', action="store_true", default=False)
        parser.add_argument('--loss', default=None)
        parser.add_argument('--orig_dataset', default="freiburg")
        return parser

    def __init__(self, conf, test_checkpoint = None, test_max=None, **kwargs):
        super().__init__()
        pl.seed_everything(RANDOM_SEED)

        self.save_hyperparameters(conf)
        self.hparams.resize = (480, 240)
        self.hparams.masking = True
        self.hparams.normalize = False
        self.test_checkpoint = test_checkpoint
        self.test_max = test_max

        self.model = SegNet(num_classes=self.hparams.num_classes)

        self.datasets = {
            "freiburg": FreiburgDataLoader,
            "cityscapes": CityscapesDataLoader,
            "kitti": KittiDataLoader,
            "own": OwnDataLoader
        }
        self.hparams.ranks = np.array(range(self.hparams.num_classes))
        self.sord = SORDLoss(n_classes=self.hparams.num_classes, masking=self.hparams.masking, ranks=self.hparams.ranks)
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.kl = KLLoss(n_classes=self.hparams.num_classes, masking=self.hparams.masking)
        self.loss = CompareLosses(n_classes=self.hparams.num_classes, masking=self.hparams.masking, ranks=self.hparams.ranks, returnloss=self.hparams.loss)

        self.train_set, self.val_set, self.test_set = self.get_dataset_splits(normalize=self.hparams.normalize)
        self.hparams.train_set, self.hparams.val_set, self.hparams.test_set = \
            len(self.train_set.dataset), len(self.val_set.dataset), len(self.test_set.dataset)

        self.orig_dataset = self.get_dataset(name=self.hparams.orig_dataset, set="test")

        # self.IoU = IoU(num_classes=self.hparams.num_classes, ignore_index=self.hparams.ignore_index)
        self.hparams.labels_orig = set(range(self.hparams.num_classes))
        self.hparams.labels_orig = list(self.hparams.labels_orig)
        self.IoU = MaskedIoU(labels=self.hparams.labels_orig)

        self.num_cls = 3 if self.hparams.mode == "convert" else self.hparams.num_classes
        self.hparams.labels_conv = set(range(self.num_cls))
        self.hparams.labels_conv = list(self.hparams.labels_conv)

        self.CM = ConfusionMatrix(labels=self.hparams.labels_conv)
        # self.IoU_conv = IoU(num_classes=self.num_cls, ignore_index=0)
        self.IoU_conv = MaskedIoU(labels=self.hparams.labels_conv)

        self.result_folder = f"results/{self.hparams.dataset}/"
        self.save_prefix = f"{timestamp}-{self.hparams.dataset}-c{self.hparams.num_classes}-{self.hparams.loss}"
        create_folder(f"{self.result_folder}/viz_per_epoch")

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # print(x.shape)
        embedding = self.model(x)
        return embedding

    def compute_loss(self, x_hat, y, loss="ce"):
        # if loss == "ce":
        #     return self.ce(x_hat, y)
        # elif loss == "sord":
        #     return self.sord(x_hat, y)
        # elif loss == "kl":
        #     return self.kl(x_hat, y)
        return self.loss(x_hat, y)

    def save_result(self, sample, pred, pred_cls, target, batch_idx=0):
        for i,(o,p,c,t) in enumerate(zip(sample,pred,pred_cls,target)):
            # print(p.shape)
            test = p.squeeze()[self.test_set.dataset.aff_idx["impossible"]] * self.hparams.ranks[0] \
             + p.squeeze()[self.test_set.dataset.aff_idx["possible"]] * self.hparams.ranks[1] \
             + p.squeeze()[self.test_set.dataset.aff_idx["preferable"]] * self.hparams.ranks[2]
            self.test_set.dataset.result_to_image(
                iter=batch_idx+i, gt=t, orig=o, pred_cls=c, pred_proba=test,
                folder=f"{self.result_folder}/viz_per_epoch",
                filename_prefix=f"{self.save_prefix}-epoch{self.current_epoch}-proba")
            # self.test_set.dataset.result_to_image(iter=batch_idx+i, pred_cls=c, folder=f"{self.result_folder}", filename_prefix=f"{self.save_prefix}-epoch{self.current_epoch}-cls")
            # self.test_set.dataset.result_to_image(iter=batch_idx+i, gt=t, folder=f"{self.result_folder}", filename_prefix=f"ref")
            # self.test_set.dataset.result_to_image(iter=batch_idx+i, orig=o, folder=f"{self.result_folder}", filename_prefix=f"orig")

    def predict(self, batch, set, save=False, batch_idx=None):
        x, y = batch
        x_hat = self.model(x)

        loss = self.compute_loss(x_hat, y, loss=self.hparams.loss)

        x_hat = torch.softmax(x_hat, dim=1)
        pred_cls = torch.argmax(x_hat, dim=1)
        iou = self.IoU(x_hat, y)

        if self.hparams.mode == "convert":
            self.log(f'{set}_iou_obj', iou, on_epoch=True)
            pred_proba_aff = self.train_set.dataset.labels_obj_to_aff(x_hat, num_cls=self.num_cls, proba=True)
            pred_cls_aff = torch.argmax(pred_proba_aff, dim=1)
            target_aff = self.train_set.dataset.labels_obj_to_aff(y, num_cls=self.num_cls)
            iou_aff = self.IoU_conv(pred_proba_aff, target_aff)
            self.log(f'{set}_iou_aff', iou_aff, on_epoch=True)
        elif self.hparams.mode == "affordances":
            self.log(f'{set}_iou_aff', iou, on_epoch=True)
        elif self.hparams.mode == "objects":
            self.log(f'{set}_iou_obj', iou, on_epoch=True)

        self.log(f'{set}_loss', loss, on_epoch=True)

        if save:
            if self.hparams.mode == "convert":
                self.save_result(sample=x, pred=pred_proba_aff, pred_cls=pred_cls_aff, target=target_aff, batch_idx=batch_idx)
            else:
                self.save_result(sample=x, pred=x_hat, pred_cls=pred_cls, target=y, batch_idx=batch_idx)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.predict(batch, set="train")
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and (not self.current_epoch % 10):
            loss = self.predict(batch, set="val", save=True, batch_idx=batch_idx)
        else:
            loss = self.predict(batch, set="val")
        return loss

    def reduce_cm(self, cms):

        labels = self.train_set.dataset.cls_labels

        cms = torch.tensor(cms, dtype=torch.long)

        cms = torch.reshape(cms, (-1, self.num_cls, self.num_cls))
        cm = torch.sum(cms, dim=0, keepdim=False)

        # ignore void class
        # cm = cm[1:, 1:]
        # cm = np.delete(cm, self.hparams.ignore_index, 0)
        if len(labels) > self.num_cls:
            labels.pop(0)
        #
        # print(cm)

        cm = cm / cm.sum(axis=1, keepdim=True) # normalize confusion matrix

        plot_confusion_matrix(cm.numpy(), labels=labels, filename=f"{self.hparams.dataset}-{self.hparams.mode}-{self.test_checkpoint}", folder=f"{self.result_folder}")
        return 0

    def test_step(self, batch, batch_idx):
        sample, target_orig = batch
        folder = f"{segnet_model.result_folder}/{self.test_checkpoint}"

        if self.test_max is None or batch_idx < self.test_max:
            # print(torch.min(sample),torch.max(sample))
            pred_orig = self.model(sample)
            loss = self.compute_loss(pred_orig, target_orig, loss=self.hparams.loss)
            pred_orig = torch.softmax(pred_orig, dim=1)
            pred_cls_orig = torch.argmax(pred_orig, dim=1)

            if self.hparams.mode == "convert":
                # print(torch.unique(torch.argmax(pred_orig, dim=1)))
                pred = self.orig_dataset.dataset.labels_obj_to_aff(pred_orig, proba=True)
                for i, p in enumerate(pred_orig):
                    proba_lst = []
                    for cls, map in enumerate(p.squeeze()):
                        if self.hparams.orig_dataset == "freiburg" and cls==0: # ignore void
                            pass
                        else:
                            proba_lst.append(map)
                    self.orig_dataset.dataset.result_to_image(
                        iter=batch_idx+i,
                        proba_lst=proba_lst,
                        folder=folder,
                        filename_prefix=f"probas_orig-{self.test_checkpoint}",
                        dataset_name=self.hparams.dataset)
            else:
                pred = pred_orig
            pred_cls = torch.argmax(pred, dim=1)

            if len(target_orig) > 1:
                target_orig = target_orig.squeeze()
            if self.hparams.mode == "convert":
                target = self.test_set.dataset.labels_obj_to_aff(target_orig)
            else:
                target = target_orig

            # print("pred",pred_cls.shape,"target",target.shape)

            for i,(o,p,c,t) in enumerate(zip(sample,pred,pred_cls,target)):
                # print(p.shape)
                proba_imposs = p.squeeze()[self.test_set.dataset.aff_idx["impossible"]]
                proba_poss = p.squeeze()[self.test_set.dataset.aff_idx["possible"]]
                proba_pref = p.squeeze()[self.test_set.dataset.aff_idx["preferable"]]
                test = proba_imposs * 0 + proba_poss * 1 + proba_pref * 2
                iter = batch_idx*self.hparams.bs + i


                for cls,map in enumerate(p.squeeze()):
                    proba_lst = []
                    if self.hparams.mode == "objects" and self.hparams.orig_dataset == "freiburg" and cls==0: # ignore void
                        pass
                    else:
                        proba_lst.append(map)
                        # self.orig_dataset.dataset.result_to_image(
                        #     iter=batch_idx+i,
                        #     proba_lst=proba_lst,
                        #     folder=folder,
                        #     filename_prefix=f"probas{cls}-{self.test_checkpoint}",
                        #     dataset_name=self.hparams.dataset)
                # logger.debug("Generating proba map")
                self.orig_dataset.dataset.result_to_image(iter=batch_idx+i, pred_proba=test, folder=folder, filename_prefix=f"proba-{self.test_checkpoint}", dataset_name=self.hparams.dataset)
                # logger.debug("Generating argmax pred")
                self.orig_dataset.dataset.result_to_image(iter=batch_idx+i, pred_cls=c, folder=folder, filename_prefix=f"cls-{self.test_checkpoint}", dataset_name=self.hparams.dataset)
                self.test_set.dataset.result_to_image(iter=batch_idx+i, gt=t, orig=o, folder=folder, filename_prefix=f"ref-dual", dataset_name=self.hparams.dataset)
                self.test_set.dataset.result_to_image(iter=batch_idx+i, orig=o, folder=folder, filename_prefix=f"orig", dataset_name=self.hparams.dataset)
                self.test_set.dataset.result_to_image(iter=batch_idx+i, gt=t, folder=folder, filename_prefix=f"gt", dataset_name=self.hparams.dataset)
                self.test_set.dataset.result_to_image(
                    iter=batch_idx+i,
                    orig=o,
                    gt=t,
                    pred_cls=c,
                    pred_proba=test,
                    folder=folder,
                    filename_prefix=f"res", dataset_name=self.hparams.dataset)
                # self.test_set.dataset.result_to_image(iter=batch_idx+i, # pred_proba=p.squeeze()[self.test_set.dataset.aff_idx["impossible"]], folder=folder, filename_prefix=f"proba0")
                # self.test_set.dataset.result_to_image(
                #     iter=batch_idx+i, gt=t, orig=o,
                #     folder=f"{self.result_folder}/viz_per_epoch",
                #     filename_prefix=f"gt")

            try:
                cm = self.CM(pred, target)
                # print(cm.shape)
                iou = self.IoU_conv(pred, target)

                self.log('test_iou', iou, on_step=False, prog_bar=False, on_epoch=True)
                self.log('cm', cm, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_cm)
            except Exception as e:
                print("Couldn't compute eval metrics",e)
            return pred


    def configure_optimizers(self):
        if self.hparams.optim == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def get_dataset(self, set, name=None, augment=None):
        if name is None:
            name = self.hparams.dataset
        if augment is None:
            augment = self.hparams.augment if set == "train" else False
        dataset = self.datasets[name](set=set, resize=self.hparams.resize, mode=self.hparams.mode, modalities=["rgb"], augment=augment)
        dataset = Subset(dataset, indices=range(len(dataset)))
        return dataset

    def get_dataset_splits(self, normalize=False):
        if self.hparams.dataset == "freiburg":
            train_set = self.get_dataset(set="train")
            test_set = self.get_dataset(set="test",augment=False)
            val_set = test_set
            # total_len = len(train_set)
            # val_len = int(0.1*total_len)
            # train_len = total_len - val_len
            # train_set, val_set = random_split(train_set, [train_len, val_len])

        elif self.hparams.dataset == "own":
            train_set = self.get_dataset(set="train")
            test_set = self.get_dataset(set="test")
            val_set = test_set
            # total_len = len(train_set)
            # val_len = int(0.1*total_len)
            # train_len = total_len - val_len
            # train_set, val_set = random_split(train_set, [train_len, val_len])

        elif self.hparams.dataset == "kitti":
            train_set = self.get_dataset(set="train")
            val_set = self.get_dataset(set="train", augment=False)
            total_len = len(train_set)
            val_len = int(0.2*total_len)
            train_len = total_len - val_len*2
            train_set, _, _ = random_split(train_set, [train_len, val_len, val_len])
            _, val_set, test_set = random_split(val_set, [train_len, val_len, val_len])
            train_set, val_set, test_set = train_set.dataset, val_set.dataset, test_set.dataset
            # print(test_set[0])

        elif self.hparams.dataset == "cityscapes":
            train_set = self.get_dataset(set="train")
            val_set = self.get_dataset(set="val")
            test_set = self.get_dataset(set="test")

        if normalize:
            mean = 0.
            std = 0.
            loader = DataLoader(train_set, batch_size=self.hparams.bs, num_workers=self.hparams.workers, shuffle=False)
            for images, _ in loader:
                batch_samples = images.size(0)  # batch size
                # print(images.shape)
                images = images.view(batch_samples, images.size(1), -1)
                # print(images.shape)
                mean += images.mean(2).sum(0)
                std += images.std(2).sum(0)

            mean /= len(loader.dataset)
            std /= len(loader.dataset)
            print("Mean and stdev",mean,std)
            tf = transforms.Compose([
                transforms.Normalize(mean=(mean,), std=(std,))
            ])
            train_set.dataset.transform = tf
            test_set.dataset.transform = tf
            val_set.dataset.transform = tf

        return train_set, val_set, test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.bs, num_workers=self.hparams.workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.bs, num_workers=self.hparams.workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.bs, num_workers=self.hparams.workers, shuffle=False)

parser = LitSegNet.add_model_specific_args(parser)
args = parser.parse_args()
if args.debug: enable_debug()

print(args)
segnet_model = LitSegNet(conf=args)

if args.prefix is None:
    args.prefix = segnet_model.save_prefix
print(args.prefix)

checkpoint_callback = ModelCheckpoint(
    dirpath='lightning_logs',
    filename=args.prefix+'-{epoch}-{val_loss:.4f}',
    verbose=True,
    monitor='val_loss',
    mode='min',
    save_last=True
)
checkpoint_callback.CHECKPOINT_NAME_LAST = f"{args.prefix}-last"

if args.train:
    logger.warning("Training phase")
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
    logger.warning("Testing phase")
    trainer = pl.Trainer.from_argparse_args(args)
    chkpt = args.test_checkpoint.split("/")[-1].replace(".ckpt", "")
    create_folder(f"{segnet_model.result_folder}/{chkpt}")
    trained_model = segnet_model.load_from_checkpoint(checkpoint_path=args.test_checkpoint, test_max = args.test_samples, test_checkpoint=chkpt, conf=args)
    trainer.test(trained_model)
