import os
import numpy as np
import random

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback

from segnet import SegNet, new_input_channels, new_output_channels
from losses import SORDLoss, KLLoss, CompareLosses
from metrics import MaskedIoU, ConfusionMatrix, Mistakes, iou_from_confmat, weight_from_target
from dataloader import FreiburgDataLoader, CityscapesDataLoader, KittiDataLoader, OwnDataLoader, ThermalVOCDataLoader, SynthiaDataLoader
from plotting import plot_confusion_matrix
from utils import create_folder, logger, enable_debug, RANDOM_SEED

from argparse import ArgumentParser
from datetime import datetime

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.set_deterministic(True)
except Exception as e:
    logger.error(e)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

timestamp = datetime.now().strftime('%Y-%m-%d %H-%M')

parser = ArgumentParser()
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--test_samples', type=int, default=None)
parser.add_argument('--test_checkpoint', default="lightning_logs/test.ckpt")
parser.add_argument('--train_checkpoint', default="lightning_logs/last.ckpt")
parser.add_argument('--prefix', default=None)
parser.add_argument('--debug', default=False, action="store_true")
parser.add_argument('--save', default=False, action="store_true")
parser.add_argument('--viz', default=False, action="store_true")
parser.add_argument('--full_dataset', default=False, action="store_true")
parser.add_argument('--update_output_layer', default=False, action="store_true")
parser.add_argument('--init', default=False, action="store_true")

import inspect

class CustomCB(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        logger.debug(f'{inspect.currentframe().f_code.co_name}')

    def on_train_epoch_end(self, trainer, pl_module):
        logger.debug(f'{inspect.currentframe().f_code.co_name}')

    def on_validation_epoch_start(self, trainer, pl_module):
        logger.debug(f'{inspect.currentframe().f_code.co_name}')

    def on_validation_epoch_end(self, trainer, pl_module):
        logger.debug(f'{inspect.currentframe().f_code.co_name}')

    def on_epoch_start(self, trainer, pl_module):
        logger.debug(f'{inspect.currentframe().f_code.co_name}')

    def on_epoch_end(self, trainer, pl_module):
        logger.debug(f'{inspect.currentframe().f_code.co_name}')

    def on_train_start(self, trainer, pl_module):
        logger.debug(f'{inspect.currentframe().f_code.co_name}')

    def on_train_end(self, trainer, pl_module):
        logger.debug(f'{inspect.currentframe().f_code.co_name}')

    def on_validation_start(self, trainer, pl_module):
        logger.debug(f'{inspect.currentframe().f_code.co_name}')

    def on_validation_end(self, trainer, pl_module):
        logger.debug(f'{inspect.currentframe().f_code.co_name}')

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
        parser.add_argument('--loss_weight', action="store_true", default=False)
        parser.add_argument('--loss', default=None)
        parser.add_argument('--orig_dataset', default="freiburg")
        parser.add_argument('--modalities', default="rgb")
        parser.add_argument('--ranks', default="1,2,3")
        parser.add_argument('--dist', default="l1")
        return parser

    def __init__(self, conf, viz=False, save=False, full_dataset=False, test_checkpoint = None, test_max=None, **kwargs):
        super().__init__()
        pl.seed_everything(RANDOM_SEED)
        self.save = save
        self.viz = viz

        self.save_hyperparameters(conf)
        self.hparams.resize = (480, 240)
        self.hparams.masking = True
        self.hparams.normalize = False
        self.test_checkpoint = test_checkpoint
        self.test_max = test_max
        self.full = full_dataset

        self.model = SegNet(num_classes=self.hparams.num_classes)
        self.hparams.modalities = self.hparams.modalities.split(",")
        logger.warning(f"modalities {self.hparams.modalities}")

        self.datasets = {
            "freiburg": FreiburgDataLoader,
            "cityscapes": CityscapesDataLoader,
            "kitti": KittiDataLoader,
            "own": OwnDataLoader,
            "thermalvoc": ThermalVOCDataLoader,
            "synthia": SynthiaDataLoader
        }

        if self.hparams.loss in ["sord","compare"]:
            self.hparams.ranks = [int(r) for r in self.hparams.ranks.split(",")]
        else:
            self.hparams.ranks = [1,2,3]


        self.train_set, self.val_set, self.test_set = self.get_dataset_splits(normalize=self.hparams.normalize)
        self.hparams.train_set, self.hparams.val_set, self.hparams.test_set = \
            len(self.train_set.dataset), len(self.val_set.dataset), len(self.test_set.dataset)

        self.orig_dataset = self.get_dataset(name=self.hparams.orig_dataset, set="test")

        self.update_settings()

    def update_settings(self):

        self.sord = SORDLoss(n_classes=self.hparams.num_classes, masking=self.hparams.masking, ranks=self.hparams.ranks, dist=self.hparams.dist)
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.kl = KLLoss(n_classes=self.hparams.num_classes, masking=self.hparams.masking)
        self.loss = CompareLosses(n_classes=self.hparams.num_classes, masking=self.hparams.masking, ranks=self.hparams.ranks, dist=self.hparams.dist, returnloss="kl")
        self.dist = Mistakes(ranks=self.hparams.ranks)
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
        self.hparams.save_prefix = f"{timestamp}-{self.hparams.dataset}-c{self.hparams.num_classes}-{self.hparams.loss}"
        if self.hparams.loss == "sord":
            self.hparams.save_prefix += f'-{",".join([str(r) for r in self.hparams.ranks])}'
            self.hparams.save_prefix += f'-{self.hparams.dist}'
        self.hparams.save_prefix += f'-{",".join(self.hparams.modalities)}'
        logger.info(self.hparams.save_prefix)
        create_folder(f"{self.result_folder}/viz_per_epoch")
        create_folder(f"{self.result_folder}/gt")
        create_folder(f"{self.result_folder}/orig")


    def update_model(self):
        channels = len(self.hparams.modalities)
        self.model = new_input_channels(self.model, channels)

    def new_output(self):
        self.model = new_output_channels(self.model, 3)
        self.hparams.num_classes = 3
        logger.debug(self.model)
        self.update_settings()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # logger.debug(x.shape)
        embedding = self.model(x)
        return embedding

    def compute_loss(self, x_hat, y, loss="ce", weight_map=None):
        if loss == "ce":
            return self.ce(x_hat, y)
        elif loss == "sord":
            return self.sord(x_hat, y, weight_map=weight_map)
        elif loss == "kl":
            return self.kl(x_hat, y, weight_map=weight_map)
        elif loss == "compare":
            return self.loss(x_hat, y, weight_map=weight_map)

    def save_result(self, sample, pred, pred_cls, target, batch_idx=0):
        for i,(o,p,c,t) in enumerate(zip(sample,pred,pred_cls,target)):
            # logger.debug(p.shape)
            if self.hparams.ranks is not None:
                test = p.squeeze()[self.test_set.dataset.aff_idx["impossible"]] * self.hparams.ranks[0] \
                 + p.squeeze()[self.test_set.dataset.aff_idx["possible"]] * self.hparams.ranks[1] \
                 + p.squeeze()[self.test_set.dataset.aff_idx["preferable"]] * self.hparams.ranks[2]
            else:
                test = None
            self.test_set.dataset.result_to_image(
                iter=batch_idx+i, gt=t, orig=o, pred_cls=c, pred_proba=test,
                folder=f"{self.result_folder}/viz_per_epoch",
                filename_prefix=f"{self.hparams.save_prefix}-epoch{self.current_epoch}-proba")
            # self.test_set.dataset.result_to_image(iter=batch_idx+i, pred_cls=c, folder=f"{self.result_folder}", filename_prefix=f"{self.hparams.save_prefix}-epoch{self.current_epoch}-cls")
            # self.test_set.dataset.result_to_image(iter=batch_idx+i, gt=t, folder=f"{self.result_folder}", filename_prefix=f"ref")
            # self.test_set.dataset.result_to_image(iter=batch_idx+i, orig=o, folder=f"{self.result_folder}", filename_prefix=f"orig")

    def predict(self, batch, set, save=False, batch_idx=None):
        x, y = batch
        x_hat = self.model(x)

        if self.hparams.loss_weight:
            weight_map = weight_from_target(y)
        else:
            weight_map = None

        loss = self.compute_loss(x_hat, y, loss=self.hparams.loss, weight_map=weight_map)

        x_hat = torch.softmax(x_hat, dim=1)
        pred_cls = torch.argmax(x_hat, dim=1)
        # iou = self.IoU(x_hat, y)

        if self.hparams.mode == "convert":
            # self.log(f'{set}_iou_obj', iou, on_epoch=True)
            pred_proba_aff = self.train_set.dataset.labels_obj_to_aff(x_hat, num_cls=self.num_cls, proba=True)
            pred_cls_aff = torch.argmax(pred_proba_aff, dim=1)
            target_aff = self.train_set.dataset.labels_obj_to_aff(y, num_cls=self.num_cls)
            # iou_aff = self.IoU_conv(pred_proba_aff, target_aff)
            # self.log(f'{set}_iou_aff', iou_aff, on_epoch=True)
            mistakes = self.dist(pred_proba_aff, target_aff, weight_map=weight_map)
        elif self.hparams.mode == "affordances":
            # self.log(f'{set}_iou_aff', iou, on_epoch=True)
            mistakes = self.dist(x_hat, y, weight_map=weight_map)
        elif self.hparams.mode == "objects":
            # self.log(f'{set}_iou_obj', iou, on_epoch=True)
            mistakes = self.dist(x_hat, y, weight_map=weight_map)

        self.log_mistakes(mistakes, prefix=set)
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
        loss = self.predict(batch, set="val")
        return loss

    def reduce_cm(self, cms, save=False):

        labels = self.train_set.dataset.cls_labels

        cms = torch.reshape(cms, (-1, self.num_cls, self.num_cls))
        cm = torch.sum(cms, dim=0, keepdim=False)

        iou_cls = iou_from_confmat(cm, num_classes=len(labels))
        logger.debug(f"CM - {cm}")
        logger.info(f"CM IoU - {100*iou_cls}")

        recall = np.diag(cm) / cm.sum(axis = 1)
        precision = np.diag(cm) / cm.sum(axis = 0)
        recall_overall = torch.mean(recall)
        precision_overall = torch.mean(precision)

        logger.info(f"precision {100*precision} ({100*precision_overall}) | recall {100*recall} ({100*recall_overall})")

        cm1 = cm / cm.sum(axis=1, keepdim=True)  # normalize confusion matrix
        cm2 = cm / cm.sum(axis=0, keepdim=True)  # normalize confusion matrix

        if save:
            if len(labels) > self.num_cls:
                labels.pop(0)
            confusionmatrix_file = f"{self.hparams.dataset}-{self.hparams.mode}-{self.test_checkpoint}"
            logger.info(f"Saving confusion matrix {confusionmatrix_file}")

            plot_confusion_matrix(cm1.numpy(), labels=labels, filename=confusionmatrix_file+"-1", folder=f"{self.result_folder}")
            plot_confusion_matrix(cm2.numpy(), labels=labels, filename=confusionmatrix_file+"-2", folder=f"{self.result_folder}")
        return 0

    def reduce_dist(self, dists):

        # logger.debug(dists)
        dist = torch.sum(dists, dim=0, keepdim=False) / dists.shape[0]
        #logger.debug(f"l1 distance {dist.item()}")

        return dist

    def reduce_acc_w(self, correct_w):
        # logger.debug(correct_w)
        acc = torch.sum(correct_w["correct_w"], dim=0, keepdim=False) / torch.sum(correct_w["samples_w"], dim=0, keepdim=False)
        return acc

    def log_mistakes(self, mistakes, prefix=""):
        underscore = '_' * int(len(prefix) > 0)
        for k, v in mistakes.items():
            if k == "correct_w":
                self.log(f"{prefix}{underscore}acc_w", {"correct_w": mistakes["correct_w"], "samples_w": mistakes["samples_w"]}, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_acc_w)
            elif "dist" in k:
                self.log(f"{prefix}{underscore}{k}", v, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_dist)
            elif k == "correct":
                self.log(f'{prefix}{underscore}acc', v, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_dist)

    def test_step(self, batch, batch_idx):
        sample, target_orig = batch
        folder = f"{segnet_model.result_folder}/{self.test_checkpoint}"

        if self.test_max is None or batch_idx < self.test_max:
            # logger.debug(torch.min(sample),torch.max(sample))
            pred_orig = self.model(sample)
            if self.hparams.loss_weight:
                weight_map = weight_from_target(target_orig)
            else:
                weight_map = None
            if self.hparams.loss == "compare": loss = self.compute_loss(pred_orig, target_orig, loss=self.hparams.loss, weight_map=weight_map)
            pred_orig = torch.softmax(pred_orig, dim=1)
            pred_cls_orig = torch.argmax(pred_orig, dim=1)

            if self.hparams.mode == "convert":
                # logger.debug(torch.unique(torch.argmax(pred_orig, dim=1)))
                pred = self.orig_dataset.dataset.labels_obj_to_aff(pred_orig, proba=True)
                for i, p in enumerate(pred_orig):
                    proba_lst = []
                    for cls, map in enumerate(p.squeeze()):
                        if self.hparams.orig_dataset == "freiburg" and cls==0: # ignore void
                            pass
                        else:
                            proba_lst.append(map)
                    # if self.save: self.orig_dataset.dataset.result_to_image(
                    #     iter=batch_idx+i,
                    #     proba_lst=proba_lst,
                    #     folder=folder,
                    #     filename_prefix=f"probas_orig-{self.test_checkpoint}",
                    #     dataset_name=self.hparams.dataset)
            else:
                pred = pred_orig
            pred_cls = torch.argmax(pred, dim=1)

            if len(target_orig) > 1:
                target_orig = target_orig.squeeze()
            if self.hparams.mode == "convert":
                target = self.test_set.dataset.labels_obj_to_aff(target_orig)
            else:
                target = target_orig

            # logger.debug("pred",pred_cls.shape,"target",target.shape)

            for i,(o,p,c,t) in enumerate(zip(sample,pred,pred_cls,target)):
                # logger.debug(p.shape)
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
                if self.save:
                    # self.orig_dataset.dataset.result_to_image(iter=batch_idx+i, pred_proba=test, folder=folder, filename_prefix=f"proba-{self.test_checkpoint}", dataset_name=self.hparams.dataset)
                    # logger.debug("Generating argmax pred")
                    self.orig_dataset.dataset.result_to_image(iter=batch_idx+i, pred_cls=c, folder=folder, filename_prefix=f"cls-{self.test_checkpoint}", dataset_name=self.hparams.dataset)
                    # self.test_set.dataset.result_to_image(iter=batch_idx+i, gt=t, orig=o, folder=folder, filename_prefix=f"ref-dual", dataset_name=self.hparams.dataset)
                    self.test_set.dataset.result_to_image(iter=batch_idx+i, orig=o, folder=f"{segnet_model.result_folder}/orig/", filename_prefix=f"orig", dataset_name=self.hparams.dataset)
                    self.test_set.dataset.result_to_image(iter=batch_idx+i, gt=t, folder=f"{segnet_model.result_folder}/gt/", filename_prefix=f"gt", dataset_name=self.hparams.dataset)
                    # self.test_set.dataset.result_to_image(
                    #     iter=batch_idx+i,
                    #     orig=o,
                    #     gt=t,
                    #     pred_cls=c,
                    #     pred_proba=test,
                    #     folder=folder,
                    #     filename_prefix=f"res", dataset_name=self.hparams.dataset)
                # self.test_set.dataset.result_to_image(iter=batch_idx+i, # pred_proba=p.squeeze()[self.test_set.dataset.aff_idx["impossible"]], folder=folder, filename_prefix=f"proba0")
                # self.test_set.dataset.result_to_image(
                #     iter=batch_idx+i, gt=t, orig=o,
                #     folder=f"{self.result_folder}/viz_per_epoch",
                #     filename_prefix=f"gt")

            #try:
            cm = self.CM(pred, target)
            # logger.debug(cm.shape)
            iou = self.IoU_conv(pred, target)

            mistakes = self.dist(pred, target, weight_map=weight_map)
            logger.debug(mistakes)
            self.log_mistakes(mistakes, prefix="test")


            self.log('test_iou', iou, on_step=False, prog_bar=False, on_epoch=True)
            self.log('cm', cm, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_cm)

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
        dataset = self.datasets[name](set=set, resize=self.hparams.resize, mode=self.hparams.mode, augment=augment, modalities=self.hparams.modalities)
        if set == "test" and self.test_max is not None:
            dataset = Subset(dataset, indices=range(self.test_max))
        else:
            dataset = Subset(dataset, indices=range(len(dataset)))
        return dataset

    def get_dataset_splits(self, normalize=False):
        if self.hparams.dataset == "freiburg":
            train_set = self.get_dataset(set="train")
            if self.full:
                test_set = self.get_dataset(set="full",augment=False)
            else:
                test_set = self.get_dataset(set="test",augment=False)
            val_set = self.get_dataset(set="test",augment=False)
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
            val_set = self.get_dataset(set="val", augment=False)
            if self.full:
                test_set = self.get_dataset(set="full", augment=False)
            else:
                test_set = self.get_dataset(set="test", augment=False)
            # total_len = len(train_set)
            # val_len = int(0.2*total_len)
            # train_len = total_len - val_len*2
            # train_set, _, _ = random_split(train_set, [train_len, val_len, val_len])
            # _, val_set, test_set = random_split(val_set, [train_len, val_len, val_len])
            # train_set, val_set, test_set = train_set.dataset, val_set.dataset, test_set.dataset

            # logger.debug(test_set[0])

        elif self.hparams.dataset == "thermalvoc":
            train_set = self.get_dataset(set="train")
            val_set = self.get_dataset(set="test", augment=False)
            test_set = val_set
            # logger.debug(test_set[0])

        elif self.hparams.dataset == "cityscapes":
            train_set = self.get_dataset(set="train")
            val_set = self.get_dataset(set="val")
            if self.full:
                test_set = self.get_dataset(set="full",augment=False)
            else:
                test_set = self.get_dataset(set="test",augment=False)


        elif self.hparams.dataset == "synthia":
            train_set = self.get_dataset(set="train")
            val_set = self.get_dataset(set="val")
            test_set = self.get_dataset(set="test",augment=False)

        if normalize:
            mean = 0.
            std = 0.
            loader = DataLoader(train_set, batch_size=self.hparams.bs, num_workers=self.hparams.workers, shuffle=False)
            for images, _ in loader:
                batch_samples = images.size(0)  # batch size
                # logger.debug(images.shape)
                images = images.view(batch_samples, images.size(1), -1)
                # logger.debug(images.shape)
                mean += images.mean(2).sum(0)
                std += images.std(2).sum(0)

            mean /= len(loader.dataset)
            std /= len(loader.dataset)
            logger.debug("Mean and stdev",mean,std)
            tf = transforms.Compose([
                transforms.Normalize(mean=(mean,), std=(std,))
            ])
            train_set.dataset.transform = tf
            test_set.dataset.transform = tf
            val_set.dataset.transform = tf

        logger.warning(f"{self.hparams.dataset} - train {len(train_set)} | val {len(val_set)} | test {len(test_set)}")
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

logger.debug(args)
segnet_model = LitSegNet(conf=args)

if args.prefix is None:
    args.prefix = segnet_model.hparams.save_prefix
logger.debug(args.prefix)

checkpoint_callback = ModelCheckpoint(
    dirpath='lightning_logs',
    filename=args.prefix+'-{epoch}-{val_loss:.4f}',
    verbose=True,
    monitor='val_loss',
    mode='min',
    save_last=True
)
checkpoint_callback.CHECKPOINT_NAME_LAST = f"{args.prefix}-last"

custom_callback = CustomCB()

lr_monitor = LearningRateMonitor(logging_interval='step')

callbacks = [lr_monitor]

if args.train:
    logger.warning("Training phase")
    wandb_logger = WandbLogger(project='segnet-freiburg', log_model = False, name = segnet_model.hparams.save_prefix)
    wandb_logger.log_hyperparams(segnet_model.hparams)
    #wandb_logger.watch(segnet_model, log='parameters', log_freq=100)


    segnet_model.update_model()
    if args.update_output_layer or args.init:
        segnet_model = segnet_model.load_from_checkpoint(checkpoint_path=args.train_checkpoint, conf=args)
        if args.update_output_layer: segnet_model.new_output()
        trainer = pl.Trainer.from_argparse_args(args,
            check_val_every_n_epoch=1,
            # ~ log_every_n_steps=10,
            logger=wandb_logger,
            checkpoint_callback=checkpoint_callback,
            callbacks=callbacks)
    else:
        trainer = pl.Trainer.from_argparse_args(args,
            check_val_every_n_epoch=1,
            # ~ log_every_n_steps=10,
            logger=wandb_logger,
            checkpoint_callback=checkpoint_callback,
            callbacks=callbacks,
            resume_from_checkpoint=args.train_checkpoint)
    trainer.fit(segnet_model)

else:
    logger.warning("Testing phase")
    trainer = pl.Trainer.from_argparse_args(args)
    chkpt = args.test_checkpoint.split("/")[-1].replace(".ckpt", "")
    create_folder(f"{segnet_model.result_folder}/{chkpt}")
    trained_model = segnet_model.load_from_checkpoint(checkpoint_path=args.test_checkpoint, test_max = args.test_samples, test_checkpoint=chkpt, save=args.save, viz=args.viz, full_dataset=args.full_dataset, conf=args)
    trained_model.update_model()
    if args.update_output_layer:
        segnet_model.new_output()
    trainer.test(trained_model)
