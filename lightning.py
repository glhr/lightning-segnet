import os
import numpy as np
import random

from metrics import MaskedIoU, ConfusionMatrix, Mistakes, iou_from_confmat, weight_from_target, RecallMetric

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback

from segnet import SegNet, new_input_channels, new_output_channels
from fusion import FusionNet
from losses import SORDLoss, KLLoss, CompareLosses
from dataloader import *
from plotting import plot_confusion_matrix, plot_scatter
from utils import create_folder, logger, enable_debug, RANDOM_SEED

from argparse import ArgumentParser
from datetime import datetime

import torchmetrics

from matplotlib import pyplot as plt

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
parser.add_argument('--test_checkpoint', default=None)
parser.add_argument('--train_checkpoint', default=None)
parser.add_argument('--prefix', default=None)
parser.add_argument('--debug', default=False, action="store_true")
parser.add_argument('--save', default=False, action="store_true")
parser.add_argument('--viz', default=False, action="store_true")
parser.add_argument('--test_set', default="test")
parser.add_argument('--update_output_layer', default=False, action="store_true")
parser.add_argument('--init', default=False, action="store_true")
parser.add_argument('--dataset_seq', default=None)
parser.add_argument('--nopredict', default=False, action="store_true")
# parser.add_argument('--accelerator', default="dp")

import inspect

class LitSegNet(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--bs', type=int, default=16)
        parser.add_argument('--lr', type=float, default=None)
        parser.add_argument('--momentum', type=int, default=None)
        parser.add_argument('--optim', type=str, default=None)
        parser.add_argument('--wd', type=float, default=0)
        parser.add_argument('--num_classes', type=int, default=3)
        parser.add_argument('--workers', type=int, default=0)
        parser.add_argument('--mode', default="affordances")
        parser.add_argument('--dataset', default="freiburg")
        parser.add_argument('--dataset_combo', default=None)
        parser.add_argument('--dataset_combo_ntrain', type=int, default=100)
        parser.add_argument('--augment', action="store_true", default=False)
        parser.add_argument('--loss_weight', action="store_true", default=False)
        parser.add_argument('--lwmap_range', default="0.1,1")
        parser.add_argument('--loss', default=None)
        parser.add_argument('--orig_dataset', default=None)
        parser.add_argument('--modalities', default="rgb")
        parser.add_argument('--init_channels', type=int, default=1)
        parser.add_argument('--depthwise_conv', action="store_true", default=False)
        parser.add_argument('--ranks', default="1,2,3")
        parser.add_argument('--dist', default="l1")
        parser.add_argument('--dist_alpha', type=int, default=1)
        parser.add_argument('--save_xp', default=None)
        parser.add_argument('--gt', default="driv")
        parser.add_argument('--noeval', default=False, action="store_true")
        return parser

    def __init__(self, conf, viz=False, save=False, test_set=None, test_checkpoint = None, test_max=None, model_only=False, num_classes = None, modalities=None, dataset_seq=None, nopredict=False, **kwargs):
        super().__init__()
        pl.seed_everything(RANDOM_SEED)
        self.save_hyperparameters(conf)

        if modalities is not None:
            self.hparams.modalities = modalities
        self.hparams.modalities = self.hparams.modalities.split(",")
        logger.warning(f"params {self.hparams}")

        init_channels = len(self.hparams.modalities) if self.hparams.init_channels is None else self.hparams.init_channels

        self.model = SegNet(
            num_classes=self.hparams.num_classes if num_classes is None else num_classes,
            n_init_features=init_channels,
            depthwise_conv=self.hparams.depthwise_conv
        )
        print(self.model)

        if not model_only:
            self.save = save
            logger.info(f"Save {self.save}")
            self.viz = viz
            self.hparams.resize = (480, 240)
            self.hparams.masking = True
            self.hparams.normalize = False
            self.hparams.lwmap_range = tuple([float(i) for i in self.hparams.lwmap_range.split(",")])
            self.test_checkpoint = test_checkpoint
            self.test_max = test_max
            self.test_set = test_set if test_set is not None else "test"
            self.dataset_seq = dataset_seq
            self.nopredict = nopredict


            self.datasets = {
                "freiburg": FreiburgDataLoader,
                "freiburgthermal": FreiburgThermalDataLoader,
                "cityscapes": CityscapesDataLoader,
                "kitti": KittiDataLoader,
                "own": OwnDataLoader,
                "thermalvoc": ThermalVOCDataLoader,
                "synthia": SynthiaDataLoader,
                "kaistped": KAISTPedestrianDataLoader,
                "kaistpedann": KAISTPedestrianAnnDataLoader,
                "multispectralseg": MIRMultispectral,
                "lostfound": LostFoundDataLoader,
                "freiburgraw": FreiburgForestRawDataLoader,
                "kittiraw": KittiRawDataLoader,
                "kittiobj": KittiObjectDataLoader,
                "cityscapesraw": CityscapesRawDataLoader,
                "rugd": RUGDDataLoader,
                "wilddash": WildDashDataLoader,
                "mapillary": MapillaryDataLoader,
                "tas500": TAS500DataLoader,
                "acdc": ACDCDataLoader,
                "idd": IDDDataLoader,
                "bdd": BDDDataLoader,
                "ycor": YCORDataLoader
            }


            if self.hparams.orig_dataset is None and self.hparams.mode in ["affordances", "objects"]:
                self.hparams.orig_dataset = self.hparams.dataset

            if not (self.hparams.dataset == "combo"):
                self.orig_dataset = self.get_dataset(name=self.hparams.orig_dataset, set=self.test_set)
            else:
                if self.hparams.dataset_combo is None:
                    self.hparams.dataset_combo = "cityscapes,mapillary,acdc,rugd,tas500,idd,bdd,ycor"

                self.hparams.dataset_combo = self.hparams.dataset_combo.split(",")

                self.orig_dataset = self.get_dataset_combo(set=self.test_set)


            self.mse = nn.MSELoss()
            self.test_acc, self.val_acc, self.train_acc = torchmetrics.Accuracy(), torchmetrics.Accuracy(), torchmetrics.Accuracy()
            self.test_recall, self.val_recall, self.train_recall = torchmetrics.Recall(num_classes=self.hparams.num_classes, average="none"), torchmetrics.Recall(num_classes=self.hparams.num_classes, average="none"), torchmetrics.Recall(num_classes=self.hparams.num_classes, average="none")
            self.test_precision, self.val_precision, self.train_precision = torchmetrics.Precision(num_classes=self.hparams.num_classes, average="none"), torchmetrics.Precision(num_classes=self.hparams.num_classes, average="none"), torchmetrics.Precision(num_classes=self.hparams.num_classes, average="none")
            self.test_mIoU, self.val_mIoU, self.train_mIoU = torchmetrics.IoU(num_classes=self.hparams.num_classes), torchmetrics.IoU(num_classes=self.hparams.num_classes), torchmetrics.IoU(num_classes=self.hparams.num_classes)
            self.test_cIoU, self.val_cIoU, self.train_cIoU = torchmetrics.IoU(num_classes=self.hparams.num_classes, reduction="none"), torchmetrics.IoU(num_classes=self.hparams.num_classes, reduction="none"), torchmetrics.IoU(num_classes=self.hparams.num_classes, reduction="none")

            self.test_recall_o = RecallMetric()
            self.precis = {
                "train": self.train_precision, "val": self.val_precision, "test": self.test_precision
            }
            self.recall = {
                "train": self.train_recall, "val": self.val_recall, "test": self.test_recall
            }
            self.accuracy = {
                "train": self.train_acc, "val": self.val_acc, "test": self.test_acc
            }
            self.mIoU = {
                "train": self.train_mIoU, "val": self.val_mIoU, "test": self.test_mIoU
            }
            self.cIoU = {
                "train": self.train_cIoU, "val": self.val_cIoU, "test": self.test_cIoU
            }

            if self.hparams.loss in ["sord","compare"]:
                self.hparams.ranks = [int(r) for r in self.hparams.ranks.split(",")]
            else:
                self.hparams.ranks = [1,2,3]

            self.train_set, self.val_set, self.test_set = self.get_dataset_splits(normalize=self.hparams.normalize)
            self.hparams.train_set, self.hparams.val_set, self.hparams.test_set = \
                len(self.train_set.dataset), len(self.val_set.dataset), len(self.test_set.dataset)

            self.test_samples = test_set
            self.update_settings()

    def update_settings(self):

        self.sord = SORDLoss(n_classes=self.hparams.num_classes, masking=self.hparams.masking, ranks=self.hparams.ranks, dist=self.hparams.dist, alpha = self.hparams.dist_alpha)
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
            self.hparams.save_prefix += f'-a{self.hparams.dist_alpha}-{self.hparams.dist}'
        if self.hparams.loss_weight:
            self.hparams.save_prefix += "-lw"
        self.hparams.save_prefix += f'-{",".join(self.hparams.modalities)}'
        logger.info(self.hparams.save_prefix)
        if self.hparams.save_xp is None:
            create_folder(f"{self.result_folder}/viz_per_epoch")
            create_folder(f"{self.result_folder}/gt")
            create_folder(f"{self.result_folder}/orig")

        if self.hparams.loss == "compare":
            create_folder(f"results/loss_weight/{self.hparams.dataset}")


    def update_model(self):
        channels = len(self.hparams.modalities) if self.hparams.init_channels is None else self.hparams.init_channels
        self.model = new_input_channels(self.model, channels)
        logger.warning(f"Model has {channels} input channels")

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

    def compute_loss(self, x_hat, y, loss="ce", weight_map=None, filename=None):
        if loss == "ce":
            return self.ce(x_hat, y)
        elif loss == "sord":
            return self.sord(x_hat, y, weight_map=weight_map)
        elif loss == "kl":
            return self.kl(x_hat, y, weight_map=weight_map)
        elif loss == "compare":
            return self.loss(x_hat, y, weight_map=weight_map, filename=filename)

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
        x, y = batch["sample"]

        if set == "test":
            with torch.no_grad():
                x_hat = self.model(x)
        else:
            x_hat = self.model(x)

        if self.hparams.loss_weight:
            weight_map = weight_from_target(y, lwmap_range=self.hparams.lwmap_range)
        else:
            weight_map = None

        loss = self.compute_loss(x_hat, y, loss=self.hparams.loss, weight_map=weight_map)
        print(x_hat, y)
        x_hat = torch.softmax(x_hat, dim=1)
        pred_cls = torch.argmax(x_hat, dim=1)
        # iou = self.IoU(x_hat, y)

        # if self.hparams.mode == "convert":
        #     # self.log(f'{set}_iou_obj', iou, on_epoch=True)
        #     pred_proba_aff = self.train_set.dataset.labels_obj_to_aff(x_hat, num_cls=self.num_cls, proba=True)
        #     pred_cls_aff = torch.argmax(pred_proba_aff, dim=1)
        #     target_aff = self.train_set.dataset.labels_obj_to_aff(y, num_cls=self.num_cls)
        #     # iou_aff = self.IoU_conv(pred_proba_aff, target_aff)
        #     # self.log(f'{set}_iou_aff', iou_aff, on_epoch=True)
        #     mistakes = self.dist(pred_proba_aff, target_aff, weight_map=weight_map)
        # elif self.hparams.mode == "affordances":
        #     # self.log(f'{set}_iou_aff', iou, on_epoch=True)
        #     mistakes = self.dist(x_hat, y, weight_map=weight_map)
        # elif self.hparams.mode == "objects":
        #     # self.log(f'{set}_iou_obj', iou, on_epoch=True)
        #     mistakes = self.dist(x_hat, y, weight_map=weight_map)
        # #
        # self.log_mistakes(mistakes, prefix=set)
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

        if self.hparams.dataset_combo is not None:
            labels = self.train_set.dataset.datasets[0].dataset.cls_labels
        else:
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

    def reduce_recall_w(self, recall):
        acc = torch.sum(recall["obstacle_recall"], dim=0, keepdim=False) / torch.sum(recall["samples_obstacle_recall"], dim=0, keepdim=False)
        return acc

    def reduce_precision_w(self, precision):
        acc = torch.sum(precision["path_precision"], dim=0, keepdim=False) / torch.sum(precision["samples_path_precision"], dim=0, keepdim=False)
        return acc

    def log_mistakes(self, mistakes, prefix=""):
        underscore = '_' * int(len(prefix) > 0)
        for k, v in mistakes.items():
            if k == "correct_w":
                if prefix != "train":
                    self.log(f"{prefix}{underscore}acc_w", {"correct_w": mistakes["correct_w"], "samples_w": mistakes["samples_w"]}, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_acc_w)
            elif k == "obstacle_recall":
                self.log(f"{prefix}{underscore}recall_w", {"obstacle_recall": mistakes["obstacle_recall"], "samples_obstacle_recall": mistakes["samples_obstacle_recall"]}, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_recall_w)
            elif k == "path_precision":
                self.log(f"{prefix}{underscore}precision_w", {"path_precision": mistakes["path_precision"], "samples_path_precision": mistakes["samples_path_precision"]}, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_precision_w)
            elif "dist" in k:
                self.log(f"{prefix}{underscore}{k}", v, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_dist)
            elif k == "correct":
                self.log(f'{prefix}{underscore}acc', v, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_dist)

    def reduce_stats(self, count):
        # print(count)
        count = torch.reshape(count, (-1, self.num_cls + 1))
        # print(count)
        count_total = torch.sum(count, dim=0, keepdim=True)
        count_total = count_total.float() / torch.sum(count_total)
        logger.info(f"GT STATS - class count: {count_total}")

        return 0

    def gt_stats(self, gt):

        # print(torch.unique(gt))
        count = torch.LongTensor([0] * (self.hparams.num_classes + 1))
        gt_values = torch.unique(gt)
        for val in gt_values:
            count[val+1] = torch.sum((gt == val))

        return count

    def input_corr(self, sample):
        n_mods = sample.shape[1]

        x = sample[:,0,::].detach().cpu().numpy().flat
        y = sample[:,1,::].detach().cpu().numpy().flat

        plot_scatter(x,y)

        # np.random.shuffle(x)

        print(np.corrcoef(y,x)[0, 1])

    def test_step(self, batch, batch_idx):
        # return self.validation_step(batch, batch_idx)

        dataset_obj = self.test_set.dataset if self.hparams.dataset_combo is None else self.test_set.dataset.datasets[0].dataset

        orig_dataset_obj = self.orig_dataset.dataset if self.hparams.dataset_combo is None else self.orig_dataset.dataset.datasets[0].dataset

        sample, target_orig = batch["sample"]

        if self.hparams.save_xp is None:
            result_folder = f"{self.result_folder}/{self.test_checkpoint}"
            gt_folder = f"{self.result_folder}/gt/"
            orig_folder = f"{self.result_folder}/orig/"
        else:
            result_folder = f"{self.result_folder}/{self.hparams.save_xp}"
            gt_folder = result_folder
            orig_folder = result_folder
            create_folder(result_folder)

        if self.test_max is None or batch_idx < self.test_max:
        # print(batch["filename"])
        # if "us0279" in batch["filename"][0]:
            # logger.info("Saving shit")
            if not self.nopredict:
                pred_orig = self.model(sample)
                if self.hparams.loss_weight:
                    weight_map = weight_from_target(target_orig, lwmap_range=self.hparams.lwmap_range)
                else:
                    weight_map = None
                filename = f'{self.hparams.dataset}/{self.hparams.dataset} - {batch["filename"][0]} - {self.test_checkpoint} - lossname'
                if self.hparams.loss == "compare": loss = self.compute_loss(pred_orig, target_orig, loss=self.hparams.loss, weight_map=weight_map, filename=filename)
                pred_orig = torch.softmax(pred_orig, dim=1)
                pred_cls_orig = torch.argmax(pred_orig, dim=1)

                if self.hparams.mode == "convert":
                    # logger.debug(torch.unique(torch.argmax(pred_orig, dim=1)))
                    pred = orig_dataset_obj.labels_obj_to_aff(pred_orig, proba=True)
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

            if self.hparams.nopredict:
                pred = target
                pred_cls = target

            # logger.debug("pred",pred_cls.shape,"target",target.shape)

            for i,(o,p,c,t) in enumerate(zip(sample,pred,pred_cls,target)):
            #     # logger.debug(p.shape)
                if not self.hparams.dataset == "combo":
                    proba_imposs = p.squeeze()[orig_dataset_obj.aff_idx["impossible"]]
                    proba_poss = p.squeeze()[orig_dataset_obj.aff_idx["possible"]]
                    proba_pref = p.squeeze()[orig_dataset_obj.aff_idx["preferable"]]
                    # expected = proba_imposs

                    expected = 1*proba_imposs + 2*proba_poss + 3*proba_pref
                    expected = expected - torch.min(expected)
                    expected = expected/torch.max(expected)
                    # print(torch.min(expected),torch.max(expected))
                    # expected = 1 - expected

                iter = batch_idx*self.hparams.bs + i

                filename = batch["filename"][i]

                # if not self.nopredict:
                #     for cls,map in enumerate(p.squeeze()):
                #         proba_lst = []
                #         proba_lst.append(map)
                            # self.orig_dataset.dataset.result_to_image(
                            #     iter=batch_idx+i,
                            #     proba_lst=proba_lst,
                            #     folder=folder,
                            #     filename_prefix=f"probas{cls}-{self.test_checkpoint}",
                            #     dataset_name=self.hparams.dataset)
                # logger.debug("Generating proba map")
                if self.save:
                    # logger.info("Saving")
                    # self.orig_dataset.dataset.result_to_image(iter=batch_idx+i, pred_proba=test, folder=folder, filename_prefix=f"proba-{self.test_checkpoint}", dataset_name=self.hparams.dataset)
                    # logger.debug("Generating argmax pred")
                    mod = ','.join(self.hparams.modalities)
                    # orig_dataset_obj.result_to_image(iter=batch_idx+i, pred_cls=c, folder=result_folder, filename_prefix=f"cls-{self.test_checkpoint}", dataset_name=self.hparams.dataset, filename = filename)
                    # self.test_set.dataset.result_to_image(iter=batch_idx+i, gt=t, orig=o, folder=folder, filename_prefix=f"ref-dual", dataset_name=self.hparams.dataset)
                    dataset_obj.result_to_image(iter=batch_idx+i, orig=o, folder=orig_folder, filename_prefix=f"orig-", dataset_name=self.hparams.dataset, modalities = self.hparams.modalities, filename = filename)
                    if not self.nopredict and self.test_checkpoint is not None:
                        dataset_obj.result_to_image(iter=batch_idx+i, overlay=c, orig=o, folder=gt_folder, filename_prefix=f"overlay-pred-{self.test_checkpoint}", dataset_name=self.hparams.dataset, filename = filename)
                        dataset_obj.result_to_image(iter=batch_idx+i, gt=c, folder=gt_folder, filename_prefix=f"pred-{self.test_checkpoint}", dataset_name=self.hparams.dataset, filename = filename)
                    if not dataset_obj.noGT:
                        # dataset_obj.result_to_image(iter=batch_idx+i, gt=t, folder=gt_folder, filename_prefix=f"gt", dataset_name=self.hparams.dataset, filename = filename)
                        dataset_obj.result_to_image(iter=batch_idx+i, overlay=t, orig=o, folder=gt_folder, filename_prefix=f"overlay-gt-{self.hparams.gt}-", dataset_name=self.hparams.dataset, filename = filename)

                    if not self.nopredict:
                        # error_map = t - c
                        # error_map[t == -1] = 0
                        # #error_map_w = 2 - error_map
                        # #dataset_obj.result_to_image(iter=batch_idx+i, pred_proba=error_map, folder=result_folder + "/error", filename_prefix=f"errorb-{self.test_checkpoint}", dataset_name=self.hparams.dataset, filename = filename)
                        # dataset_obj.result_to_image(iter=batch_idx+i, pred_proba=error_map, folder=result_folder + "/error", filename_prefix=f"errorw-{self.test_checkpoint}", dataset_name=self.hparams.dataset, filename = filename, colorize=True)

                        if not self.hparams.dataset == "combo":
                            dataset_obj.result_to_image(iter=batch_idx+i, pred_proba=expected, folder=result_folder + "/exp", filename_prefix=f"exp-{self.test_checkpoint}", dataset_name=self.hparams.dataset, filename = filename, colorize=False)
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

            if not dataset_obj.noGT and not self.nopredict and not self.hparams.noeval:
                #try:
                # pass
                # cm = self.CM(pred, target)
                # logger.debug(cm.shape)
                # iou = self.IoU_conv(pred, target)

                mistakes = self.dist(pred, target, weight_map=weight_map)
                self.log_mistakes(mistakes, prefix="test")

                mistakes = dict()
                #print(target)
                target_cls = target[target>=0]
                pred_cls = torch.argmax(pred, dim=1)[target>=0]
                pred_prob = torch.max(pred, dim=1)[0][target >= 0]
                mistakes["mse"] = self.mse(pred_cls.float(),target_cls.float())
                # logger.debug(mistakes)

                # self.log(f'test_acc', v, on_step=False, prog_bar=False, on_epoch=True)

                set = "test"
                #inp = pred.contiguous()
                inp = pred_prob.view(-1, )

                t = target_cls.view(-1, )
                c = pred_cls.view(-1, )
                correct_pred = inp[c == t].detach().cpu().numpy()
                incorrect_pred = inp[c != t].detach().cpu().numpy()
                #print(correct_pred, incorrect_pred)
                bins = np.arange(0, 1, 0.05)  # fixed bin size
                counts_c, bins_c = np.histogram(correct_pred, bins=bins)
                counts_i, bins_i = np.histogram(incorrect_pred, bins=bins)




                # torch.set_deterministic(False)
                acc = self.accuracy[set](pred_cls, target_cls)
                # miou = self.mIoU[set](pred_cls, target_cls)
                # ciou = self.cIoU[set](pred_cls, target_cls)
                # recall = self.recall[set](pred_cls, target_cls)
                # precision = self.precis[set](pred_cls, target_cls)
                #
                # self.log(f'{set}_mse', mistakes["mse"], on_epoch=True)
                self.log(f'{set}_accuracy', acc, on_epoch=True)
                #
                # self.log(f'{set}_mIoU', miou, on_epoch=True)
                # self.log(f'{set}_recall_r', recall[0], on_epoch=True)
                # self.log(f'{set}_precision_g', precision[2], on_epoch=True)
                # self.log(f'{set}_cIoU_1', ciou[0], on_epoch=True)
                # self.log(f'{set}_cIoU_2', ciou[1], on_epoch=True)
                # self.log(f'{set}_cIoU_3', ciou[2], on_epoch=True)
                # torch.set_deterministic(True)



                #self.log('test_iou', iou, on_step=False, prog_bar=False, on_epoch=True)
                #self.log('cm', cm, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_cm)

            return {
                "correct_hist": counts_c,
                "incorrect_hist": counts_i,
                "bins": bins_c
            }

        # else:
        #     count = self.gt_stats(target_orig)
        #     # corr = self.input_corr(sample)
        #     self.log('gt_cls_count', count, on_step=False, prog_bar=False, on_epoch=True, reduce_fx=self.reduce_stats)


    def test_epoch_end(self, outputs):
        bins = outputs[0]["bins"]  # fixed bin size
        #print(len(outputs[0]["correct_hist"]))
        correct_pred = np.array([item["correct_hist"] for item in outputs if item is not None])
        correct_pred = np.sum(correct_pred, axis=0)

        incorrect_pred = np.array([item["incorrect_hist"] for item in outputs if item is not None])
        incorrect_pred = np.sum(incorrect_pred, axis=0)
        #print(correct_pred)
        #correct_pred =
        #incorrect_pred = outputs["incorrect_hist"]

        #print(bins, correct_pred)
        np.save(f"acdc-night-correct_pred.npy", correct_pred)
        np.save(f"acdc-night-incorrect_pred.npy", incorrect_pred)
        np.save(f"acdc-night-bins.npy", bins)
        counts_i, bins_i, _ = plt.hist(bins[:-1], bins, weights = correct_pred, alpha=0.5, color = "green")

        counts_i, bins_i, _ = plt.hist(bins[:-1], bins, weights = incorrect_pred, alpha=0.5, color = "red")
        plt.show()

    def configure_optimizers(self):
        if self.hparams.optim == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def get_dataset(self, set, name=None, augment=None):
        logger.info(f"Loading {set} set")
        if name is None:
            name = self.hparams.dataset
        if augment is None:
            augment = self.hparams.augment if set == "train" else False
        logger.info(self.hparams.modalities)
        logger.info(f"{set} augment: {augment}")
        logger.info(self.dataset_seq)
        logger.info(self.hparams.gt)
        dataset = self.datasets[name](set=set, resize=self.hparams.resize, mode=self.hparams.mode, augment=augment, modalities=self.hparams.modalities, viz=(self.viz and set == "train"), dataset_seq=self.dataset_seq, rgb=(self.hparams.init_channels > 1), gt=self.hparams.gt)
        if set == "test" and self.test_max is not None:
            dataset = Subset(dataset, indices=range(self.test_max))
        else:
            dataset = Subset(dataset, indices=range(len(dataset)))
        return dataset

    def get_dataset_combo(self, set, augment=None):
        subsets = []
        if augment is None:
            augment = self.hparams.augment if set == "train" else False

        n_samples = self.hparams.dataset_combo_ntrain if set == "train" else int(self.hparams.dataset_combo_ntrain/9)
        total_length = 0

        for name in self.hparams.dataset_combo:
            # print(name, set)
            dataset = self.datasets[name](set=set, resize=self.hparams.resize, mode=self.hparams.mode, augment=augment, modalities=self.hparams.modalities, viz=(self.viz and set == "train"), dataset_seq=self.dataset_seq, sort=False, rgb=(self.hparams.init_channels > 1))
            random_indices = np.random.choice(range(len(dataset)), replace=True, size=n_samples)
            subsets.append(Subset(dataset, indices=range(len(dataset))))
            total_length += len(dataset)

        combo = ConcatDataset(subsets)
        out = Subset(combo, indices=range(total_length))
        #print(dir(out.dataset),out.dataset.datasets[0].dataset)

        return out

    def get_dataset_splits(self, normalize=False):
        dataset_func = self.get_dataset_combo if self.hparams.dataset == "combo" else self.get_dataset
        train_set = dataset_func(set="train")
        if self.test_set is not None:
            test_set = dataset_func(set=self.test_set, augment=self.hparams.augment)
        else:
            test_set = dataset_func(set="test", augment=self.hparams.augment)
        val_set = dataset_func(set="val",augment=False)

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
        return DataLoader(self.test_set, batch_size=self.hparams.bs, num_workers=self.hparams.workers, shuffle=(self.test_samples in ["train"]))


if __name__ == '__main__':
    parser = LitSegNet.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.debug: enable_debug()

    logger.debug(args)
    segnet_model = LitSegNet(conf=args, viz=args.viz, dataset_seq=args.dataset_seq, save=args.save, save_xp=args.save_xp, test_set=args.test_set, nopredict=args.nopredict, test_max = args.test_samples)

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

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [lr_monitor]

    if args.train:
        logger.warning("Training phase")
        wandb_logger = WandbLogger(project='segnet-freiburg', log_model = False, name = segnet_model.hparams.save_prefix)
        wandb_logger.log_hyperparams(segnet_model.hparams)
        #wandb_logger.watch(segnet_model, log='parameters', log_freq=100)



        if args.update_output_layer or args.init:
            segnet_model = segnet_model.load_from_checkpoint(checkpoint_path=args.train_checkpoint, conf=args)
            segnet_model.update_model()
            if args.update_output_layer: segnet_model.new_output()
            trainer = pl.Trainer.from_argparse_args(args,
                check_val_every_n_epoch=1,
                # ~ log_every_n_steps=10,
                logger=wandb_logger,
                callbacks=callbacks + [checkpoint_callback],
                accelerator= "dp"
            )
        else:
            segnet_model.update_model()
            trainer = pl.Trainer.from_argparse_args(args,
                check_val_every_n_epoch=1,
                # ~ log_every_n_steps=10,
                logger=wandb_logger,
                callbacks=callbacks + [checkpoint_callback],
                resume_from_checkpoint=args.train_checkpoint,
                accelerator= "dp")
        trainer.fit(segnet_model)

    else:
        logger.warning("Testing phase")
        trainer = pl.Trainer.from_argparse_args(args,
        move_metrics_to_cpu=True)
        if args.test_checkpoint is not None:
            chkpt = args.test_checkpoint.split("/")[-1].replace(".ckpt", "")
            if args.save_xp is None:
                create_folder(f"{segnet_model.result_folder}/{chkpt}")
            trained_model = segnet_model.load_from_checkpoint(checkpoint_path=args.test_checkpoint, test_max = args.test_samples, test_checkpoint=chkpt, save=args.save, viz=args.viz, test_set=args.test_set, conf=args, dataset_seq=args.dataset_seq, nopredict=args.nopredict)
        else:
            trained_model = segnet_model
        if args.update_output_layer:
            trained_model.new_output()

        trained_model.eval()
        trainer.test(trained_model)
