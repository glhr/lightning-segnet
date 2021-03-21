import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import jaccard_score

from utils import logger, enable_debug


def flatten_tensors(inp, target):
        # ~ print(inp.shape, target.shape)
        # ~ print(inp, target)
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)
        # ~ print(inp.shape, target.shape)
        # ~ print(inp, target)
        return inp, target

class KLLoss(nn.Module):
    def __init__(self, n_classes, masking=False):
        super().__init__()
        self.num_classes = n_classes
        self.masking = masking

    def forward(self, output, target, debug=False):
        output, target = flatten_tensors(output, target)
        if debug: print(output,target)

        if self.masking:
            mask = target.ge(0)
            # print(mask, mask.shape)
            # print(output.shape,target.shape)
            output = output[mask]
            target = target[mask]

        n_samples = target.shape[0]
        if debug: print(output,target)
        target = F.one_hot(target, num_classes=self.num_classes).float()
        if debug: print(output,target)
        output = torch.nn.LogSoftmax(dim=-1)(output)
        loss = nn.KLDivLoss(reduction='none')(output, target)
        loss = torch.sum(loss)/n_samples
        return loss

class MaskedIoU(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = list(labels)
        logger.info(f"IoU labels: {self.labels}")

    def forward(self, output, target, debug=False):

        output, target = flatten_tensors(output, target)
        output = torch.argmax(output, dim=-1)

        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        iou_micro = jaccard_score(target, output, labels=self.labels, average='micro', zero_division=0)

        if debug:
            iou_macro = jaccard_score(target, output, labels=self.labels, average='macro', zero_division=0)
            iou_cls = jaccard_score(target, output, labels=self.labels, average=None, zero_division=0)
            logger.debug(f"MaskedIoU inputs: target {target}, pred {output}")
            logger.debug(f"MaskedIoU micro {iou_micro} | macro {iou_macro}")
            logger.debug(f"MaskedIoU per class {iou_cls}")
        else:
            logger.debug(f"MaskedIoU micro {iou_micro}")

        return iou_micro



class SORDLoss(nn.Module):
    """
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf
    """

    def __init__(self, n_classes, ranks=None, masking=False):
        super().__init__()
        self.num_classes = n_classes
        if ranks is not None and len(ranks) == self.num_classes:
            self.ranks = ranks
        else:
            self.ranks = np.arange(0, self.num_classes)
        self.masking = masking
        logger.info(f"SORD ranks: {self.ranks}")

    def forward(self, output, target, debug=False, mod_input=None):

        logger.debug(f"SORD - before flatten: target shape {target.shape} | output shape {output.shape}")
        output, target = flatten_tensors(output, target)
        logger.debug(f"SORD - after flatten: target shape {target.shape} | output shape {output.shape}")

        if self.masking:
            logger.debug(f"SORD - before masking: target shape {target.shape} | output shape {output.shape}")
            mask = target.ge(0)
            logger.debug(f"SORD - mask shape: {mask.shape}")
            # print(mask, mask.shape)
            # print(output.shape,target.shape)
            output = output[mask]
            target = target[mask]
            logger.debug(f"SORD - after masking: target shape {target.shape} | output shape {output.shape}")

        n_samples = target.shape[0]

        if debug: print("output",output)
        ranks = torch.tensor(self.ranks, dtype=output.dtype, device=output.device, requires_grad=False).repeat(output.size(0), 1)
        if debug: print("ranks",ranks)
        target = target.unsqueeze(1).repeat(1, self.num_classes)
        if debug: print("target",target)
        soft_target = -nn.L1Loss(reduction='none')(target, ranks)  # should be of size N x num_classes
        if debug: print("l1 target",soft_target)
        soft_target = torch.softmax(soft_target, dim=-1)
        if debug: print("soft target",soft_target)
        # output = torch.log(soft_target)
        # flatten label and prediction tensors

        if mod_input is not None:
            output = mod_input.long().view(-1,).unsqueeze(1).repeat(1, self.num_classes)
            output = -nn.L1Loss(reduction='none')(output, ranks)  # should be of size N x num_classes
            if debug: print("output",output)
            output = torch.softmax(output, dim=-1)
            if debug: print("output",output)
            output = torch.log(output)
        else:
            output = torch.nn.LogSoftmax(dim=-1)(output)

        loss = nn.KLDivLoss(reduction='none')(output, soft_target)
        #print(n_samples)
        loss = torch.sum(loss)/n_samples
        return loss


if __name__ == '__main__':

    input = torch.tensor([[ [[0.0]], [[1.0]],  [[0.0]]],[ [[0.0]], [[1.0]],  [[0.0]]]], requires_grad=True)
    target = torch.tensor([[[1]],[[1]]])
    # ~ output = ce(input, target)
    # ~ print(input,target,output)
    # ~ output.backward()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default="pref")
    parser.add_argument('--gt', default="pref")
    parser.add_argument('--debug', default=True, action="store_true")
    args = parser.parse_args()
    print(args)

    if args.debug: enable_debug()

    onehot = {
        "pref": [0.0, 0.0, 1.0],
        "poss": [0.0, 1.0, 0.0],
        "imposs": [1.0, 0.0, 0.0]
    }
    level = {
        "pref": 2,
        "poss": 1,
        "imposs": 0,
        "void": -1
    }

    input = torch.tensor([onehot[args.pred],onehot[args.pred],onehot[args.pred],onehot[args.pred]], requires_grad=True)
    target = torch.tensor([level[args.gt],level[args.gt],level[args.gt],level[args.gt]], dtype=torch.long)
    # ~ print(target, input, "CE ->", output)
    # ~ input = torch.randn(1, 3, requires_grad=True)
    # ~ target = torch.empty(1, dtype=torch.long).random_(3)

    # output = ce(input, target)
    # output.backward()
    # print(target, input, "CE ->", output)

    # ~ input, target = flatten_tensors(input, target)
    # ~ input = torch.nn.LogSoftmax(dim=-1)(input)
    cm = np.zeros((3, 3))
    sord = SORDLoss(n_classes = 3, ranks=[level["imposs"],level["poss"],level["pref"]], masking=True)
    print("SORD",sord(input, target))

    # for p,pred in enumerate(level.keys()):
    #     for g,gt in enumerate(level.keys()):
    #         input = torch.tensor([onehot[pred]], requires_grad=True)
    #         target = torch.tensor([level[gt]], dtype=torch.long)
    #         mod_input = torch.tensor([level[pred]], dtype=torch.long)
    #         loss = sord(input, target, debug=True, mod_input=mod_input)
    #         print("SORD ->", loss)
    #         cm[g][p] = loss.item()
    # print(cm)
    #
    # rankings = "|"+"|".join([str(l) for l in level.values()])+"|"
    #
    # from plotting import plot_confusion_matrix
    # plot_confusion_matrix(cm, labels=["impossible","possible","preferable"], filename=f"sordloss-{rankings}", folder="results/sordloss", vmax=None, cmap="Blues", cbar=True, annot=False, vmin=0)
    #
    # level = {
    #     "pref": 2,
    #     "poss": 1,
    #     "imposs": 0
    # }
    # rankings = "|"+"|".join([str(l) for l in level.values()])+"|"
    #
    # cm = np.zeros((3, 3))
    # ce = nn.CrossEntropyLoss(ignore_index = -1)
    # for p,pred in enumerate(level.keys()):
    #     for g,gt in enumerate(level.keys()):
    #         input = torch.tensor([onehot[pred]], requires_grad=True)
    #         target = torch.log_softmax(torch.tensor([onehot[gt]]),dim=-1)
    #         input = torch.log_softmax(input, dim=-1)
    #         print(input)
    #         loss = nn.KLDivLoss(reduction='mean',log_target=True)(input, target)
    #         print("CE ->", loss)
    #         cm[g][p] = loss.item()
    # print(cm)
    # plot_confusion_matrix(cm, labels=["impossible","possible","preferable"], filename=f"celoss-{rankings}", folder="results/sordloss", vmax=None, cmap="Blues", cbar=True, annot=False, vmin=0)

    kl = KLLoss(n_classes = 3, masking=True)
    loss = kl(input, target)
    print("KL",loss)

    iou = MaskedIoU(labels=[0,1,2])
    print(iou(input,target))
