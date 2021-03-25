import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logger, enable_debug

import matplotlib.pyplot as plt


def flatten_tensors(inp, target):
    # ~ print(inp.shape, target.shape)
    # ~ print(inp, target)
    target = target.long()
    num_classes = inp.size()[1]

    i0 = 1
    i1 = 2

    while i1 < len(inp.shape):  # this is ugly but torch only allows to transpose two axes at once
        inp = inp.transpose(i0, i1)
        i0 += 1
        i1 += 1

    inp = inp.contiguous()
    inp = inp.view(-1, num_classes)

    target = target.view(-1,)
    # ~ print(inp.shape, target.shape)
    # ~ print(inp, target)
    return inp, target


def viz_loss(target, output, losses, bs, nclasses):

    fig, axes = plt.subplots(ncols=nclasses+2, nrows=bs, sharex=True, sharey=True,
                             figsize=(6, 3))
    for i, (loss_name, loss) in enumerate(losses.items()):
        loss_reshaped = torch.reshape(loss,(bs,240,480,-1))
        logger.debug(f"target {target.shape} | loss {loss.shape} | reshaped {loss_reshaped.shape}")

        batch = 0
        loss_viz = torch.sum(loss_reshaped[batch].squeeze(), axis=-1).numpy()
        axes[i][0].imshow(target[batch], cmap=plt.cm.gray)
        axes[i][0].axis('off')
        for cls in range(0, nclasses):
            axes[i][cls+1].imshow(output[batch][cls], cmap=plt.cm.gray, vmin=0, vmax=1)
            axes[i][cls+1].axis('off')
        im = axes[i][nclasses+1].imshow(loss_viz, cmap=plt.cm.gray)
        fig.colorbar(im, ax=axes[i][nclasses+1])
        axes[i][nclasses+1].axis('off')
        axes[i][nclasses+1].set_title(loss_name)
        print("unique loss values",np.unique(loss_viz))

    # for r in axes:
    #     for c in r:
    #         axes[r][c].axis('off')


    plt.axis('off')
    plt.tight_layout()
    plt.show()


class CompareLosses(nn.Module):
    def __init__(self, n_classes, masking, ranks, returnloss):
        super().__init__()
        self.num_classes = n_classes
        self.masking = masking
        self.ranks = ranks
        self.kl = KLLoss(n_classes=n_classes, masking=masking)
        self.sord = SORDLoss(n_classes=n_classes, masking=masking, ranks=ranks)
        self.returnloss = returnloss

    def forward(self, output, target, debug=True, viz=True):
        target = torch.fliplr(target)
        for i in range(target.shape[0]):
            for cls in range(0, self.num_classes):
                output[i][cls] = 0.0
            output[i][2] = 1.0

        losses = {
            "kl": self.kl(output_orig=output, target_orig=target, debug=debug, viz=False, reduce=False),
            "sord": self.sord(output_orig=output, target_orig=target, debug=debug, viz=False, reduce=False),
        }
        viz_loss(target, output, losses, bs=target.shape[0], nclasses=self.num_classes)
        return losses[self.returnloss]


class KLLoss(nn.Module):
    def __init__(self, n_classes, masking=False):
        super().__init__()
        self.num_classes = n_classes
        self.masking = masking

    def forward(self, output_orig, target_orig, debug=False, viz=True, reduce=True):
        bs = target_orig.shape[0]
        output, target = flatten_tensors(output_orig, target_orig)
        if debug: print(output,target)

        if self.masking:
            mask = target.ge(0)
            # print(mask, mask.shape)
            # print(output.shape,target.shape)
            output = output[mask]
            target = target[mask]

        n_samples = target.shape[0]
        logger.debug(f"KLLoss n_samples {n_samples}")
        if debug: print(output,target)
        target = F.one_hot(target, num_classes=self.num_classes).float()
        if debug: print(output,target)
        output = torch.nn.LogSoftmax(dim=-1)(output)
        loss = nn.KLDivLoss(reduction='none')(output, target)
        if viz:
            viz_loss(target_orig, output_orig, loss, bs, self.num_classes, title="KLLoss")

        if reduce:
            loss = torch.sum(loss)/n_samples
        return loss





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

    def forward(self, output_orig, target_orig, debug=False, mod_input=None, viz=True, reduce=True):

        bs = target_orig.shape[0]
        target = torch.clone(target_orig)
        #if debug: print("target_orig",target_orig,torch.unique(target_orig))
        for i,r in enumerate(self.ranks):
            target[target_orig==i] = r
            if debug: print(f"{i} to {r}")
        logger.debug(f"SORD - before flatten: target shape {target_orig.shape} | output shape {output_orig.shape}")
        output, target = flatten_tensors(output_orig, target)
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
        logger.debug(f"SORDLoss n_samples {n_samples}")

        if debug: print("output",output)
        ranks = torch.tensor(self.ranks, dtype=output.dtype, device=output.device, requires_grad=False).repeat(output.size(0), 1)
        if debug: print("ranks",ranks)
        target = target.unsqueeze(1).repeat(1, self.num_classes)
        if debug: print("target",target,torch.unique(target))
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
        if reduce:
            loss = torch.sum(loss)/n_samples
        return loss


if __name__ == '__main__':

    from metrics import MaskedIoU

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
