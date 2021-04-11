import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logger, enable_debug, color_ramp
import metrics

import matplotlib.pyplot as plt

import scipy

def test_loss(alpha, dist, debug=False):
    logger.warning(f"dist {dist}, alpha {alpha}")
    n_cls = 3
    input = torch.as_tensor([n_cls-1])
    input = F.one_hot(input, num_classes=n_cls).float()
    # input = input.expand(2, 1, n_cls)
    logger.debug(input)
    ranks = [1,2,3]
    sord = SORDLoss(n_classes = n_cls, ranks=ranks, masking=True, dist=dist, alpha = alpha)
    kl = KLLoss(n_classes = n_cls, masking=True)

    # target = torch.tensor([0])
    # target = target.expand(2, 1, 1)
    input = torch.tensor([[ [[0.0]], [[1.0]],  [[0.0]]],[ [[0.0]], [[1.0]],  [[0.0]]], [ [[0.0]], [[1.0]],  [[0.0]]]], requires_grad=True)
    target = torch.tensor([[[0]],[[1]],[[2]]])
    # logger.debug(target)
    # logger.debug("KL", kl(input, target, debug=True)/n_cls)
    soft,loss = sord(input, target, debug=debug, reduce=False)
    loss = loss/n_cls

    soft = soft.detach().cpu().numpy()
    entropy = scipy.stats.entropy(soft, axis=-1)
    logger.info(f"soft target {soft}")
    logger.info(f"entropy {entropy}")

def flatten_tensors(inp, target, weight_map=None):
    # ~ logger.debug(inp.shape, target.shape)
    # ~ logger.debug(inp, target)
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
    # ~ logger.debug(inp.shape, target.shape)
    # ~ logger.debug(inp, target)
    if weight_map is not None:
        # logger.debug(type(target), type(weight_map))
        weight_map = weight_map.view(-1,)
        return inp, target, weight_map
    else:
        return inp, target


def expected_value(p, ranks=[0,1,2]):
    p = torch.transpose(p, 0, 1)
    proba_imposs = p[ranks[0]]
    proba_poss = p[ranks[1]]
    proba_pref = p[ranks[2]]
    return proba_imposs * 0 + proba_poss * 1 + proba_pref * 2

idx = 0
def viz_loss(output, losses, bs, nclasses, target=None, use_w = None, weight_map=None, show={"gt","argmax","loss"}):
    global idx
    idx += 1
    single_row = len(losses) == 1
    single_col = len(show) == 1

    fig, axes = plt.subplots(ncols=len(show), nrows=len(losses), sharex=True, sharey=True,
                             figsize=(5.33*len(show), 2.7*len(losses)))

    cols = ["gt","argmax","loss","weight_map"]

    if use_w is None:
        use_w = {k:True for k,v in losses.items()}


    for i, (loss_name, (loss_target,loss)) in enumerate(losses.items()):

        if single_row: ax = axes
        else: ax = axes[i]

        batch = 0

        col = 0
        for _, elem_name in enumerate(cols):

            if elem_name in show:
                if single_col: ax = ax
                else: ax = ax[col]

            if elem_name == "gt" and elem_name in show:
                # logger.debug(target.shape)
                if target is None:
                    target = expected_value(loss_target)
                # logger.debug(target.shape)
                target = torch.reshape(target, (bs,240,480,-1))
                # logger.debug(torch.unique(target[batch]))
                ax.imshow(target[batch], cmap=color_ramp, vmin=-1, vmax=2, interpolation='none')
                ax.axis('off')
                # for cls in range(0, nclasses):
                #     axes[i][cls+1].imshow(output[batch][cls], cmap=plt.cm.gray, vmin=0, vmax=1)
                #     axes[i][cls+1].axis('off')

            elif elem_name == "argmax" and elem_name in show:
                pred = output[batch][0]*0 + output[batch][1]*1 + output[batch][2]*2
                pred = torch.argmax(output[batch], dim = 0)
                ax.imshow(pred, cmap=color_ramp, vmin=0, vmax=2)
                ax.axis('off')

            elif elem_name == "loss" and elem_name in show:
                loss_reshaped = torch.reshape(loss,(bs,240,480,-1))

                target = torch.reshape(target, (bs,240,480,-1))
                if weight_map is not None: weight_map = torch.reshape(weight_map, (bs,240,480,-1))
                mask = target[batch].ge(0)

                if weight_map is None or not use_w[loss_name]:
                    nsamples = torch.sum(torch.ones_like(target[batch][mask]),axis=-1)
                    logger.debug(f"unweighted - nsamples {nsamples}")
                    loss_viz = torch.sum(loss_reshaped[batch].squeeze(), axis=-1).numpy()
                    logger.debug(f"loss viz MAX {np.max(loss_viz)}")
                    loss_viz = loss_viz/nsamples
                else:
                    logger.debug(weight_map.shape)
                    nsamples = torch.sum(weight_map[batch][mask],axis=-1)
                    logger.debug(f"weight map - nsamples {nsamples}")
                    loss_viz = loss_reshaped[batch].squeeze().numpy()
                    logger.debug(f"loss viz MAX {np.max(loss_viz)}")
                    loss_viz = loss_viz/nsamples
                # logger.debug(loss_reshaped[batch].shape, f"loss sum {np.sum(loss_viz)}")
                im = ax.imshow(loss_viz, cmap=plt.cm.jet)
                cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
                cbar.ax.locator_params(nbins=5)
                ax.axis('off')
                # axes[i][2].set_title(loss_name)
                # logger.debug("unique loss values",np.unique(loss_viz))

                logger.debug(f"loss {loss.shape} | reshaped {loss_reshaped.shape}")

            elif elem_name == "weight_map" and elem_name in show:
                im = ax.imshow(weight_map[batch], cmap=plt.cm.jet)
                cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
                cbar.ax.locator_params(nbins=5)
                ax.axis('off')

            if elem_name in show:
                col += 1
    # for r in axes:
    #     for c in r:
    #         axes[r][c].axis('off')


    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"results/loss_weight/cityscapes/{loss_name} - cityscapes - test{idx} - 2021-03-27 14-54-cityscapes-c3-kl-rgb-epoch=191-val_loss=0.0958.png")


def prepare_sample(output_orig, target_orig, weight_map=None, masking=True):
    bs = target_orig.shape[0]
    if weight_map is not None:
        output, target, weight_map = flatten_tensors(output_orig, target_orig, weight_map)
    else:
        output, target = flatten_tensors(output_orig, target_orig)

    return bs, output, target, weight_map

class CompareLosses(nn.Module):
    def __init__(self, n_classes, masking, ranks, dist, returnloss):
        super().__init__()
        self.num_classes = n_classes
        self.masking = masking
        self.ranks = ranks
        self.kl = KLLoss(n_classes=self.num_classes, masking=self.masking)
        self.sord = SORDLoss(n_classes=self.num_classes, masking=self.masking, ranks=self.ranks, dist=dist)
        self.returnloss = returnloss

    def forward(self, output, target, weight_map=None, debug=True, viz=True):
        # target = torch.fliplr(target)
        # for i in range(target.shape[0]):
            # driveable = torch.zeros_like(target[i])
            # driveable[:100,:] = 1
            # for cls in range(0, self.num_classes):
            #     output[i][cls] = driveable
            #
            # output[i][1] = 0
            # output[i][0] = driveable
            # output[i][2] = torch.logical_not(driveable)

        # weight_map = None

        losses = {
            #"kldiv": self.kl(output_orig=output, target_orig=torch.clone(target), weight_map=None, debug=debug, reduce=False),
            "kldiv_w": self.kl(output_orig=output, target_orig=torch.clone(target), weight_map=weight_map, debug=debug, reduce=False),
            #"sord": self.sord(output_orig=output, target_orig=torch.clone(target), weight_map=weight_map, debug=debug, reduce=False),
        }
        use_w = {
            "kldiv": False,
            "kldiv_w": True
        }
        viz_loss(output, losses = losses, use_w = use_w, weight_map = weight_map, bs=target.shape[0], nclasses=self.num_classes, show={"loss"}, target=target)
        return losses["kldiv_w"][1]


class KLLoss(nn.Module):
    def __init__(self, n_classes, masking=False):
        super().__init__()
        self.num_classes = n_classes
        self.masking = masking

    def forward(self, output_orig, target_orig, weight_map=None, debug=False, reduce=True):

        # logger.debug(torch.unique(target_orig))
        bs, output, target, weight_map = prepare_sample(output_orig, target_orig, weight_map=weight_map, masking=self.masking)

        if self.masking:
            mask = target.ge(0)
            loss = torch.zeros_like(output)
            target[~mask] = 0
        else:
            mask = torch.ones_like(target)

        n_samples = torch.sum(mask)
        # n_samples = torch.sum(weight_map,axis=-1)
        logger.debug(f"KLLoss n_samples {n_samples}")

        if debug: logger.debug(f"{output},{target}")
        target = F.one_hot(target, num_classes=self.num_classes).float()
        if debug: logger.debug(f"{output},{target}")
        output = torch.nn.LogSoftmax(dim=-1)(output)
        loss[mask] = nn.KLDivLoss(reduction='none')(output, target)[mask]
        if weight_map is not None:
            #logger.debug(loss.shape, weight_map.shape)
            #logger.debug(torch.min(weight_map).item(),torch.max(weight_map).item())
            if debug: logger.debug(f"{loss.shape},{weight_map.shape}")
            if debug: logger.debug(f"{weight_map[:5]}, {torch.max(weight_map)} before weight map {loss[:5]}, {torch.max(loss)}")
            loss *= weight_map.unsqueeze(1).repeat(1, self.num_classes)
            if debug: logger.debug(f"after weight map {loss[:5]} {torch.max(loss)}")

        loss_reduced = torch.sum(loss)/n_samples
        logger.debug(f"KL loss reduced {loss_reduced}")

        if reduce:
            return loss_reduced

        return target, loss

class Distance(nn.Module):
    def __init__(self, dist):
        super().__init__()
        self.dist = dist
        self.mse = nn.MSELoss(reduction='none')
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, target, ranks, alpha=1):
        # logger.debug(target)
        if self.dist == "logl2":
            dist = torch.pow(torch.abs(alpha*(torch.log(target.float()) - torch.log(ranks.float()))),2)
        elif self.dist == "logl1":
            dist = torch.abs(torch.log((target.float() - torch.log(ranks.float()))))
        elif self.dist == "l2":
            dist = self.mse(alpha*target, alpha*ranks)
            #logger.debug(self.dist)
        elif self.dist == "l1":
            dist = self.l1(alpha*target, alpha*ranks)
        return dist

class SORDLoss(nn.Module):
    """
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf
    """

    def __init__(self, n_classes, ranks=None, masking=False, dist="l1", alpha=1):
        super().__init__()
        self.num_classes = n_classes
        if ranks is not None and len(ranks) == self.num_classes:
            self.ranks = ranks
        else:
            self.ranks = np.arange(1, self.num_classes+1)
        self.masking = masking
        logger.info(f"SORD ranks: {self.ranks}")

        self.alpha = alpha
        self.dist = Distance(dist=dist)
        logger.info(f"SORD alpha {self.alpha}")

    def forward(self, output_orig, target_orig, weight_map=None, debug=False, mod_input=None, reduce=True):

        target = torch.clone(target_orig)
        logger.debug(f"target {target_orig} {torch.unique(target_orig)}")
        # if debug: logger.debug("target_orig",target_orig,torch.unique(target_orig))
        for i,r in enumerate(self.ranks):
            target[target_orig==i] = r
            if debug: logger.debug(f"{i} to {r}")
        # logger.debug(torch.unique(target_orig))
        logger.debug(f"target {target} {torch.unique(target)}")

        logger.debug(f"SORD - before flatten: target shape {target_orig.shape} | output shape {output_orig.shape}")

        bs, output, target, weight_map = prepare_sample(output_orig, target, weight_map=weight_map, masking=self.masking)
        logger.debug(f"SORD - after flatten: target shape {target.shape} | output shape {output.shape}")

        if self.masking:
            logger.debug("SORD masking")
            mask = target.ge(0)
            loss = torch.zeros_like(output)
            target[~mask] = 0
        else:
            mask = torch.ones_like(target)

        n_samples = torch.sum(mask)
        logger.debug(f"SORDLoss n_samples {n_samples}")

        if debug: logger.debug(f"output {output}")
        ranks = torch.tensor(self.ranks, dtype=output.dtype, device=output.device, requires_grad=False).repeat(output.size(0), 1)
        if debug: logger.debug(f"ranks {ranks}")
        target = target.unsqueeze(1).repeat(1, self.num_classes)
        if debug: logger.debug(f"target {target} {torch.unique(target)}")
        soft_target = - self.dist(target, ranks, self.alpha)  # should be of size N x num_classes
        if debug: logger.debug(f"dist target {soft_target}")
        soft_target = torch.softmax(soft_target, dim=-1)
        if debug: logger.debug(f"soft target {soft_target}")
        # output = torch.log(soft_target)
        # flatten label and prediction tensors

        if mod_input is not None:
            output = mod_input.long().view(-1,).unsqueeze(1).repeat(1, self.num_classes)
            output = -self.dist(reduction='none')(output, ranks)  # should be of size N x num_classes
            if debug: logger.debug(f"output {output}")
            output = torch.softmax(output, dim=-1)
            if debug: logger.debug(f"output {output}")
            output = torch.log(output)
        else:
            output = torch.nn.LogSoftmax(dim=-1)(output)

        loss[mask] = nn.KLDivLoss(reduction='none')(output, soft_target)[mask]
        if weight_map is not None:
            if debug: logger.debug(f"{loss.shape},{weight_map.shape}")
            if debug: logger.debug(f"{torch.unique(weight_map)} before weight map {torch.unique(loss)}")
            loss *= weight_map.unsqueeze(1).repeat(1, self.num_classes)
            if debug: logger.debug(f"after weight map {torch.unique(loss)}")

        #logger.debug(n_samples)
        if reduce:
            loss = torch.sum(loss)/n_samples
            return loss
        return soft_target, loss


if __name__ == '__main__':

    from metrics import MaskedIoU

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pred', default="pref")
    # parser.add_argument('--gt', default="pref")
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--dist', default="l1")
    args = parser.parse_args()
    logger.debug(args)
    if args.debug: enable_debug()

    test_loss(alpha=args.alpha, dist=args.dist, debug=args.debug)

    # from metrics import MaskedIoU
    #
    # input = torch.tensor([[ [[0.0]], [[1.0]],  [[0.0]]],[ [[0.0]], [[1.0]],  [[0.0]]]], requires_grad=True)
    # target = torch.tensor([[[1]],[[1]]])
    # # ~ output = ce(input, target)
    # # ~ logger.debug(input,target,output)
    # # ~ output.backward()
    #

    #
    # if args.debug: enable_debug()
    #
    # onehot = {
    #     "pref": [0.0, 0.0, 1.0],
    #     "poss": [0.0, 1.0, 0.0],
    #     "imposs": [1.0, 0.0, 0.0]
    # }
    # level = {
    #     "pref": 2,
    #     "poss": 1,
    #     "imposs": 0,
    #     "void": -1
    # }
    #
    # input = torch.tensor([onehot[args.pred],onehot[args.pred],onehot[args.pred],onehot[args.pred]], requires_grad=True)
    # target = torch.tensor([level[args.gt],level[args.gt],level[args.gt],level[args.gt]], dtype=torch.long)
    # # ~ logger.debug(target, input, "CE ->", output)
    # # ~ input = torch.randn(1, 3, requires_grad=True)
    # # ~ target = torch.empty(1, dtype=torch.long).random_(3)
    #
    # # output = ce(input, target)
    # # output.backward()
    # # logger.debug(target, input, "CE ->", output)
    #
    # # ~ input, target = flatten_tensors(input, target)
    # # ~ input = torch.nn.LogSoftmax(dim=-1)(input)
    # cm = np.zeros((3, 3))
    # sord = SORDLoss(n_classes = 3, ranks=[level["imposs"],level["poss"],level["pref"]], masking=True)
    # logger.debug("SORD",sord(input, target))
    #
    # # for p,pred in enumerate(level.keys()):
    # #     for g,gt in enumerate(level.keys()):
    # #         input = torch.tensor([onehot[pred]], requires_grad=True)
    # #         target = torch.tensor([level[gt]], dtype=torch.long)
    # #         mod_input = torch.tensor([level[pred]], dtype=torch.long)
    # #         loss = sord(input, target, debug=True, mod_input=mod_input)
    # #         logger.debug("SORD ->", loss)
    # #         cm[g][p] = loss.item()
    # # logger.debug(cm)
    # #
    # # rankings = "|"+"|".join([str(l) for l in level.values()])+"|"
    # #
    # # from plotting import plot_confusion_matrix
    # # plot_confusion_matrix(cm, labels=["impossible","possible","preferable"], filename=f"sordloss-{rankings}", folder="results/sordloss", vmax=None, cmap="Blues", cbar=True, annot=False, vmin=0)
    # #
    # # level = {
    # #     "pref": 2,
    # #     "poss": 1,
    # #     "imposs": 0
    # # }
    # # rankings = "|"+"|".join([str(l) for l in level.values()])+"|"
    # #
    # # cm = np.zeros((3, 3))
    # # ce = nn.CrossEntropyLoss(ignore_index = -1)
    # # for p,pred in enumerate(level.keys()):
    # #     for g,gt in enumerate(level.keys()):
    # #         input = torch.tensor([onehot[pred]], requires_grad=True)
    # #         target = torch.log_softmax(torch.tensor([onehot[gt]]),dim=-1)
    # #         input = torch.log_softmax(input, dim=-1)
    # #         logger.debug(input)
    # #         loss = nn.KLDivLoss(reduction='mean',log_target=True)(input, target)
    # #         logger.debug("CE ->", loss)
    # #         cm[g][p] = loss.item()
    # # logger.debug(cm)
    # # plot_confusion_matrix(cm, labels=["impossible","possible","preferable"], filename=f"celoss-{rankings}", folder="results/sordloss", vmax=None, cmap="Blues", cbar=True, annot=False, vmin=0)
    #
    # kl = KLLoss(n_classes = 3, masking=True)
    # loss = kl(input, target)
    # logger.debug("KL",loss)
    #
    # iou = MaskedIoU(labels=[0,1,2])
    # logger.debug(iou(input,target))
