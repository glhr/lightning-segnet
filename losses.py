import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class RegressionLoss(nn.Module):
    def forward(self, output, target):
        return nn.MSELoss()(output.squeeze(), target)


class ClassificationLoss(nn.Module):
    def forward(self, output, target, weights):
        return nn.NLLLoss(weights)(output, target.long())


class UnimodalUniformOTLoss(nn.Module):
    """
    https://arxiv.org/pdf/1911.02475.pdf
    """

    def __init__(self, n_classes):
        super().__init__()
        self.num_classes = n_classes
        self.csi = 0.15
        self.e = 0.05
        self.tau = 1.

    def forward(self, output, target):
        output = torch.softmax(output, -1)
        ranks = torch.arange(0, self.num_classes, dtype=output.dtype, device=output.device).repeat(output.size(0), 1)
        target_repeated = target.unsqueeze(1).repeat(1, self.num_classes)
        p = torch.softmax(torch.exp(-torch.abs(ranks - target_repeated) / self.tau), dim=-1)
        target_onehot = torch.nn.functional.one_hot(target.unsqueeze(0).long(), self.num_classes).squeeze()
        uniform_term = 1. / self.num_classes
        soft_target = (1 - self.csi - self.e) * target_onehot + self.csi * p + self.e * uniform_term
        loss = nn.L1Loss()(torch.cumsum(output, dim=1), torch.cumsum(soft_target, dim=1))
        return loss


class DLDLLoss(nn.Module):
    """
    https://arxiv.org/pdf/1611.01731.pdf
    """

    def __init__(self, n_classes):
        super().__init__()
        self.num_classes = n_classes

    def forward(self, output, target):
        output = torch.nn.LogSoftmax(dim=-1)(output)
        normal_dist = Normal(torch.arange(0, self.num_classes).to(output.device), torch.ones(self.num_classes).to(output.device))
        soft_target = torch.softmax(normal_dist.log_prob(target.unsqueeze(1)).exp(), -1)
        return nn.KLDivLoss()(output, soft_target)


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

class SORDLoss(nn.Module):
    """
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf
    """

    def __init__(self, n_classes, ranks = [0,1,2]):
        super().__init__()
        self.num_classes = n_classes
        if ranks is not None and len(ranks) == self.num_classes:
            self.ranks = ranks
        else:
            self.ranks = np.arange(0, self.num_classes)

    def forward(self, output, target, debug=False):

        #flatten label and prediction tensors
        if debug: print("output",output)
        # print(torch.unique(target))
        output, target = flatten_tensors(output, target)
        output = torch.nn.LogSoftmax(dim=-1)(output)

        # mask = target.ge(0)
        # # print(mask, mask.shape)
        # # print(output.shape,target.shape)
        # output = output[mask]
        # target = target[mask]

        if debug: print("output",output)
        ranks = torch.tensor(self.ranks, dtype=output.dtype, device=output.device, requires_grad=False).repeat(output.size(0), 1)
        if debug: print("ranks",ranks)
        target = target.unsqueeze(1).repeat(1, self.num_classes)
        if debug: print("target",target)
        soft_target = -nn.L1Loss(reduction='none')(target, ranks)  # should be of size N x num_classes
        if debug: print("l1 target",soft_target)
        soft_target = torch.softmax(soft_target, dim=-1)
        if debug: print("soft target",soft_target)
        return nn.KLDivLoss(reduction='mean')(output, soft_target)


class OTLossSoft(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.num_classes = n_classes

    def forward(self, output, target):
        ranks = torch.arange(0, self.num_classes, dtype=output.dtype, device=output.device, requires_grad=False).repeat(output.size(0), 1)
        target = target.unsqueeze(1).repeat(1, self.num_classes)
        soft_target = -nn.L1Loss(reduction='none')(target, ranks)  # should be of size N x num_classes
        soft_target = torch.softmax(soft_target, dim=-1)  # like in SORD
        loss = nn.L1Loss()(torch.cumsum(output, dim=1), torch.cumsum(soft_target, dim=1))  # like in Liu 2019
        return loss


class OTLoss(nn.Module):

    def __init__(self, n_classes, cost='linear'):
        super().__init__()
        self.num_classes = n_classes
        C0 = np.expand_dims(np.arange(n_classes), 0).repeat(n_classes, axis=0) / self.num_classes
        C1 = np.expand_dims(np.arange(n_classes), 1).repeat(n_classes, axis=1) / self.num_classes

        C = np.abs(C0 - C1)
        if cost == 'quadratic':
            C = C ** 2
        elif cost == 'linear':
            pass
        self.C = torch.tensor(C).float()

    def forward(self, output_probs, target_class):
        C = self.C.cuda(output_probs.device)
        costs = C[target_class.long()]
        transport_costs = torch.sum(costs * output_probs, dim=1)
        result = torch.mean(transport_costs)
        return result

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
    args = parser.parse_args()
    print(args)

    onehot = {
        "pref": [0.0, 0.0, 1.0],
        "poss": [0.0, 1.0, 0.0],
        "imposs": [1.0, 0.0, 0.0]
    }
    level = {
        "pref": 2,
        "poss": 1,
        "imposs": 0
    }

    input = torch.tensor([onehot[args.pred]], requires_grad=True)
    target = torch.tensor([level[args.gt]], dtype=torch.long)
    # ~ print(target, input, "CE ->", output)
    # ~ input = torch.randn(1, 3, requires_grad=True)
    # ~ target = torch.empty(1, dtype=torch.long).random_(3)

    # output = ce(input, target)
    # output.backward()
    # print(target, input, "CE ->", output)

    # ~ input, target = flatten_tensors(input, target)
    # ~ input = torch.nn.LogSoftmax(dim=-1)(input)
    cm = np.zeros((3, 3))
    sord = SORDLoss(n_classes = 3, ranks=[level["imposs"],level["poss"],level["pref"]])

    for p,pred in enumerate(level.keys()):
        for g,gt in enumerate(level.keys()):
            input = torch.tensor([onehot[pred]], requires_grad=True)
            target = torch.tensor([level[gt]], dtype=torch.long)
            loss = sord(input, target, debug=True)
            print("SORD ->", loss)
            cm[g][p] = loss.item()
    print(cm)

    rankings = "|"+"|".join([str(l) for l in level.values()])+"|"

    from plotting import plot_confusion_matrix
    plot_confusion_matrix(cm, labels=["impossible","possible","preferable"], filename=f"sordloss-{rankings}", folder="", vmax=None, cmap="Blues", cbar=True, annot=False, vmin=0)

    level = {
        "pref": 2,
        "poss": 1,
        "imposs": 0
    }
    rankings = "|"+"|".join([str(l) for l in level.values()])+"|"

    cm = np.zeros((3, 3))
    ce = nn.CrossEntropyLoss(ignore_index = -1)
    for p,pred in enumerate(level.keys()):
        for g,gt in enumerate(level.keys()):
            input = torch.tensor([onehot[pred]], requires_grad=True)
            target = torch.tensor([level[gt]], dtype=torch.long)
            loss = ce(input, target)
            print("CE ->", loss)
            cm[g][p] = loss.item()
    print(cm)
    plot_confusion_matrix(cm, labels=["impossible","possible","preferable"], filename=f"celoss-{rankings}", folder="", vmax=None, cmap="Blues", cbar=True, annot=False, vmin=0)
