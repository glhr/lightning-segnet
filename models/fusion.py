from torch import nn as nn
import torch

import numpy as np

class SSMACustom(nn.Module):
    """Short summary.

    Parameters
    ----------
    features : int
        feature size
    bottleneck : int
         compression factor. use 16 by default.
    out : int
        default=None. for late fusion, set this to the number of classes
    late_dilation : int
        for dilated convolution, only used for late fusion
    fusion_activ : str
        "softmax" or "sigmoid"
    branches : 2
        number of inputs to fuse

    Attributes
    ----------
    final : type
        Description of attribute `final`.
    link : type
        Description of attribute `link`.
    sm : type
        Description of attribute `sm`.
    final_conv : type
        Description of attribute `final_conv`.
    branches

    """
    def __init__(self, features, bottleneck, out=None, late_dilation=1, fusion_activ="softmax", branches=2):
        super(SSMACustom, self).__init__()

        reduce_size = int(features / bottleneck)
        if out is None:
            self.final = False
            dilation = 1
        else:
            self.final = True
            dilation = late_dilation
        double_features = int(branches * features)
        self.link = nn.Sequential(
            nn.Conv2d(double_features, reduce_size, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv2d(reduce_size, double_features, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
        )
        self.sm = nn.Softmax(dim=1) if fusion_activ == "softmax" else nn.Sigmoid()

        if self.final:
            self.final_conv = nn.Sequential(
                nn.Conv2d(features, out, kernel_size=3, stride=1, padding=1)
            )
            nn.init.xavier_uniform_(self.final_conv[0].weight)
        else:
            self.final_conv = None

        nn.init.kaiming_normal_(self.link[0].weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.link[2].weight)

        self.branches = branches


    def forward(self, m_lst):
        """Short summary.

        Parameters
        ----------
        m_lst : list of tensors
            list of input feature maps to fuse (one per modality)

        Returns
        -------
        tensor
            combined feature map

        """
        #print(m_lst[0].shape, m_lst[1].shape)
        i_12 = torch.cat(m_lst, dim=1)
        #print(i1.shape,i2.shape, i_12.shape)

        i_12_w = self.link(i_12)
        b,c,h,w = i_12_w.shape
        i_12_w = i_12_w.view(b,self.branches,int(c/self.branches),h,w)
        #print(i_12_w.shape)
        i_12_w = self.sm(i_12_w)
        #print(i_12_w.shape)

        #x_12 = torch.sum(i_12_w, dim=1)
        x_12 = torch.unbind(i_12_w, dim=1)

        #print(i1.shape, x_12[0].shape)
        #print(i1.long()[0][0][0][:5], i2.long()[0][0][0][:5])
        fused = m_lst[0] * x_12[0]
        for f in range(1,self.branches):
            fused += (m_lst[f] * x_12[f])
        #print(fused.long()[0][0][0][:5])
        if self.final:
            fused = self.final_conv(fused)

        return fused
