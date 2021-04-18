from torch import nn as nn
import torch

from segnet import SegNet

from utils import logger
from plotting import display_img
import numpy as np

class FusionNet(nn.Module):
    """PyTorch module for 'AdapNet++' and 'AdapNet++ with fusion architecture' """

    def __init__(self, encoders, decoder, classifier, filter_config, pooling_fusion="rgb"):
        super(FusionNet, self).__init__()

        self.fusion = False
        self.filter_config = filter_config
        self.pooling_fusion = pooling_fusion
        logger.info(pooling_fusion)

        logger.debug(len(encoders), encoders)

        if len(encoders) > 1:
            self.encoder_mod1 = encoders[0]
            # self.encoder_mod1.res_n50_enc.layer3[2].dropout = False
            self.encoder_mod2 = encoders[1]
            # self.encoder_mod2.res_n50_enc.layer3[2].dropout = False
            # self.ssma_s1 = SSMA(24, 6)
            # self.ssma_s2 = SSMA(24, 6)
            self.ssma_res = SSMA(512, 16)
            if self.pooling_fusion == "fuse":
                self.pooling_fusion_block = nn.ModuleList()
                for f in self.filter_config:
                    self.pooling_fusion_block.append(PoolingFusion(f))
            self.fusion = True
        else:
            self.encoder_mod1 = encoders[0]

        self.eASPP = eASPP()
        self.decoder = decoder
        self.classifier = classifier


    def init_decoder(self):
        for d in self.decoder.children():
            for layer in d.features:
                #print(layer)
                if hasattr(layer, 'reset_parameters'):
                    #print("before reset", layer.weight[0])
                    layer.reset_parameters()
                    #print("after reset",layer.weight[:5])
                    if isinstance(layer, nn.Conv2d):
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                        #print("after kaiming",layer.weight[:5])
        nn.init.kaiming_uniform_(self.classifier.weight)

    def encoder_path(self, encoder, feat):
        indices = []
        unpool_sizes = []
        feats = []
        for i in range(0, 5):
            (feat, ind), size = encoder[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)
            feats.append(feat)
        return feats, indices, unpool_sizes

    def decoder_path(self, decoder, feat, indices, unpool_sizes):
        for i in range(0, 5):
            feat = decoder[i](feat, indices[4 - i], unpool_sizes[4 - i])
        return feat

    def forward(self, mod):
        """Forward pass

        In the case of AdapNet++, only 1 modality is used (either the RGB-image, or the Depth-image). With 'AdapNet++
        with fusion architecture' two modalities are used (both the RGB-image and the Depth-image).

        :param mod1: modality 1
        :param mod2: modality 2
        :return: final output and auxiliary output 1 and 2
        """
        logger.debug(f"{mod.shape}, {mod[:,0,:,:].unsqueeze(1).shape}")

        if self.fusion:
            # logger.info("FUSING SHIT :D")
            feat_1, indices_1, unpool_sizes_1 = self.encoder_path(self.encoder_mod1, mod[:,0,:,:].unsqueeze(1))
            feat_2, indices_2, unpool_sizes_2 = self.encoder_path(self.encoder_mod2, mod[:,1,:,:].unsqueeze(1))
            #m2_x, m2_s2, m2_s1 = self.encoder_mod2(mod2)
            #skip2 = self.ssma_s2(skip2, m2_s2)
            #skip1 = self.ssma_s1(skip1, m2_s1)
            feat = self.ssma_res(feat_1[-1], feat_2[-1])
        else:
            feat_1, indices_1, unpool_sizes_1 = self.encoder_path(self.encoder_mod1, mod)
            feat = feat_1[-1]

        # logger.info(self.pooling_fusion)

        if self.pooling_fusion == "fuse":
        # decoder path, upsampling with corresponding indices and size
            idx_fused = []
            for i,layer_idx in enumerate(indices_1):
                # print(indices_1[i].shape, indices_1[i][0][0][:5], indices_2[i][0][0][:5])
                # print(indices_1[i].shape)
                # combo = torch.stack((indices_1[i],indices_2[i]))
                # print(combo.shape)
                # print("id1",indices_1[0][0][0][0][:5])
                # print("id2",indices_2[0][0][0][0][:5])
                indices_fused = self.pooling_fusion_block[i](feat_1[i], feat_2[i], indices_1[i], indices_2[i])
                # print(mean.shape, mean[0][0][:5])
                # print(mean.shape)
                idx_fused.append(indices_fused)
                # print("idx",idx_fused[0][0][0][0][:5])
            # logger.debug(f"idx {torch.stack((indices_1)).shape}")
            # c
            # logger.debug(f"cat {cat[0]} {cat.shape}")
            indices = idx_fused
        elif self.pooling_fusion == "rgb":
            indices = indices_1

        # decoder path, upsampling with corresponding indices and size
        feat = self.decoder_path(self.decoder, feat, indices, unpool_sizes_1)

        return self.classifier(feat)

        #aux1, aux2, res = self.decoder(m1_x, skip1, skip2)
        #return aux1, aux2, res

class eASPP(nn.Module):
    """PyTorch Module for eASPP"""

    def __init__(self):
        """Constructor
        Initializes the 5 branches of the eASPP network.
        """

        super(eASPP, self).__init__()

        # branch 1
        self.branch1_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.branch1_bn = nn.BatchNorm2d(256)

        self.branch234 = nn.ModuleList([])
        self.branch_rates = [3, 6, 12]
        for rate in self.branch_rates:
            # branch 2
            branch = nn.Sequential(
                nn.Conv2d(512, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            self.branch234.append(branch)
        for i, sequence in enumerate(self.branch234):
            for ii, layer in enumerate(sequence):
                if str(type(layer)) == "<class 'torch.nn.modules.conv.Conv2d'>":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

        # branch 5
        self.branch5_conv = nn.Conv2d(512, 256, 1)
        nn.init.kaiming_uniform_(self.branch5_conv.weight, nonlinearity="relu")
        self.branch5_bn = nn.BatchNorm2d(256)

        # final layer
        self.eASPP_fin_conv = nn.Conv2d(512, 256, kernel_size=1)
        nn.init.kaiming_uniform_(self.eASPP_fin_conv.weight, nonlinearity="relu")
        self.eASPP_fin_bn = nn.BatchNorm2d(256)

    def forward(self, x):
        """Forward pass
        :param x: input from encoder (in stage 1) or from fused encoders (in stage 2 and 3)
        :return: feature maps to be forwarded to decoder
        """
        # branch 1: 1x1 convolution
        out = torch.relu(self.branch1_bn(self.branch1_conv(x)))

        # branch 2-4: atrous pooling
        y = self.branch234[0](x)
        out = torch.cat((out, y), 1)
        y = self.branch234[1](x)
        out = torch.cat((out, y), 1)
        y = self.branch234[2](x)
        out = torch.cat((out, y), 1)

        # branch 5: image pooling
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.relu(self.branch5_bn(self.branch5_conv(x)))
        x = nn.Upsample((24, 48), mode="bilinear")(x)
        out = torch.cat((out, x), 1)

        return torch.relu(self.eASPP_fin_bn(self.eASPP_fin_conv(out)))


class PoolingFusion(nn.Module):
    def __init__(self, channels, bottleneck=16):
        """Constructor
        :param features: number of feature maps
        :param bottleneck: bottleneck compression rate
        """
        super(PoolingFusion, self).__init__()

        reduce_size = int(channels / bottleneck)
        self.link = nn.Sequential(
            nn.Conv2d(channels*2, reduce_size, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(reduce_size, channels*2, kernel_size=1, stride=1),

        )

        nn.init.kaiming_uniform_(self.link[0].weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.link[2].weight, nonlinearity="relu")

        self.sm = nn.Softmax(dim=1)

    def forward(self, m1, m2, i1, i2):
        """Forward pass
        :param x1: input data from encoder 1
        :param x2: input data from encoder 2
        :return: Fused feature maps
        """

        i_12 = torch.cat((m1, m2), dim=1)
        #print(i1.shape,i2.shape, i_12.shape)


        i_12_w = self.link(i_12)
        b,c,h,w = i_12_w.shape
        i_12_w = i_12_w.view(b,2,int(c/2),h,w)
        #print(i_12_w.shape)
        i_12_w = self.sm(i_12_w)
        #print(i_12_w.shape)

        #x_12 = torch.sum(i_12_w, dim=1)
        x_12 = torch.unbind(i_12_w, dim=1)

        #print(i1.shape, x_12[0].shape)
        fused = (i1 * x_12[0]) + (i2 * x_12[1])
        #print(torch.unique(fused.long()))

        return fused.long()

class SSMA(nn.Module):
    """PyTorch Module for SSMA"""

    def __init__(self, features, bottleneck):
        """Constructor
        :param features: number of feature maps
        :param bottleneck: bottleneck compression rate
        """
        super(SSMA, self).__init__()
        reduce_size = int(features / bottleneck)
        double_features = int(2 * features)
        self.link = nn.Sequential(
            nn.Conv2d(double_features, reduce_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(reduce_size, double_features, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(double_features, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features)
        )

        nn.init.kaiming_uniform_(self.link[0].weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.link[2].weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.final_conv[0].weight, nonlinearity="relu")

    def forward(self, x1, x2):
        """Forward pass
        :param x1: input data from encoder 1
        :param x2: input data from encoder 2
        :return: Fused feature maps
        """
        x_12 = torch.cat((x1, x2), dim=1)

        x_12_est = self.link(x_12)
        x_12 = x_12 * x_12_est
        x_12 = self.final_conv(x_12)

        return x_12

if __name__ == "__main__":
    segnet = SegNet(num_classes=3)
    encoder = segnet.encoders
    #print(encoder)
    fusion = FusionNet( encoders=[encoder], decoder=segnet.decoders, classifier=segnet.classifier)
    print(fusion)
    print(SSMA(features=512, bottleneck=3))
