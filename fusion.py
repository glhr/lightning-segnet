from torch import nn as nn
import torch

from segnet import SegNet

from utils import logger
from plotting import display_img
import numpy as np

class FusionNet(nn.Module):
    """PyTorch module for 'AdapNet++' and 'AdapNet++ with fusion architecture' """

    def __init__(self, segnet_models=None, num_classes=3):
        super(FusionNet, self).__init__()

        self.fusion = False

        if segnet_models is None:
            segnet_models = [
                SegNet(num_classes=3),
                SegNet(num_classes=3)
            ]
        if len(segnet_models) > 1:
            self.encoder_mod1 = segnet_models[0].encoders
            # self.encoder_mod1.res_n50_enc.layer3[2].dropout = False
            self.encoder_mod2 = segnet_models[1].encoders
            # self.encoder_mod2.res_n50_enc.layer3[2].dropout = False
            # self.ssma_s1 = SSMA(24, 6)
            # self.ssma_s2 = SSMA(24, 6)
            self.ssma_res = SSMA(segnet_models[0].filter_config[-1], bottleneck=16)

            self.decoder_mod1 = segnet_models[0].decoders
            self.decoder_mod2 = segnet_models[1].decoders
            self.classifier = SSMA(segnet_models[0].filter_config[0], out=num_classes)

            self.fusion = True
        else:
            self.encoder_mod1 = segnet_models[0].encoders
            self.decoder_mod1 = segnet_models[0].decoders
            self.classifier = segnet_models[0].classifier


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
            # print(feat_1[-1].shape)
            feat_2, indices_2, unpool_sizes_2 = self.encoder_path(self.encoder_mod2, mod[:,1,:,:].unsqueeze(1))
            #m2_x, m2_s2, m2_s1 = self.encoder_mod2(mod2)
            #skip2 = self.ssma_s2(skip2, m2_s2)
            #skip1 = self.ssma_s1(skip1, m2_s1)
            feat = self.ssma_res(feat_1[-1], feat_2[-1])
        else:
            feat_1, indices_1, unpool_sizes_1 = self.encoder_path(self.encoder_mod1, mod)
            feat = feat_1[-1]

        # logger.info(self.pooling_fusion)

        if self.fusion:
            feat1 = self.decoder_path(self.decoder_mod1, feat, indices_1, unpool_sizes_1)
            feat2 = self.decoder_path(self.decoder_mod2, feat, indices_2, unpool_sizes_2)
            out = self.classifier(feat1, feat2)
            # print(out.shape)
            return out
        else:
            # decoder path, upsampling with corresponding indices and size
            feat = self.decoder_path(self.decoder, feat, indices_1, unpool_sizes_1)
            return self.classifier(feat)

        #aux1, aux2, res = self.decoder(m1_x, skip1, skip2)
        #return aux1, aux2, res

class SSMA(nn.Module):
    """PyTorch Module for SSMA"""

    def __init__(self, features, bottleneck=16, out=None):
        """Constructor
        :param features: number of feature maps
        :param bottleneck: bottleneck compression rate
        """
        super(SSMA, self).__init__()

        reduce_size = int(features / bottleneck)
        if out is None:
            self.final = False
            out = features
        else:
            self.final = True
        double_features = int(2 * features)
        self.link = nn.Sequential(
            nn.Conv2d(double_features, reduce_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(reduce_size, double_features, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(double_features, out, kernel_size=3, stride=1, padding=1),
        )

        if not self.final:
            self.bn = nn.BatchNorm2d(features)

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

        if not self.final:
            x_12 = self.bn(x_12)

        return x_12

if __name__ == "__main__":
    segnet = SegNet(num_classes=3)
    encoder = segnet.encoders
    #print(encoder)
    fusion = FusionNet( encoders=[encoder], decoder=segnet.decoders, classifier=segnet.classifier)
    print(fusion)
    print(SSMA(features=512, bottleneck=3))
