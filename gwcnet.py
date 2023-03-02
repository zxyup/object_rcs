import time
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from submodule import *


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)


    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)

        return gwc_feature


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        # padding = 3 - 1 - 1 = 1
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=0, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))
            # torch.Size([1, 64, 24, 60, 80])
        # padding = 3 - 1 - 1 = 1
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=0, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))
            # torch.Size([1, 32, 48, 120, 160])

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv2 = self.conv2(self.conv1(x))
        
        conv5 = F.relu(F.pad(self.conv5(self.conv4(self.conv3(conv2))), [0, 1]*3) + self.redir2(conv2), inplace=True)

        conv6 = F.relu(F.pad(self.conv6(conv5), [0, 1]*3) + self.redir1(x), inplace=True)

        return conv6


class GwcNet(nn.Module):
    def __init__(self, maxdisp):
        super(GwcNet, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = 40
        self.concat_channels = 0
        self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    def forward(self, left, right):
        # batch optimization, 通过批次叠加加快计算
        comb = torch.cat([left, right], dim=0)
        features = self.feature_extraction(comb)
        features_left = features[0].unsqueeze(dim=0)
        features_right = features[1].unsqueeze(dim=0)
        # pure calculate
        volume = build_gwc_volume(features_left, features_right, self.maxdisp // 4,
                                      self.num_groups)
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)
        cost3 = self.classif3(out3)
        cost3 = F.interpolate(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparity_regression(pred3, self.maxdisp)

        return pred3


def GwcNet_G(d):
    return GwcNet(d)

