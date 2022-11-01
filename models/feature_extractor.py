from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodules import *
from models.biconv import *


class feature_extraction(nn.Module):

    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv0 = binary_firstconv(8, 64, 7, 1, 3, 1)
        self.firstconv1 = binary_convbn_relu(64, 64, 5, 2, 2, 1)
        self.firstconv2 = binary_convbn_relu(64, 32, 3, 1, 1, 1)
        self.layer1 = self._make_layer(BasicBlock, 32, 1, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, 2, 1, 1)  # 1/4
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 2, 1, 1)  # 1/8
        self.layer4 = self._make_layer(BasicBlock, 128, 1, 1, 1, 1)
        self.reduce = binary_convbn_relu(128, 64, 3, 1, 1, 1)

        self.conv1a = BasicConv(64, 96, kernel_size=3, stride=2, pad=1)
        self.conv2a = BasicConv(96, 128, kernel_size=3, stride=2, pad=1)
        self.conv3a = BasicConv(128, 160, kernel_size=3, stride=2, pad=1)

        self.IBC2d3a = IBC2d(160, 128, stride=1)
        self.IBC2d2a = IBC2d(128, 96, stride=1)
        self.IBC2d1a = IBC2d(96, 64, stride=1)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(DownSample(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = encode_input(x)
        x = self.firstconv0(x)
        x = self.firstconv1(x)
        x = self.firstconv2(x)
        x = self.layer1(x)
        conv0a = x
        x = self.layer2(x)
        x = self.layer3(x)
        feat0 = x
        x = self.layer4(x)
        feat1 = x
        x = self.reduce(x)
        feat2 = x
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        x = self.IBC2d3a(x, rem2)
        x = self.IBC2d2a(x, rem1)
        x = self.IBC2d1a(x, rem0)
        feat3 = x
        gwc_feature = torch.cat((feat0, feat1, feat2, feat3), dim=1)
        return conv0a, gwc_feature
