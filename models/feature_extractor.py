from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodules import *
from models.biconv import *


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        if is_3d:
            if deconv:
                self.conv = binary_convbn_transpose_3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = binary_convbn_3d_relu(in_channels, out_channels, bias=False, **kwargs)

        else:
            if deconv:
                self.conv = binary_deconvbn_2d_relu(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = binary_convbn_relu(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class IBC2d(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True, stride=2):
        super(IBC2d, self).__init__()
        self.concat = concat
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=3, stride=stride, pad=1)
        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, pad=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, pad=1)

    def forward(self, x, rem):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x = self.conv1(x)
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


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
