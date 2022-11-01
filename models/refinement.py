from __future__ import print_function
import torch.nn as nn
from models.submodules import *
import torch
import matplotlib.pyplot as plt
from models.warp import disp_warp
import torch.nn.functional as F
from models.biconv import *


class HourglassRefinement(SubModule):

    def __init__(self):
        super(HourglassRefinement, self).__init__()

        in_channels = 2

        self.conv1 = binary_convbn_relu(in_channels, 16, 3, 1, 1)
        self.conv2 = binary_convbn_relu(1, 16, 3, 1, 1)

        self.conv_start = binary_convbn_relu(32, 32, 3, 1, 1)

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, pad=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, pad=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, pad=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, pad=1)

        self.IBC2d4a = IBC2d(128, 96, stride=1)
        self.IBC2d3a = IBC2d(96, 64, stride=1)
        self.IBC2d2a = IBC2d(64, 48, stride=1)
        self.IBC2d1a = IBC2d(48, 32, stride=1)

        self.final_conv = binary_conv(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear')
            disp = disp * scale_factor
        warped_right = disp_warp(right_img, disp)[0]
        error = warped_right - left_img
        concat1 = torch.cat((error, left_img), dim=1)
        conv1 = self.conv1(concat1)
        conv2 = self.conv2(disp)
        x = torch.cat((conv1, conv2), dim=1)

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)

        x = self.IBC2d4a(x, rem3)
        x = self.IBC2d3a(x, rem2)
        x = self.IBC2d2a(x, rem1)
        x = self.IBC2d1a(x, rem0)

        residual_disp = self.final_conv(x)
        disp = F.relu(disp + residual_disp, inplace=True)

        return disp
