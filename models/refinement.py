from __future__ import print_function
import torch.nn as nn
from models.submodules import *
import torch
import matplotlib.pyplot as plt
from models.warp import disp_warp
import torch.nn.functional as F
from models.biconv import *


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        # print("relu:",relu,"bn:",bn)
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


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True, stride=2, mdconv=False):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, pad=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, pad=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, pad=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


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

        self.deconv4a = Conv2x(128, 96, stride=1)
        self.deconv3a = Conv2x(96, 64, stride=1)
        self.deconv2a = Conv2x(64, 48, stride=1)
        self.deconv1a = Conv2x(48, 32, stride=1)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, stride=1)
        self.deconv3b = Conv2x(96, 64, stride=1)
        self.deconv2b = Conv2x(64, 48, stride=1)
        self.deconv1b = Conv2x(48, 32, stride=1)

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
        rem4 = x

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x = self.deconv4b(x, rem3)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x = self.deconv3b(x, rem2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x = self.deconv2b(x, rem1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        x = self.deconv1b(x, rem0)

        residual_disp = self.final_conv(x)
        disp = F.relu(disp + residual_disp, inplace=True)

        return disp
