from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from models.submodules import SubModule
from models.biconv import *


class IBC3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, pad):
        super(IBC3d, self).__init__()
        self.conv = binary_convbn_3d_relu(in_channels, out_channels, kernel_size, stride, pad)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False, recompute_scale_factor=True)
        x = self.conv(x)
        return x


class HourGlass(SubModule):

    def __init__(self, inplanes=16):
        super(HourGlass, self).__init__()

        self.conv1 = binary_convbn_3d_relu(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1)
        self.conv2 = binary_convbn_3d_relu(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)
        self.conv1_1 = binary_convbn_3d_relu(inplanes * 2, inplanes * 4, kernel_size=3, stride=2, pad=1)
        self.conv2_1 = binary_convbn_3d_relu(inplanes * 4, inplanes * 4, kernel_size=3, stride=1, pad=1)
        self.conv3 = binary_convbn_3d_relu(inplanes * 4, inplanes * 8, kernel_size=3, stride=2, pad=1)
        self.conv4 = binary_convbn_3d_relu(inplanes * 8, inplanes * 8, kernel_size=3, stride=1, pad=1)
        self.ibc3d3 = IBC3d(inplanes * 8, inplanes * 4, kernel_size=3, stride=1, pad=1)
        self.ibc3d2 = IBC3d(inplanes * 4, inplanes * 2, kernel_size=3, stride=1, pad=1)
        self.ibc3d1 = IBC3d(inplanes * 2, inplanes, kernel_size=3, stride=1, pad=1)
        self.last_for_guidance = binary_convbn_3d_relu(inplanes, 32, kernel_size=3, stride=1, pad=1)
        self.weight_init()


class CoeffsPredictor(HourGlass):

    def __init__(self, hourglass_inplanes=16):
        super(CoeffsPredictor, self).__init__(hourglass_inplanes)

    def forward(self, input):
        output0 = self.conv1(input)
        output0_a = self.conv2(output0) + output0
        output0 = self.conv1_1(output0_a)
        output0_c = self.conv2_1(output0) + output0
        output0 = self.conv3(output0_c)
        output0 = self.conv4(output0) + output0
        output1 = self.ibc3d3(output0) + output0_c
        output1 = self.ibc3d2(output1) + output0_a
        output1 = self.ibc3d1(output1)
        coeffs = self.last_for_guidance(output1).permute(0, 2, 1, 3, 4).contiguous()
        return coeffs
