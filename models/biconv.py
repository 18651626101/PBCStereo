import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np


class BinaryActivation(nn.Module):

    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + \
            (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + \
            (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * \
            (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class LearnableBias(nn.Module):

    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class LearnableBias_3D(nn.Module):

    def __init__(self, out_chn):
        super(LearnableBias_3D, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + \
            cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        return y


class HardBinary_ConvTranspose2d(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size, stride=1, padding=1):
        super(HardBinary_ConvTranspose2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (in_chn, out_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + \
            cliped_weights
        y = F.conv_transpose2d(x, binary_weights, stride=self.stride, padding=self.padding)
        return y


class HardBinaryConv_3D(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size, stride=1, padding=1):
        super(HardBinaryConv_3D, self).__init__()
        self.stride = (1, stride, stride)
        self.padding = (padding, padding, padding)
        self.number_of_weights = in_chn * out_chn * \
            kernel_size * kernel_size*kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(torch.mean(abs(real_weights), dim=4, keepdim=True), dim=3, keepdim=True),
                                               dim=2,
                                               keepdim=True),
                                    dim=1,
                                    keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + \
            cliped_weights
        y = F.conv3d(x, binary_weights, stride=self.stride, padding=self.padding)
        return y


class HardBinary_ConvTranspose3d(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size, padding, output_padding, stride):
        super(HardBinary_ConvTranspose3d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.number_of_weights = in_chn * out_chn * \
            kernel_size * kernel_size * kernel_size
        self.shape = (in_chn, out_chn, kernel_size, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights, 1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(torch.mean(abs(real_weights), dim=4, keepdim=True), dim=3, keepdim=True),
                                               dim=2,
                                               keepdim=True),
                                    dim=1,
                                    keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + \
            cliped_weights
        y = F.conv_transpose3d(x, binary_weights, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
        return y


class binary_conv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
        super(binary_conv, self).__init__()
        self.move0 = LearnableBias(in_planes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_planes, out_planes, kernel_size, stride, pad)

    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        return out


class binary_convbn(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
        super(binary_convbn, self).__init__()
        self.move0 = LearnableBias(in_planes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_planes, out_planes, kernel_size, stride, pad)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.move1 = LearnableBias(out_planes)

    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)
        out = self.move1(out)
        return out


class binary_convbn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
        super(binary_convbn_relu, self).__init__()
        self.move0 = LearnableBias(in_planes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_planes, out_planes, kernel_size, stride, pad)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.move1 = LearnableBias(out_planes)
        self.prelu = nn.PReLU(out_planes)
        self.move2 = LearnableBias(out_planes)

    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)
        return out


class binary_convbn_2d_Tanh(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
        super(binary_convbn_2d_Tanh, self).__init__()
        self.conv0 = binary_convbn(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False)
        self.move0 = nn.Tanh()

    def forward(self, x):
        out = self.conv0(x)
        out = self.move0(out)
        return out


class binary_deconvbn_2d_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
        super(binary_deconvbn_2d_relu, self).__init__()
        self.move0 = LearnableBias(in_planes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinary_ConvTranspose2d(in_planes, out_planes, kernel_size, stride, pad)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.move1 = LearnableBias(out_planes)
        self.prelu = nn.PReLU(out_planes)
        self.move2 = LearnableBias(out_planes)

    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)
        return out


class binary_convbn_3d_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, pad):
        super(binary_convbn_3d_relu, self).__init__()
        self.move0 = LearnableBias_3D(in_planes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv_3D(in_planes, out_planes, kernel_size, stride, pad)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.move1 = LearnableBias_3D(out_planes)
        self.prelu = nn.PReLU(out_planes)
        self.move2 = LearnableBias_3D(out_planes)

    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)
        return out


class binary_convbn_transpose_3d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, padding, output_padding, stride, bias):
        super(binary_convbn_transpose_3d, self).__init__()
        self.move0 = LearnableBias_3D(in_planes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinary_ConvTranspose3d(in_planes, out_planes, kernel_size, padding, output_padding, stride)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.move1 = LearnableBias_3D(out_planes)
        self.prelu = nn.PReLU(out_planes)
        self.move2 = LearnableBias_3D(out_planes)

    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)
        return out


class DownSample(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(DownSample, self).__init__()
        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)
        return out


def encode_input(x):
    assert len(x.shape) == 4
    x = x.int()
    b = x.shape[0]
    w = x.shape[2]
    h = x.shape[3]
    x = x.repeat(1, 8, 1, 1)
    x = x.reshape(b, 8, w, h)
    for j in range(b):
        for i in range(8):
            x[j][i] = (x[j][i] & 2**i)
    x = x.float()
    return x


class binary_firstconv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
        super(binary_firstconv, self).__init__()
        self.binary_conv = HardBinaryConv(in_planes, out_planes, kernel_size, stride, pad)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.move1 = LearnableBias(out_planes)
        self.prelu = nn.PReLU(out_planes)

    def forward(self, x):
        out = self.binary_conv(x)
        out = self.bn1(out)
        out = self.move1(out)
        out = self.prelu(out)
        return out
