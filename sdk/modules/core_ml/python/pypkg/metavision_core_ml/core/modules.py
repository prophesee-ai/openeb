# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Reusable building blocks for neural networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWiseSeparableConv2d(nn.Sequential):
    """Depthwise Separable Convolution followed by pointwise 1x1 Convolution.

    A convolution is called depthwise separable when the normal convolution is split into two convolutions:
    depthwise convolution and pointwise convolution.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): separable conv receptive field
        stride (int): separable conv stride.
        padding (int): separable conv padding.
        depth_multiplier (int): Factor by which we multiply the *in_channels* to get the number of output_channels
            in the depthwise convolution.
        **kwargs: Additional keyword arguments passed to the first convolution operator.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False,
                 depth_multiplier=1,
                 **kwargs):
        super(DepthWiseSeparableConv2d, self).__init__(
            nn.Conv2d(in_channels, int(in_channels * depth_multiplier), kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=in_channels, bias=bias, **kwargs),
            nn.Conv2d(int(depth_multiplier * in_channels), out_channels, kernel_size=1, stride=1, padding=0,
                      dilation=1, groups=1, bias=bias)
        )


class ConvLayer(nn.Sequential):
    """Building Block Convolution Layer

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): conv receptive field
        stride (int): conv stride
        dilation (int): conv dilation
        bias (bool): whether or not to add a bias
        norm (str): type of the normalization
        activation (str): type of non-linear activation
        separable (bool): whether to use separable convolution
        **kwargs: Additional keyword arguments passed to the convolution operator.
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1,
                 bias=True, norm="BatchNorm2d", activation='ReLU', separable=False, **kwargs):

        conv_func = DepthWiseSeparableConv2d if separable else nn.Conv2d
        self.out_channels = out_channels
        self.separable = separable
        if not separable and "depth_multiplier" in kwargs:
            kwargs.pop('depth_multiplier')

        normalizer = nn.Identity() if norm == 'none' else getattr(nn, norm)(out_channels)

        super(ConvLayer, self).__init__(
            conv_func(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=padding, bias=bias, **kwargs),
            normalizer,
            getattr(nn, activation)()
        )


class PreActBlock(nn.Module):
    """
    Squeeze-Excite Block from:
    Squeeze-and-Excitation Networks (Hu et al.)

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        stride (int): convolution stride.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        assert out_channels % 4 == 0
        super(PreActBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = ConvLayer(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, padding=0, bias=False)
            )
        # SE layers
        self.fc1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # Squeeze
        w = F.adaptive_avg_pool2d(out, (1, 1))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w
        out = out + self.downsample(x)
        return out


class ResBlock(nn.Module):
    """
    Residual Convolutional Block

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        stride (int): convolutional stride
        norm (str): type of normalization
    """

    def __init__(self, in_channels, out_channels, stride=1, norm="BatchNorm2d"):
        super(ResBlock, self).__init__()
        bias = norm == 'none'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
            norm=norm,
        )
        self.conv2 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm=norm,
            bias=False,
        )

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                stride=stride,
                norm=norm,
                bias=False,
                activation="Identity",
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.downsample(x)
        out = F.relu(out)
        return out
