# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F

from metavision_core_ml.core.temporal_modules import VideoSequential, ConvRNN, seq_wise
from metavision_core_ml.core.unet import unet_layers, Unet
from metavision_core_ml.core.modules import ConvLayer


class MergeSkip(nn.Module):
    """
    Merge with skip connection
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, skip):
        x = seq_wise(F.interpolate)(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, skip), dim=2)
        return x


def pytestcase_feed_forward_unet():
    """
    We build a feedforward unet
    """
    num_layers = 3
    base = 4
    downs = [base * 2**(i + 1) for i in range(num_layers)]
    ups = [base * 2**(num_layers - i) for i in range(num_layers)]

    down = nn.Identity()
    up = MergeSkip()

    # Trial Architecture
    def enc_fun(x, y):
        return VideoSequential(ConvLayer(x, y, stride=2))

    def midd_fun(x, y):
        return VideoSequential(ConvLayer(x, y))

    def dec_fun(x, y):
        return VideoSequential(ConvLayer(x, y))

    encoders, decoders = unet_layers(enc_fun, midd_fun, dec_fun, base, downs, ups[0] * 2, ups)
    net = Unet(encoders, decoders, down, up)

    t, b, c, h, w = 4, 3, base, 64, 64
    x = torch.randn(t, b, c, h, w)
    y = net(x)
    assert y.shape == torch.Size((t, b, ups[-1], h, w))


def pytestcase_recurrent_unet():
    """
    We build a recurrent unet
    """
    num_layers = 3
    base = 4
    downs = [base * 2**(i + 1) for i in range(num_layers)]
    ups = [base * 2**(num_layers - i) for i in range(num_layers)]

    down = VideoSequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False))
    up = MergeSkip()

    # Trial Architecture
    def enc_fun(x, y):
        return ConvRNN(x, y, separable=True, separable_hidden=True)

    def midd_fun(x, y):
        return VideoSequential(ConvLayer(x, y, separable=True))

    def dec_fun(x, y):
        return VideoSequential(ConvLayer(x, y, separable=True))

    encoders, decoders = unet_layers(enc_fun, midd_fun, dec_fun, base, downs, ups[0] * 2, ups)
    net = Unet(encoders, decoders, down, up)

    t, b, c, h, w = 4, 3, base, 64, 64
    x = torch.randn(t, b, c, h, w)
    y = net(x)
    assert y.shape == torch.Size((t, b, ups[-1], h, w))


def pytestcase_grunet():
    """
    We build a GRU unet
    """
    num_layers = 3
    base = 4
    downs = [base * 2**(i + 1) for i in range(num_layers)]
    ups = [base * 2**(num_layers - i) for i in range(num_layers)]

    down = nn.Identity()
    up = MergeSkip()

    # Trial Architecture
    def enc_fun(x, y):
        return VideoSequential(ConvLayer(x, y, stride=2))

    def midd_fun(x, y):
        return ConvRNN(x, y, cell="gru")

    def dec_fun(x, y):
        return VideoSequential(ConvLayer(x, y))

    encoders, decoders = unet_layers(enc_fun, midd_fun, dec_fun, base, downs, ups[0] * 2, ups)
    net = Unet(encoders, decoders, down, up)

    t, b, h, w = 4, 3, 64, 64
    x = torch.randn(t, b, base, h, w)
    y = net(x)
    assert y.shape == torch.Size((t, b, ups[-1], h, w))
