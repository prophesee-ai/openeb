# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# pylint: disable=C0321

import torch
import torch.nn as nn
import torch.nn.functional as F

from metavision_core_ml.core.temporal_modules import VideoSequential, ConvRNN, seq_wise
from metavision_core_ml.core.unet import unet_layers, Unet
from metavision_core_ml.core.modules import ConvLayer, ResBlock, PreActBlock


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


class EventToVideo(nn.Module):
    """
    High Speed and High Dynamic Range Video with an Event Camera
    Rebecq et al.
    Every resize is done using bilinear sampling of factor 2
    (even though you could use a different resize)
    Args:
        in_channels (int):
        out_channels (int):
        num_layers (int):
        base (int):
        cell (str): type of rnn cell
    """

    def __init__(
            self, in_channels, out_channels, num_layers=3, base=4, cell='lstm', separable=False,
            separable_hidden=False, archi="all_rnn"):
        super().__init__()

        # out_channels for last encoder using NE Encoders = base * 2**NE
        downs = [base * 2**(i + 1) for i in range(num_layers)]
        ups = [base * 2**(num_layers - i) for i in range(num_layers)]

        print('down channels: ', downs)
        print('up channels: ', ups)
        print('archi: ', archi)

        down = VideoSequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False))
        up = MergeSkip()

        if archi == 'all_rnn':
            def enc_fun(x, y): return ConvRNN(x, y, cell=cell)
            def midd_fun(x, y): return VideoSequential(ResBlock(x, y), ResBlock(y, y))
            def dec_fun(x, y): return ConvRNN(x, y, cell=cell)
        else:
            raise NotImplementedError("archi not available")

        encoders, decoders = unet_layers(enc_fun, midd_fun, dec_fun, base, downs, ups[0] * 2, ups)
        self.unet = Unet(encoders, decoders, down, up)

        self.head = VideoSequential(ConvLayer(in_channels, base, 5, 1, 2))
        self.predictor = VideoSequential(nn.Conv2d(ups[-1], out_channels, 1, 1, 0))
        self.flow = VideoSequential(nn.Conv2d(ups[-1], 2, 1, 1, 0))

    def forward(self, x):
        y = self.head(x)
        y = self.unet(y)
        return y

    def predict_gray(self, y):
        return torch.sigmoid(self.predictor(y))

    def predict_flow(self, y):
        return self.flow(y)

    def reset(self, mask):
        for module in self.unet.modules():
            if hasattr(module, "reset"):
                module.reset(mask)
