# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Base unet code
U-Net: Convolutional Networks for Biomedical Image Segmentation
Olaf Ronneberger, Philipp Fischer, Thomas Brox

Notes:
- User is responsible for creating the layers,
they should have in_channels, out_channels in argument (they must be pre-filled)
- User is responsible for making sure spatial sizes agree.
"""
import torch
import torch.nn as nn


def unet_layers(
    down_block,
    middle_block,
    up_block,
    input_size=5,
    down_filter_sizes=[
        32,
        64],
        middle_filter_size=128,
        up_filter_sizes=[
            64,
            32,
        8]):
    """Builds unet layers
    can be used to build unet layers (but you are not forced to)

    Here we make sure to connect the last upsampled feature-map to the input!

    X         -        Y = Conv([X,Up(U2)])
     \                /
      D1      -      U2 = Conv([D1,Up(U1)])
       \             /
        D2    -     U1 = Conv([D2,Up(M)])
         \         /
          D3  -   M


    All block types are partial functions expecting in_channels, out_channels as first two parameters.

    Args:
        down_block: encoder's block type
        middle_block: bottleneck's block type
        up_block: decoder's block type
        input_size: in_channels
        down_filter_sizes: out_channels per encoder
        middle_filter_size: bottleneck's channels
        up_fitler_sizes: decoder's channels
    """
    assert len(down_filter_sizes) <= len(up_filter_sizes)
    encoders = []
    encoders_channels = [input_size]
    last_channels = input_size
    for cout in down_filter_sizes:
        enc = down_block(encoders_channels[-1], cout)
        encoders.append(enc)
        encoders_channels.append(cout)

    middle = middle_block(encoders_channels[-1], middle_filter_size)
    decoders = [middle]
    decoders_channels = [middle_filter_size]
    for i, cout in enumerate(up_filter_sizes):
        # note index = -2! last encoder is at the same scale of current input, this is not what we want!
        cin = decoders_channels[-1] + encoders_channels[-i - 2]
        decoders.append(up_block(cin, cout))
        decoders_channels.append(cout)

    return encoders, decoders


class Unet(nn.Module):
    """Ultra-Generic Unet

    Args:
        encoders: list of encoder layers
        decoders: list of decoder layers
        down_layer: layer to resize input
        up_layer: layer to resize + merge
    """

    def __init__(self, encoders, decoders, down, up):
        super(Unet, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.up = up
        self.down = down

    def forward(self, x):
        enc = [x]
        for encoder in self.encoders:
            x = self.down(x)
            x = encoder(x)
            enc.append(x)

        dec = [self.decoders[0](x)]
        for i, decoder in enumerate(self.decoders[1:]):
            skip = enc[-i - 2]
            x = self.up(dec[-1], skip)
            x = decoder(x)
            dec.append(x)

        return dec[-1]
