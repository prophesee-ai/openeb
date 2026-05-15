# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# pylint: disable=C0321

"""
we build several unets and time them
"""
import torch
import torch.nn as nn

from metavision_core_ml.core.temporal_modules import VideoSequential
from metavision_core_ml.core.unet import unet_layers, Unet
from metavision_core_ml.core.unet_utils import UpsampleConvLayer, DownsampleConvRNN, UpsampleConvRNN
from metavision_core_ml.core.modules import ConvLayer
from metavision_core_ml.utils.torch_ops import cuda_tick

from einops.layers.torch import Rearrange


def sep_enc_rnn_dec_ff(in_channels, out_channels, num_layers, base):
    downs = [base * 2**(i + 1) for i in range(num_layers)]
    ups = [base * 2**(num_layers - i) for i in range(num_layers)]
    ups[-1] = out_channels
    # Trial Architecture
    def down(x, y): return DownsampleConvRNN(x, y, separable=True, separable_hidden=True)
    def midd(x, y): return VideoSequential(ConvLayer(x, y, separable=True))
    def up(x, y): return UpsampleConvLayer(x, y, separable=True)
    enc, dec = unet_layers(down, midd, up, in_channels, downs, ups[0] * 2, ups)
    net = Unet(enc, dec)
    return net


def plain_enc_rnn_dec_ff(in_channels, out_channels, num_layers, base):
    downs = [base * 2**(i + 1) for i in range(num_layers)]
    ups = [base * 2**(num_layers - i) for i in range(num_layers)]
    ups[-1] = out_channels
    # Trial Architecture
    def down(x, y): return DownsampleConvRNN(x, y, separable=False, separable_hidden=False)
    def midd(x, y): return VideoSequential(ConvLayer(x, y, separable=False))
    def up(x, y): return UpsampleConvLayer(x, y, separable=False)
    enc, dec = unet_layers(down, midd, up, in_channels, downs, ups[0] * 2, ups)
    net = Unet(enc, dec)
    return net


def sep_enc_ff_dec_rnn(in_channels, out_channels, num_layers, base):
    downs = [base * 2**(i + 1) for i in range(num_layers)]
    ups = [base * 2**(num_layers - i) for i in range(num_layers)]
    ups[-1] = out_channels
    # Trial Architecture
    def down(x, y): return VideoSequential(ConvLayer(x, y, separable=True))
    def midd(x, y): return VideoSequential(ConvLayer(x, y, separable=True))
    def up(x, y): return UpsampleConvRNN(x, y, separable=True, separable_hidden=False)
    enc, dec = unet_layers(down, midd, up, in_channels, downs, ups[0] * 2, ups)
    net = Unet(enc, dec)
    return net


def plain_enc_ff_dec_rnn(in_channels, out_channels, num_layers, base):
    downs = [base * 2**(i + 1) for i in range(num_layers)]
    ups = [base * 2**(num_layers - i) for i in range(num_layers)]
    ups[-1] = out_channels
    # Trial Architecture
    def down(x, y): return VideoSequential(ConvLayer(x, y, separable=True))
    def midd(x, y): return VideoSequential(ConvLayer(x, y, separable=False))
    def up(x, y): return UpsampleConvRNN(x, y, separable=False, separable_hidden=False)
    enc, dec = unet_layers(down, midd, up, in_channels, downs, ups[0] * 2, ups)
    net = Unet(enc, dec)
    return net


def measure_runtime(t, b, c, h, w, net, device, n_iters=10):
    x = torch.randn(t, b, c, h, w)
    x = x.to(device)
    net.to(device)
    y = net(x)
    start = cuda_tick()
    for i in range(n_iters):
        y = net(x)
    end = cuda_tick()
    return (end-start)/n_iters


def benchmark(name, t=10, b=1, h=512, w=512, in_channels=5, out_channels=1, num_layers=3, base=8, device='cuda:0'):
    net = globals()[name](in_channels, out_channels, num_layers, base)
    runtime = measure_runtime(t, b, in_channels, h, w, net, device)
    print('runtime: ', runtime)


if __name__ == '__main__':
    import fire
    fire.Fire(benchmark)
