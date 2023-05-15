# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Layers involving recursion or temporal aspects.
"""
import torch
import torch.nn as nn
from typing import Tuple
from functools import partial

from .modules import ConvLayer, DepthWiseSeparableConv2d


def time_to_batch(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Collapses a five dimensional Tensor to four dimensional tensor
    by putting sequence samples in the batch dimension.

    Args:
        x (torch.Tensor): (num_time_bins, batch_size, channel_count, height, width)

    Returns:
        x (torch.Tensor): shape (num_time_bins * batch_size, channel_count, height, width)
        batch_size (int): number of separate sequences part of the Tensor.
    """
    t, n, c, h, w = x.size()
    x = x.view(n * t, c, h, w)
    return (x, n)


def batch_to_time(x: torch.Tensor, n: int) -> torch.Tensor:
    """Reverts a 5 dimensional Tensor that has been collapsed with time_to_batch to its original form

    Args:
        x (torch.Tensor): shape (num_time_bins * batch_size, channel_count, height, width)
        batch_size (int): number of separate sequences part of the Tensor.

    Returns:
        x (torch.Tensor): shape (num_time_bins, batch_size, channel_count, height, width)
    """
    nt, c, h, w = x.size()
    time = nt // n
    x = x.view(time, n, c, h, w)
    return x


def seq_wise(function):
    """Decorator to apply 4 dimensional tensor functions on 5 dimensional temporal tensor input."""
    def sequence_function(x5, *args, **kwargs):
        x4, batch_size = time_to_batch(x5)
        y4 = function(x4, *args, **kwargs)
        return batch_to_time(y4, batch_size)
    return sequence_function


class SequenceWise(nn.Module):
    def __init__(self, module, ndims=5):
        """
        Wrapper Module that allows the wrapped Module to be applied on sequential Tensors
        of shape 5 (num_time_bins, batch_size, channel_count, height, width)

        Attributes:
            module (torch.nn.Module): Module to wrap to be able to apply non sequential model on
                tensor of 5 dimensions.

        Args:
            module (torch.nn.Module): Module to wrap to be able to apply non sequential model on
                tensor of 5 dimensions.
        """
        super(SequenceWise, self).__init__()
        self.ndims = ndims
        self.module = module

        if hasattr(self.module, "out_channels"):
            self.out_channels = self.module.out_channels

    def forward(self, x):
        if x.dim() == self.ndims - 1:
            return self.module(x)
        else:
            x4, batch_size = time_to_batch(x)
            y4 = self.module(x4)
            return batch_to_time(y4, batch_size)

    def __repr__(self):
        module_str = self.module.__repr__()
        str = f"{self.__class__.__name__} (\n{module_str})"
        return str


class VideoSequential(nn.Sequential):
    """
    Wrapper Module that allows to call a torch.nn.Sequential object
    on shape 5 (num_time_bins, batch_size, channel_count, height, width)

    Difference with SequenceWise is that this handles a list of module.
    You can build this like a Sequential Object.

    Example:
        >> video_net = VideoSequential(nn.Conv2d(3,16,3,1,1),
                                       nn.ReLU())
        >> t,b,c,h,w = 3,2,3,128,128
        >> x = torch.randn(t,b,c,h,w)
        >> y = video_net(x)
    """

    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        if x.dim() == 4:
            return super().forward(x)
        else:
            return seq_wise(super().forward)(x)


class RNNCell(nn.Module):
    """
    Abstract class that has memory. serving as a base class to memory layers.

    Args:
        hard (bool): Applies hard gates to memory updates function.
    """

    def __init__(self, hard):
        super(RNNCell, self).__init__()
        self.set_gates(hard)

    def set_gates(self, hard):
        if hard:
            self.sigmoid = self.hard_sigmoid
            self.tanh = self.hard_tanh
        else:
            self.sigmoid = torch.sigmoid
            self.tanh = torch.tanh

    def hard_sigmoid(self, x_in):
        x = x_in * 0.5 + 0.5
        y = torch.clamp(x, 0.0, 1.0)
        return y

    def hard_tanh(self, x):
        y = torch.clamp(x, -1.0, 1.0)
        return y

    def reset(self):
        raise NotImplementedError()


class ConvLSTMCell(RNNCell):
    """ConvLSTMCell module, applies sequential part of LSTM.

    LSTM with matrix multiplication replaced by convolution
    See Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
    (Shi et al.)

    Args:
        hidden_dim (int): number of output_channels of hidden state.
        kernel_size (int): internal convolution receptive field.
        conv_func (fun): functional that you can replace if you want to interact with your 2D state differently.
        hard (bool): applies hard gates.
    """

    def __init__(self, hidden_dim, kernel_size, conv_func=nn.Conv2d, hard=False):
        super(ConvLSTMCell, self).__init__(hard)
        self.hidden_dim = hidden_dim

        self.conv_h2h = conv_func(in_channels=self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=1,
                                  bias=True)

        self.register_buffer("prev_h", torch.zeros((1, self.hidden_dim, 0, 0)), persistent=False)
        self.register_buffer("prev_c", torch.zeros((1, self.hidden_dim, 0, 0)), persistent=False)
    @torch.jit.export
    def get_dims_NCHW(self):
        return self.prev_h.size()

    def forward(self, x):
        assert x.dim() == 5

        xseq = x.unbind(0)

        assert self.prev_h.size() == self.prev_c.size()

        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_h.size()
        input_N, input_C, input_H, input_W = xseq[0].size()
        assert input_C == 4 * hidden_C
        assert hidden_C == self.hidden_dim

        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            device = x.device
            self.prev_h = torch.zeros((input_N, self.hidden_dim, input_H, input_W)).type_as(x)
            self.prev_c = torch.zeros((input_N, self.hidden_dim, input_H, input_W)).type_as(x)
        self.prev_h.detach_()
        self.prev_c.detach_()

        result = []
        for t, xt in enumerate(xseq):
            assert xt.dim() == 4

            tmp = self.conv_h2h(self.prev_h) + xt

            cc_i, cc_f, cc_o, cc_g = torch.split(tmp, self.hidden_dim, dim=1)
            i = self.sigmoid(cc_i)
            f = self.sigmoid(cc_f)
            o = self.sigmoid(cc_o)
            g = self.tanh(cc_g)

            c = f * self.prev_c + i * g
            h = o * self.tanh(c)
            result.append(h.unsqueeze(0))

            self.prev_h = h
            self.prev_c = c

        res = torch.cat(result, dim=0)
        return res

    @torch.jit.export
    def reset(self, mask):
        """Sets the memory (or hidden state to zero), normally at the beginning of a new sequence.

        `reset()` needs to be called at the beginning of a new sequence. The mask is here to indicate which elements
        of the batch are indeed new sequences. """
        if self.prev_h.numel() == 0:
            return
        batch_size, _, _, _ = self.prev_h.size()
        if batch_size == len(mask):
            assert batch_size == mask.numel()
            mask = mask.reshape(-1, 1, 1, 1)
            assert mask.shape == torch.Size([len(self.prev_h), 1, 1, 1])
            self.prev_h.detach_()
            self.prev_c.detach_()
            self.prev_h = self.prev_h*mask.to(device=self.prev_h.device)
            self.prev_c = self.prev_c*mask.to(device=self.prev_c.device)

    @torch.jit.export
    def reset_all(self):
        """Resets memory for all sequences in one batch."""
        self.reset(torch.zeros((len(self.prev_h), 1, 1, 1)).type_as(self.prev_h))


class ConvGRUCell(RNNCell):
    """
    ConvGRUCell module, applies sequential part of the Gated Recurrent Unit.

    GRU with matrix multiplication replaced by convolution
    See Chung, Junyoung, et al. "Empirical evaluation of gated recurrent neural networks on sequence modeling.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output_channels of hidden state.
        kernel_size (int): internal convolution receptive field.
        padding (int): padding parameter for the convolution
        conv_func (fun): functional that you can replace if you want to interact with your 2D state differently.
        hard (bool): applies hard gates.

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, conv_func=nn.Conv2d, hard=False,
                 stride=1, dilation=1):
        super(ConvGRUCell, self).__init__(hard)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rz = conv_func(in_channels=self.in_channels + self.out_channels, out_channels=2 * self.out_channels,
                                 kernel_size=kernel_size, padding=1)
        self.conv_f = conv_func(in_channels=self.in_channels + self.out_channels, out_channels=self.out_channels,
                                kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.register_buffer("prev_h", torch.zeros((1, self.out_channels, 0, 0)), persistent=False)
    def forward(self, xt):
        """
        xt size: (T, B,C,H,W)
        return size: (T, B,C',H,W)
        """
        hidden_N, hidden_C, hidden_H, hidden_W = self.prev_h.size()
        input_N, input_C, input_H, input_W = xt[0].size()
        if hidden_N != input_N or hidden_H != input_H or hidden_W != input_W:
            self.prev_h = torch.zeros((input_N, self.out_channels, input_H, input_W)).type_as(xt)

        self.prev_h.detach_()

        result = []
        for xi in xt:
            assert xi.dim() == 4

            z, r = self.conv_rz(torch.cat((self.prev_h, xi), dim=1)).split(self.out_channels, 1)
            update_gate = self.sigmoid(z)
            reset_gate = self.sigmoid(r)

            f = self.conv_f(torch.cat((self.prev_h * reset_gate, xi), dim=1))
            input_gate = self.tanh(f)

            self.prev_h = (1 - update_gate) * self.prev_h + update_gate * input_gate

            result.append(self.prev_h)
        return torch.cat([r[None] for r in result], dim=0)

    @torch.jit.export
    def reset(self, mask):
        """Sets the memory (or hidden state to zero), normally at the beginning of a new sequence.

        `reset()` needs to be called at the beginning of a new sequence. The mask is here to indicate which elements
        of the batch are indeed new sequences. """
        batch_size, _, _, _ = self.prev_h.size()
        if batch_size == len(mask) and self.prev_h.device == mask.device:
            assert mask.shape == torch.Size([len(self.prev_h), 1, 1, 1])
            self.prev_h.detach_()
            self.prev_h = self.prev_h*mask.to(device=self.prev_h.device)


class ConvRNN(nn.Module):
    """ConvRNN module. ConvLSTM cell followed by a feed forward convolution layer.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): separable conv receptive field
        stride (int): separable conv stride.
        padding (int): padding.
        separable (boolean): if True, uses depthwise separable convolution for the forward convolutional layer.
        separable_hidden (boolean): if True, uses depthwise separable convolution for the hidden convolutional layer.
        cell (string): RNN cell type, currently gru and lstm only are supported.
        **kwargs: additional parameters for the feed forward convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 cell='lstm', separable=False, separable_hidden=False, **kwargs):
        assert cell.lower() in ('lstm', 'gru'), f"Only 'gru' or 'lstm' cells are supported, got {cell}"
        super(ConvRNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_x2h = SequenceWise(
            ConvLayer(in_channels, 4 * out_channels if cell == "lstm" else in_channels, kernel_size=kernel_size,
                      activation='Identity', stride=stride, padding=padding, dilation=dilation, separable=separable,
                      **kwargs))

        if separable_hidden:
            conv_hidden = partial(DepthWiseSeparableConv2d, stride=1)
        else:
            conv_hidden = nn.Conv2d

        if cell.lower() == "lstm":
            self.timepool = ConvLSTMCell(out_channels, 3, conv_func=conv_hidden)
        else:
            self.timepool = ConvGRUCell(in_channels, out_channels, kernel_size=3, conv_func=conv_hidden)

    def forward(self, x):
        y = self.conv_x2h(x)
        h = self.timepool(y)
        return h

    def reset(self, mask=torch.zeros((1,), dtype=torch.float32)):
        """Resets memory of the network."""
        self.timepool.reset(mask)

    @torch.jit.export
    def reset_all(self):
        self.timepool.reset_all()
