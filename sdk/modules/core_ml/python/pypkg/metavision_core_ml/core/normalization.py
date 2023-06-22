# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class PositionalNorm(nn.Module):
    """
    Implementation of Positional norm (without moment shorcut)
    https://arxiv.org/pdf/1907.04312.pdf
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(PositionalNorm, self).__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True).add(self.eps).sqrt()
        out = (x - mean) / std
        if self.affine:
            out = self.weight.view(-1, 1, 1) * out + self.bias.view(-1, 1, 1)
        return out

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)


class LocalContextNorm2d(nn.Module):
    """
    Adapts pytorch's batchnorm implementation to do Local context/feature normalization https://arxiv.org/pdf/1912.05845.pdf
    Can be seen as extension of batch norm but mean and std are computed locally around each pixel
    """

    def __init__(self, num_features, window_size=(1, 9, 9), c_group=1, eps=1e-5, do_submatrick=False, affine=True):
        """
        Parameters
        ----------
            num_features : int
                total number of channels
            window_size : int/tuple
                size of the window around each pixel to normalize (better when odd numbers) default value not best
                one/two ints -> per channel spatial dims only norm
                three ints -> across channels and spatial dims
            c_group: int
                number of channels per group, if window size is three long, its first value takes precedence (for older trained models)
                with do_submatrick=False, this norm does not actually compute by group!
            eps: float
                small factor to avoid zero division
            do_submatrick: bool
                whether to use submatrixSumTrick in the computation
                This does not seem to work most of the time, use with c_group a multiple of 4 (so 4 or 8) for raw images seems to work
        """
        super(LocalContextNorm2d, self).__init__()
        self.num_features = num_features
        if isinstance(window_size, int) or len(window_size) == 1:
            w = window_size if isinstance(window_size, int) else window_size[0]
            self.window_size = (c_group, w, w)
        elif len(window_size) == 2:
            self.window_size = (c_group, *window_size)
        elif len(window_size) == 3:
            self.window_size = window_size
        else:
            raise ValueError(f"Window size for Local Context must be 3D at most, got {len(self.window_size)}")
        if self.window_size[0] > num_features:
            LOGGER.warning(
                f"Window size in channel dimension ({self.window_size[0]}) can't exceed total number of channels ({num_features})")
            LOGGER.warning(f"Reducing channel window size")
            self.window_size = (num_features, *self.window_size[1:])

        self.eps = eps
        self.affine = affine
        self.faster_comp = do_submatrick
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        dim = x.dim()
        if dim != 3 and dim != 4:
            raise ValueError(
                f"Input to LocalContextNorm must be of shape (B, C, H, W) or (C, H, W), got {dim}-dimensional input")
        if dim == 3:
            x.unsqueeze(0)
        B, C, H, W = x.shape

        # Checking for attribute to work with older saved models
        if not(hasattr(self, "faster_comp")) or not self.faster_comp:
            cur_window = (min(C, self.window_size[0]), min(H, self.window_size[1]), min(W, self.window_size[2]))
            # old behaviour without submatrix trick
            if cur_window[0] == 1:
                # Separately on each channel
                def pool(inp): return F.avg_pool2d(inp, (cur_window[1], cur_window[2]), padding=(
                    cur_window[1] // 2, cur_window[2] // 2), stride=1, count_include_pad=False)
            else:
                def pool(inp): return F.avg_pool3d(inp, cur_window,
                                                   padding=(cur_window[0] // 2, cur_window[1] // 2, cur_window[2] // 2),
                                                   stride=1, count_include_pad=False)

            mean = pool(x)
            # slice out extra values for even windows
            mean = mean[:, :C, :H, :W]
            sq_diff = torch.square(x - mean) + self.eps
            std = torch.sqrt(pool(sq_diff))
            std = std[:, :C, :H, :W]
            out = (x - mean) / std
        else:
            c_group = self.window_size[0]
            if c_group > C:
                LOGGER.warning(
                    f"Can't compute LCN with more channels per group (given {c_group}) than total channels ({C} here)")
                LOGGER.warning(f"Reverting to 1 group ({C} channels per group), equivalent to \"local\" Layer norm")
                c_group = C
            elif C % c_group:
                LOGGER.warning(f"Number of channels per group ({c_group}) must divide total number of channels ({C})")
                LOGGER.warning(f"Reverting to 1 channel per group ({C} groups), equivalent to \"local\" Instance Norm")
                LOGGER.warning(
                    f"Next time try it with 2 or 3 channels per group, or check the number of features in your model conv layers")
                c_group = 1
                self.window_size = (1, *self.window_size[1:])
            if self.window_size[1] >= H or self.window_size[2] >= W:
                g = C // c_group
                x = x.view(B, g, -1)
                mean = x.mean(-1, keepdim=True)
                var = x.var(-1, keepdim=True)
                out = (x - mean) / (var + self.eps).sqrt()
                out = out.view(B, C, H, W)
            else:
                out = submatrixSumTrick(x, self.window_size[1:], c_group, self.eps)

        if self.affine:
            out = self.weight.view(-1, 1, 1) * out + self.bias.view(-1, 1, 1)
        if dim == 3:
            out.squeeze(0)
        return out

    def extra_repr(self):
        return '{window_size}, num_features={num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)
