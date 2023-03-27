# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Miscellaneous operations in Pytorch
"""
import os
import numpy as np
import torch
import time
import math
import kornia


def cuda_tick():
    """
    Measures time for torch
    operations on gpu.
    """
    torch.cuda.synchronize()
    return time.time()


def cuda_time(func):
    """
    Decorator for Pytorch ops
    on gpu.

    Args:
        func: method to time
    """
    def wrapper(*args, **kwargs):
        start = cuda_tick()
        out = func(*args, **kwargs)
        end = cuda_tick()
        rt = end - start
        freq = 1. / rt
        if freq > 0:
            print(freq, ' it/s @ ', func)
        else:
            print(rt, ' s/it @ ', func)
        return out
    return wrapper


def normalize_tiles(tensor, num_stds=6, num_dims=2, real_min_max=True):
    """
    Normalizes tiles, allows us to have normalized views
    (we filter outliers + standardize)

    Args:
        tensor: tensor input, we assume last 2 dims are H,W
    Returns:
        tensor: normalized tensor
    """
    shape = tensor.shape[:-num_dims]
    trflat = tensor.view(*shape, -1)

    mu, std = trflat.mean(dim=-1), trflat.std(dim=-1) * num_stds
    mu = mu[(...,) + (None,) * num_dims]
    std = std[(...,) + (None,) * num_dims]

    low, high = mu - std, mu + std

    tensor = torch.min(tensor, high)
    tensor = torch.max(tensor, low)

    if real_min_max:
        trflat = tensor.view(*shape, -1)
        low, high = trflat.min(dim=-1)[0], trflat.max(dim=-1)[0]
        low = low[(...,) + (None,) * num_dims]
        high = high[(...,) + (None,) * num_dims]

    return (tensor - low) / (high - low + 1e-5)


def viz_flow(flow):
    """
    Visualizes flow in rgb colors

    Args:
        flow: (B,2,H,W) tensor
    Returns:
        rgb: (B,3,H,W) tensor
    """
    b, c, h, w = flow.shape
    mag = torch.linalg.norm(flow, dim=1)
    ang = torch.atan2(flow[:, 1], flow[:, 0])
    mask = ang < 0
    ang[mask] = ang[mask] + 2*math.pi  # ang<0 -> [pi;2*pi]
    hsv = torch.zeros((b, 3, h, w), dtype=torch.float32, device=flow.device)
    hsv[:, 0] = ang
    hsv[:, 1] = 1
    hsv[:, 2] = mag / mag.max()
    rgb = kornia.color.hsv_to_rgb(hsv)
    return (255*rgb).byte()
