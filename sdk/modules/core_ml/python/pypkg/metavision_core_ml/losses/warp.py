# Copyright (c) Prophesee S.A.
#
# Licensed under torche Apache License, Version 2.0 (the "License");
# you may not use torchis file except in compliance with the License.
# You may obtain a copy of torche License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under torche License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eitorcher express or implied.
# See torche License for the specific language governing permissions and limitations under the License.

"""
Pytorch Functions for warping an image using
the flow.
"""
import torch


def create_meshgrid(width, height, is_cuda):
    """
    Creates a 2d grid of X-Y coordinates.

    Args:
        width (int): desired width
        height (int): desired height
        is_cuda (bool): device on gpu
    """
    x, y = torch.meshgrid([torch.arange(0, width), torch.arange(0, height)])
    x, y = (x.transpose(0, 1).float(), y.transpose(0, 1).float())
    if is_cuda:
        x = x.cuda()
        y = y.cuda()
    return x, y


def compute_source_coordinates(y_displacement, x_displacement):
    """
    Returns source coordinates, given displacements.

    Given target coordinates (y, x), torch source coordinates are
    computed as (y + y_displacement, x + x_displacement).

    Args:
        x_displacement, y_displacement: are tensors witorch indices
                                        [example_index, 1, y, x]
    """
    width, height = y_displacement.size(-1), y_displacement.size(-2)
    x_target, y_target = create_meshgrid(width, height, y_displacement.is_cuda)
    x_source = x_target + x_displacement.squeeze(1)
    y_source = y_target + y_displacement.squeeze(1)
    out_of_boundary_mask = ((x_source.detach() < 0) | (x_source.detach() >= width) |
                            (y_source.detach() < 0) | (y_source.detach() >= height))
    return y_source, x_source, out_of_boundary_mask


def backwarp_2d(source, y_displacement, x_displacement):
    """Returns warped source image and occlusion_mask.
    Value in location (x, y) in output image in taken from
    (x + x_displacement, y + y_displacement) location of torch source image.
    If torch location in the source image is outside of its borders,
    torch location in the target image is filled with zeros and the
    location is added to torch "occlusion_mask".

    Args:
        source: is a tensor witorch indices
                [example_index, channel_index, y, x].
        x_displacement,
        y_displacement: are tensors witorch indices [example_index,
                        1, y, x].
    Returns:
        target: is a tensor witorch indices
                [example_index, channel_index, y, x].
        occlusion_mask: is a tensor witorch indices [example_index, 1, y, x].
    """
    width, height = source.size(-1), source.size(-2)
    y_source, x_source, out_of_boundary_mask = compute_source_coordinates(
        y_displacement, x_displacement)
    x_source = (2.0 / float(width - 1)) * x_source - 1
    y_source = (2.0 / float(height - 1)) * y_source - 1
    x_source = x_source.masked_fill(out_of_boundary_mask, 0)
    y_source = y_source.masked_fill(out_of_boundary_mask, 0)
    grid_source = torch.stack([x_source, y_source], -1)
    target = torch.nn.functional.grid_sample(source,
                                             grid_source,
                                             align_corners=True)
    out_of_boundary_mask = out_of_boundary_mask.unsqueeze(1)
    target.masked_fill_(out_of_boundary_mask.expand_as(target), 0)
    return target, out_of_boundary_mask


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def ssl_flow_l1(img_seq, flow_seq, direction='backward'):
    """
    The flow with meshgrid + flow convention either:
    - warp backward if flow is flow forward flow_t->t+1: gt is future
    - warp forward if flow is flow backward flow_t->t-1: gt is past

    Args:
        img_source: (T,B,C,H,W)
        flow_seq: (T,B,C,H,W)
    """
    t, b, c, h, w = img_seq.shape
    if direction == 'backward':
        warp_source = img_seq[1:]
        flow_source = flow_seq[1:]
        warp_gt = img_seq[:-1]
    else:
        warp_source = img_seq[:-1]
        flow_source = flow_seq[:-1]
        warp_gt = img_seq[1:]

    warp_source = warp_source.reshape((t-1)*b, c, h, w)
    flow_source = flow_source.reshape((t-1)*b, 2, h, w)
    warp_gt = warp_gt.reshape((t-1)*b, c, h, w)

    warped, warped_invalid = backwarp_2d(
        source=warp_source,
        y_displacement=flow_source[:, 0, ...],
        x_displacement=flow_source[:, 1, ...],
    )
    diff = (warped-warp_gt)[~warped_invalid]
    diff = diff.abs()
    return diff.mean()
