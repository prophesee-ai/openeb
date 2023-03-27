# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
util functions for color space conversions
"""

import numpy as np
import torch
from kornia.color import yuv_to_rgb, rgb_to_yuv, rgb_to_grayscale, linear_rgb_to_rgb, rgb_to_linear_rgb, raw_to_rgb, CFA


BAYER2KORNIA_DICT = {'RGGB': 0, 'GRBG': 1, 'BGGR': 2,
                     'GBRG': 3}


def get_num_channels(input_colorspace):
    """
    util function returning number of channels of an image
    for a given color space
        Args:
            input_colorspace (string):
    """
    if input_colorspace in ['linear_rgb', 'srgb', 'yuv', 'lab']:
        return 3
    elif input_colorspace in ['linear_gray', 'raw']:
        return 1
    else:
        raise NotImplementedError(f'unsupported color space: {input_colorspace}')


def bayer2kornia(bayer_pattern):
    """
    convert human readable bayer pattern string (e.g. 'BGGR') to kornia enum
        Args:
            bayer_pattern (string), one among RGGB, GRBG, BGGR, GBRG

    """
    bayer_pattern_up = bayer_pattern.upper()
    assert bayer_pattern_up in ['RGGB', 'GRBG', 'BGGR',
                                'GBRG'], 'kornia only supports 4 types of bayer pattern: RGGB, GRBG, BGGR, GBRG'

    return BAYER2KORNIA_DICT[bayer_pattern_up]


def raw_numpy_to_linear_rgb(input_image, device, bayer_pattern='BGGR', bpp=10):
    """
    convert a numpy raw image to linear rgb
    output values are in [0,255]
    """
    kornia_cfa = bayer2kornia(bayer_pattern)
    cfa = CFA(kornia_cfa)
    max_val_raw_bayer = 2**(bpp) - 1

    # load raw image
    image_th = torch.from_numpy(input_image.astype(np.float32)).to(device)
    assert (image_th <= max_val_raw_bayer).all(), "The bpp is wrong, the image contains larger values"
    # convert bayer to rgb
    image_rgb = raw_to_rgb(image_th[None, None] / max_val_raw_bayer, cfa)

    # If train images are gamma encoded, we should do the same here
    # from now on we assume train images are in linear domain
    # if do_gamma_encoding:
    #     gamma = 2.2  # TODO set as parameter
    #     image_rgb = (torch.pow(image_rgb[0] / 1024, 1/gamma)*255)[None]

    # NB in eval_slow_mo_kpis.py, images for kpis are converted to RAW format, here we keep rgb
    image_rgb = ((image_rgb[0])*255)[None]

    return image_rgb


# rgb to xyz and lab functions adapted from here:
# https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py

# Color conversion code
def rgb_to_xyz(rgb, is_linear_rgb=False):
    """
    converts tensor of rgb images (N,3,H,W) to xyz color space
    rgb values should be in [0,1], otherwise they are clipped
    if is_linear_rgb = True, it is assumed rgb values are linear (i.e. gamma decoded)
    otherwise will apply decoding using srgb standard: https://en.wikipedia.org/wiki/SRGB
    """
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
    # [0.212671, 0.715160, 0.072169],
    # [0.019334, 0.119193, 0.950227]])

    rgb = torch.clamp(rgb, 0., 1.)
    if not is_linear_rgb:
        rgb = rgb_to_linear_rgb(rgb)
        # mask = (rgb > .04045).type(torch.FloatTensor)
        # if(rgb.is_cuda):
        #     mask = mask.cuda()

        # rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:, 0, :, :]+.357580*rgb[:, 1, :, :]+.180423*rgb[:, 2, :, :]
    y = .212671*rgb[:, 0, :, :]+.715160*rgb[:, 1, :, :]+.072169*rgb[:, 2, :, :]
    z = .019334*rgb[:, 0, :, :]+.119193*rgb[:, 1, :, :]+.950227*rgb[:, 2, :, :]
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)

    return out


def xyz_to_rgb(xyz, return_linear_rgb=False):
    """
    converts tensor of xyz images (N,3,H,W) to rgb color space
    if return_linear_rgb = True, will return rgb linear values (i.e. gamma decoded)
    otherwise will apply gamma encoding using srgb standard: https://en.wikipedia.org/wiki/SRGB
    N.B. output values are clipped to [0,1]
    """
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134*xyz[:, 0, :, :]-1.53715152*xyz[:, 1, :, :]-0.49853633*xyz[:, 2, :, :]
    g = -0.96925495*xyz[:, 0, :, :]+1.87599*xyz[:, 1, :, :]+.04155593*xyz[:, 2, :, :]
    b = .05564664*xyz[:, 0, :, :]-.20404134*xyz[:, 1, :, :]+1.05731107*xyz[:, 2, :, :]

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1).clamp_(
        0., 1.)  # clip since can reach small negative number, which causes NaNs

    if not return_linear_rgb:
        # mask = (rgb > .0031308).type(torch.FloatTensor)
        # if(rgb.is_cuda):
        #     mask = mask.cuda()

        # rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)
        rgb = linear_rgb_to_rgb(rgb)

    return rgb


def xyz_to_lab(xyz):
    """
    converts tensor of xyz images (N,3,H,W) to Lab color space
    it uses D65 white point
    https://en.wikipedia.org/wiki/CIELAB_color_space
    """
    # 0.95047, 1., 1.08883 for D65 white point
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:, 1, :, :]-16.
    a = 500.*(xyz_int[:, 0, :, :]-xyz_int[:, 1, :, :])
    b = 200.*(xyz_int[:, 1, :, :]-xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)

    return out


def lab_to_xyz(lab):
    """
    converts tensor of lab images (N,3,H,W) to xyz color space
    it uses D65 white point
    https://en.wikipedia.org/wiki/CIELAB_color_space
    """
    y_int = (lab[:, 0, :, :]+16.)/116.
    x_int = (lab[:, 1, :, :]/500.) + y_int
    z_int = y_int - (lab[:, 2, :, :]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out*sc

    return out


def rgb_to_lab(rgb, is_linear_rgb=False):
    """
    converts tensor of rgb images (N,3,H,W) to lab color space
    input tensor should be in [0,1], output tensor is normalized in [0,1] for L
    and [-1,1] for a and b
    (NB a,b values are theoretically unbounded, so values outside [-1,1] are clipped)
    """

    lab = xyz_to_lab(rgb_to_xyz(rgb, is_linear_rgb))
    l_cent = 0
    l_norm = 100  # normalize in [0,1] for L
    ab_norm = 127  # normalize in [-1,1] for ab
    l_rs = (lab[:, [0], :, :]-l_cent)/l_norm
    # l_rs = torch.clamp_(l_rs, 0, 1)
    ab_rs = lab[:, 1:, :, :]/ab_norm
    ab_rs = torch.clamp_(ab_rs, -1, 1)
    out = torch.cat((l_rs, ab_rs), dim=1)
    return out


def lab_to_rgb(lab_rs, return_linear_rgb=False):
    """
    converts tensor of Lab images (N,3,H,W) to rgb color space
    input tensor L should be in [0,1], a and b in [-1,1],
    values outside this range will be clamped
    output tensor is in [0,1]
    """
    l_cent = 0
    l_norm = 100
    ab_norm = 127
    l = lab_rs[:, [0], :, :]*l_norm + l_cent  # this is in [0,1], map to [0,100]
    l = torch.clamp_(l, 0, 100)
    ab = lab_rs[:, 1:, :, :]*ab_norm  # this is in [-1,1], map to [-127,127]
    ab = torch.clamp_(ab, -ab_norm, ab_norm)
    lab = torch.cat((l, ab), dim=1)
    out = xyz_to_rgb(lab_to_xyz(lab), return_linear_rgb)
    return out


def from_linear_rgb(input_tensor, output_colorspace):
    """
    convert rgb images to target color space
    input_tensor: torch tensor (N,3,H,W) of RGB images, with values in [0,1]
    output_colorspace: string, target color space (for now only 'yuv' and 'lab' is supported)
    """
    N, C, H, W = input_tensor.shape
    assert C == 3
    assert input_tensor.min() >= 0
    assert input_tensor.max() <= 1
    if output_colorspace == 'linear_rgb':
        return input_tensor
    if output_colorspace == 'linear_gray':
        return rgb_to_grayscale(input_tensor, rgb_weights=torch.tensor([0.2126, 0.7152, 0.0722]))
    if output_colorspace == 'srgb':
        return linear_rgb_to_rgb(input_tensor)
    elif output_colorspace == 'yuv':
        # NB: yuv cspace requires non=linear rgb input
        return rgb_to_yuv(linear_rgb_to_rgb(input_tensor))
    elif output_colorspace == 'lab':
        # FIXME: here we could use kornia but it would mean 3 useless calls to linear_rgb_to_rgb/rgb_to_linear_rgb
        return rgb_to_lab(input_tensor, is_linear_rgb=True)
    else:
        raise NotImplementedError(f'unsupported color space: {output_colorspace}')


def to_linear_rgb(input_tensor, input_colorspace):
    """
    convert color images to linear rgb color space 
    (for now only 'srgb', 'yuv' and 'lab' input is supported)
    input_tensor: torch tensor (N,C,H,W)
        for yuv input, Y channel should be in [0,1], while U and V in [-0.5, 0.5]
        for lab input, L channel should be in [0,1], while a and b in [-1, 1]
    input_colorspace: string, color space of input images
    """
    if input_colorspace == 'linear_rgb':
        return input_tensor
    if input_colorspace == 'linear_gray':  # just replicate along channels
        return input_tensor.repeat([1, 3, 1, 1])
    elif input_colorspace == 'srgb':
        return rgb_to_linear_rgb(input_tensor)
    elif input_colorspace == 'yuv':
        # NB: if values are outside range, they are clipped
        y = torch.clamp_(input_tensor[:, [0], :, :], 0, 1)
        uv = torch.clamp_(input_tensor[:, 1:, :, :], -0.5, 0.5)
        yuv_tensor = torch.cat((y, uv), dim=1)
        # For small values of Y' it is possible to get R, G, or B values that are negative
        srgb_tensor = yuv_to_rgb(yuv_tensor).clamp_(0, 1)
        return rgb_to_linear_rgb(srgb_tensor)
    elif input_colorspace == 'lab':
        # NB: if values are outside range, they are clipped
        l = torch.clamp_(input_tensor[:, [0], :, :], 0, 1)
        ab = torch.clamp_(input_tensor[:, 1:, :, :], -1, 1)
        lab_tensor = torch.cat((l, ab), dim=1)
        # FIXME: here we could use kornia but it would mean 3 useless calls to linear_rgb_to_rgb/rgb_to_linear_rgb
        return lab_to_rgb(lab_tensor, return_linear_rgb=True)
    else:
        raise NotImplementedError(f'unsupported color space: {input_colorspace}')


def to_srgb(input_tensor, input_colorspace):
    """
    convert color images to srgb color space 
    (for now only 'linear_rgb', 'yuv' and 'lab' input are supported)
    input_tensor: torch tensor (N,C,H,W)
        for yuv input, Y channel should be in [0,1], while U and V in [-0.5, 0.5]
        for lab input, L channel should be in [0,1], while a and b in [-1, 1]
    input_colorspace: string, color space of input images
    NB: this function is calling to_linear_rgb followed by linear_rgb_to_rgb
    for some color spaces, such as YUV, it might be inefficeint
    """
    if input_colorspace == 'srgb':
        return input_tensor
    else:
        return linear_rgb_to_rgb(to_linear_rgb(input_tensor, input_colorspace))


def from_srgb(input_tensor, output_colorspace):
    """
    convert color images from srgb color space 
    (for now only 'linear_rgb', 'yuv' and 'lab' input are supported)
    input_tensor: torch tensor (N,C,H,W)
        for yuv input, Y channel should be in [0,1], while U and V in [-0.5, 0.5]
        for lab input, L channel should be in [0,1], while a and b in [-1, 1]
    output_colorspace: string, color space of input images
    NB: this function is calling rgb_to_linear_rgb followed by from_linear_rgb
    for some color spaces, such as YUV, it might be inefficeint 
    """
    if output_colorspace == 'srgb':
        return input_tensor
    else:
        return from_linear_rgb(rgb_to_linear_rgb(input_tensor), output_colorspace)


def linear_rgb_to_gray(input_tensor):
    """
    convert linear RGB images to linear Y
    """
    return from_linear_rgb(input_tensor, 'linear_gray')


def srgb_to_gray(input_tensor):
    """
    convert sRGB images to non-linear Y'
    """
    # in this case kornia uses rgb_weights = [0.299, 0.587, 0.114]
    return rgb_to_grayscale(input_tensor)
