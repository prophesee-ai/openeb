# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
6-DOF motion in front of image plane
All in numpy + OpenCV
Applies continuous homographies to your picture in time.
Also you can get the optical flow for this motion.
"""
from __future__ import absolute_import

import numpy as np
import cv2
import rawpy
import kornia
import torch
import random

from metavision_core_ml.data.camera_poses import CameraPoseGenerator
from kornia.color import raw_to_rgb, rgb_to_raw, CFA
from metavision_core_ml.utils.color_utils import bayer2kornia


class RawPlanarMotionStream(object):
    """
    Generates a planar motion in front of the raw image

    Returns the image as 4 channels unshuffled H/2, W/2

    Args:
        image_filename (str): path to raw image
        height (int): expected height (or crop). Must be a even
        width (int): expected width (or crop). Must be even
        max_frames (int): number of frames to stream
        bayer_pattern (str): type of bayer pattern
        bpp (int): number of bits per pixel
        black_level (int): black level
        pause_probability (float): probability to add a pause during the stream
        max_optical_flow_threshold (float): maximum optical flow between two consecutive frames
        max_interp_consecutive_frames (int): maximum number of interpolated frames between two consecutive frames
    """

    def __init__(self, image_filename, height, width, max_frames=1000,
                 bayer_pattern="BGGR", bpp=10, black_level=0,
                 pause_probability=0.5,
                 max_optical_flow_threshold=1., max_interp_consecutive_frames=20):
        assert height % 2 == 0
        assert width % 2 == 0
        self.height_unshuffled = height // 2
        self.width_unshuffled = width // 2
        self.max_frames = max_frames
        self.bayer_pattern = bayer_pattern
        self.bpp = bpp
        self.black_level = black_level
        self.max_val_raw = 2**bpp - 1
        self.filename = image_filename
        raw_frame = rawpy.imread(image_filename).raw_image
        assert raw_frame.min() >= 0
        assert raw_frame.max() <= self.max_val_raw
        raw_frame = np.clip(raw_frame, black_level, self.max_val_raw) - black_level

        H, W = raw_frame.shape
        margin_h, margin_w = (H - height) // 2, (W - width) // 2
        assert margin_h >= 0
        assert margin_w >= 0

        self.width = width
        self.height = height

        offset_h, offset_w = random.randint(0, margin_h) * 2, random.randint(0, margin_w) * 2
        self.offset_h = offset_h
        self.offset_w = offset_w
        raw_frame = raw_frame[offset_h:offset_h + height, offset_w:offset_w + width]

        assert raw_frame.shape == (2 * self.height_unshuffled, 2 * self.width_unshuffled)
        raw_frame_th = torch.from_numpy(raw_frame.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0).float() / self.max_val_raw

        self.pixel_unshuffle = torch.nn.PixelUnshuffle(2)

        self.cfa = kornia.color.CFA(bayer2kornia(bayer_pattern))
        self.frame_rgb = raw_to_rgb(raw_frame_th, self.cfa)
        assert self.frame_rgb.shape == (1, 3, self.height, self.width)

        self.camera = CameraPoseGenerator(self.height_unshuffled, self.width_unshuffled, self.max_frames, pause_probability,
                                          max_optical_flow_threshold=max_optical_flow_threshold,
                                          max_interp_consecutive_frames=max_interp_consecutive_frames)
        self.iter = 0
        self.dt = np.random.randint(10000, 20000)


    def get_unshuffled_size(self):
        """
        Returns the size of the unshuffled raw frame: (1, 4, H / 2, W / 2), where (H, W) is the size of the initial raw
        """
        return (self.height_unshuffled, self.width_unshuffled)

    def get_shuffled_size(self):
        """
        Returns the size of the shuffled (initial) raw: (1, 1, H, W)
        """
        assert self.height_unshuffled == self.height // 2
        assert self.width_unshuffled == self.width // 2
        return (2 * self.height_unshuffled, 2 * self.width_unshuffled)

    def pos_frame(self):
        return self.iter

    def __len__(self):
        return self.max_frames

    def __next__(self):
        if self.iter >= len(self.camera):
            raise StopIteration

        G_0to2, ts = self.camera()
        G_0to2_th = torch.from_numpy(G_0to2).unsqueeze(dim=0).float()


        out_rgb = kornia.geometry.warp_perspective(src=self.frame_rgb, M=G_0to2_th,
                                                dsize=(self.height, self.width),
                                                mode="bilinear", padding_mode="reflection")
        assert out_rgb.shape == (1, 3, self.height, self.width)

        out_raw = rgb_to_raw(out_rgb, self.cfa)
        assert out_raw.shape == (1, 1, self.height, self.width)
        out = self.pixel_unshuffle(out_raw)
        assert out.shape == (1, 4, self.height_unshuffled, self.width_unshuffled)

        self.iter += 1
        ts *= self.dt

        return out, ts

    def __iter__(self):
        return self

    def get_relative_homography(self, time_step):
        rvec1, tvec1 = self.camera.rvecs[time_step], self.camera.tvecs[time_step]
        rvec2, tvec2 = self.camera.rvecs[self.iter-1], self.camera.tvecs[self.iter-1]
        H_2_1 = self.camera.get_transform(rvec2, tvec2, rvec1, tvec1, self.height_unshuffled, self.width_unshuffled)
        return H_2_1
