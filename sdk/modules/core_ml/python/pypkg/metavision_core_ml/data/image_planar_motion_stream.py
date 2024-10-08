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

from metavision_core_ml.data.camera_poses import CameraPoseGenerator
from metavision_core_ml.utils.files import is_image, is_video, is_tiff_image

import torch
from metavision_core_ml.utils.color_utils import from_linear_rgb
import random
import skimage.io


class PlanarMotionStream(object):
    """
    Generates a planar motion in front of the image

    Args:
        image_filename (str): path to image
        height (int): desired height
        width (int): desired width
        max_frames (int): number of frames to stream
        rgb (bool): color images or gray
        infinite (bool): border is mirrored
        pause_probability (float): probability to add a pause during the stream
        max_optical_flow_threshold (float): maximum optical flow between two consecutive frames
        max_interp_consecutive_frames (int): maximum number of interpolated frames between two consecutive frames
        crop_image (bool): crop images or resize them
        saturation_max_factor (float): multiplicative factor of saturated pixels (only for tiff 16 bits images. Use 1.0 to disable)
    """

    def __init__(self, image_filename, height, width, max_frames=1000, rgb=False, infinite=True,
                 pause_probability=0.5,
                 max_optical_flow_threshold=2., max_interp_consecutive_frames=20, crop_image=False,
                 saturation_max_factor=1.0):
        self.height = height
        self.width = width
        self.crop_image = crop_image
        self.max_frames = max_frames
        self.rgb = rgb
        self.filename = image_filename
        if is_tiff_image(image_filename):
            assert saturation_max_factor >= 1.0, f"Error: saturation_max_factor should be >= 1.0 for tiff images ({saturation_max_factor})"
            img_tiff = skimage.io.imread(image_filename)
            H, W, C = img_tiff.shape
            assert C == 3
            lrgb_frame_np = (img_tiff.transpose(2, 0, 1).astype(np.float32) / (2**16 - 1))[None].astype(np.float16)
            lrgb_frame = torch.from_numpy(lrgb_frame_np)
            B, C, H, W = lrgb_frame.shape
            assert B == 1 and C == 3

            lab_frame = from_linear_rgb(lrgb_frame, "lab")
            assert lab_frame.shape == (1, 3, H, W)
            mask_saturated = lab_frame[:, [0]] >= 1.
            nb_saturated_pixels = mask_saturated.sum()
            if nb_saturated_pixels > 0:
                lrgb_frame[0, :, mask_saturated.squeeze()] *= 1.0 + random.random() * (saturation_max_factor - 1.0)

            frame = lrgb_frame.numpy()
            assert frame.ndim == 4
            assert frame.shape == (B, C, H, W)
            frame = np.ascontiguousarray(frame.squeeze(axis=0).transpose(1, 2, 0))
            assert frame.shape == (H, W, C)
        elif is_image(image_filename):
            assert saturation_max_factor == 1.0, f"Error: saturation_max_factor should be 1.0 for non-tiff images ({saturation_max_factor})"
            if not self.rgb:
                frame = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
            else:
                frame = cv2.imread(image_filename)[..., ::-1]
        else:
            raise ValueError(f"Unhandled type of file: {image_filename} (should be image or npy image)")
        self.frame = frame
        self.frame_height, self.frame_width = self.frame.shape[:2]
        if self.height == -1 or self.width == -1:
            self.height, self.width = self.frame_height, self.frame_width
        self.camera = CameraPoseGenerator(self.frame_height, self.frame_width, self.max_frames, pause_probability,
                                          max_optical_flow_threshold=max_optical_flow_threshold,
                                          max_interp_consecutive_frames=max_interp_consecutive_frames)
        self.iter = 0
        self.border_mode = cv2.BORDER_REFLECT101 if infinite else cv2.BORDER_CONSTANT
        self.dt = np.random.randint(10000, 20000)
        self.xy1 = None

    def get_size(self):
        return (self.height, self.width)

    def pos_frame(self):
        return self.iter

    def __len__(self):
        return self.max_frames

    def __next__(self):
        if self.iter >= len(self.camera):
            raise StopIteration

        G_0to2, ts = self.camera()

        out = cv2.warpPerspective(
            self.frame,
            G_0to2,
            dsize=(self.frame_width, self.frame_height),
            borderMode=self.border_mode,
        )
        self.iter += 1
        ts *= self.dt

        if self.crop_image and out.shape[0] >= self.height and out.shape[1] >= self.width:
            margin_height_top = int((out.shape[0] - self.height) // 2.0)
            margin_height_bottom = (out.shape[0] - self.height) - margin_height_top
            margin_width_left = int((out.shape[1] - self.width) // 2.0)
            margin_width_right = (out.shape[1] - self.width) - margin_width_left
            out = out[margin_height_top:-margin_height_bottom or None,
                      margin_width_left:-margin_width_right or None]
        else:
            out = cv2.resize(out, (self.width, self.height), 0, 0, cv2.INTER_AREA)
        if self.rgb:
            assert out.shape == (self.height, self.width, 3)
        else:
            assert out.shape == (self.height, self.width)
        return out, ts

    def __iter__(self):
        return self

    def get_relative_homography(self, time_step):
        rvec1, tvec1 = self.camera.rvecs[time_step], self.camera.tvecs[time_step]
        rvec2, tvec2 = self.camera.rvecs[self.iter-1], self.camera.tvecs[self.iter-1]
        H_2_1 = self.camera.get_transform(rvec2, tvec2, rvec1, tvec1, self.height, self.width)
        return H_2_1
