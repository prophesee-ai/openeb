# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
6-DOF motion in front of image plane and returns corners positions
All in numpy + OpenCV
Applies continuous homographies to your picture in time.
"""
from __future__ import absolute_import

import numpy as np
import cv2

from metavision_core_ml.data.image_planar_motion_stream import PlanarMotionStream
from metavision_core_ml.corner_detection.utils import get_harris_corners_from_image, project_points


class CornerPlanarMotionStream(PlanarMotionStream):
    """
    Generates a planar motion in front of the image, returning both images and Harris' corners

    Args:
        image_filename: path to image
        height: desired height
        width: desired width
        max_frames: number of frames to stream
        rgb: color images or gray
        infinite: border is mirrored
        pause_probability: probability of stream to pause
        draw_corners_as_circle: if true corners will be 2 pixels circles
    """

    def __init__(self, image_filename, height, width, max_frames=1000, rgb=False, infinite=True,
                 pause_probability=0.5, draw_corners_as_circle=True):
        super().__init__(image_filename, height, width, max_frames=max_frames, rgb=rgb, infinite=infinite,
                         pause_probability=pause_probability)
        self.iter = 0
        self.corners = get_harris_corners_from_image(self.frame)
        self.draw_corners_as_circle = draw_corners_as_circle
        if self.draw_corners_as_circle:
            self.image_of_corners = np.zeros((self.frame_height, self.frame_width))
            rounded_corners = np.round(self.corners).astype(np.int16)
            if len(rounded_corners) > 0:
                for x, y, z in rounded_corners:
                    cv2.circle(self.image_of_corners, (x, y), 2, (255, 255, 255), -1)

    def __next__(self):
        if self.iter >= len(self.camera):
            raise StopIteration

        G_0to2, ts = self.camera()

        corners = np.zeros((self.height, self.width))
        if len(self.corners) != 0:
            if self.draw_corners_as_circle:
                corners = cv2.warpPerspective(
                    self.image_of_corners,
                    G_0to2,
                    dsize=(self.frame_width, self.frame_height),
                    borderMode=self.border_mode,
                )
                corners = cv2.resize(corners, (self.width, self.height), 0, 0, cv2.INTER_AREA)
            else:
                projected_corners = project_points(self.corners,
                                                   G_0to2,
                                                   self.width,
                                                   self.height,
                                                   self.frame_width,
                                                   self.frame_height)
                corners_rounded = np.round(projected_corners).astype(np.int16)
                corners[corners_rounded[:, 1], corners_rounded[:, 0]] = 1
        out = cv2.warpPerspective(
            self.frame,
            G_0to2,
            dsize=(self.frame_width, self.frame_height),
            borderMode=self.border_mode,
        )
        out = cv2.resize(out, (self.width, self.height), 0, 0, cv2.INTER_AREA)

        self.iter += 1
        ts *= self.dt

        return out, corners, ts
