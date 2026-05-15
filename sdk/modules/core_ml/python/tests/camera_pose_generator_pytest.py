# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for the camera pose generator
"""
import os
import numpy as np
import cv2

import metavision_core_ml.data.camera_poses as cam


def pytestcase_interpolation():
    max_frames = 100
    height = 480
    width = 640
    opt_flow_threshold = 1.0
    K = np.array(
        [[width / 2, 0, width / 2], [0, height / 2, height / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    Kinv = np.linalg.inv(K)

    nt = np.array([0, 0, 1], dtype=np.float32).reshape(1, 3)

    np.random.seed(0)
    for _ in range(10):
        signal = cam.generate_smooth_signal(6, max_frames//10).T
        rvecs = signal[:, :3]
        tvecs = signal[:, 3:]

        depth = np.random.uniform(1.0, 2.0)
        rvecs, tvecs, times, max_flow = cam.interpolate_poses(
            rvecs, tvecs, nt, depth, K, Kinv, height, width, opt_flow_threshold=opt_flow_threshold,
            max_frames_per_bin=-1)

        _, _, _, max_flow2 = cam.interpolate_poses(rvecs, tvecs, nt, depth, K, Kinv, height, width)

        assert max_flow2 <= opt_flow_threshold


def pytestcase_generate_pose_equivalence():
    max_frames = 100
    height = 480
    width = 640
    opt_flow_threshold = 1.0
    K = np.array(
        [[width / 2, 0, width / 2], [0, height / 2, height / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    Kinv = np.linalg.inv(K)
    nt = np.array([0, 0, 1], dtype=np.float32).reshape(1, 3)
    signal = cam.generate_smooth_signal(6, max_frames//10).T
    rvecs = signal[:, :3]
    tvecs = signal[:, 3:]

    depth = np.random.uniform(1.0, 2.0)
    homographies = cam.generate_homographies(rvecs, tvecs, nt, depth)

    for i in range(len(rvecs)):
        rvec, tvec = rvecs[i], tvecs[i]
        homography = cam.generate_homography(rvec, tvec, nt, depth)
        assert np.allclose(homography, homographies[i])


def pytestcase_generate_transform_equivalence():
    max_frames = 100
    height = 480
    width = 640
    opt_flow_threshold = 1.0
    K = np.array(
        [[width / 2, 0, width / 2], [0, height / 2, height / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    Kinv = np.linalg.inv(K)
    nt = np.array([0, 0, 1], dtype=np.float32).reshape(1, 3)
    signal = cam.generate_smooth_signal(6, max_frames//10).T
    rvecs = signal[:, :3]
    tvecs = signal[:, 3:]

    depth = np.random.uniform(1.0, 2.0)

    # checking math
    i = 0
    j = len(rvecs)-1
    h_0_1 = cam.generate_homography(rvecs[i], tvecs[i], nt, depth)
    h_0_2 = cam.generate_homography(rvecs[j], tvecs[j], nt, depth)
    h_1_2 = cam.get_transform(rvecs[i], tvecs[i], rvecs[j], tvecs[j], nt, depth)

    # check that 1->2 = 0->1->2
    h_0_2_bis = h_1_2.dot(h_0_1)
    rel_error = np.abs(h_0_2 - h_0_2_bis) / np.abs(h_0_2)
    assert np.allclose(h_0_2, h_0_2_bis), rel_error

    h_0_1 = cam.generate_image_homography(rvecs[i], tvecs[i], nt, depth, K, Kinv)
    h_0_2 = cam.generate_image_homography(rvecs[j], tvecs[j], nt, depth, K, Kinv)
    h_1_2 = cam.get_image_transform(rvecs[i], tvecs[i], rvecs[j], tvecs[j], nt, depth, K, Kinv)

    # check that 1->2 = 0->1->2
    h_0_2_bis = h_1_2.dot(h_0_1)
    rel_error = np.abs(h_0_2 - h_0_2_bis) / np.abs(h_0_2)
    assert np.allclose(h_0_2, h_0_2_bis), rel_error
