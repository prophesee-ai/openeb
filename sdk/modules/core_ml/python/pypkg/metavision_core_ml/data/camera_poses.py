# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Camera Pose Generator:

This module allows to define a trajectory of camera poses
and generate continuous homographies by interpolating when
maximum optical flow is beyond a predefined threshold.
"""
from __future__ import absolute_import

import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from numba import jit

DTYPE = np.float32


@jit
def interpolate_times_tvecs(tvecs, key_times, inter_tvecs, inter_times, nums):
    """
    Interpolates between key times times & translation vectors

    Args:
        tvecs (np.array): key translation vectors  (N, 3)
        key_times (np.array): key times (N, )
        inter_tvecs (np.array): interpolated translations (nums.sum(), 3)
        inter_times (np.array): interpolated times (nums.sum(),)
        nums (np.array): number of interpolation point between key points (N-1,)
                         nums[i] is the number of points between key_times[i] (included) and key_times[i+1] (excluded)
                         minimum is 1, which corresponds to key_times[i]
    """
    n = 0
    for i in range(nums.shape[0]):
        num = nums[i]
        tvec1 = tvecs[i]
        tvec2 = tvecs[i + 1]
        time1 = key_times[i]
        time2 = key_times[i + 1]
        for j in range(num):
            a = j / num
            ti = time1 * (1 - a) + a * time2
            inter_times[n] = ti
            inter_tvecs[n] = tvec1 * (1 - a) + a * tvec2
            n += 1


def generate_homography(rvec, tvec, nt, depth):
    """
    Generates a single homography

    Args:
        rvec (np.array): rotation vector
        tvec (np.array): translation vector
        nt (np.array): normal to camera
        depth (float): depth to camera
    """
    R = cv2.Rodrigues(rvec)[0].T
    H = R - np.dot(tvec.reshape(3, 1), nt) / depth
    return H


def generate_image_homography(rvec, tvec, nt, depth, K, Kinv):
    """
    Generates a single image homography

    Args:
        rvec (np.array): rotation vector
        tvec (np.array): translation vector
        nt (np.array): normal to camera
        depth (float): depth
        K (np.array): intrisic matrix
        Kinv (np.array): inverse intrinsic matrix
    """
    H = generate_homography(rvec, tvec, nt, depth)
    G = np.dot(K, np.dot(H, Kinv))
    G /= G[2, 2]
    return G


def generate_homographies_from_rotation_matrices(rot_mats, tvecs, nt, depth):
    """
    Generates multiple homographies from rotation matrices

    Args:
        rot_mats (np.array): N,3,3 rotation matrices
        tvecs (np.array): N,3 translation vectors
        nt (np.array): normal to camera
        depth (float): depth to camera
    """
    rot_mats = np.moveaxis(rot_mats, 2, 1)
    t = np.einsum('ik,jd->ikd', tvecs, nt)
    h = rot_mats - t / depth
    return h


def generate_homographies(rvecs, tvecs, nt, d):
    """
    Generates multiple homographies from rotation vectors

    Args:
        rvecs (np.array): N,3 rotation vectors
        tvecs (np.array): N,3 translation vectors
        nt (np.array): normal to camera
        d (float): depth
    """
    rot_mats = R.from_rotvec(rvecs).as_matrix()
    return generate_homographies_from_rotation_matrices(rot_mats=rot_mats, tvecs=tvecs, nt=nt, depth=d)


def generate_image_homographies_from_homographies(h, K, Kinv):
    """
    Multiplies homography left & right by intrinsic matrix

    Args:
        h (np.array): homographies N,3,3
        K (np.array): intrinsic
        Kinv (np.ndarray): inverse intrinsic
    """
    g = np.einsum('ikc,cd->ikd', h, Kinv)
    g = np.einsum('kc,jcd->jkd', K, g)
    g /= g[:, 2:3, 2:3]
    return g


def get_transform(rvec1, tvec1, rvec2, tvec2, nt, depth):
    """
    Get geometric Homography between 2 poses

    Args:
        rvec1 (np.array): rotation vector 1
        tvec1 (np.array): translation vector 1
        rvec2 (np.array): rotation vector 2
        tvec2 (np.array): translation vector 2
        nt (np.array): plane normal
        depth (float): depth from camera
    """
    H_0_1 = generate_homography(rvec1, tvec1, nt, depth)
    H_0_2 = generate_homography(rvec2, tvec2, nt, depth)
    H_1_2 = H_0_2.dot(np.linalg.inv(H_0_1))
    return H_1_2


def get_image_transform(rvec1, tvec1, rvec2, tvec2, nt, depth, K, Kinv):
    """
    Get image Homography between 2 poses (includes cam intrinsics)

    Args:
        rvec1 (np.array): rotation vector 1
        tvec1 (np.array): translation vector 1
        rvec2 (np.array): rotation vector 2
        tvec2 (np.array): translation vector 2
        nt (np.array): plane normal
        depth (float): depth from camera
        K (np.array): intrinsic
        Kinv (np.ndarray): inverse intrinsic
    """
    H_0_1 = generate_image_homography(rvec1, tvec1, nt, depth, K, Kinv)
    H_0_2 = generate_image_homography(rvec2, tvec2, nt, depth, K, Kinv)
    H_1_2 = H_0_2.dot(np.linalg.inv(H_0_1))
    return H_1_2


def interpolate_poses(rvecs, tvecs, nt, depth, K, Kinv, height, width, opt_flow_threshold=2, max_frames_per_bin=20):
    """
    Interpolate given poses

    Args:
        rvecs (np.array): N,3 rotation vectors
        tvecs (np.array): N,3 translation vectors
        nt (np.array): plane normal
        depth (float): depth to camera
        K (np.array): camera intrinsic
        Kinv (np.array): inverse camera intrinsic
        height (int): height of image
        width (int): width of image
        opt_flow_threshold (float): maximum flow threshold
        max_frames_per_bin (int): maximum number of pose interpolations between two consecutive poses
                                  of the original list of poses
    """
    max_frames = len(rvecs)
    key_times = np.linspace(0, max_frames - 1, max_frames, dtype=np.float32)  # (N,)

    rotations = R.from_rotvec(rvecs)

    # all homographies
    h_0_2 = generate_homographies_from_rotation_matrices(rotations.as_matrix(), tvecs, nt, depth)  # (N, 3, 3)
    hs = generate_image_homographies_from_homographies(h_0_2, K, Kinv)  # (N, 3, 3)

    h_0_1 = hs[:-1]  # (N-1, 3, 3)
    h_0_2 = hs[1:]  # (N-1, 3, 3)
    h_0_1 = np.einsum('ijk,ikc->ijc', h_0_2, np.linalg.inv(h_0_1))  # (N-1, 3, 3)

    # 4 corners
    uv1 = np.array([[0, 0, 1], [0, height - 1, 1], [width - 1, 0, 1], [width - 1, height - 1, 1]])  # (4, 3)

    # maximum flows / image
    xyz = np.einsum('jk,lck->ljc', uv1, h_0_1)  # equivalent to uv1.dot(h_0_1.T) for each 3x3 in h_0_1   (N-1, 4, 3)

    uv2 = xyz / xyz[..., 2:3]  # (N-1, 4, 3)
    flows = uv2[..., :2] - uv1[..., :2]  # (N-1, 4, 2)
    flow_mags = np.sqrt(flows[..., 0]**2 + flows[..., 1]**2)  # (N-1, 4)
    max_flows = flow_mags.max(axis=1)  # (N-1,)

    # interpolate
    nums = 1 + np.ceil(max_flows / opt_flow_threshold)
    if max_frames_per_bin > 0:
        nums = np.minimum(max_frames_per_bin, np.maximum(1, nums))
    nums = nums.astype(np.int32)
    total = nums.sum()

    interp_tvecs = np.zeros((total, 3), dtype=np.float32)
    times = np.zeros((total,), dtype=np.float32)
    interpolate_times_tvecs(tvecs, key_times, interp_tvecs, times, nums)

    slerp = Slerp(key_times, rotations)
    interp_rvecs = slerp(times).as_rotvec()  # (nums.sum(), 3)

    return interp_rvecs, interp_tvecs, times, max_flows.max()


def get_flow(rvec1, tvec1, rvec2, tvec2, nt, depth, K, Kinv, height, width):
    """
    Computes Optical Flow between 2 poses

    Args:
        rvec1 (np.array): rotation vector 1
        tvec1 (np.array): translation vector 1
        rvec2 (np.array): rotation vector 2
        tvec2 (np.array): translation vector 2
        nt (np.array): plane normal
        depth (float): depth from camera
        K (np.array): intrisic matrix
        Kinv (np.array): inverse intrisic matrix
        height (int): height of image
        width (int): width of image
        infinite (bool): plan is infinite or not
    """
    # 1. meshgrid of image 1
    uv1 = get_grid(height, width).reshape(height * width, 3)

    # adapt K with new height, width
    H_0_1 = generate_image_homography(rvec1, tvec1, nt, depth, K, Kinv)
    H_0_2 = generate_image_homography(rvec2, tvec2, nt, depth, K, Kinv)

    # 2. apply H_0_2.dot(H_1_0) directly
    H_1_0 = H_0_2.dot(np.linalg.inv(H_0_1))

    xyz = uv1.dot(H_1_0.T)
    uv2 = xyz / xyz[:, 2:3]

    flow = uv2[:, :2] - uv1[:, :2]
    flow = flow.reshape(height, width, 2)
    return flow


def get_grid(height, width):
    """
    Computes a 2d meshgrid

    Args:
        height (int): height of grid
        width (int): width of grid
    """
    x, y = np.linspace(0, width - 1, width, dtype=DTYPE), np.linspace(0, height - 1, height, dtype=np.float32)
    x, y = np.meshgrid(x, y)
    x, y = x[:, :, None], y[:, :, None]
    o = np.ones_like(x)
    xy = np.concatenate([x, y, o], axis=2)
    return xy


def generate_smooth_signal(num_signals, num_samples, min_speed=1e-4, max_speed=1e-1):
    """
    Generates a smooth signal

    Args:
        num_signals (int): number of signals to generate
        num_samples (int): length of multidimensional signal
        min_speed (float): minimum rate of change
        max_speed (float): maximum rate of change
    """
    t = np.linspace(0, num_samples - 1, num_samples)

    num_bases = 10
    samples = 0 * t
    samples = samples[None, :].repeat(num_signals, 0)
    for i in range(num_bases):
        speed = np.random.uniform(min_speed, max_speed, (num_signals,))[:, None]
        phase = np.random.uniform(np.pi, 10 * np.pi)
        test = np.sin((speed * t[None, :]) + phase)
        samples += test / num_bases

    # final
    speed = np.random.uniform(min_speed, max_speed, (num_signals,))[:, None]
    test = np.sin((speed * t[None, :]))
    test = (test + 1) / 2
    samples *= test
    return samples.astype(DTYPE)


def add_random_pause(signal, max_pos_size_ratio=0.3):
    """
    Adds a random pause in a multidimensional signal

    Args:
        signal (np.array): TxD signal
        max_pos_size_ratio (float): size of pause relative to signal Length
    """
    num_pos = len(signal)
    max_pause_size = int(max_pos_size_ratio * num_pos)
    min_pos = int(0.1 * num_pos)
    pause_size = max_pause_size
    pause_pos = np.random.randint(min_pos, num_pos - pause_size)
    value = signal[pause_pos:pause_pos + 1].repeat(pause_size, 0)
    cat = np.concatenate((signal[:pause_pos - 1], value, signal[pause_pos:]), axis=0)
    return cat


class CameraPoseGenerator(object):
    """
    CameraPoseGenerator generates a series of continuous homographies
    with interpolation.

    Args:
        height (int): height of image
        width (int): width of image
        max_frames (int): maximum number of poses
        pause_probability (float): probability that the sequence contains a pause
        max_optical_flow_threshold (float): maximum optical flow between two consecutive frames
        max_interp_consecutive_frames (int): maximum number of interpolated frames between two consecutive frames
    """

    def __init__(self, height, width, max_frames=300, pause_probability=0.5,
                 max_optical_flow_threshold=2., max_interp_consecutive_frames=20):
        self.K = np.array(
            [[width / 2, 0, width / 2], [0, height / 2, height / 2], [0, 0, 1]],
            dtype=DTYPE,
        )
        self.Kinv = np.linalg.inv(self.K)

        self.nt = np.array([0, 0, 1], dtype=DTYPE).reshape(1, 3)

        signal = generate_smooth_signal(6, max_frames).T
        if np.random.rand() < pause_probability:
            signal = add_random_pause(signal)
        rvecs = signal[:, :3]
        tvecs = signal[:, 3:]

        self.depth = np.random.uniform(1.0, 2.0)
        self.rvecs, self.tvecs, self.times, max_flow = interpolate_poses(
            rvecs, tvecs, self.nt, self.depth, self.K, self.Kinv, height, width,
            opt_flow_threshold=max_optical_flow_threshold,
            max_frames_per_bin=max_interp_consecutive_frames)

        self.time = 0
        self.max_frames = max_frames
        assert len(self.rvecs) >= max_frames
        self.rvecs, self.tvecs, self.times = self.rvecs[:max_frames], self.tvecs[:max_frames], self.times[:max_frames]

    def __len__(self):
        """
        Returns number of poses
        """
        return len(self.rvecs)

    def __call__(self):
        """
        Returns next homography
        """
        rvec2 = self.rvecs[self.time]
        tvec2 = self.tvecs[self.time]
        ts = self.times[self.time]
        H = generate_image_homography(rvec2, tvec2, self.nt, self.depth, self.K, self.Kinv)
        self.time += 1
        return H, ts

    def get_image_transform(self, rvec1, tvec1, rvec2, tvec2):
        """
        Get Homography between 2 poses

        Args:
            rvec1 (np.array): rotation vector 1
            tvec1 (np.array): translation vector 1
            rvec2 (np.array): rotation vector 2
            tvec2 (np.array): translation vector 2
        """
        return get_image_transform(rvec1, tvec1, rvec2, tvec2, self.nt, self.depth, self.K, self.Kinv)

    def get_flow(self, rvec1, tvec1, rvec2, tvec2, height, width):
        """
        Computes Optical flow between 2 poses

        Args:
            rvec1 (np.array): rotation vector 1
            tvec1 (np.array): translation vector 1
            rvec2 (np.array): rotation vector 2
            tvec2 (np.array): translation vector 2
            nt (np.array): plane normal
            depth (float): depth from camera
            height (int): height of image
            width (int): width of image
        """
        return get_flow(rvec1, tvec1, rvec2, tvec2, self.nt, self.depth, self.K, self.Kinv, height, width)
