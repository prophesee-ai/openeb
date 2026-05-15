# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# pylint: disable=E0633
# pylint: disable=E1121

"""
cpu + cuda kernels for gpu simulation of cutoff
"""

import math
from numba import njit
from numba import cuda


@cuda.jit()
def _cuda_kernel_dynamic_moving_average(
    images, num_frames_cumsum, prev_log_images_0, prev_log_images_1,
        prev_image_times, image_times, first_times, cutoff_rates, min_pixel_range=20, max_pixel_incr=20, log_eps=1e-7):
    """
    Dynamic Blurring of sequence + Log-space conversion
    (GPU code)
    """
    b, y, x = cuda.grid(3)
    height, width, _ = images.shape
    batch_size = len(num_frames_cumsum)

    if b < batch_size and y < height and x < width:
        first_time = int(first_times[b])
        cutoff_hz = cutoff_rates[b]
        cutoff_hz = cutoff_hz if cutoff_hz > 0 else 1e4
        tau = math.pi * 2 * cutoff_hz
        start_f = num_frames_cumsum[b - 1] if b else 0
        end_f = num_frames_cumsum[b]
        last_image_ts = prev_image_times[b]

        pixel = images[y, x]
        log_pixel = math.log(pixel[start_f] / 255.0 + log_eps)
        if first_time:
            log_state0 = log_pixel
            log_state1 = log_pixel
        else:
            log_state0 = prev_log_images_0[b, y, x]
            log_state1 = prev_log_images_1[b, y, x]

        for t in range(start_f, end_f):
            ind = t - start_f

            dt_s = 1e-6 * (image_times[b, ind] - last_image_ts)
            eps = (pixel[t] + min_pixel_range) / (255 + max_pixel_incr) * dt_s * tau
            eps = max(0, min(eps, 1))

            log_state0 = (1 - eps) * log_state0 + eps * math.log(pixel[t] / 255.0 + log_eps)
            log_state1 = (1 - eps) * log_state1 + eps * log_state0

            if t > start_f or not first_time:
                pixel[t] = log_state1
            else:
                pixel[start_f] = log_pixel

            last_image_ts = image_times[b, ind]
        prev_log_images_0[b, y, x] = log_state0
        prev_log_images_1[b, y, x] = log_state1


@njit()
def _cpu_kernel_dynamic_moving_average(
    images, num_frames_cumsum, prev_log_images_0, prev_log_images_1,
        prev_image_times, image_times, first_times, cutoff_rates, min_pixel_range=20, max_pixel_incr=20, log_eps=1e-7):
    """
    Dynamic Blurring of sequence + Log-space conversion
    (CPU code)
    """
    height, width = images.shape[:2]
    batch_size = len(num_frames_cumsum)
    for b in range(batch_size):
        first_time = int(first_times[b])
        cutoff_hz = cutoff_rates[b]
        cutoff_hz = cutoff_hz if cutoff_hz > 0 else 1e4
        tau = math.pi * 2 * cutoff_hz
        start_f = num_frames_cumsum[b - 1] if b else 0
        end_f = num_frames_cumsum[b]

        for y in range(height):
            for x in range(width):
                last_image_ts = prev_image_times[b]
                pixel = images[y, x]
                log_pixel = math.log(pixel[start_f] / 255.0 + log_eps)
                if first_time:
                    log_state0 = log_pixel
                    log_state1 = log_pixel
                else:
                    log_state0 = prev_log_images_0[b, y, x]
                    log_state1 = prev_log_images_1[b, y, x]

                for t in range(start_f, end_f):
                    ind = t - start_f

                    dt_s = 1e-6 * (image_times[b, ind] - last_image_ts)
                    eps = (pixel[t] + min_pixel_range) / (255. + max_pixel_incr) * dt_s * tau

                    eps = max(0, min(eps, 1))

                    log_state0 = (1 - eps) * log_state0 + eps * math.log(pixel[t] / 255.0 + log_eps)
                    log_state1 = (1 - eps) * log_state1 + eps * log_state0

                    if t > start_f or not first_time:
                        pixel[t] = log_state1
                    else:
                        pixel[start_f] = log_pixel

                    last_image_ts = image_times[b, ind]
                prev_log_images_0[b, y, x] = log_state0
                prev_log_images_1[b, y, x] = log_state1
