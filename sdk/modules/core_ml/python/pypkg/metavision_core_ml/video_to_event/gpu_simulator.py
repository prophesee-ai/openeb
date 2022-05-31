# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
More efficient reimplementation.
The main difference is cuda kernels & possibility to
directly stream the voxel grid.
"""
# pylint: disable=access-member-before-definition
# pylint: disable=undefined-variable

import torch
import torch.nn as nn
import numpy as np

from numba import cuda

from .cutoff_kernels import _cuda_kernel_dynamic_moving_average, _cpu_kernel_dynamic_moving_average
from .events_kernels import _cuda_kernel_count_events, _cpu_kernel_count_events
from .events_kernels import _cpu_kernel_fill_events, _cuda_kernel_fill_events
from .events_kernels import _cuda_kernel_voxel_grid_sequence, _cpu_kernel_voxel_grid_sequence


class GPUEventSimulator(nn.Module):
    """
    GPU Event Simulator of events from frames & timestamps.

    Implementation is based on the following publications:

    - Video to Events: Recycling Video Datasets for Event Cameras: Daniel Gehrig et al.
    - V2E: From video frames to realistic DVS event camera streams: Tobi Delbruck et al.

    Args:
        batch_size (int): number of video clips / batch
        height (int): height
        width (int): width
        c_mu (float): threshold average
        c_std (float): threshold standard deviation
        refractory period (int): time before event can be triggered again
        leak_rate_hz (float): frequency of reference voltage leakage
        cutoff_hz (float): frequency for photodiode latency
        shot_noise_hz (float): frequency of shot noise events
    """

    def __init__(
            self, batch_size, height, width, c_mu=0.1, c_std=0.022, refractory_period=10, leak_rate_hz=0, cutoff_hz=0,
            shot_noise_hz=0):
        super().__init__()
        shape1 = (batch_size, height, width)
        shape2 = (2, batch_size, height, width)
        self.register_buffer("log_states", torch.zeros(shape1, dtype=torch.float64))
        self.register_buffer("prev_log_images", torch.zeros(shape1, dtype=torch.float32))
        self.register_buffer("counts", torch.zeros(shape1, dtype=torch.int32))
        self.register_buffer("timestamps", torch.zeros(shape1, dtype=torch.float32))

        thresholds = torch.randn(shape2, dtype=torch.float32) * c_std + c_mu
        self.register_buffer("thresholds", thresholds)
        self.register_buffer("prev_image_ts", torch.zeros((batch_size), dtype=torch.float32))
        self.register_buffer("filtering_prev_image_ts", torch.zeros((batch_size), dtype=torch.float32))
        self.thresholds.clamp_(0.01, 1.0)

        self.threshold_mu = c_mu
        self.threshold_std = c_std
        self.leak_rate_micro_hz = leak_rate_hz * 1e-6
        self.cutoff_hz = cutoff_hz
        self.shot_noise_micro_hz = shot_noise_hz * 1e-6
        self.refractory_period = refractory_period

        self.register_buffer("prev_log_images_0", torch.zeros(shape1, dtype=torch.float32))
        self.register_buffer("prev_log_images_1", torch.zeros(shape1, dtype=torch.float32))

        self.register_buffer("rng_states", torch.zeros(shape1, dtype=torch.float32))

        # generate array of noise parameters
        self.register_buffer("refractory_periods", torch.LongTensor([self.refractory_period] * batch_size))
        self.register_buffer("cutoff_rates", torch.tensor([self.cutoff_hz] * batch_size, dtype=torch.float32))
        self.register_buffer("leak_rates", torch.tensor([self.leak_rate_micro_hz] * batch_size, dtype=torch.float32))
        self.register_buffer("shot_rates", torch.tensor([self.shot_noise_micro_hz] * batch_size, dtype=torch.float32))
        self.register_buffer("threshold_mus", torch.tensor([c_mu] * batch_size, dtype=torch.float32))
        self.register_buffer("rgb_to_gray", torch.tensor([0.33, 0.33, 0.33], dtype=torch.float32))

    def get_size(self):
        return self.thresholds.shape[-2:]

    def randomize_broken_pixels(self, first_times, video_proba=1e-2, crazy_pixel_proba=5e-4, dead_pixel_proba=5e-3):
        """
        Simulates dead & crazy pixels

        Args:
            first_times: B video just started flags
            video_proba: probability to simulate broken pixels
        """
        height, width = self.thresholds.shape[-2:]
        device = first_times.device
        for i in range(len(first_times)):
            if first_times[i].item():
                # crazy pixels
                if np.random.rand() < video_proba:
                    mask = torch.rand(height, width).to(device) < crazy_pixel_proba
                    self.thresholds[:, i, mask] = 1e-3
                # dead pixels
                if np.random.rand() < video_proba:
                    mask = torch.rand(height, width).to(device) < dead_pixel_proba
                    self.thresholds[:, i, mask] = 1e3

    def randomize_thresholds(self, first_times, th_mu_min=0.05, th_mu_max=0.2, th_std_min=0.001, th_std_max=0.01):
        """
        Re-Randomizes thresholds per video

        Args:
            first_times: B video just started flags
            th_mu_min:
            th_mu_max:
            th_std_min:
            th_std_max:
        """
        batch_size = len(first_times)
        ft = first_times[:, None, None]
        for i in range(batch_size):
            if ft[i].item():
                mu = np.random.uniform(th_mu_min, th_mu_max)
                std = np.random.uniform(th_std_min, th_std_max)
                self.threshold_mus[i] = mu
                self.thresholds[:, i].normal_(mean=mu, std=std)
                self.thresholds[:, i].clamp_(0.01, 1.0)

    def randomize_cutoff(self, first_times, cutoff_min=0, cutoff_max=900):
        """
        Randomizes the cutoff rates per video

        Args:
            first_times: B video just started flags
            cutoff_min: in hz
            cutoff_max: in hz
        """
        ft = first_times
        cutoff_rates = torch.zeros_like(self.cutoff_rates).uniform_(cutoff_min, cutoff_max)
        self.cutoff_rates = ft * cutoff_rates + (1 - ft) * self.cutoff_rates

    def randomize_leak(self, first_times, leak_min=0, leak_max=1):
        """
        Randomizes the leak rates per video

        Args:
            first_times: B video just started flags
            leak_min: in hz
            leak_max: in hz
        """
        ft = first_times
        rates = torch.zeros_like(self.leak_rates).uniform_(leak_min * 1e-6, leak_max * 1e-6)
        self.leak_rates = ft * rates + (1 - ft) * self.leak_rates

    def randomize_shot(self, first_times, shot_min=0, shot_max=1):
        """
        Randomizes the shot noise per video

        Args:
            shot_min: in hz
            shot_max: in hz
        """
        ft = first_times
        rates = torch.zeros_like(self.shot_rates).uniform_(shot_min * 1e-6, shot_max * 1e-6)
        self.shot_rates = ft * rates + (1 - ft) * self.shot_rates

    def randomize_refractory_periods(self, first_times, ref_min=10, ref_max=1000):
        """
        Randomizes the refractory period per video

        Args:
            first_times: B video just started flags
            ref_min: in microseconds
            ref_max: in microseconds
        """
        ft = first_times
        rates = torch.zeros_like(self.shot_rates).uniform_(ref_min, ref_max)
        self.refractory_periods = ft * rates + (1 - ft) * self.refractory_periods

    def forward(self):
        raise NotImplementedError

    def _kernel_call(self, log_images, video_len, image_ts, first_times, cuda_kernel, cpu_kernel, args_list, *args,
                     reset_rng_states=True):
        """
        generic functions to call simulation and feature computation kernels.

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            cuda_kernel (function): numba.cuda jitted function (defined in events_kernel.py)
            cpu_kernel (function): numba jitted function (defined in events_kernel.py)
            args_list (Tensor list): additional Tensor arguments that the kernel might take as argument.
            *args: additional flags for the kernel
        """
        device = log_images.device
        height, width = log_images.shape[:2]
        batch_size = len(video_len)

        if reset_rng_states:
            self.rng_states.uniform_()
        # prepare args
        args_list = args_list + [
            log_images, video_len.cumsum(0), image_ts, first_times, self.rng_states, self.log_states,
            self.prev_log_images, self.timestamps, self.thresholds, self.prev_image_ts, self.refractory_periods,
            self.leak_rates, self.shot_rates, self.threshold_mus]

        if device.type == "cuda":
            args_list = [v.to(device) for v in args_list]
            cu_args = [cuda.as_cuda_array(v) for v in args_list] + list(args)

            block_dim = (1, 16, 16)
            sizes = (batch_size, height, width)
            grid_dim = tuple(int(np.ceil(a / b)) for a, b in zip(sizes, block_dim))

            # kernel
            cuda_kernel[grid_dim, block_dim](*cu_args)
        else:
            args_list = [v.numpy() for v in args_list] + list(args)

            # kernel
            cpu_kernel(*args_list)

    @torch.no_grad()
    def get_events(self, log_images, video_len, image_ts, first_times):
        """
        Retrieves the AER event list in a pytorch array.

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
        Returns:
            events: N,5 in batch_index, x, y, polarity, timestamp (micro-seconds)
        """
        height, width = log_images.shape[:2]
        batch_size = len(video_len)

        self.counts[...] = 0

        self._kernel_call(log_images, video_len, image_ts, first_times, _cuda_kernel_count_events,
                          _cpu_kernel_count_events, [self.counts], False, reset_rng_states=True)
        # come-up with offset
        event_counts = self.counts
        cumsum = event_counts.view(-1).cumsum(dim=0)
        total_num_events = cumsum[-1].item()
        offsets = cumsum.view(batch_size, height, width) - event_counts

        events = torch.full((total_num_events, 5), fill_value=-10, device=cumsum.device, dtype=torch.int32)

        self._kernel_call(log_images, video_len, image_ts, first_times, _cuda_kernel_fill_events,
                          _cpu_kernel_fill_events, [events, offsets], True, reset_rng_states=False)

        # update values
        self.prev_image_ts = image_ts[:, -1]

        return events

    @torch.no_grad()
    def count_events(self, log_images, video_len, image_ts, first_times, reset=True, persistent=True):
        """
        Estimates the number of events per pixel.

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            reset: do reset the count variable
        Returns:
            counts: B,H,W
        """

        if reset:
            self.counts[...] = 0

        self._kernel_call(log_images, video_len, image_ts, first_times, _cuda_kernel_count_events,
                          _cpu_kernel_count_events, [self.counts], persistent, reset_rng_states=True)

        # update values
        self.prev_image_ts = image_ts[:, -1]

        return self.counts

    @torch.no_grad()
    def event_volume(self, log_images, video_len, image_ts, first_times, nbins, mode='bilinear', split_channels=False):
        """
        Computes a volume of discretized images formed after the events, without
        storing the AER events themselves. We go from simulation directly to this
        space-time quantized representation. You can obtain the event-volume of
        [Unsupervised Event-based Learning of Optical Flow, Zhu et al. 2018] by
        specifying the mode to "bilinear" or you can obtain a stack of histograms
        if mode is set to "nearest".

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            nbins (int): number of time-bins for the voxel grid
            mode (str): bilinear or nearest
            split_channels: if True positive and negative events have a distinct channels instead of doing their
                difference in a single channel.
        """
        prev_times = self.prev_image_ts * (1 - first_times) + image_ts[:, 0] * first_times
        end_times = image_ts[:, -1]

        target_timestamps = torch.cat((prev_times[:, None], end_times[:, None]), 1).long()

        return self.event_volume_sequence(log_images, video_len, image_ts, target_timestamps,
                                          first_times, nbins, mode, split_channels).squeeze(0)

    @torch.no_grad()
    def event_volume_sequence(
            self,
            log_images,
            video_len,
            image_ts,
            target_timestamps,
            first_times,
            nbins,
            mode='bilinear',
            split_channels=False):
        """
        Computes a volume of discretized images formed after the events, without
        storing the AER events themselves. We go from simulation directly to this
        space-time quantized representation. You can obtain the event-volume of
        [Unsupervised Event-based Learning of Optical Flow, Zhu et al. 2018] by
        specifying the mode to "bilinear" or you can obtain a stack of histograms
        if mode is set to "nearest".
        Here, we also receive a sequence of target timestamps to cut non uniformly the event volumes.

        Args:
            log_images (Tensor): shape (H, W, total_num_frames) tensor containing the video frames
            video_len (Tensor): shape (B,) len of each video in the batch.
            images_ts (Tensor): shape (B, max(video_len)) timestamp associated with each frame.
            first_times (Tensor): shape (B) whether the video is a new one or the continuation of one.
            nbins (int): number of time-bins for the voxel grid
            mode (str): bilinear or nearest
            split_channels: if True positive and negative events have a distinct channels instead of doing their
                difference in a single channel.
        """
        height, width = log_images.shape[:2]
        num_channels = nbins * 2 if split_channels else nbins
        batch_size = len(video_len)
        batch_times = target_timestamps.shape[1] - 1

        device = log_images.device
        voxel_grid = torch.zeros(
            (batch_times, batch_size, num_channels, height, width),
            dtype=torch.float32, device=device)

        # arbitrary voxel start times and durations
        voxel_start_times = self.prev_image_ts * (1 - first_times) + image_ts[:, 0] * first_times
        voxel_durations = image_ts[:, -1] - voxel_start_times

        # prepare args
        args = [voxel_grid, target_timestamps]

        self._kernel_call(
            log_images, video_len, image_ts, first_times, _cuda_kernel_voxel_grid_sequence,
            _cpu_kernel_voxel_grid_sequence, args, True, mode == 'bilinear', split_channels, reset_rng_states=True)

        self.prev_image_ts = image_ts[:, -1]

        voxel_grid = voxel_grid.view(batch_times, batch_size, num_channels, height, width)
        return voxel_grid

    @torch.no_grad()
    def log_images(self, u8imgs, eps=1e-7):
        """
        Converts byte images to log

        Args:
            u8imgs (torch.Tensor): B,C,H,W,T byte images
            eps (float): epsilon factor
        """
        return torch.log(u8imgs.float() / 255.0 + eps)

    @torch.no_grad()
    def dynamic_moving_average(self, images, num_frames, timestamps, first_times, eps=1e-7):
        """

        Converts byte images to log and
        performs a pass-band motion blur of incoming images.
        This simulates the latency of the photodiode w.r.t to incoming
        light dynamic.

        Args:
            images (torch.Tensor): H,W,T byte or float images in the 0 to 255 range
            num_frames (torch.Tensor): shape (B,) len of each video in the batch.
            timestamps (torch.Tensor): B,T timestamps
            first_times (torch.Tensor): B flags
            eps (float): epsilon factor
        """
        fl_images = images.float()

        if self.cutoff_rates.sum() == 0:  # this broke the sample
            return self.log_images(images, eps=eps)

        height, width = images.shape[:2]
        batch_size = len(num_frames)

        args = [fl_images, num_frames.cumsum(0), self.prev_log_images_0, self.prev_log_images_1,
                self.filtering_prev_image_ts, timestamps, first_times, self.cutoff_rates]

        if images.device.type == "cuda":
            cu_args = [cuda.as_cuda_array(v) for v in args]
            block_dim = (1, 16, 16)
            sizes = (batch_size, height, width)
            grid_dim = tuple(int(np.ceil(a / b)) for a, b in zip(sizes, block_dim))
            _cuda_kernel_dynamic_moving_average[grid_dim, block_dim](*cu_args, eps)
        else:
            args = [v.numpy() for v in args]
            _cpu_kernel_dynamic_moving_average(*args, eps)

        self.filtering_prev_image_ts = timestamps[:, -1]

        return fl_images
