# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Here we reuse the GPUSimulator from OpenEB to stream synthetic events and corners.
"""
import torch
import numpy as np

from metavision_core_ml.corner_detection.corner_video_stream_dataset import make_corner_video_dataset
from metavision_core_ml.video_to_event.gpu_simulator import GPUEventSimulator


def collect_target_images(gray_images, timestamps, video_len, target_indices, num_heatmaps):
    """
    Collect target frames + timestamps at target indices
    and rearranges them into T,B,C,H,W tensor

    Args:
        gray_images (tensor): H,W,T format (videos are concatenated along 3rd dimension
        timestamps (tensor): B,T
        video_len (tensor): B lengths
        target_indices (tensor): B,M indices
        num_heatmaps (int): number of heatmaps
    """
    num_frames_cumsum = video_len.numpy().cumsum(0)
    target_gray = []
    target_times = []
    for i in range(len(target_indices)):
        row_img = []
        row_ts = []
        start_f = int(num_frames_cumsum[i - 1]) if i else 0
        end_f = int(num_frames_cumsum[i])
        for t in range(len(target_indices[i])):
            tt = target_indices[i, t].item()
            idx = start_f + tt
            assert start_f <= idx < end_f
            channel_img = []
            for index_multi_timestep in range(num_heatmaps):
                idx_channel = idx + index_multi_timestep - (num_heatmaps - 1)
                assert start_f <= idx_channel < end_f
                img = gray_images[..., idx_channel].clone()
                channel_img.append(img[None, None])
            row_img.append(torch.cat(channel_img, axis=1))
            row_ts.append(timestamps[i, tt])
        row_img = torch.cat(row_img)
        target_gray.append(row_img[:, None])
        target_times.append(torch.FloatTensor(row_ts)[None, :])
    target_images = torch.cat(target_gray, 1)  # T,B,C,H,W
    target_times = torch.cat(target_times, 0)
    return target_images, target_times


class GPUEBSimCorners(object):
    """
    Simulated Events on GPU returns events, images and corners

    Args:
        dataloader: video-clips dataloader
        simulator: gpu-simulator
        batch_times: number of rounds per batch
        event_volume_depth: number of timesteps per round
        device: hardware to run simulation on
    """

    def __init__(self, dataloader, simulator, batch_times, event_volume_depth, randomize_noises, device,
                 number_of_heatmaps, height, width, batch_size):
        self.dataloader = dataloader
        self.simulator = simulator
        self.device = device
        self.batch_times = batch_times
        self.event_volume_depth = event_volume_depth
        self.do_randomize_noises = randomize_noises
        self.number_of_heatmaps = number_of_heatmaps
        self.height, self.width = height, width
        self.batch_size = batch_size

    @classmethod
    def from_params(cls, folder,
                    num_workers,
                    batch_size,
                    batch_times,
                    event_volume_depth,
                    height,
                    width,
                    min_frames_per_video,
                    max_frames_per_video,
                    number_of_heatmaps,
                    randomize_noises=False,
                    device='cuda:0'):
        """
        Creates the simulator from parameters
        Args:
            folder: folder of images
            num_workers: number of workers
            batch_size: size of batch
            batch_times: time dimension per batch
            event_volume_depth: number of channels in event volume
            height: height of images
            width: width of images
            min_frames_per_video: minimum number of frames per video
            max_frames_per_video: maximum number of frames per video
            number_of_heatmaps: number of heatmaps of corners locations returned
            randomize_noises: whether or not to randomize the noise of the simulator
            device: location of data

        Returns:
            GPUEBSimCorners class instantiated
        """
        print('randomize noises: ', randomize_noises)
        dataloader = make_corner_video_dataset(
            folder, num_workers, batch_size, height, width, min_frames_per_video, max_frames_per_video,
            number_of_heatmaps=number_of_heatmaps, batch_times=batch_times)
        event_gpu = GPUEventSimulator(batch_size, height, width)
        event_gpu.to(device)
        return cls(dataloader, event_gpu, batch_times, event_volume_depth, randomize_noises, device,
                   number_of_heatmaps, height, width, batch_size)

    def randomize_noises(self, first_times):
        """
        Randomizes noise in the simulator consistent with the batches
        Args:
            first_times: whether or not the video in the batch is new
        """
        batch_size = len(first_times)
        self.simulator.randomize_thresholds(first_times, th_mu_min=0.3,
                                            th_mu_max=0.4, th_std_min=0.08, th_std_max=0.01)
        for i in range(batch_size):
            if first_times[i].item():
                cutoff_rate = 70 if np.random.rand() < 0.1 else 1e6
                leak_rate = 0.1 * 1e-6 if np.random.rand() < 0.1 else 0
                shot_rate = 10 * 1e-6 if np.random.rand() < 0.1 else 0
                refractory_period = np.random.uniform(10, 200)
                self.simulator.cutoff_rates[i] = cutoff_rate
                self.simulator.leak_rates[i] = leak_rate
                self.simulator.shot_rates[i] = shot_rate
                self.simulator.refractory_periods[i] = refractory_period

    def __iter__(self):
        for batch in self.dataloader:
            gray_images = batch['images'].squeeze(0).to(self.device)
            gray_corners = batch['corners'].squeeze(0).to(self.device)
            first_times = batch['first_times'].to(self.device)
            timestamps = batch['timestamps'].to(self.device)
            video_len = batch["video_len"].to(self.device)
            target_indices = batch['target_indices']
            prev_ts = self.simulator.prev_image_ts.clone()
            prev_ts = prev_ts * (1 - first_times) + timestamps[:, 0] * first_times

            if self.do_randomize_noises:
                self.randomize_noises(batch['first_times'])

            log_images = self.simulator.dynamic_moving_average(gray_images, video_len, timestamps, first_times)

            target_images, target_times = collect_target_images(
                gray_images, timestamps, batch['video_len'], target_indices, self.number_of_heatmaps)
            target_corners, target_times = collect_target_images(
                gray_corners, timestamps, batch['video_len'], target_indices, self.number_of_heatmaps)

            all_times = torch.cat((prev_ts[:, None], target_times.to(self.device)), dim=1).long()
            inputs = self.simulator.event_volume_sequence(
                log_images, video_len, timestamps, all_times, first_times, self.event_volume_depth)

            reset = 1 - first_times[:, None, None, None]
            out_batch = {'inputs': inputs,
                         'images': target_images,
                         'corners': target_corners,
                         'reset': reset}

            yield out_batch
