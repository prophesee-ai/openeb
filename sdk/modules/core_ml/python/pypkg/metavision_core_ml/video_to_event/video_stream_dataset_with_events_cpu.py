# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Image stream data loader
"""
# pylint: disable=not-callable, line-too-long

import torch
import numpy as np
from kornia.geometry.transform import resize

from metavision_core_ml.data.stream_dataloader import StreamDataset, StreamDataLoader
from metavision_core_ml.data.scheduling import build_metadata
from .video_stream_dataset import VideoDatasetIterator
from metavision_core_ml.video_to_event.simulator import EventSimulator
from metavision_core_ml.utils.color_utils import from_srgb
# pylint: disable=not-callable


def get_random_simu_param(param, random_event_noise_prob, default_value):
    """
    randomize a single noise params for events simulator.
    param must be a list of 2 elements, or a constant. 
    In the first case, uses noisy settings with random_event_noise_prob probability, 
    the rest of the time uses "clean" settings
    If input parameter is a scaler, no randomization is done.
    """
    if isinstance(param, list):
        assert len(param) == 2
        if np.random.rand() < random_event_noise_prob:
            param_rand = np.random.uniform(
                param[0],
                param[1])
        else:
            param_rand = default_value
    else:
        param_rand = param

    return param_rand


class VideoDatasetWithEventsIterator(object):
    """
    Dataset Iterator streaming images and timestamps and events

    Gamma correction is applied using default kornia parameters (see kornia.color.rgb_to_linear_rgb)
    to use linear images before calling the video_to_event simulator

    Args:
        dic_params_video_dataset (dict): dictionary with the following keys
            metadata (object): path to picture or video
            height (int): height of input images / video clip
            width (int): width of input images / video clip
            rgb (bool): stream rgb videos
            mode (str): mode of batch sampling 'frames','delta_t','random'
            min_tbins (int): minimum number of frames per batch step
            max_tbins (int): maximum number of frames per batch step
            min_dt (int): minimum duration of frames per batch step
            max_dt (int): maximum duration of frames per batch step
            batch_times (int): number of timesteps of training sequences
            max_number_of_batches_to_produce (int): maximum number of batches to produce
        dic_params_events_simulator (dict): dictionary with the following keys
            ev_height (int): height of the image used by the event simulator
            ev_width (int): width of the image used by the event simulator
            min_Cp (float): minimum value for positive contrast threshold
            max_Cp (float): maximum value for positive contrast threshold
            min_Cn (float): minimum value for negative contrast threshold
            max_Cn (float): maximum value for negative contrast threshold
            min_refractory_period (int): minimum value for refractory period
            max_refractory_period (int): maximum value for refractory period
            min_sigma_threshold (float): minimal value for standard deviation of threshold array
            max_sigma_threshold (float): maximal value for standard deviation of threshold array
            cutoff_hz (float): cutoff frequency for photodiode latency simulation, if 2 elements vector, will randomly sample in [min_val, mav_val]
            leak_rate_hz (float): frequency of reference value leakage, if 2 elements vector, will randomly sample in [min_val, mav_val]
            shot_noise_rate_hz (float): frequency for shot noise events, if 2 elements vector, will randomly sample in [min_val, mav_val]
            random_event_noise_prob (float): probability to sample random noise or use clean settings
        discard_events_between_batches (bool): if True, events between last frame of previous batch and first frame
                                               of current batch will be discarded. If False, they are kept
    """

    def __init__(self, metadata, dic_params_video_dataset, dic_params_events_simulator,
                 discard_events_between_batches=True):
        self.discard_events_between_batches = discard_events_between_batches
        self.video_dataset_iterator = VideoDatasetIterator(metadata=metadata, **dic_params_video_dataset)
        self._init_simulator(**dic_params_events_simulator)

    def _init_simulator(
            self, ev_height=None, ev_width=None, min_Cp=0.1, max_Cp=0.1, min_Cn=0.1, max_Cn=0.1,
            min_refractory_period=10, max_refractory_period=10, min_sigma_threshold=0., max_sigma_threshold=0,
            cutoff_hz=0, leak_rate_hz=0, shot_noise_rate_hz=0, random_event_noise_prob=0.1):
        if ev_height is None or ev_width is None:
            assert ev_height is None and ev_width is None
            ev_height, ev_width = self.video_dataset_iterator.get_size()

        # randomize event camera biases
        Cp = np.random.uniform(min_Cp, max_Cp)
        Cn = np.random.uniform(min_Cn, max_Cn)
        refractory_period = 10**np.random.uniform(
            np.log10(min_refractory_period),
            np.log10(max_refractory_period))
        sigma_threshold = np.random.uniform(min_sigma_threshold, max_sigma_threshold)

        # randomize noise params (optionally)
        cutoff_hz_random, leak_rate_hz_random, shot_noise_rate_hz_random = self.get_random_noise_params(
            cutoff_hz, leak_rate_hz, shot_noise_rate_hz, random_event_noise_prob)

        self.simu = EventSimulator(height=ev_height, width=ev_width,
                                   Cp=Cp, Cn=Cn, refractory_period=refractory_period,
                                   sigma_threshold=sigma_threshold,
                                   cutoff_hz=cutoff_hz_random,
                                   leak_rate_hz=leak_rate_hz_random,
                                   shot_noise_rate_hz=shot_noise_rate_hz_random)

    def get_events_size(self):
        return self.simu.get_size()

    def get_random_noise_params(self, cutoff_hz, leak_rate, shot_rate, random_event_noise_prob):
        """
        randomize noise params for events simulator.
        input must be a list of 2 elements, or a constant. 
        In the first case, uses noisy settings with random_event_noise_prob probability, 
        the rest of the time uses "clean" settings
        If input parameter is a scaler, no randomization is done.
        """
        cutoff_hz_rand = get_random_simu_param(cutoff_hz, random_event_noise_prob, 1e6)
        leak_rate_rand = get_random_simu_param(leak_rate, random_event_noise_prob, 0)
        shot_rate_random = get_random_simu_param(shot_rate, random_event_noise_prob, 0)

        return cutoff_hz_rand, leak_rate_rand, shot_rate_random

    def __iter__(self):
        video_dataset_iter = iter(self.video_dataset_iterator)
        self.simu.reset()
        for batch in video_dataset_iter:
            images_grayscale = from_srgb(batch[0].squeeze(dim=0).permute(3, 0, 1, 2).float().clone() / 255.,
                                         output_colorspace='linear_gray') * 255.
            images_grayscale = images_grayscale.squeeze(dim=1)
            assert images_grayscale.min() >= 0. and images_grayscale.max() <= 255.
            if (images_grayscale.max() <= 1).all():
                print(f"Warning: No pixel >= 1 (expected range [0, 255]). Image is all black ?")

            T, img_H, img_W = images_grayscale.shape
            ev_H, ev_W = self.simu.get_size()
            if img_H != ev_H or img_W != ev_W:
                # Using antialias (low-pass filter) improves the final result
                # it removes fake events simulated from aliased edges
                images_grayscale = resize(images_grayscale, size=(ev_H, ev_W),
                                          interpolation='bilinear', antialias=True)

            is_start_of_sequence = batch[3]

            for t in range(T):
                current_image_grayscale = images_grayscale[t].numpy().copy()
                current_ts = batch[1][0][t].item()
                total = self.simu.image_callback(current_image_grayscale, current_ts)

                # This is called when t==0 to clear the events generated from
                # the last image in the previous sequence and the first image in the current sequence
                if self.discard_events_between_batches and (t == 0) and not is_start_of_sequence:
                    events = self.simu.get_events()
                    self.simu.flush_events()
            # Store the events in the current sequence and clear them afterwards
            events = self.simu.get_events()
            self.simu.flush_events()
            simu_params = {"mean_Cp": self.simu.get_mean_Cp(),
                           "mean_Cn": self.simu.get_mean_Cn(),
                           "Cps": self.simu.Cps.copy(),
                           "Cns": self.simu.Cns.copy(),
                           "cutoff_hz": self.simu.cutoff_hz,
                           "leak_rate_hz": self.simu.leak_rate_hz,
                           "shot_noise_rate_hz": self.simu.shot_noise_rate_hz,
                           "refractory_period": self.simu.refractory_period,
                           }
            # Batch returns from VideoDatasetIterator and contains images and image timestamps
            # Now add events and some parameters in the tuple
            yield batch + (events, simu_params)


def pad_collate_fn_images_and_events(data_list):
    """
    Here we pad with last image/ timestamp to get a contiguous batch
    """
    images, timestamps, target_indices, first_times, events_cd, simu_params = zip(*data_list)
    video_len = [item.shape[-1] for item in images]
    max_len = max([item.shape[-1] for item in images])
    b = len(images)
    c, h, w = images[0].shape[1:-1]
    out_images = torch.zeros((c, h, w, sum(video_len)), dtype=images[0].dtype)
    out_timestamps = torch.zeros((b, max_len), dtype=timestamps[0].dtype)
    target_indices = torch.cat(target_indices).int()
    current_ind = 0
    for i in range(b):
        video = images[i]
        ilen = video.shape[-1]
        out_images[..., current_ind: current_ind + ilen] = video
        current_ind += ilen
        out_timestamps[i, :ilen] = timestamps[i]
        out_timestamps[i, ilen:] = timestamps[i][:, ilen - 1:].unsqueeze(1)

    first_times = torch.FloatTensor(first_times)
    return {'images': out_images, 'timestamps': out_timestamps, 'target_indices': target_indices,
            'first_times': first_times, 'video_len': torch.tensor(video_len, dtype=torch.int32),
            'events_cd': events_cd, 'simu_params': simu_params}


def make_video_dataset_with_events_cpu(
        path, batch_size, num_workers, min_length, max_length, dic_params_video_dataset, dic_params_event_simulator,
        seed=None):
    metadatas = build_metadata(path, min_length, max_length)
    print('scheduled streams: ', len(metadatas))

    def iterator_fun(metadata):
        return VideoDatasetWithEventsIterator(
            metadata, dic_params_video_dataset=dic_params_video_dataset,
            dic_params_events_simulator=dic_params_event_simulator, discard_events_between_batches=True)
    dataset = StreamDataset(metadatas, iterator_fun, batch_size, "data", None, seed)
    dataloader = StreamDataLoader(dataset, num_workers, pad_collate_fn_images_and_events)
    dataloader.dic_params_video_dataset = dic_params_video_dataset
    dataloader.dic_params_event_simulator = dic_params_event_simulator
    return dataloader
