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
import torch
import numpy as np

from metavision_core_ml.data.video_stream import TimedVideoStream
from metavision_core_ml.data.image_planar_motion_stream import PlanarMotionStream
from metavision_core_ml.data.stream_dataloader import StreamDataset, StreamDataLoader
from metavision_core_ml.data.scheduling import build_metadata
from metavision_core_ml.utils.files import is_image, is_video


class VideoDatasetIterator(object):
    """
    Dataset Iterator streaming images and timestamps

    Args:
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
        pause_probability (float): probability to add a pause (no events) (works only with PlanarMotionStream)
        max_optical_flow_threshold (float): maximum allowed optical flow between two consecutive frames (works only with PlanarMotionStream)
        max_interp_consecutive_frames (int): maximum number of interpolated frames between two consecutive frames (works only with PlanarMotionStream)
        max_number_of_batches_to_produce (int): maximum number of batches to produce
        crop_image (bool): crop images or resize them
    """

    def __init__(self, metadata, height, width, rgb, mode='frames', min_tbins=3, max_tbins=10,
                 min_dt=3000, max_dt=50000, batch_times=1,
                 pause_probability=0.5,
                 max_optical_flow_threshold=2., max_interp_consecutive_frames=20,
                 max_number_of_batches_to_produce=None, crop_image=False):
        assert mode in ["frames", "delta_t", "random"]
        if is_video(metadata.path):
            self.image_stream = TimedVideoStream(metadata.path, height, width, rgb=rgb,
                                                 start_frame=metadata.start_frame, max_frames=len(metadata))
        elif is_image(metadata.path):
            self.image_stream = PlanarMotionStream(metadata.path, height, width, len(metadata), rgb=rgb,
                                                   pause_probability=pause_probability,
                                                   max_optical_flow_threshold=max_optical_flow_threshold,
                                                   max_interp_consecutive_frames=max_interp_consecutive_frames,
                                                   crop_image=crop_image)
        self.height = height
        self.width = width
        self.rgb = rgb
        self.metadata = metadata
        self.mode = ['frames', 'delta_t'][np.random.randint(0, 2)] if mode == 'random' else mode
        self.num_tbins = np.random.randint(min_tbins, max_tbins)
        self.delta_t = np.random.randint(min_dt, max_dt)
        self.batch_times = batch_times
        self.max_number_of_batches_to_produce = max_number_of_batches_to_produce
        self.min_tbins = min_tbins

    def get_size(self):
        return self.height, self.width

    def __iter__(self):
        out = []
        times = []
        target_indices = []
        first_time = True
        last_time = None
        n_times_dt = 0
        t0 = None
        nb_batches_produced = 0
        for i, (img, ts) in enumerate(self.image_stream):

            if img.ndim == 3:
                img = np.moveaxis(img, 2, 0)
            else:
                img = img[None]

            out.append(img[None, ..., None])  # B,C,H,W,T or B,H,W,T
            times.append(ts)
            if last_time is None:
                last_time = ts
            dt = times[-1] - last_time
            if self.mode == 'delta_t' and dt >= self.delta_t:
                n_times_dt += 1
                last_time = times[-1]
                target_indices.append(len(out) - 1)
            elif self.mode == 'frames' and len(out) % self.num_tbins == 0:
                target_indices.append(len(out) - 1)

            if (self.mode == 'frames' and len(out) == (self.batch_times * self.num_tbins)) or (
                    self.mode == 'delta_t' and n_times_dt >= self.batch_times):

                sequence = torch.from_numpy(np.concatenate(out, axis=-1))  # (1, C, H, W, T) or (1, H, W, T)
                timestamps = torch.FloatTensor(times)[None, :]  # (1,T)
                assert target_indices[-1] == len(out) - 1
                assert len(target_indices) == self.batch_times
                target_indices = torch.FloatTensor(target_indices)[None, :]  # (1, self.batch_times=1)
                yield sequence, timestamps, target_indices, first_time
                nb_batches_produced += 1

                if self.max_number_of_batches_to_produce and nb_batches_produced >= self.max_number_of_batches_to_produce:
                    return

                out = []
                times = []
                target_indices = []
                first_time = False
                n_times_dt = 0

        if len(out) >= self.min_tbins:
            sequence = torch.from_numpy(np.concatenate(out, axis=-1))
            timestamps = torch.FloatTensor(times)[None, :]  # B,T
            target_indices.append(len(out) - 1)

            if len(target_indices) < self.batch_times:
                target_indices = target_indices + [target_indices[-1]] * (self.batch_times - len(target_indices))

            assert len(target_indices) == self.batch_times
            target_indices = torch.FloatTensor(target_indices)[None, :]  # B,T
            yield sequence, timestamps, target_indices, first_time


def pad_collate_fn(data_list):
    """
    Here we pad with last image/ timestamp to get a contiguous batch
    """
    images, timestamps, target_indices, first_times = zip(*data_list)
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
    return {
        'images': out_images,
        'timestamps': out_timestamps,
        'target_indices': target_indices,
        'first_times': first_times,
        'video_len': torch.tensor(
            video_len,
            dtype=torch.int32)}


def make_video_dataset(
        path, num_workers, batch_size, height, width, min_length, max_length, mode='frames',
        min_frames=5, max_frames=30, min_delta_t=5000, max_delta_t=50000, rgb=False, seed=None, batch_times=1,
        pause_probability=0.5, max_optical_flow_threshold=2., max_interp_consecutive_frames=20,
        max_number_of_batches_to_produce=None, crop_image=False):
    """
    Makes a video / moving picture dataset.

    Args:
        path (str): folder to dataset
        batch_size (int): number of video clips / batch
        height (int): height
        width (int): width
        min_length (int): min length of video
        max_length (int): max length of video
        mode (str): 'frames' or 'delta_t'
        min_frames (int): minimum number of frames per batch
        max_frames (int): maximum number of frames per batch
        min_delta_t (int): in microseconds, minimum duration per batch
        max_delta_t (int): in microseconds, maximum duration per batch
        rgb (bool): retrieve frames in rgb
        seed (int): seed for randomness
        batch_times (int): number of time steps in training sequence
        pause_probability (float): probability to add a pause during the sequence (works only with PlanarMotionStream)
        max_optical_flow_threshold (float): maximum allowed optical flow between two consecutive frames (works only with PlanarMotionStream)
        max_interp_consecutive_frames (int): maximum number of interpolated frames between two consecutive frames (works only with PlanarMotionStream)
        max_number_of_batches_to_produce (int): maximum number of batches to produce. Makes sure the stream will not
                                                produce more than this number of consecutive batches using the same
                                                image or video.
    """
    metadatas = build_metadata(path, min_length, max_length)
    print('scheduled streams: ', len(metadatas))

    def iterator_fun(metadata):
        return VideoDatasetIterator(
            metadata, height, width, rgb=rgb, mode=mode, min_tbins=min_frames,
            max_tbins=max_frames, min_dt=min_delta_t, max_dt=max_delta_t, batch_times=batch_times,
            pause_probability=pause_probability, max_optical_flow_threshold=max_optical_flow_threshold,
            max_interp_consecutive_frames=max_interp_consecutive_frames, crop_image=crop_image)
    dataset = StreamDataset(metadatas, iterator_fun, batch_size, "data", None, seed)
    dataloader = StreamDataLoader(dataset, num_workers, pad_collate_fn)
    # TODO: one day unify make_video_dataset and make_video_dataset_with_events_cpu
    dic_params_video_dataset = {"height": height, "width": width,
                                "min_tbins": min_frames, "max_tbins": max_frames,
                                "rgb": rgb, "pause_probability": pause_probability,
                                "max_number_of_batches_to_produce": max_number_of_batches_to_produce}

    dataloader.dic_params_video_dataset = dic_params_video_dataset
    return dataloader
