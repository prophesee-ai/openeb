# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Image and Corner stream data loader
"""
import torch
import numpy as np

from metavision_core_ml.data.corner_planar_motion_stream import CornerPlanarMotionStream
from metavision_core_ml.data.stream_dataloader import StreamDataset, StreamDataLoader
from metavision_core_ml.data.scheduling import build_metadata


class CornerVideoDatasetIterator(object):
    """
    Dataset Iterator streaming images, timestamps and corners

    Args:
        metadata (object): path to picture or video
        height (int): height of input images / video clip
        width (int): width of input images / video clip
        rgb (bool): stream rgb videos
        number_of_heatmaps (int): The number of heatmaps containing corner locations
        batch_times (int): number of timesteps of training sequences
    """

    def __init__(self, metadata, height, width, rgb, number_of_heatmaps=10, batch_times=1):
        self.image_stream = CornerPlanarMotionStream(metadata.path, height, width, len(metadata), rgb=rgb)
        self.height = height
        self.width = width
        self.rgb = rgb
        self.metadata = metadata
        self.mode = 'frames'
        self.number_of_heatmaps = number_of_heatmaps
        self.batch_times = batch_times

    def __iter__(self):
        img_out = []
        corners_out = []
        times = []
        target_indices = []
        first_time = True
        last_time = None
        for i, (img, corners, ts) in enumerate(self.image_stream):
            if img.ndim == 3:
                img = np.moveaxis(img, 2, 0)
                corners = np.moveaxis(corners, 2, 0)
            else:
                img = img[None]
                corners = corners[None]

            img_out.append(img[None, ..., None])  # B,C,H,W,T or B,H,W,T
            corners_out.append(corners[None, ..., None])
            times.append(ts)
            if last_time is None:
                last_time = ts

            if len(img_out) % self.number_of_heatmaps == 0:
                target_indices.append(len(img_out)-1)

            if len(corners_out) == (self.batch_times*self.number_of_heatmaps):
                image_sequence = torch.from_numpy(np.concatenate(img_out, axis=-1))
                corner_sequence = torch.from_numpy(np.concatenate(corners_out, axis=-1))
                timestamps = torch.FloatTensor(times)[None, :]  # B,T
                assert target_indices[-1] == len(img_out)-1
                assert len(target_indices) == self.batch_times
                target_indices = torch.FloatTensor(target_indices)[None, :]  # B,T
                yield image_sequence, corner_sequence, timestamps, target_indices, first_time
                img_out = []
                corners_out = []
                times = []
                target_indices = []
                first_time = False


def pad_collate_fn(data_list):
    """
    Here we pad with last image/ timestamp to get a contiguous batch
    """
    images, corners, timestamps, target_indices, first_times = zip(*data_list)
    video_len = [item.shape[-1] for item in images]
    max_len = max([item.shape[-1] for item in images])
    b = len(images)
    c, h, w = images[0].shape[1:-1]
    out_images = torch.zeros((c, h, w, sum(video_len)), dtype=images[0].dtype)
    out_corners = torch.zeros((c, h, w, sum(video_len)), dtype=images[0].dtype)
    out_timestamps = torch.zeros((b, max_len), dtype=timestamps[0].dtype)
    target_indices = torch.cat(target_indices).int()
    current_ind = 0
    for i in range(b):
        video = images[i]
        ilen = video.shape[-1]
        out_images[..., current_ind: current_ind + ilen] = video
        out_corners[..., current_ind: current_ind + ilen] = corners[i]
        current_ind += ilen
        out_timestamps[i, :ilen] = timestamps[i]
        out_timestamps[i, ilen:] = timestamps[i][:, ilen - 1:].unsqueeze(1)

    first_times = torch.FloatTensor(first_times)
    return {'images': out_images,
            'corners': out_corners,
            'timestamps': out_timestamps,
            'target_indices': target_indices,
            'first_times': first_times,
            'video_len': torch.tensor(video_len, dtype=torch.int32)}


def make_corner_video_dataset(path, num_workers, batch_size, height, width, min_length, max_length,
                              number_of_heatmaps=10, rgb=False, seed=None, batch_times=1):
    """
    Makes a video/ moving picture dataset.

    Args:
        path (str): folder to dataset
        batch_size (int): number of video clips / batch
        height (int): height
        width (int): width
        min_length (int): min length of video
        max_length (int): max length of video
        mode (str): 'frames' or 'delta_t'
        num_tbins (int): number of bins in event volume
        number_of_heatmaps (int): number of corner heatmaps predicted by the network
        rgb (bool): retrieve frames in rgb
        seed (int): seed for randomness
        batch_times (int): number of time steps in training sequence
    """
    metadata = build_metadata(path, min_length, max_length, denominator=number_of_heatmaps*batch_times)
    print('scheduled streams: ', len(metadata))

    def iterator_fun(metadata):
        return CornerVideoDatasetIterator(
            metadata, height, width, rgb=rgb, number_of_heatmaps=number_of_heatmaps, batch_times=batch_times)
    dataset = StreamDataset(metadata, iterator_fun, batch_size, "data", None, seed)
    dataloader = StreamDataLoader(dataset, num_workers, pad_collate_fn)
    return dataloader
