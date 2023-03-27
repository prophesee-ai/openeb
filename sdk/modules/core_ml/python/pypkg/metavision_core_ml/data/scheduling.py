# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Scheduling System for Videos
"""
import os
import random
import numpy as np
import skvideo.io
import json
from metavision_core_ml.utils.files import grab_videos, grab_images


class Metadata(object):
    """
    Represents part of a file to be read.

    Args:
        path (str): path to video
        start_frame (int): first frame to seek to
        end_frame (int): last frame to read
    """

    def __init__(self, path, start_frame, end_frame):
        self.path = path
        self.start_frame = start_frame
        self.end_frame = end_frame

    def __len__(self):
        return self.end_frame - self.start_frame

    def split(self, num_frames):
        out = []
        for i in range(self.start_frame, self.end_frame, num_frames):
            j = i + num_frames
            out.append(Metadata(self.path, i, j))
        return out

    def __repr__(self):
        return f"Metadata:\n  path: {self.path}\n  start_frame: {self.start_frame}\n  end_frame: {self.end_frame}"


def build_video_metadata(folder):
    """
    Builds Metadata from videos

    Args:
        folder (str): path to videos (only looks in current directory, not subfolders)
    """
    paths_images = grab_images(folder, recursive=False)
    paths = grab_videos(folder, recursive=False)
    assert len(paths_images) == 0 or len(
        paths) == 0, f"Error {folder} contains both videos and images"
    # If there are no videos in the folder, we don't dump the json file
    if not len(paths):
        return []
    info_json_path = os.path.join(folder, 'video_info.json')
    if os.path.exists(info_json_path):
        with open(info_json_path, 'r') as f:
            info = json.load(f)
    else:
        try:
            info = {os.path.basename(path): int(skvideo.io.ffprobe(path)["video"]["@nb_frames"]) for path in paths}
        except BaseException:
            return []
        with open(info_json_path, 'w') as f:
            json.dump(info, f)
    out = []
    for path in paths:
        num_frames = info[os.path.basename(path)]
        out.append(Metadata(path, 0, num_frames))
    return out


def build_image_metadata(folder, min_size, max_size, denominator=1):
    """
    Build Metadata from images

    Args:
        folder (str): path to pictures
        min_size (int): minimum number of frames
        max_size (int): maximum number of frames
        denominator (int): num_frames will always be a multiple of denominator.
                           It is used to avoid having batches that are missing some frames and need to be padded. This
                           happens when the number of time steps is not a multiple of num_frames.
    """
    paths = grab_images(folder, recursive=False)
    sizes = np.round(np.random.randint(min_size, max_size, size=len(paths)) / float(denominator)) * denominator
    sizes = sizes.astype(np.int)
    sizes[sizes == 0] = denominator
    out = []
    for path, num_frames in zip(paths, sizes):
        out.append(Metadata(path, 0, num_frames))
    return out


def split_video_metadata(metadatas, min_size, max_size):
    """
    Split video metadata into smaller ones.

    Args:
        metadatas (list): list of metadata objects
        min_size (int): minimum number of frames
        max_size (int): maximum number of frames
    """
    sizes = np.random.randint(min_size, max_size, size=len(metadatas))
    out = []
    for md, size in zip(metadatas, sizes):
        out += md.split(size)
    random.shuffle(out)
    return out


def build_metadata(folder, min_length, max_length, denominator=1):
    """
    Builds Metadata for Videos and Images

    Args:
        folder (str): path to videos or images
        min_length (int): minimum number of frames
        max_length (int): maximum number of frames
        denominator (int): denominator of number of frames for image metadata
    """
    metadata = build_video_metadata(folder)
    metadata = split_video_metadata(metadata, min_length, max_length)
    if not len(metadata):
        print('no video, grabbing pictures')
        metadata = build_image_metadata(folder, min_length, max_length, denominator=denominator)
    return metadata
