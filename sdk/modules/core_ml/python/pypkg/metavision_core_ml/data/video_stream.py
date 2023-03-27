# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
Video .mp4 or .avi Iterator
Right now backend is OpenCV
"""
import os
import random
import numpy as np
import cv2
import skvideo.io


class TimedVideoStream(object):
    """
    Video iterator opening both a video stream and a file of timestamps.
    If it does not exist, it generates them with a regular period of the frequency described in the metadata
    of the video or 1/override_fps if given.
    Timestamps are delivered in microseconds.

    Args:
        input_path (str): Path to the file to read.
        height (int): Height of the output frames.
        width (int): Width of the output frames.
        start_frame (int): First frame to seek to
        max_frames (int): Maximum number of frames loaded (if greater than the number of frames in the video it will return all of them).
        rgb (bool): Whether the output should be in rgb or greyscale.
        override_fps (int): Frequency of the generated timestamps Hz (if timestamp file is not available)
        If equal to 0, the frequency will be taken from the video's metadata
    """

    def __init__(
            self, video_filename, height=-1, width=-1, start_frame=0, max_frames=0, rgb=False, override_fps=0):

        self.video_filename = video_filename
        self.rgb = rgb

        metadata = skvideo.io.ffprobe(video_filename)
        self.num_frames = int(metadata["video"]["@nb_frames"])

        self.start_frame = start_frame
        self.max_frames = (self.num_frames - start_frame) if max_frames <= 0 else min(max_frames,
                                                                                      self.num_frames - start_frame)
        self.end_frame = start_frame + self.max_frames
        self.height, self.width = int(metadata["video"]["@height"]), int(metadata["video"]["@width"])

        avg_frame_rate = eval(metadata["video"]["@avg_frame_rate"])
        if override_fps:
            self.freq = override_fps
        else:
            self.freq = avg_frame_rate

        self.duration_s = float(metadata["video"]['@duration'])

        if width < 0:
            width = self.width
        if height < 0:
            height = self.height

        self.start_ts = start_frame / avg_frame_rate
        assert self.start_ts <= float(metadata["video"]["@duration"])

        self.input_dict = {'-ss': str(self.start_ts)} if start_frame else {}
        self.outputdict = {"-s": str(width) + "x" + str(height)}
        if not rgb:
            self.outputdict["-pix_fmt"] = "gray"

        ts_path = os.path.splitext(video_filename)[0] + '_ts.npy'
        if os.path.exists(ts_path):
            assert not override_fps, "Parameter override_fps should not be given if _ts.npy file is provided"
            self.timestamps_us = (np.load(ts_path) * 1e6).round()
        else:
            timestamps_s, step = np.linspace(0, self.num_frames / self.freq, self.num_frames,
                                             endpoint=False, retstep=True)
            assert abs(step - 1 / self.freq) < 1e-6
            self.timestamps_us = (timestamps_s * 1e6).round()

        self.timestamps_us = self.timestamps_us[self.start_frame:self.end_frame]

    def get_size(self):
        """Function returning the size of the imager which produced the events.

        Returns:
            Tuple of int (height, width) which might be (None, None)"""
        return self.height, self.width

    def __iter__(self):
        # Initialized here because vreader does not reinitialize when calling iter
        self._it_video = skvideo.io.vreader(
            self.video_filename,
            num_frames=self.max_frames,
            inputdict=self.input_dict,
            outputdict=self.outputdict)  # Gets the closest frame before start_ts
        self._it_timestamps = iter(self.timestamps_us)
        return self

    def __next__(self):
        im = next(self._it_video)
        if not self.rgb:
            im = im.squeeze()
        ts = next(self._it_timestamps)
        return im, ts

    def __len__(self):
        return len(self.timestamps_us)
