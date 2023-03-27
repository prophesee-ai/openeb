# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
This wrapper shows and/or writes a video
"""
import cv2
from skvideo.io import FFmpegWriter
import os


class ShowWrite(object):
    """
    ShowWrite either shows and/or writes a video

    Args:
        window_name (str): if empty, will not show
        video_path (str): if empty, will not write
        crf (int): compression factor for video output
    """

    def __init__(self, window_name, video_path, crf=30):
        assert window_name or video_path
        self.window_name = window_name
        if window_name:
            self.window = cv2.namedWindow(window_name)
        if video_path:
            dirname = os.path.dirname(video_path)
            if dirname and not os.path.isdir(dirname):
                os.makedirs(dirname)
            self.video_out = FFmpegWriter(video_path, outputdict={
                '-vcodec': 'libx264',  # use the h.264 codec
                '-crf': str(crf),  # set the constant rate factor to 0, which is lossless
                '-preset': 'veryslow'  # the slower the better compression, in princple, try
                # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
            })
        self.called = False
        self.last_ts = None

    def __call__(self, image, delay=5):
        key = None
        if hasattr(self, "window"):
            cv2.imshow(self.window_name, image)
            key = cv2.waitKey(delay)
        if hasattr(self, "video_out"):
            if image.ndim == 3:
                image = image[..., ::-1]
            self.video_out.writeFrame(image)
        self.called = True
        return key

    def __del__(self):
        if hasattr(self, "video_out") and self.called:
            self.video_out.close()
        if hasattr(self, "window"):
            cv2.destroyWindow(self.window_name)
