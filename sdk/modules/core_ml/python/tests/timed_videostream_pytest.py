# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for the TimedVideoStream Interface
"""
import os
import pytest
import numpy as np
# Temporary solution to fix the numpy deprecated alias in skvideo: https://github.com/scikit-video/scikit-video/issues/154#issuecomment-1445239790
# Will be deleted in MV-2134 when skvideo makes the correction
np.float = np.float64
np.int = np.int_
import skvideo.io
import cv2
import random

from metavision_core_ml.data.video_stream import TimedVideoStream


def pytestcase_test_seek(tmpdir):
    """Checks number of frames is correct and we are able to seek properly"""
    tmp_output_video_filename = str(tmpdir.join("dummy_video.mp4"))
    if os.path.exists(tmp_output_video_filename):
        os.remove(tmp_output_video_filename)
    assert not os.path.exists(tmp_output_video_filename)

    # Begin write frames
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 200)
    fontScale = 1
    fontColor = (1, 0, 0)
    lineType = 2

    video_writer = skvideo.io.FFmpegWriter(tmp_output_video_filename, outputdict={
        '-vcodec': 'libx264',  # use the h.264 codec
                   '-crf': '0',  # set the constant rate factor to 0, which is lossless
                   '-preset': 'veryslow'  # the slower the better compression, in princple, try
        # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    })

    all_images = []
    nb_frames = 100
    for i in range(nb_frames):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        idx_str = str(i).zfill(6)

        cv2.putText(image, idx_str,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        all_images.append(image[..., ::-1])

        video_writer.writeFrame(image[:, :, ::-1])
    video_writer.close()
    assert os.path.isfile(tmp_output_video_filename)
    # End write frames

    tvs = TimedVideoStream(video_filename=tmp_output_video_filename, start_frame=0, rgb=True)
    assert len(tvs) == nb_frames
    it_tvs = iter(tvs)
    nb_loaded_images = 0
    for image, ts in it_tvs:
        assert (np.abs(image.astype(np.float) - all_images[nb_loaded_images].astype(np.float)) <= 1).all()
        nb_loaded_images += 1
    assert nb_loaded_images == nb_frames

    for i in range(10):
        nb_loaded_images = 0
        start_frame = random.randint(0, nb_frames - 1)
        tvs = TimedVideoStream(video_filename=tmp_output_video_filename, start_frame=start_frame, rgb=True)
        assert len(tvs) == nb_frames - start_frame
        it_tvs = iter(tvs)
        for image, ts in it_tvs:
            assert (np.abs(image.astype(np.float) -
                           all_images[start_frame + nb_loaded_images].astype(np.float)) <= 1).all()
            nb_loaded_images += 1
        assert nb_loaded_images == nb_frames - start_frame

    os.remove(tmp_output_video_filename)
    assert not os.path.exists(tmp_output_video_filename)
