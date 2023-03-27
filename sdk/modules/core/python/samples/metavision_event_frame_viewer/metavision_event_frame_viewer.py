# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Code sample to display the content of DIFF3D and HISTO3D event frames
"""

from metavision_core.event_io.raw_info import raw_histo_header_bits_per_channel
from metavision_core.event_io import EventFrameIterator

import os
import numpy as np
import cv2
from skvideo.io import FFmpegWriter


def display_event_frames(input_path, output_video_path=None, disable_display=False):
    if output_video_path:
        assert not os.path.exists(output_video_path), f"Error: path {output_video_path} already exists"
    video_writer = None
    if output_video_path:
        video_writer = FFmpegWriter(output_video_path)

    mv_it = EventFrameIterator(input_path=input_path)
    frame_type = mv_it.get_frame_type()
    if frame_type == "DIFF3D":
        if not disable_display:
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    elif frame_type == "HISTO3D":
        if not disable_display:
            cv2.namedWindow("img_neg", cv2.WINDOW_NORMAL)
            cv2.namedWindow("img_pos", cv2.WINDOW_NORMAL)
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        bits_n, bits_p = raw_histo_header_bits_per_channel(input_path)
    else:
        raise NotImplementedError(f"Unsupported type of frame: {frame_type}")

    stop = False
    for frame_idx, frame in enumerate(mv_it):
        if frame_type == "DIFF3D":
            print(f"frame_idx: {frame_idx}, shape: {frame.shape}, min: {frame.min()}, max: {frame.max()}")
            assert frame.dtype == np.int8
            assert frame.shape == (320, 320)
            img = frame.astype(np.uint8) + 128
            if not disable_display:
                cv2.imshow("img", img)
            if video_writer:
                video_writer.writeFrame(img)

        elif frame_type == "HISTO3D":
            print(f"frame_idx: {frame_idx}, shape: {frame.shape},    " +
                  f"negative channel (min,max): ({frame[..., 0].min()},{frame[..., 0].max()})    " +
                  f"positive channel (min,max): ({frame[..., 1].min()},{frame[..., 1].max()})    ")
            assert frame.dtype == np.uint8
            assert frame.shape == (320, 320, 2)
            assert (frame[:, :, 0] < 2**bits_n).all()
            assert (frame[:, :, 1] < 2**bits_p).all()

            img_neg = np.ascontiguousarray(frame[..., 0]) * 2**(8 - bits_n)
            img_pos = np.ascontiguousarray(frame[..., 1]) * 2**(8 - bits_p)
            img = np.concatenate([img_neg[..., None], img_pos[..., None],
                                  np.zeros((320, 320, 1), dtype=np.uint8)], axis=2)
            if not disable_display:
                cv2.imshow("img_neg", img_neg)
                cv2.imshow("img_pos", img_pos)
                cv2.imshow("img", img[..., ::-1])
            if video_writer:
                video_writer.writeFrame(img)
        else:
            raise NotImplementedError(f"Unsupported type of frame: {frame_type}")

        if not disable_display:
            k = cv2.waitKey(1)
            if k == ord('q'):
                print("!! STOP !!")
                break

    if not disable_display:
        cv2.destroyAllWindows()
    if video_writer:
        video_writer.close()


if __name__ == "__main__":
    import fire
    fire.Fire(display_event_frames)
