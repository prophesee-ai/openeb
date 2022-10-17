# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from metavision_core.event_io import AdaptiveRateEventsIterator
from metavision_sdk_core import AdaptiveRateEventsSplitterAlgorithm
import numpy as np
import cv2
import os
from skvideo.io import FFmpegWriter


def events_to_diff_image(events, sensor_size, strict_coord=True):
    """
    Place events into an image using numpy
    """
    xs = events["x"]
    ys = events["y"]
    ps = events["p"] * 2 - 1

    mask = (xs < sensor_size[1]) * (ys < sensor_size[0]) * (xs >= 0) * (ys >= 0)
    if strict_coord:
        assert (mask == 1).all()
    coords = np.stack((ys*mask, xs*mask))
    ps *= mask

    try:
        abs_coords = np.ravel_multi_index(coords, sensor_size)
    except ValueError:
        raise ValueError("Issue with input arrays! coords={}, min_x={}, min_y={}, max_x={}, max_y={}, coords.shape={}, sum(coords)={}, sensor_size={}".format(
            coords, min(xs), min(ys), max(xs), max(ys), coords.shape, np.sum(coords), sensor_size))

    img = np.bincount(abs_coords, weights=ps, minlength=sensor_size[0]*sensor_size[1])
    img = img.reshape(sensor_size)
    return img


def split_into_frames(filename_raw, thr_var_per_event=5e-4, downsampling_factor=2, disable_display=False,
                      filename_output_video=None):
    """ This function loads a sequence, splits it into sharp event frames, and displays the result

    This approach is an alternative to fixed delta_t (where events are gathered over a fixed time window)
    or fixed N events (where a constant number of events are gathered). Here, the number of events per slice
    (as well as the slice duration) is adaptive and depends on the content of the events stream.

    It will generate a sequence of reasonably sharp event frames. Those could be used in a variable duration
    processing pipeline (for example detection and tracking, or optical flow computation). We could also consider
    dropping some of the frames to cope with limited computational budget (detection).

    Args:
        filename_raw (str): input sequence filename to process
        thr_var_per_event (float): minimum variance per event to reach before generating a new frame
        downsampling_factor (int): reduction factor to apply to input frame. Original coordinates will be
                                   multiplied by 2**(-downsampling_factor)
        disable_display (boolean): disable the output window
        filename_output_video (str): writes an mp4 output video of the generated event frames
    """

    assert downsampling_factor == int(downsampling_factor), "Error: downsampling_factor must be an integer"
    assert downsampling_factor >= 0, "Error: downsampling_factor must be >= 0"

    mv_adaptive_rate_iterator = AdaptiveRateEventsIterator(input_path=filename_raw,
                                                           thr_var_per_event=thr_var_per_event,
                                                           downsampling_factor=downsampling_factor)

    height, width = mv_adaptive_rate_iterator.get_size()

    if filename_output_video is None:
        video_process = None
    else:
        assert not os.path.exists(filename_output_video)
        video_process = FFmpegWriter(filename_output_video)

    if video_process or not disable_display:
        img_bgr = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)

    for events in mv_adaptive_rate_iterator:
        assert events.size > 0
        start_ts = events[0]["t"]
        end_ts = events[-1]["t"]
        print("frame: {} -> {}   delta_t: {}   fps: {}   nb_ev: {}".format(start_ts, end_ts,
                                                                           end_ts - start_ts,
                                                                           1e6 / (end_ts - start_ts),
                                                                           events.size))
        if video_process or not disable_display:
            img = events_to_diff_image(events, sensor_size=(height, width))
            img_bgr[...] = 0
            img_bgr[img < 0, 0] = 255
            img_bgr[img > 0, 1] = 255

            chunk_start_ts = events[0]["t"]
            chunk_end_ts = events[-1]["t"]
            delta_t_frame = chunk_end_ts - chunk_start_ts + 1
            frame_txt = "ts: {} -> {}  delta_t: {}  fps: {}  (nb_ev): {}".format(chunk_start_ts, chunk_end_ts,
                                                                                 delta_t_frame,
                                                                                 int(1.e6/delta_t_frame),
                                                                                 events.size)
            img_bgr[20:45, ...] = 0
            cv2.putText(img_bgr,
                        frame_txt,
                        (int(0.05 * width), 40),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 100))

        if video_process:
            video_process.writeFrame(img_bgr.astype(np.uint8)[..., ::-1])
        if not disable_display:
            cv2.imshow("img", img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if video_process:
        video_process.close()
    if not disable_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import fire
    fire.Fire(split_into_frames)
