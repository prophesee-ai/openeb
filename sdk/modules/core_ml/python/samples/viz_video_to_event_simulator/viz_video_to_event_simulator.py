# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Displays events produced by the frame-to-event simulator.
Streams either from a moving picture or a video with timestamps.
The application buffers 3 iterators:
- Image or Video Streamer
- EventSimulator
- FixedTime/CountBufferizer

and finally applies some event tensorization.
"""

from __future__ import absolute_import

import os
import time
import cv2
import argparse

import numpy as np
from metavision_core_ml.video_to_event.simulator import EventSimulator
from metavision_core.event_io.event_bufferizer import FixedCountBuffer
from metavision_core.event_io import DatWriter
from metavision_core_ml.preprocessing import viz_events
from metavision_core_ml.data.video_stream import TimedVideoStream
from metavision_core_ml.data.image_planar_motion_stream import PlanarMotionStream


def parse_args(only_default_values=False):
    parser = argparse.ArgumentParser(description='Run a simple event based simulator on a video or an image',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', help='path to a video or an image from which we will produce'
                        ' the corresponding events ')

    parser.add_argument('--n_events', type=int, default=50000,
                        help='number of events to display at once')
    parser.add_argument('--height_width', nargs=2, default=None, type=int,
                        help="if set, scales the input image to the requested values.")
    parser.add_argument('--crop_image', action="store_true", help='crop images instead of resizing them.')
    parser.add_argument("--no_display", dest="display", action="store_false", help='disable the graphical return.')
    parser.add_argument("--verbose", action="store_true", help='set to have the speed of the simulator in ev/s')
    parser.add_argument('-o', "--output", help="if provided, will write the events in the corresponding path")
    parser.add_argument('-fps', '--override_fps', default=0, type=float,
                        help="if positive, overrides the framerate of the input video. Useful for slow motion videos.")

    simulator_options = parser.add_argument_group('Simulator parameters')
    simulator_options.add_argument("--Cp", default="0.15", type=float,
                                   help="mean for positive event contrast threshold distribution")
    simulator_options.add_argument("--Cn", default="0.10", type=float,
                                   help="mean value for negative event contrast threshold distribution")
    simulator_options.add_argument(
        "--refractory_period", default=1, type=float,
        help="time interval (in us), after firing an event during which a pixel won't emit a new event.")
    simulator_options.add_argument(
        "--sigma_threshold", type=float, default="0.001", help="standard deviation for threshold"
        "distribution across the array of pixels. The higher it is the less reliable the imager.")
    simulator_options.add_argument("--cutoff_hz", default=0, type=float,
                                   help="cutoff frequency for photodiode latency simulation")
    simulator_options.add_argument("--leak_rate_hz", type=float, default=0,
                                   help="frequency of reference value leakage")
    simulator_options.add_argument("--shot_noise_rate_hz", default=10, type=float,
                                   help="frequency for shot noise events")

    return parser.parse_args()


def main(args):
    [height, width] = [-1, -1] if args.height_width is None else args.height_width
    path = args.path
    crop_image = args.crop_image
    assert os.path.exists(path), f"{path} doesn't exist!"
    assert os.path.isfile(path), f"{path} is not a file"
    start = time.time()
    if os.path.splitext(path)[1] in [".jpg", ".JPG", ".png", ".PNG"]:
        image_stream = PlanarMotionStream(path, height, width, crop_image)
    else:
        image_stream = TimedVideoStream(path, height, width, override_fps=args.override_fps)

    if args.height_width is None:
        height, width = image_stream.get_size()

    n_events = args.n_events
    fixed_buffer = FixedCountBuffer(n_events)
    simu = EventSimulator(height, width, args.Cp, args.Cn, args.refractory_period, cutoff_hz=args.cutoff_hz,
                          sigma_threshold=args.sigma_threshold, shot_noise_rate_hz=args.shot_noise_rate_hz)

    if args.display:
        cv2.namedWindow('events', cv2.WINDOW_NORMAL)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    if args.output:
        writer = DatWriter(args.output, height=height, width=width)

    for img, ts in image_stream:
        total = simu.image_callback(img, ts)

        if total < n_events:
            continue

        events = simu.get_events()
        simu.flush_events()

        events = fixed_buffer(events)

        if not len(events):
            continue

        end = time.time()
        dt = events['t'][-1] - events['t'][0]

        image_rgb = viz_events(events, width=width, height=height)

        if args.verbose:
            num_evs = len(events)
            max_evs = np.unique(events["x"] * height + events['y'], return_counts=True)[1].max()
            print(
                f"runtime: {(end-start)*1000:.5f} ms, max ev/pixel: {max_evs}, total Mev: {num_evs * 1e-6:.5f}, dt: {dt} us")

        if args.display:
            cv2.imshow('events', image_rgb[..., ::-1])
            cv2.imshow('image', img)
            key = cv2.waitKey(5)
            if key == 27:
                break
        if args.output:
            writer.write(events)

        start = time.time()

    if args.display:
        cv2.destroyAllWindows()

    if args.output:
        writer.close()


if __name__ == '__main__':
    main(parse_args())
