# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
This application demonstrates how to use Metavision SDK Stream module to slice events from a camera
"""

import numpy as np

from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_stream import Camera, CameraStreamSlicer, FileConfigHints, SliceCondition
from metavision_sdk_ui import MTWindow, BaseWindow, EventLoop, UIAction, UIKeyEvent


def parse_args():
    """
    Parse command line arguments
    """
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Code sample showing how to use the Metavision CameraStreamSlicer to slice the events from a camera"
            " or a file into slices of a fixed number of events or a fixed duration."))

    # Base options
    parser.add_argument(
        '-i', '--input-event-file',
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
    parser.add_argument('-s', '--camera-serial-number',
                        help="Serial number of the camera to be used")
    parser.add_argument('-r', '--real-time-playback', action="store_true",
                        help="Flag to play record at recording speed")

    # Slicing options
    parser.add_argument('-m', '--slicing-mode', type=str,
                        choices=['N_EVENTS', 'N_US', 'MIXED'],
                        default='N_US', help="Slicing mode (i.e. N_EVENTS, N_US, MIXED)")
    parser.add_argument('-t', '--delta-ts', type=int, default=10000,
                        help="Slice duration in microseconds (default=10000us)")
    parser.add_argument('-n', '--delta-n-events', type=int, default=100000,
                        help="Number of events in a slice (default=100000)")

    args = parser.parse_args()

    if args.slicing_mode == 'IDENTITY':
        args.slice_condition = SliceCondition.make_identity()
    elif args.slicing_mode == 'N_EVENTS':
        args.slice_condition = SliceCondition.make_n_events(args.delta_n_events)
    elif args.slicing_mode == 'N_US':
        args.slice_condition = SliceCondition.make_n_us(args.delta_ts)
    elif args.slicing_mode == 'MIXED':
        args.slice_condition = SliceCondition.make_mixed(args.delta_ts, args.delta_n_events)
    else:
        raise ValueError(f"Invalid slicing mode: {args.slicing_mode}")

    return args


def build_slicer(args):
    """
    Build the CameraStreamSlicer from the command line arguments

    Args:
        args: Command line arguments

    Returns: The CameraStreamSlicer instance
    """
    # [CAMERA_INIT_BEGIN]
    if args.camera_serial_number:
        camera = Camera.from_serial(args.camera_serial_number)
    elif args.input_event_file:
        hints = FileConfigHints()
        hints.real_time_playback(args.real_time_playback)
        camera = Camera.from_file(args.input_event_file, hints)
    else:
        camera = Camera.from_first_available()
    # [CAMERA_INIT_END]

    # [SLICER_INIT_BEGIN]
    slicer = CameraStreamSlicer(camera.move(), args.slice_condition)
    # [SLICER_INIT_END]

    return slicer


def main():
    args = parse_args()
    slicer = build_slicer(args)
    width = slicer.camera().width()
    height = slicer.camera().height()
    frame = np.zeros((height, width, 3), np.uint8)

    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # [SLICER_LOOP_BEGIN]
        for slice in slicer:
            EventLoop.poll_and_dispatch()

            print(f"ts: {slice.t}, new slice of {slice.events.size} events")

            BaseFrameGenerationAlgorithm.generate_frame(slice.events, frame)
            window.show_async(frame)

            if window.should_close():
                break
        # [SLICER_LOOP_END]


if __name__ == "__main__":
    main()
