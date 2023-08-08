# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Example of using Metavision SDK Core Python API for visualizing Time Surface of events
"""

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import TimeSurfaceProducerAlgorithmMergePolarities, MostRecentTimestampBuffer
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
import numpy as np
import cv2
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Time Surface sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-raw-file', dest='input_path', default="",
        help="Path to input RAW file. If not specified, the live stream of the first available camera is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    last_processed_timestamp = 0

    # Events iterator on Camera or RAW file
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=10000)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Helper iterator to emulate realtime
    if not is_live_camera(args.input_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # Window - Graphical User Interface
    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        time_surface = MostRecentTimestampBuffer(rows=height, cols=width, channels=1)
        ts_prod = TimeSurfaceProducerAlgorithmMergePolarities(width=width, height=height)

        def cb_time_surface(timestamp, data):
            nonlocal last_processed_timestamp
            nonlocal time_surface
            last_processed_timestamp = timestamp
            time_surface = data

        ts_prod.set_output_callback(cb_time_surface)
        img = np.empty((height, width), dtype=np.uint8)

        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()
            ts_prod.process_events(evs)
            time_surface.generate_img_time_surface(last_processed_timestamp, 10000, img)
            window.show_async(cv2.applyColorMap(img, cv2.COLORMAP_JET))

            if window.should_close():
                break

if __name__ == "__main__":
    main()
