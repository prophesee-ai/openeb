# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Code sample demonstrating how to use Metavision SDK to display events from a CSV file.
"""


import numpy as np
import pandas as pd
from metavision_sdk_base import EventCD, EventCDBuffer
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


# CSV file contains events written in the following format:
# x1,y1,t1,p1
# x2,y2,t2,p2
# ...
# xn,yn,tn,pn

events_chunksize = 5000
accumulation_time_us = 10000


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision CSV Viewer sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-csv-file', dest='input_path', type=str,
                        help="Path to input CSV file.", required=True)
    parser.add_argument('--width', dest='width', type=int, default=640,
                        help="Width of the sensor associated to the CSV file.")
    parser.add_argument('--height', dest='height', type=int, default=480,
                        help="Height of the sensor associated to the CSV file.")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    print("Code sample demonstrating how to use Metavision SDK to display events from a CSV file.")

    # Event Frame Generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(args.width, args.height, accumulation_time_us)

    # Window - Graphical User Interface
    with MTWindow(title="Metavision CSV Viewer", width=args.width, height=args.height, mode=BaseWindow.RenderMode.BGR) as window:
        def on_cd_frame_cb(ts, cd_frame):
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # Parse CSV using Pandas
        reader = pd.read_csv(args.input_path, delimiter=',', chunksize=events_chunksize, header=None, names=[
                             'x', 'y', 'p', 't'], dtype={'x': np.ushort, 'y': np.ushort, 'p': np.short, 't': np.longlong})
        events_buf = EventCDBuffer(events_chunksize)
        np_evs = events_buf.numpy()
        # Read CSV by chunks
        for chunk in reader:
            buf = chunk.to_numpy()
            buf_size = int(buf.size/4)

            # The last chunk won't necessarily be of size 'events_chunksize'
            if buf_size != np_evs.size:
                events_buf = EventCDBuffer(buf_size)
                np_evs = events_buf.numpy()

            np_evs['x'] = buf[:, 0]
            np_evs['y'] = buf[:, 1]
            np_evs['p'] = buf[:, 2]
            np_evs['t'] = buf[:, 3]

            # Feed events to the Frame Generator
            event_frame_gen.process_events(np_evs)
            if window.should_close():
                break


if __name__ == "__main__":
    main()
