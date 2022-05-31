# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""
h5io for event storage

Example WRITE:
--------------
>>  python3 samples/metavision_h5_io/metavision_h5_io.py write path/to/test.raw path/to/test.h5

Example READ:
-------------
>> python3 samples/metavision_h5_io/metavision_h5_io.py read path/to/test.h5
"""
import os
import argparse
import time
import numpy as np

from tqdm import tqdm
from metavision_core.event_io.h5_io import H5EventsWriter
from metavision_core.event_io.events_iterator import EventsIterator
from metavision_core.event_io.meta_event_producer import MetaEventBufferProducer
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision HDF5 IO sample')

    parser.add_argument('io', type=str, choices=['read', 'write'],
                        help='read or write operation')

    parser.add_argument('path', type=str,
                        help="Path to input file.")

    parser.add_argument('--out_path', type=str, default="",
                        help="Path to output h5 file.")
    parser.add_argument('--start_ts', type=int, default=0,
                        help="start time")
    parser.add_argument('--max_duration', type=int, default=-1,
                        help="start time")
    parser.add_argument('--accumulation_time_us', type=int, default=10000,
                        help="window accumulation time")
    parser.add_argument('--compression_backend', type=str, default="zlib",
                        help="Path to output h5 file.")

    parser.add_argument(
        '-m', '--mode', choices=["delta_t", "n_events", "mixed"],
        type=str, default='delta_t', help='bufferization strategy: delta_t, n_events or mixed')
    parser.add_argument(
        '-d', '--delta-t', type=int, default=10000, help='fixed duration for the bufferization')
    parser.add_argument(
        '-n', '--n-events', type=int, default=10000, help='fixed n events for the bufferization')
    args = parser.parse_args()
    return args


def write(src_path, out_path, start_ts, mode, n_events, delta_t, max_duration, backend):
    if not out_path:
        out_path = os.path.split(src_path, '.raw')[0] + '.h5'
    if max_duration <= 0:
        max_duration = None
    mv_it = EventsIterator(src_path, start_ts, mode, delta_t, n_events, max_duration)
    height, width = mv_it.get_size()
    h5_writer = H5EventsWriter(out_path, height, width, backend)
    for events in tqdm(mv_it):
        h5_writer.write(events)
    h5_writer.close()


def read(src_path, start_ts, mode, n_events, delta_t, max_duration, accumulation_time_us):
    ext = os.path.splitext(src_path)[1]
    max_duration = None if max_duration <= 0 else max_duration
    mv_it = EventsIterator(src_path, start_ts, mode, delta_t, n_events, max_duration)
    height, width = mv_it.get_size()
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us)
    cd_frame = np.zeros((height, width, 3), dtype=np.uint8)
    with MTWindow(title="Metavision H5 IO", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
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

        for i, events in enumerate(tqdm(mv_it)):
            event_frame_gen.process_events(events)
            if window.should_close():
                break


if __name__ == '__main__':
    ARGS = parse_args()
    if ARGS.io == 'read':
        read(ARGS.path, ARGS.start_ts, ARGS.mode, ARGS.n_events,
             ARGS.delta_t, ARGS.max_duration, ARGS.accumulation_time_us)
    else:
        write(ARGS.path, ARGS.out_path, ARGS.start_ts, ARGS.mode, ARGS.n_events,
              ARGS.delta_t, ARGS.max_duration, ARGS.compression_backend)
