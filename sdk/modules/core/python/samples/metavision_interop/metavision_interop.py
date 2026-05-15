# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Code sample to show how to work simply with other event-based storage formats.

We provide here 3 examples to show how to read other popular event formats:
- rosbag: .bag files
- csv: .txt files (also handled as a .zip compressed archive)
- aedat4: .aedat4 files
"""

import argparse
import numpy as np
import cv2
from extended_events_iterator import ExtendedEventsIterator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision interoperability sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path', default="",
        help="Path to input event file. The format can be either .bag (ros), .zip, .txt or .aedat4")
    parser.add_argument(
        '-m', '--mode', choices=["delta_t", "n_events", "mixed"],
        type=str, default='delta_t', help='bufferization strategy: delta_t, n_events or mixed')
    parser.add_argument(
        '-d', '--delta-t', type=int, default=5000, help='fixed duration for the bufferization')
    parser.add_argument(
        '-n', '--n-events', type=int, default=5000, help='fixed n events for the bufferization')
    args = parser.parse_args()
    return args


def read_exotic_format(path, mode, delta_t, n_events):
    """
    Reads alternative formats of AER events.

    Args:
        path (str): path to input file (can be .bag, .txt, .zip or .aedat4)
        mode (str): "delta_t", "n_events" or "mixed" mode to read events
        delta_t (int): fixed duration
        n_events (int): fixed number of events
    """
    reader = ExtendedEventsIterator(path, mode=mode, delta_t=delta_t, n_events=n_events)
    height, width = reader.get_size()
    img = np.zeros((height, width, 3), dtype=np.uint8)
    print('height, width: ', height, width)
    for events in reader:
        img[...] = 0
        if len(events):
            x, y, p = events['x'], events['y'], events['p']
            img[y[p == 1], x[p == 1], 1] = 255
            img[y[p == 0], x[p == 0], 2] = 255
        cv2.imshow('img', img)
        cv2.waitKey(5)


if __name__ == '__main__':
    read_exotic_format(**parse_args().__dict__)
