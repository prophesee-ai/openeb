# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
CSV Reader

This class can read the CSV files from the HDR dataset available at http://rpg.ifi.uzh.ch/E2VID.html produced
for the paper "High Speed and High Dynamic Range Video with an Event Cameras"

The reader handles .txt files in CSV format (t, x, y, p) and .zip compressed archive of those .txt files
"""

import numpy as np
import pandas as pd
from metavision_sdk_base import EventCD


class CSVBaseReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path, n_events=10000, start_index=0):
        header = next(iter(pd.read_csv(path, nrows=0)))
        width, height = header.split(' ')
        self.height, self.width = int(height), int(width)
        self.iterator = pd.read_csv(path, delim_whitespace=True, header='infer',
                                    names=['t', 'x', 'y', 'pol'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                    engine='c',
                                    skiprows=start_index + 1, chunksize=n_events, nrows=None, memory_map=True)
        self.event_buffer = np.zeros((int(1e6),), dtype=EventCD)

    def is_done(self):
        return False

    def __del__(self):
        pass

    def get_size(self):
        return self.height, self.width

    def __iter__(self):
        return self

    def seek_time(self, ts):
        if ts != 0:
            raise Exception('time seek in csv not implemented')

    def __next__(self):
        try:
            event_window = self.iterator.__next__().values
        except BaseException:
            raise StopIteration
        num = len(event_window)
        y = self.height - 1 - event_window[:, 2].copy()
        t = (event_window[:, 0] * 1e6)
        self.event_buffer[:num]['x'] = event_window[:, 1]
        self.event_buffer[:num]['y'] = y
        self.event_buffer[:num]['t'] = t
        self.event_buffer[:num]['p'] = event_window[:, 3] > 0
        return self.event_buffer[:num]
