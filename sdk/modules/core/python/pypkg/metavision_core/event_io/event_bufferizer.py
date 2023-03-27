# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
This class receives events, buffers them,
and yields them according to a common criterion like
- event_count
- fixed_time
- array coverage
- maximum flow
etc.
"""

import numpy as np
from metavision_sdk_base import EventCD


class FixedTimeBuffer(object):
    """Fixed Time Bufferizer
    Sends data once delta_t is accumulated
    Otherwise Sends None.
    Args:
        delta_t (int): time trigger
    """

    def __init__(self, delta_t):
        self.delta_t = delta_t
        self.current_time = 0
        self.no_events = np.zeros((0,), dtype=EventCD)
        self.event_buffer = []

    def __call__(self, events):
        if not len(events):
            return self.no_events
        dt = events['t'][-1]-self.current_time
        if dt > self.delta_t:
            expected_time = self.current_time + self.delta_t
            idx = np.searchsorted(events["t"], expected_time)
            ev_dtype = events[0].dtype
            tmp = np.concatenate(self.event_buffer+[events[:idx]]).astype(ev_dtype)
            self.event_buffer = [events[idx:]]
            self.current_time += self.delta_t
            return tmp
        else:
            self.event_buffer.append(events)
        return self.no_events


class FixedCountBuffer(object):
    """Fixed Count Buffer
    Sends data once max_count events are accumulated
    Args:
        max_count (int): count trigger
    """

    def __init__(self, max_count):
        self.max_count = max_count
        self.no_events = np.zeros((0,), dtype=EventCD)
        self.event_buffer = []
        self.count = 0

    def __call__(self, events):
        if not len(events):
            return self.no_events
        count = self.count + len(events)
        if count > self.max_count:
            idx = int(self.max_count - self.count)
            ev_dtype = events[0].dtype
            tmp = np.concatenate(self.event_buffer+[events[:idx]]).astype(ev_dtype)
            self.event_buffer = [events[idx:]]
            self.count = len(self.event_buffer[0])
            return tmp
        else:
            self.count = count
            self.event_buffer.append(events)
        return self.no_events
