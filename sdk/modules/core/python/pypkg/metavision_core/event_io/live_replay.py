# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Live Replay Events Iterator, this is for user friendliness
"""

import os
import time
from metavision_core.event_io.events_iterator import EventsIterator


def is_live_camera(input_path):
    """
    Checks if input_path is a live camera

    Args:
        input_path (str): path to the file to read. if `path` is an empty string or a camera serial number,
        this function will return true.
    """
    return isinstance(input_path, str) and not os.path.exists(input_path)


class LiveReplayEventsIterator(object):
    """
    LiveReplayEventsIterator allows replaying a record in "live" ("real-time") condition or at a
    speed-up (or slow-motion) factor of real-time.

    Args:
        events_iterator (EventsIterator): event iterator
        replay_factor (float): if greater than 1.0 we replay with slow-motion,
        otherwise this is a speed-up over real-time.
    """

    def __init__(self, events_iterator, replay_factor=1.0):
        assert isinstance(events_iterator, EventsIterator)
        self.iterator = events_iterator
        self.replay_factor = replay_factor
        self.clock = 0

    @property
    def start_ts(self):
        return self.iterator.start_ts

    @property
    def delta_t(self):
        return self.iterator.delta_t

    def get_size(self):
        return self.iterator.get_size()

    def __iter__(self):
        self.clock = time.time()
        for events in self.iterator:
            ts = self.iterator.get_current_time()
            clock_time = time.time() - self.clock
            diff_s = self.replay_factor * ts * 1e-6 - clock_time
            if diff_s > 0:
                time.sleep(diff_s)
            yield events
