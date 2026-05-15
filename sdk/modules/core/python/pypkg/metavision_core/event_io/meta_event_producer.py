# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
This class Resamples an event producer to stream by N events or DT duration
"""

import numpy as np
from metavision_sdk_base import EventCD
from metavision_sdk_core import SharedCdEventsBufferProducer as EventsBufferProducer
from collections import deque


class MetaEventBufferProducer(object):
    """
    Resamples an event producer to stream
    by N events or DT duration.

    Args:
        event_producer (object): any object that streams numpy buffer of EventCD
        mode (str): "delta_t", "n_events" or "mixed" trigger ways
        delta_t (int): fixed duration
        n_events (int): fixed count
        start_ts (int): start time
        relative_timestamps (bool): retrieve events with first time of buffer substracted
    """

    def __init__(self, event_producer, mode='delta_t', delta_t=10000, n_events=10000, start_ts=0,
                 relative_timestamps=False):
        self.event_producer = event_producer
        self.mode = mode
        self.delta_t = delta_t
        self.n_events = n_events
        self.current_time = start_ts
        self.start_ts = start_ts
        self.relative_timestamps = relative_timestamps
        self.cur_iter = None
        self.done = False

    @property
    def path(self):
        return self.event_producer.path

    def is_done(self):
        return self.event_producer.is_done()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __del__(self):
        pass

    def seek_time(self, ts):
        """
        seek in time, depends on event producer's seek implementation.
        """
        self.event_producer.seek_time(ts)

    def get_size(self):
        """
        Resolution of the sensor that produced the events.
        """
        return self.event_producer.get_size()

    def load_delta_t(self, delta_t):
        if self.cur_iter is None:
            self.cur_iter = iter(self)
        return next(self.cur_iter)

    def load_n_events(self, n_events):
        if self.cur_iter is None:
            self.cur_iter = iter(self)
        return next(self.cur_iter)

    def load_mixed(self, n_events, delta_t):
        if self.cur_iter is None:
            self.cur_iter = iter(self)
        return next(self.cur_iter)

    def _initialize(self):
        """
        Initializes Event Buffer Producer and FIFO
        """
        n_events = self.n_events
        delta_t = int(self.delta_t)
        if self.mode == "delta_t":
            n_events = 0
        elif self.mode == 'n_events':
            n_events = self.n_events
            delta_t = 0
        self.buffer_producer = EventsBufferProducer(
            self._process_batch, event_count=n_events, time_slice_us=delta_t)
        self._event_buffer = deque()
        self.seek_time(self.start_ts)

    def __iter__(self):
        """
        Initializes and iterates over event buffers
        """
        self._initialize()
        previous_time = self.start_ts
        self.done = False

        def empty_buffer():
            nonlocal previous_time

            while len(self._event_buffer) > 0:
                self.current_time += self.delta_t
                current_time, output = self._event_buffer.popleft()
                # If there are no events during delta_t, an empty buffer is yielded
                while current_time - previous_time > self.delta_t and self.mode == 'delta_t':
                    yield np.empty(0, dtype=EventCD)
                    previous_time += self.delta_t
                if self.relative_timestamps:
                    output['t'] -= previous_time
                previous_time = current_time
                yield output

        for i, events in enumerate(self.event_producer):
            self.buffer_producer.process_events(events)

            for events in empty_buffer():
                yield events

        # Empty buffers
        self.buffer_producer.flush()
        for events in empty_buffer():
            yield events

        self.done = True
        self.cur_iter = None

    def is_done(self):
        return self.done

    def _process_batch(self, ts, batch):
        self._event_buffer.append((ts, batch))
