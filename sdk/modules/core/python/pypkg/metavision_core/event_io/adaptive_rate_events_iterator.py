# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Adaptive rate events iterator built around the EventsIterator class
"""
from .events_iterator import EventsIterator
from metavision_sdk_core import AdaptiveRateEventsSplitterAlgorithm

import os


class AdaptiveRateEventsIterator(object):
    """
    AdaptiveRateEventsIterator is small convenience class to iterate through a recording

    It will produce reasonably sharp slices of events of variable time duration and variable number of events,
    depending on the content of the stream itself.

    Internally, it uses a compation of variance per event as a criterion for the sharpness of the current slice
    of events.
    An additional criterion is the proportion of active pixels containing both positive and negative events.

    Args:
        input_path (str): Path to the file to read
        thr_var_per_event (float): minimum variance per pixel value to reach before considering splitting the slice.
        downsampling_factor (int): performs a downsampling of the input before computing the statistics. Original
                                   coordinates will be multiplied by 2**(-downsampling_factor)


    Examples:
        >>> for events in AdaptiveRateEventsIterator("beautiful_record.raw"):
        >>>     assert events.size > 0
        >>>     start_ts = events[0]["t"]
        >>>     end_ts = events[-1]["t"]
        >>>     print("frame: {} -> {}   delta_t: {}   fps: {}   nb_ev: {}".format(start_ts, end_ts,
                                                                                   end_ts - start_ts,
                                                                                   1e6 / (end_ts - start_ts),
                                                                                   events.size))
    """

    def __init__(self, input_path, thr_var_per_event=5e-4, downsampling_factor=2):
        assert os.path.isfile(input_path)
        assert downsampling_factor == int(downsampling_factor), "Error: downsampling_factor must be an integer"
        assert downsampling_factor >= 0, "Error: downsampling_factor must be >= 0"
        assert (downsampling_factor & (downsampling_factor - 1)) == 0, "Error: downsampling_factor must be a power of 2"
        self.input_path = input_path
        self.thr_var_per_event = thr_var_per_event
        self.downsampling_factor = downsampling_factor
        self.mv_iterator = EventsIterator(input_path=self.input_path, mode="n_events", n_events=100)
        self.height, self.width = self.mv_iterator.get_size()

    def __iter__(self):
        self.mv_iterator = EventsIterator(input_path=self.input_path, mode="n_events", n_events=100)
        self.height, self.width = self.mv_iterator.get_size()
        self.events_splitter = AdaptiveRateEventsSplitterAlgorithm(
            height=self.height, width=self.width,
            thr_var_per_event=self.thr_var_per_event,
            downsampling_factor=self.downsampling_factor)
        self.events_buff = self.events_splitter.get_empty_output_buffer()
        self.mv_iterator_ = iter(self.mv_iterator)
        return self

    def __next__(self):
        while True:
            ev100 = next(self.mv_iterator_)
            if not self.events_splitter.process_events(ev100):
                continue
            self.events_splitter.retrieve_events(self.events_buff)
            events = self.events_buff.numpy()
            return events
        raise StopIteration

    def get_size(self):
        return self.height, self.width
