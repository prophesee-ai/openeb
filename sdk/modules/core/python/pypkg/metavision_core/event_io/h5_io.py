# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Reads & Seeks into an HDF5 event file
"""

import h5py
import numpy as np
from metavision_sdk_base import EventCD, EventExtTrigger


class HDF5EventsReader(object):
    """
    Reads & Seeks into an HDF5 event file

    Args:
        src_name (str): input path
    """

    def __init__(self, path):
        self.path = path
        self.file = h5py.File(path, "r")
        # This is hard-coded, it is related to how the indexes table is created.
        self.indexes_period_us = 2000

        self.events_CD = self.file['CD']['events']
        self.indexes_CD = self.file['CD']['indexes']
        self.events_EXT = self.file['EXT_TRIGGER']['events']
        self.indexes_EXT = self.file['EXT_TRIGGER']['indexes']
        assert self.events_CD.dtype == EventCD, (
            f"The data type of CD events is {self.events_CD.dtype}, doesn't match {EventCD}!"
        )
        assert self.events_EXT.dtype == EventExtTrigger, (
            f"The data type of external triggered events is "
            f"{self.events_EXT.dtype}, doesn't match {EventExtTrigger}!"
        )
        self.total_num_events_CD = len(self.events_CD)
        self.total_num_indexes_CD = len(self.indexes_CD)

        if self.total_num_events_CD == 0:
            print("WARNING: The file is empty, containing no events!")
            self.current_time = 0
            self.current_idx = 0
            self.done = True
        else:
            if "offset" in self.indexes_CD.attrs.keys():
                self.ts_offset = int(self.indexes_CD.attrs["offset"])
            else:
                self.ts_offset = 0

            self.first_ev_t = self.events_CD[0]["t"]
            self.last_ev_t = self.events_CD[-1]["t"]
            self.last_idx_CD_t = self.indexes_CD[-1]["ts"]

            self.total_num_events_EXT = len(self.events_EXT)
            if self.total_num_events_EXT > 0:
                self._has_events_EXT = True
                self.first_ext_ev_t = self.events_EXT[0]["t"]
                self.last_ext_ev_t = self.events_EXT[-1]["t"]
            else:
                self._has_events_EXT = False

            self.current_time = 0
            self.current_idx = 0
            self.done = False

    def is_done(self):

        return self.done

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __del__(self):
        self.file.close()

    def seek_time(self, ts):
        """
        Move the position to the event whose timestamp is before and closest to ts.

        Assume the timetable of indexes_CD is:
        (    0,    -1), (    0,    19), (  797,  2010), ( 1513,  4002),
        ( 2234,  6000), ( 3067,  8000), ( 3919, 10000), ( 4715, 12008),
        ( 5495, 14003), ( 6324, 16002), ( 7195, 18009), ( 7964, 20000)

        When ts = 12333, table_idx = 12333 // 2000 = 6, then events[3919: 5495] will be loaded and its timestamp range 
        is (10000, 14003), the pointer will be moved to the event whose timestamp is before and closest to 12333.

        When ts = 1998, table_idx = 1998 // 2000 = 0, then events[0: 797] will be loaded and its timestamp range 
        is (19, 2010), the pointer will be moved to the event whose timestamp is before and closest to 1998.

        """
        if self.total_num_events_CD > 0:
            assert ts < self.last_ev_t, "the seek timestamp is even beyond the max timestamp of the events!"
            table_idx = (ts + self.ts_offset) // self.indexes_period_us

            table_idx = min(table_idx, self.total_num_indexes_CD-1)
            idx_CD_seek = min(table_idx + 2, self.total_num_indexes_CD-1)

            if table_idx >= 0:
                begin_ev_idx = int(self.indexes_CD[table_idx][0])
                end_ev_idx = int(self.indexes_CD[idx_CD_seek][0])

                if begin_ev_idx == end_ev_idx:
                    self.current_idx = begin_ev_idx
                else:
                    events = self.events_CD[begin_ev_idx:end_ev_idx]
                    additive_idx = np.searchsorted(events["t"], ts, side='left')

                    assert events["t"][additive_idx] >= ts, (
                        "The timestamp of the current event should be larger than "
                        f"sought time events['t'][additive_idx] "
                        f"({events['t'][additive_idx]})  ts ({ts})"
                    )
                    self.current_idx = begin_ev_idx + additive_idx
            else:
                self.current_idx = 0
            self.current_time = ts

    def get_size(self):
        """
        Resolution of the sensor that produced the events.
        The format of the resolution is 'widthxheight', such as '640x480'
        """
        # The resolution is stored in String
        assert "geometry" in self.file.attrs.keys()
        geometry = self.file.attrs["geometry"].split("x")
        width = int(geometry[0])
        height = int(geometry[1])
        return height, width

    def load_delta_t(self, delta_t):
        """
        Load events whose timestamp ranges (current_time, current_time + delta_t)
        """
        delta_t = int(delta_t)
        if delta_t < 1:
            raise ValueError("load_delta_t(): Delta_t must be at least 1 microsecond: {}".format(delta_t))

        expected_time = self.current_time + delta_t
        # See if it is the last chunk of events to read
        if expected_time >= self.last_ev_t:
            self.done = True
            events = self.events_CD[self.current_idx:]
            self.current_time = events["t"][-1]
        else:
            previous_idx = self.current_idx
            self.seek_time(expected_time)
            # We don't use the timestamp of events to decide the current_time
            # because we want to always load events inside [n*dt, (n+1)*dt]
            if self.current_idx > previous_idx:
                events = self.events_CD[previous_idx: self.current_idx]
            # It is possible there are no events between [n*dt, (n+1)*dt]
            else:
                return np.empty((0,), dtype=EventCD)
            self.current_time = expected_time

        return events

    def load_n_events(self, n_events):
        """
        Continue loading n events once a time
        """
        n_events = int(n_events)
        assert n_events > 0, "The amount of events to slice must be larger than 0."
        num_events_left = self.total_num_events_CD - self.current_idx
        # See if it is the last chunk of events to read
        if n_events >= num_events_left:
            self.done = True
            events = self.events_CD[self.current_idx:]
            self.current_time = events["t"][-1]
        else:
            events = self.events_CD[self.current_idx: self.current_idx+n_events]
            self.current_time = events["t"][-1]
            self.current_idx += n_events

        return events

    def load_mixed(self, n_events, delta_t):
        """
        Try loading n events, if the duration of these events is larger than delta_t,
        then only keep part of the events which stay inside the time range(delta_t).
        """
        n_events = int(n_events)
        assert n_events > 0, "The amount of events to slice must be larger than 0."
        delta_t = int(delta_t)
        if delta_t < 1:
            raise ValueError("load_delta_t(): Delta_t must be at least 1 microsecond: {}".format(delta_t))
        previous_time = self.current_time
        previous_idx = self.current_idx

        if self.current_time + delta_t <= self.first_ev_t:
            self.current_time += delta_t
            return np.empty((0,), dtype=EventCD)

        events = self.load_n_events(n_events)

        if events["t"][-1] - previous_time > delta_t:
            if events["t"][0] - previous_time > delta_t:
                return np.empty((0,), dtype=EventCD)
            index = np.searchsorted(events['t'], previous_time + delta_t, side='left')
            events = events[:index]
            self.current_time = previous_time + delta_t
            self.current_idx = previous_idx + index

        return events

    def get_ext_trigger_events(self):
        """
        Load external events which are triggered before the current time.
        """
        if self._has_events_EXT:
            if self.current_time < self.first_ext_ev_t:
                return np.empty((0,), dtype=EventExtTrigger)
            elif self.current_time >= self.last_ext_ev_t:
                return self.events_EXT[:]
            else:
                table_idx = (self.current_time + self.ts_offset) // self.indexes_period_us
                ext_ev_t = self.indexes_EXT[table_idx+1]["ts"]
                if self.current_time + self.ts_offset < ext_ev_t:
                    ext_ev_idx = int(self.indexes_EXT[table_idx]["id"])
                    return self.events_EXT[:ext_ev_idx+1]
                else:
                    ext_ev_idx = int(self.indexes_EXT[table_idx+1]["id"])
                    return self.events_EXT[:ext_ev_idx+1]
        else:
            return np.empty((0,), dtype=EventExtTrigger)
