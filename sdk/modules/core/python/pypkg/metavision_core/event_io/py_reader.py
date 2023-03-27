# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
This class loads events from DAT or NPY files
"""

import os
import numpy as np

from . import dat_tools as dat
from . import npy_tools as npy_format


class EventBaseReader(object):
    """
    EventBaseReader base class to pure python event readers.

    EventBaseReader allows reading a file of events while maintaining a position of the cursor.
    Further manipulations like advancing the cursor or going backward are allowed.

    Attributes:
        path (string): Path to the file being read
        current_time (int): Indicating the position of the cursor in the file in us
        duration_s (int): Indicating the total duration of the file in seconds

    Args:
        event_file (str): file containing events
    """

    def __init__(self, event_file):
        self._binary_format = None
        self._file = None
        self._start = None
        self.ev_type = None
        self._ev_size = None
        self._size = None
        self._dtype = None
        self._decode_dtype = None
        self.path = event_file
        self._extension = self.path.split('.')[-1]
        self.open_file()

        # size
        self._file.seek(0, os.SEEK_END)
        self._end = self._file.tell()
        self._ev_count = (self._end - self._start) // self._ev_size
        self.current_time = 0
        if self._ev_count == 0:
            print("WARNING: The events file is empty!!!")
            self.done = True
            self.duration_s = 0
        else:
            self.done = False
            self._file.seek(self._start)
            # If the current time is t, it means that next event that will be loaded has a
            # timestamp superior or equal to t (event with timestamp exactly t is not loaded yet)
            self.first_ev_t = self.get_first_ev_timestamp()
            self.last_ev_t = self.get_last_ev_timestamp()
            self.duration_s = (self.last_ev_t - self.first_ev_t) * 1e-6

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def open_file(self):
        raise NotImplementedError()

    def reset(self):
        """Resets at beginning of file."""
        self._file.seek(self._start)
        self.done = False if self._ev_count else True
        self.current_time = 0

    def event_count(self):
        """Getter on event_count.

        Returns:
            An int indicating the total number of events in the file
        """
        return self._ev_count

    def current_event_index(self):
        """Returns the number of event already loaded"""
        return (self._file.tell() - self._start) // self._ev_size

    def get_size(self):
        """Function returning the size of the imager which produced the events.

        Returns:
            Tuple of int (height, width) which might be (None, None)"""
        return self._size

    def is_done(self):
        """Returns True if the end of the file has been reached."""

        return self.done

    def __repr__(self):
        """String representation of a `DatReader` object.

        Returns:
            string describing the DatReader state and attributes
        """

        wrd = 'DatReader: {}\n'.format(self.path)
        wrd += '-----------\n'
        if self._extension == 'dat':
            wrd += 'Event Type: {}\n'.format(self._binary_format.EV_STRINGS[self.ev_type])
        elif self._extension == 'npy':
            wrd += 'Event Type: numpy array element\n'
        wrd += 'Event Size: {} bytes\n'.format(self._ev_size)
        wrd += 'Event Count: {}\n'.format(self._ev_count)
        wrd += 'Duration: {} s \n'.format(self.duration_s)
        wrd += '-----------\n'
        return wrd

    def load_n_events(self, n_events):
        """
        Loads batch of n events.

        Args:
            n_events (int): Number of events that will be loaded

        Returns:
            events (numpy array): structured numpy array containing the events.

        """
        assert n_events > 0, "The number of events to slice is lower than 0!!!"
        n_events = int(n_events)
        event_buffer = np.empty((n_events), dtype=self._decode_dtype)

        pos = self._file.tell()
        count = (self._end - pos) // self._ev_size
        if n_events >= count:
            n_events = count
            self._binary_format.stream_events(self._file, event_buffer, self._dtype, n_events)
            self.done = True
        else:
            self._binary_format.stream_events(self._file, event_buffer, self._dtype, n_events)
            self._file.seek(pos + n_events * self._ev_size)

        events = event_buffer[:n_events]
        self.current_time = events["t"][-1]
        return events

    def load_delta_t(self, delta_t):
        """
        Loads events corresponding to a slice of time, starting from the DatReader's `current_time`.

        Args:
            delta_t (int): slice duration (in us).

        Returns:
            events (numpy array): structured numpy array containing the events.

        Note that current time will be incremented by `delta_t`.
        If an event is timestamped at exactly current_time it will not be loaded.
        """
        delta_t = int(delta_t)
        if delta_t < 1:
            raise ValueError("load_delta_t(): Delta_t must be at least 1 micro-second: {}".format(delta_t))

        if self.done or (self._file.tell() >= self._end):
            self.done = True
            return np.empty((0,), dtype=self._decode_dtype)

        expected_time = self.current_time + delta_t
        tmp_time = self.current_time
        start = self._file.tell()
        pos = start
        nevs = 0
        batch = 100000
        event_buffer = []
        # data is read by buffers until enough events are read or until the end of the file
        while tmp_time < expected_time and pos < self._end:
            count = (min(self._end, pos + batch * self._ev_size) - pos) // self._ev_size
            buffer = np.empty((count,), dtype=self._decode_dtype)
            self._binary_format.stream_events(self._file, buffer, self._dtype, count)
            tmp_time = buffer["t"][-1]
            event_buffer.append(buffer)
            nevs += count
            pos = self._file.tell()
        if tmp_time >= expected_time:
            self.current_time = expected_time
        else:
            self.current_time = tmp_time
        assert len(event_buffer) > 0
        if event_buffer[0]["t"][0] >= expected_time:
            self._file.seek(start)
            return np.empty((0,), dtype=self._decode_dtype)
        idx = np.searchsorted(event_buffer[-1]["t"], expected_time)
        event_buffer[-1] = event_buffer[-1][:idx]
        ev_dtype = event_buffer[0].dtype
        event_buffer = np.concatenate(event_buffer).astype(ev_dtype)
        idx = len(event_buffer)
        self._file.seek(start + idx * self._ev_size)
        self.done = self._file.tell() >= self._end

        return event_buffer

    def load_mixed(self, n_events, delta_t):
        """
        Loads batch of n events or delta_t microseconds, whichever comes first.

        Args:
            n_events (int): Maximum number of events that will be loaded.
            delta_t (int): Maximum allowed slice duration (in us).

        Returns:
            events (numpy array): structured numpy array containing the events.

        Note that current time will be incremented to reach the timestamp of the first event not loaded yet.
        However if the maximal time slice duration is reached, current time will be increased by delta_t instead.
        """
        event_buffer = np.empty((n_events), dtype=self._decode_dtype)
        previous_time = self.current_time

        pos = self._file.tell()
        count = (self._end - pos) // self._ev_size
        if count <= n_events:
            n_events = count
            self._binary_format.stream_events(self._file, event_buffer, self._dtype, n_events)
            self.done = True
        else:
            self._binary_format.stream_events(self._file, event_buffer, self._dtype, n_events)
            self._file.seek(pos + n_events * self._ev_size)
        events = event_buffer[:n_events]
        self.current_time = events["t"][-1]

        # let's check is the delta_t condition already met
        if self.current_time - previous_time >= delta_t:
            self.current_time = previous_time + delta_t
            if events["t"][0] - previous_time >= delta_t:
                self._file.seek(pos)
                return np.empty((0,), dtype=self._decode_dtype)
            # then we only need a subset of the events.
            index = np.searchsorted(events['t'], previous_time + delta_t)
            events = events[:index]
            self._file.seek(pos + index * self._ev_size)

        return events

    def seek_event(self, n_events):
        """
        Seeks in the file by `n_events` events

        Args:
            n_events (int): seek in the file the nth events

        """
        assert n_events > 0, f"Impossible to seek {n_events}th event!!!"
        if n_events >= self._ev_count:
            # we put the cursor one event before and read the last event
            # which puts the file cursor at the right place
            # current_time is set to the last event timestamp
            self._file.seek(self._start + (self._ev_count - 1) * self._ev_size)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)["t"][0]

        else:
            # we put the cursor at the *(n_events-1)*nth event
            self._file.seek(self._start + (n_events - 1) * self._ev_size)
            # we read the timestamp of the following event (this change the position in the file)
            self.current_time = np.fromfile(self._file, dtype=self._dtype, count=1)["t"][0]

        self.done = self._file.tell() >= self._end

    def seek_time(self, expected_time, term_criterion=100000):
        """Goes to the time expected_time inside the file.
        This is implemented using a binary search algorithm.

        Args:
            expected_time (int): Expected time
            term_criterion (int): Binary search termination criterion in nb of events

        Once the binary search has found a buffer of size *term_criterion* events, containing the
        *expected_time*. It will load them in memory and perform a `searchsorted`_ from numpy, so that the end
        of the binary search doesn't take to many iterations in python.

        .. _searchsorted:
            https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html
        """
        assert expected_time >= 0, "The seeked time should be at least above than or equal to 0!!!"
        if self._ev_count > 0:
            expected_time = int(expected_time)
            if expected_time > self.last_ev_t:
                self._file.seek(self._end)
                self.done = True
                self.current_time = self.last_ev_t
                return
            elif expected_time < self.first_ev_t:
                self._file.seek(self._start)
                self.current_time = expected_time
                return

            if expected_time <= 0:
                self.reset()
                return

            low = 1
            high = self._ev_count

            # binary search
            assert term_criterion > 0, "The termination criterion must be larger than 0!!!"
            while high - low > term_criterion:
                middle = (low + high) // 2

                self.seek_event(middle)
                mid = self.current_time

                if mid > expected_time:
                    high = middle
                elif mid < expected_time:
                    low = middle
                else:
                    # this implies that current_time = expected_time
                    self.done = self._file.tell() >= self._end
                    return
            # we now know that it is between low and high
            self.seek_event(low)
            final_buffer = np.fromfile(self._file, dtype=self._dtype, count=high - low + 1)["t"]
            final_index = np.searchsorted(final_buffer, expected_time)

            self.seek_event(low + final_index)
            self.current_time = expected_time
            self.done = self._file.tell() >= self._end

    def get_last_ev_timestamp(self):
        """
        Returns the timestamp of the last event in us
        """
        if not self._ev_count:
            return 0
        # save the state of the class
        pos = self._file.tell()
        current_time = self.current_time
        done = self.done
        # read the last event's timestamp
        self._file.seek(self._start + (self._ev_count - 1) * self._ev_size)
        time = np.fromfile(self._file, dtype=self._dtype, count=1)["t"][0]
        # restore the state
        self._file.seek(pos)
        self.current_time = current_time
        self.done = done

        return time

    def get_first_ev_timestamp(self):
        """
        Returns the timestamp of the first event in us
        """
        if not self._ev_count:
            return 0
        # save the state of the class
        pos = self._file.tell()
        current_time = self.current_time
        done = self.done
        # read the first event's timestamp
        self._file.seek(self._start)
        time = np.fromfile(self._file, dtype=self._dtype, count=1)["t"][0]
        # restore the state
        self._file.seek(pos)
        self.current_time = current_time
        self.done = done

        return time

    def __del__(self):
        self._file.close()


class EventNpyReader(EventBaseReader):
    """
    EventNpyReader class to read NPY long files.

    Attributes:
        path (string): Path to the file being read
        current_time (int): Indicating the position of the cursor in the file in us
        duration_s (int): Indicating the total duration of the file in seconds

    Args:
        event_file (str): file containing events
    """

    def __init__(self, event_file):
        super().__init__(event_file)

    def open_file(self):
        assert self._extension == "npy", 'input file path = {}'.format(self.path)
        self._binary_format = npy_format
        self._file = open(self.path, "rb")
        self._start, self.ev_type, self._ev_size, self._size = self._binary_format.parse_header(self._file)
        assert self._ev_size != 0
        self._dtype = self.ev_type
        self._decode_dtype = self.ev_type


class EventDatReader(EventBaseReader):
    """
    EventDatReader class to read DAT long files.
    DAT files are a binary format with events stored
    with polarity, x and y casted into a uint32 and timestamp on another uint32.
    This format still exists in many of our datasets, so this file is used to support it.

    Attributes:
        path (string): Path to the file being read
        current_time (int): Indicating the position of the cursor in the file in us
        duration_s (int): Indicating the total duration of the file in seconds

    Args:
        event_file (str): file containing events
    """

    def __init__(self, event_file):
        super().__init__(event_file)

    def open_file(self):
        assert self._extension == "dat", 'input file path = {}'.format(self.path)
        self._binary_format = dat
        self._file = open(self.path, "rb")
        self._start, self.ev_type, self._ev_size, self._size = self._binary_format.parse_header(self._file)
        assert self._ev_size != 0
        assert isinstance(self._ev_size, int)
        self._dtype = self._binary_format.EV_TYPES[self.ev_type]
        self._decode_dtype = self._binary_format.DECODE_DTYPES[self.ev_type]
