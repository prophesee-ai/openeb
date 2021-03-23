# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Simple Iterator built around the Metavision Reader classes.
"""
from .raw_reader import RawReaderBase
from .py_reader import EventDatReader


class EventsIterator(object):
    """
    EventsIterator is a small convenience class to iterate through either a camera, a DAT file
    or a RAW file.

    Attributes:
        reader : class handling the file or camera.
        delta_t (int): Duration of served event slice in us.
        max_duration (int): If not None, maximal duration of the iteration in us.
        end_ts (int): If max_duration is not None, last timestamp to consider.
        relative_timestamps (boolean): Whether the timestamp of served events are relative to the current
            reader timestamp, or since the beginning of the recording.

    Args:
        input_path (str): Path to the file to read. If `path` is an empty string or a camera ID it will try to open
            that camera instead.
        start_ts (int): First timestamp to consider.
        mode (string): Load by timeslice of number of events. Either "delta_t" or "n_events"
        delta_t (int): Duration of served event slice in us.
        n_events (int): Number of events in the timeslice.
        max_duration (int): If not None, maximal duration of the iteration in us.
        relative_timestamps (boolean): Whether the timestamp of served events are relative to the current
            reader timestamp, or since the beginning of the recording.
        **kwargs: Arbitrary keyword arguments passed to the underlying RawReaderBase or
            EventDatReader.

    Examples:
        >>> for ev in EventsIterator("beautiful_record.raw", delta_t=1000000, max_duration=1e6*60):
        >>>     print("Rate : {:.2f}Mev/s".format(ev.size * 1e-6))
    """

    def __init__(self, input_path, start_ts=0, mode="delta_t", delta_t=10000, n_events=10000,
                 max_duration=None, relative_timestamps=False, **kwargs):

        self.start_ts = int(start_ts)
        assert delta_t >= 0 and n_events >= 0
        assert mode.lower() in ('delta_t', 'n_events')
        self.delta_t = delta_t if mode == "delta_t" else 0
        self.n_events = n_events if mode == "n_events" else 0
        self.max_duration = max_duration
        self.end_ts = self.max_duration + self.start_ts if max_duration is not None else None
        self.relative_timestamps = relative_timestamps

        if type(input_path) == type(""):
            if input_path.endswith(".dat"):
                self.reader = EventDatReader(input_path, **kwargs)
            else:
                self.reader = RawReaderBase(input_path, delta_t=self.delta_t, ev_count=self.n_events, **kwargs)
        else:
            # we assume input_path is a actually device
            self.reader = RawReaderBase.from_device(input_path, delta_t=self.delta_t, ev_count=self.n_events, **kwargs)

        if mode == "delta_t":
            self._load = lambda: self.reader.load_delta_t(self.delta_t)
        else:
            self._load = lambda: self.reader.load_n_events(self.n_events)
        self._ran = False

    @classmethod
    def from_device(cls, device, start_ts=0, n_events=10000, delta_t=50000, mode="delta_t", max_duration=None,
                    relative_timestamps=False, **kwargs):
        """Alternate way of constructing an EventsIterator from an already initialized HAL device.

        Args:
            device (device): Hal device object initialized independently.
            start_ts (int): First timestamp to consider.
            mode (string): Load by timeslice of number of events. Either "delta_t" or "n_events"
            delta_t (int): Duration of served event slice in us.
            n_events (int): Number of events in the timeslice.
            max_duration (int): If not None, maximal duration of the iteration in us.
            relative_timestamps (boolean): Whether the timestamp of served events are relative to the current
                reader timestamp, or since the beginning of the recording.
            **kwargs: Arbitrary keyword arguments passed to the underlying RawReaderBase.

        Examples:
            >>> from metavision_core.event_io.raw_reader import initiate_device
            >>> device = initiate_device(path=args.input_path)
            >>> # call any methods on device
            >>> mv_it = EventsIterator.from_device(device=device)
        """
        return cls(device, start_ts=start_ts, n_events=n_events, delta_t=delta_t, mode=mode, max_duration=max_duration,
                   relative_timestamps=relative_timestamps, **kwargs)

    def get_size(self):
        """Function returning the size of the imager which produced the events.

        Returns:
            Tuple of int (height, width) which might be (None, None)"""
        return self.reader.get_size()

    def __repr__(self):
        string = "EventsIterator({})\n".format(self.reader.path)
        string += "delta_t {} us\n".format(self.delta_t)
        string += "starts_ts {} us end_ts {}".format(self.start_ts, self.end_ts)
        return string

    def __iter__(self):
        if self._ran:
            self.reader.reset()
        self._ran = True
        self.reader.seek_time(self.start_ts)
        return self

    def __next__(self):
        if self.reader.is_done():
            raise StopIteration
        else:
            ts = self.reader.current_time
            if self.end_ts is not None:
                if ts >= self.end_ts:
                    raise StopIteration

            events = self._load()

            if self.end_ts is not None and events.size and events['t'][-1] >= self.end_ts:
                events = events[events["t"] < self.end_ts]
            if self.relative_timestamps:
                events['t'] -= ts
            return events

    def __del__(self):
        self.reader.__del__()
