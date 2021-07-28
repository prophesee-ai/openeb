# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
This class loads events from RAW files

 the interface is close to DatReader but not everything could be implemented
    - no backward seek functionality
    - cd events dtype contains a 2 byte offset
"""

import os
from collections import deque
import numpy as np

from metavision_hal import DeviceDiscovery, RawFileConfig
from metavision_sdk_core import SharedCdEventsBufferProducer as EventsBufferProducer

from metavision_sdk_base import EventCD
from metavision_sdk_base import EventExtTrigger


def initiate_device(path, do_time_shifting=True, use_external_triggers=[]):
    """
    Constructs a device either from a file if the path ends with RAW or with the camera ID.

    This device can be used in conjunction with `RawReader.from_device` or `EventsIterator.from_device`
    to create a RawReader or an EventsIterator with a customized HAL device.

    Args:
        path (str): either path do a RAW file (having a .raw or .RAW extension) or a camera ID. leave blank to take
            the first available camera.
        do_time_shifting (bool): in case of a file, makes the timestamps start close to 0mus.
        use_external_triggers (int List): list of integer values corresponding to the channels of external trigger
            to be activated (only relevant for a live camera).
    Returns:
        device: a HAL Device.

    """

    if path.lower().endswith(".raw"):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        file_stream_config = RawFileConfig()

        file_stream_config.do_time_shifting = do_time_shifting
        file_stream_config.n_events_to_read = 100000
        device = DeviceDiscovery.open_raw_file(path, file_stream_config)
        if device is None:
            raise OSError(f"Incorrect raw input file : {path} !")
    else:
        # if the id is not correctly zero padded we try to recover
        device = None
        device = DeviceDiscovery.open(path)
        if device is None:
            raise OSError(f"Failed to open camera {path} !")
        # External triggers
        if use_external_triggers:
            trigger_in = device.get_i_trigger_in()
            if trigger_in is None:
                raise OSError(f"Failed to open EventExtTrigger facility from camera {path} !")
            for trigger in use_external_triggers:
                trigger_in.enable(trigger)
    return device


class RawReaderBase(object):
    def __init__(self, record_base, device=None, do_time_shifting=True, ev_count=0, delta_t=50000,
                 initiate_device=True, use_external_triggers=[]):
        self._event_buffer = deque([])

        self._event_ext_trigger_buffer = np.empty(int(1e6), dtype=EventExtTrigger)
        self.path = record_base
        self.do_time_shifting = do_time_shifting
        self.ev_count = int(ev_count)
        self.delta_t = int(delta_t)
        self.current_time = -1
        self.use_external_triggers = use_external_triggers

        def process_batch_ext_trigger(batch):
            """put external trigger events in internal buffer"""
            beg = self._event_ext_trigger_buffer_end
            self._event_ext_trigger_buffer_end += len(batch)
            self._event_ext_trigger_buffer[beg:self._event_ext_trigger_buffer_end] = batch

        self.process_batch_ext_trigger = process_batch_ext_trigger
        self.do_initiate_device = initiate_device
        self.device = device
        self.reset()

    @classmethod
    def from_device(cls, device, ev_count=0, delta_t=50000):
        return cls("", device=device, ev_count=ev_count, delta_t=delta_t, initiate_device=False)

    def __del__(self):
        if self.i_device_control is not None:
            self.i_device_control.stop()
        if self.i_events_stream is not None:
            self.i_events_stream.stop()
            self.i_events_stream.stop_log_raw_data()
        del self.device
        del self._event_ext_trigger_buffer
        del self._event_buffer
        del self.i_events_stream
        del self.i_device_control
        del self.i_decoder

    def _process_batch(self, ts, batch):
        # in case of fast forward incoming events are discarded
        if ts > self._seek_time:
            self._event_buffer.append((ts, batch))
        else:
            self._current_event_index += len(batch)

    def _run(self):
        """"decode a packet"""
        if self._decode_done:
            return False
        if self.i_events_stream.poll_buffer() < 0:
            # This would only happen in a live camera or when the file is done
            self._decode_done = True
            self.buffer_producer.flush()
            return False
        data = self.i_events_stream.get_latest_raw_data()
        self.i_decoder.decode(data)

        return True

    def _advance(self, final_time, skip=False):
        """decodes events until final time.

        Args:
            final_time (int): Timestamps in us until the search stops"""
        final_time = int(final_time)

        if self.current_time > final_time:
            raise RuntimeError('cannot seek backward in RAW file')
        if final_time > self._last_loaded_ts():
            if skip:
                self._seek_time = final_time
            while not self._decode_done and final_time > self._last_loaded_ts():
                self._run()

    def __repr__(self):
        string = "RawReader({})\n".format(self.path)
        string += "current time : {:d}us done : {}\n".format(int(self.current_time), str(self.done))
        string += "current event index : {:d}\n".format(int(self._current_event_index))
        return string

    def reset(self):
        """Resets at beginning of file."""
        # Metavision HAL initialization sequence
        if self.do_initiate_device:
            self.device = initiate_device(self.path, do_time_shifting=self.do_time_shifting,
                                          use_external_triggers=self.use_external_triggers)

        i_geometry = self.device.get_i_geometry()
        if i_geometry is None:
            raise OSError("No I_Geometry facility")

        self.width = i_geometry.get_width()
        self.height = i_geometry.get_height()

        self.i_events_stream = self.device.get_i_events_stream()
        if self.i_events_stream is None:
            raise OSError("No I_EventsStream facility")

        self.i_decoder = self.device.get_i_decoder()

        if self.i_decoder is None:
            raise OSError("No I_Decoder facility")

        self.i_event_cd_decoder = self.device.get_i_event_cd_decoder()
        if self.i_event_cd_decoder is None:
            raise OSError("No I_EventDecoder<Metavision::EventCD> facility")

        self.buffer_producer = EventsBufferProducer(self._process_batch, event_count=self.ev_count,
                                                    time_slice_us=self.delta_t)

        self.i_event_cd_decoder.set_add_decoded_native_vevent_callback(
            self.buffer_producer.get_process_events_callback())

        self.i_eventdecoder_ext_trigger = self.device.get_i_event_ext_trigger_decoder()
        if self.i_eventdecoder_ext_trigger is not None:
            self.i_eventdecoder_ext_trigger.add_event_buffer_callback(self.process_batch_ext_trigger)

        self.i_events_stream.start()
        self.i_device_control = self.device.get_i_device_control()

        if self.i_device_control is not None:
            self.i_device_control.start()
            self.i_device_control.reset()

        self._reset_state_vars()
        self._reset_buffer()

    def _reset_state_vars(self):
        # reset state variables
        self.done = False
        self._decode_done = False
        self._seek_time = -1
        self.current_time = 0
        self._current_event_index = 0

        # resets the memory buffer pointer for event ext trigger
        self._event_ext_trigger_buffer_end = 0

    def _reset_buffer(self):
        self._event_buffer = deque([])

    def get_ext_trigger_events(self):
        """Returns all external trigger events that have been loaded until now in the record"""
        return self._event_ext_trigger_buffer[:self._event_ext_trigger_buffer_end]

    def clear_ext_trigger_events(self):
        """Deletes previously stored external trigger events"""
        self._event_ext_trigger_buffer_end = 0

    def current_event_index(self):
        """Returns the number of event already loaded"""
        return self._current_event_index

    def get_size(self):
        """Function returning the size of the imager which produced the events.

        Returns:
            Tuple of int (height, width)"""
        return self.height, self.width

    def is_done(self):
        """
        indicates if all events have been loaded and if the rolling buffer is empty
        """
        self.done = self._decode_done and not self._event_buffer
        return self.done

    def seek_time(self, final_time):
        """
        seeks into the RAW file until current_time >= final_time.

        Args:
            final_time (int): Timestamp in us at which the search stops (only multiples of delta_t are supported.).
        """
        self._advance(final_time, skip=True)
        if self._seek_time > 0 and len(self._event_buffer):
            for index, (ts, evs) in enumerate(self._event_buffer):
                if ts - self.delta_t < self._seek_time:
                    self._event_buffer[index] = (ts, evs[evs['t'] >= self._seek_time])
        self.current_time = final_time

    def _last_loaded_ts(self):
        """returns the timestamp of the last loaded event and -1 if None are in the buffer"""
        if self._event_buffer:
            return self._event_buffer[-1][0]
        else:
            return 0

    def _load_next_buffer(self, increase_time_by_delta_t=False):
        """
        Loads a batch of events from the queue.

        Args:
            increase_time_by_delta_t (bool): if True increases "current time" by delta_t, otherwise sets current time
                to the last loaded event.

        Returns:
            events
        """

        while not (self._decode_done or len(self._event_buffer)):
            self._run()
        # if enough events have been decoded, we take the first buffer in the queue.
        current_ts, events = self._event_buffer.popleft()
        # update variables describing the object state.
        self._current_event_index += events.size

        if increase_time_by_delta_t:
            self.current_time += self.buffer_producer.get_processing_n_us()
        else:
            if events.size:
                self.current_time = events[-1]['t']

        return events

    def load_n_events(self, ev_count):
        """
        Loads a batch of *ev_count* events.

        Args:
            ev_count (int): Number of events to load

        Returns:
            events
        """

        return self._load_next_buffer(increase_time_by_delta_t=False)

    def load_delta_t(self, delta_t):
        """
        Loads all the events contained in the next *delta_t* microseconds.

        Args:
            delta_t (int): Interval of time in us since last loading, within which events are loaded

        Returns:
            events
        """

        return self._load_next_buffer(increase_time_by_delta_t=True)


class RawReader(RawReaderBase):
    """
    RawReader loads events from a RAW file.

    RawReader allows to read a file of events while maintaning a position of the cursor.
    Further manipulations like advancing the cursor in time are posible.

    Attributes:
        path (string): Path to the file being read. If `path` is an empty string or a camera ID it will try to open
            that camera instead.
        current_time (int): Indicating the position of the cursor in the file in us.
        duration_s (int): Indicating the total duration of the file in seconds.
        do_time_shifting (bool): If True the origin of time is a few us from the first events.
            Otherwise it is when the camera was started.

    Args:
        record_base (string): Path to the record being read.
        do_time_shifting (bool): If True the origin of time is a few us from the first events.
            Otherwise it is when the camera was started.
        use_external_triggers (int List): list of integer values corresponding to the channels of external trigger
            to be activated (only relevant for a live camera).
    """

    def __init__(self, record_base, max_events=int(1e8), do_time_shifting=True,
                 device=None, initiate_device=True, use_external_triggers=[]):
        super().__init__(record_base, device=device, do_time_shifting=do_time_shifting,
                         initiate_device=initiate_device, use_external_triggers=use_external_triggers)
        self._event_buffer = np.empty(max_events, dtype=EventCD)

    @classmethod
    def from_device(cls, device, max_events=int(1e8)):
        """
        Alternate way of constructing an RawReader from an already initialized HAL device.

        Args:
            device (device): Hal device object initialized independently.

        Examples:
            >>> device = initiate_device(path=args.input_path)
            >>> # call any methods on device
            >>> reader = RawReader.from_device(device=device)
        """
        return cls("", device=device, max_events=max_events, initiate_device=False)

    # callbacks
    def _process_batch(self, ts, batch):
        # in case of fast forward incoming events are discarded
        if self._seek_time > batch[-1]['t']:
            self._current_event_index += (self._end_buffer - self._begin_buffer) + len(batch)
            self._begin_buffer = self._end_buffer
            return
        # otherwise the rolling buffer parameters are updated
        begin_buffer = self._end_buffer
        self._end_buffer += len(batch)
        # and events are copied in one go or two goes depending on whether we need to "roll"
        # around the buffer
        if self._end_buffer <= self._event_buffer.size:
            self._event_buffer[begin_buffer:self._end_buffer] = batch
        else:
            n_evs_before_end_buffer = self._event_buffer.size - begin_buffer
            self._event_buffer[begin_buffer:] = batch[:n_evs_before_end_buffer]
            remaining_size = batch.size - n_evs_before_end_buffer
            if remaining_size > self._begin_buffer:
                raise ValueError('RawReader buffer size too small. Please increase max_events')
            self._event_buffer[:remaining_size] = batch[n_evs_before_end_buffer:]

            self._end_buffer = remaining_size

    def __repr__(self):
        string = super().__repr__()
        string += "_begin_buffer {},_end_buffer_ {},  buffer_size {}".format(
            self._begin_buffer, self._end_buffer, self._event_buffer.size)
        return string

    def _count_ev_loaded(self):
        """helper function to count loaded events in rolling buffer"""
        if self._end_buffer >= self._begin_buffer:
            return self._end_buffer - self._begin_buffer
        else:
            return self._event_buffer.size - self._begin_buffer + self._end_buffer

    def _are_enough_ev_loaded(self, ev_count):
        """helper function to check if there are enough events in rolling buffer"""
        loaded_count = self._count_ev_loaded()

        return loaded_count >= ev_count

    def _last_loaded_ts(self):
        """returns the timestamp of the last loaded event and -1 if None are in the buffer"""
        if self._end_buffer == self._begin_buffer:
            return -1
        return int(self._event_buffer[self._end_buffer - 1]['t'])

    def _reset_buffer(self):
        # resets memory buffer "pointers"
        self._begin_buffer, self._end_buffer = 0, 0

    def load_n_events(self, ev_count):
        """
        Loads a batch of *ev_count* events.

        Args:
            ev_count (int): Number of events to load

        Returns:
            events
        """

        while not (self._decode_done or self._are_enough_ev_loaded(ev_count)):
            self._run()

        # if all events are decoded, there is only this classes buffer left.
        if self._decode_done:
            ev_count = min(ev_count, self._count_ev_loaded())

        if self._begin_buffer + ev_count < self._event_buffer.size:
            events = self._event_buffer[self._begin_buffer:self._begin_buffer + ev_count]
            self._begin_buffer += ev_count
        else:
            events = np.concatenate(
                (self._event_buffer[self._begin_buffer:],
                 self._event_buffer[:ev_count - self._event_buffer.size + self._begin_buffer]))
            self._begin_buffer = ev_count - self._event_buffer.size + self._begin_buffer
        # update variables describing the Class state
        self._current_event_index += events.size
        self.current_time = self._event_buffer[
            self._begin_buffer]['t']
        self.is_done()
        return events

    def load_delta_t(self, delta_t):
        """
        Loads all the events contained in the next *delta_t* microseconds.

        Args:
            delta_t (int): Interval of time in us since last loading, within which events are loaded

        Returns:
            events
        """

        delta_t = int(delta_t)
        final_time = self.current_time + delta_t

        self._advance(final_time, skip=False)

        # return events that have timestamps between [current_time, current_time+dt[
        ev_count = self._count_ev_loaded()

        if self._begin_buffer + ev_count < self._event_buffer.size:
            index = np.searchsorted(
                self._event_buffer[self._begin_buffer:self._begin_buffer + ev_count]['t'], final_time)
            events = self._event_buffer[self._begin_buffer:self._begin_buffer + index]
            self._begin_buffer += index
        else:
            events = np.concatenate(
                (self._event_buffer[self._begin_buffer:],
                 self._event_buffer[:ev_count - self._event_buffer.size + self._begin_buffer]))
            index = np.searchsorted(events[:]['t'], final_time)
            if self._begin_buffer + index < self._event_buffer.size:
                self._begin_buffer += index
            else:
                self._begin_buffer = self._end_buffer - (ev_count - index)
        # update variables describing the Class state
        self._current_event_index += events[:index].size
        self.current_time = final_time
        self.is_done()
        return events[:index]

    def seek_time(self, final_time):
        """
        seeks into the RAW file until current_time >= final_time.

        Args:
            final_time (int): Timestamp in us at which the search stops.
        """
        final_time = int(final_time)
        self._advance(final_time, skip=True)

        # adjust the beginning of the buffer to the right final_time
        ev_count = self._count_ev_loaded()
        begin_buffer_mem = self._begin_buffer

        if self._begin_buffer + ev_count <= self._event_buffer.size:
            self._begin_buffer += np.searchsorted(
                self._event_buffer[self._begin_buffer:self._begin_buffer + ev_count]['t'], final_time)
        else:
            events = np.concatenate(
                (self._event_buffer[self._begin_buffer:],
                 self._event_buffer[:ev_count - self._event_buffer.size + self._begin_buffer]))
            index = np.searchsorted(events[:]['t'], final_time)
            if self._begin_buffer + index < self._event_buffer.size:
                self._begin_buffer += index
            else:
                self._begin_buffer = self._end_buffer - (ev_count - index)
        # update variables describing the Class state
        self._current_event_index += self._begin_buffer - begin_buffer_mem
        self.current_time = final_time
        self.is_done()

    def is_done(self):
        """
        indicates if all event have been loaded and if the rolling buffer is empty
        """
        self.done = self._decode_done and (self._begin_buffer >= self._end_buffer)
        return self.done
