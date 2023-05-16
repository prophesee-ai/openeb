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
    Constructs a device either from a file if the path ends with RAW or with the camera serial number.

    This device can be used in conjunction with `RawReader.from_device` or `EventsIterator.from_device`
    to create a RawReader or an EventsIterator with a customized HAL device.

    Args:
        path (str): either path to a RAW file (having a .raw or .RAW extension) or a camera serial number.
            leave blank to take the first available camera.
        do_time_shifting (bool): in case of a file, makes the timestamps start close to 0us.
        use_external_triggers (Channel List): list of channels of external trigger to be activated (only relevant for a live camera).
            On most systems, only one (MAIN) channel can be enabled.
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
        if hasattr(self, "i_events_stream") and self.i_events_stream is not None:
            self.i_events_stream.stop()
            self.i_events_stream.stop_log_raw_data()
        if hasattr(self, "i_events_stream"):
            del self.i_events_stream
        if hasattr(self, "i_events_stream_decoder"):
            del self.i_events_stream_decoder
        if hasattr(self, "i_event_cd_decoder"):
            del self.i_event_cd_decoder
        if hasattr(self, "i_eventdecoder_ext_trigger"):
            del self.i_eventdecoder_ext_trigger
        if hasattr(self, "device"):
            del self.device
        if hasattr(self, "buffer_producer"):
            del self.buffer_producer
        if hasattr(self, "_event_buffer"):
            del self._event_buffer
        if hasattr(self, "_event_ext_trigger_buffer"):
            del self._event_ext_trigger_buffer

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.__del__()

    def _process_batch(self, ts, batch):
        # in case of fast forward incoming events are discarded
        if ts > self._seek_time and len(batch) > self._seek_event:
            self._event_buffer.append((ts, batch))
        else:
            self._current_event_index += len(batch)
            self._seek_event -= len(batch)

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
        self.i_events_stream_decoder.decode(data)

        return True

    def _advance(self, n_events=0, delta_t=0, drop_events=False):
        """
        decodes events until either n_events or delta_t events are decoded.

        Args:
            n_events (int): number of events to decode.
            delta_t (int): duration in us of events to decode
            drop_events (boolean): if True drop the decoded events until the desired point.
        """
        final_time = int(delta_t + self.current_time)
        if self.current_time > final_time:
            raise RuntimeError('cannot seek backward in RAW file')

        def _are_enough_ev_loaded(final_time, n_events):
            enough = final_time == self.current_time or final_time < self._last_loaded_ts()
            return enough and ((not n_events) or
                               (n_events < self._count_ev_loaded()) or
                               (self._seek_event > 0 and self._seek_event < self._count_ev_loaded()))

        if drop_events:
            if delta_t:
                self._seek_time = final_time
            if n_events:
                self._seek_event = n_events

        while not (self._decode_done or _are_enough_ev_loaded(final_time, n_events)):
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

        self.i_events_stream_decoder = self.device.get_i_events_stream_decoder()

        if self.i_events_stream_decoder is None:
            raise OSError("No I_Events_Stream_Decoder facility")

        self.i_event_cd_decoder = self.device.get_i_event_cd_decoder()
        if self.i_event_cd_decoder is None:
            raise OSError("No I_EventDecoder<Metavision::EventCD> facility")

        self.buffer_producer = EventsBufferProducer(self._process_batch, event_count=self.ev_count,
                                                    time_slice_us=self.delta_t)

        self.i_event_cd_decoder.add_event_buffer_native_callback(
            self.buffer_producer.get_process_events_callback())

        self.i_eventdecoder_ext_trigger = self.device.get_i_event_ext_trigger_decoder()
        if self.i_eventdecoder_ext_trigger is not None:
            self.i_eventdecoder_ext_trigger.add_event_buffer_callback(self.process_batch_ext_trigger)

        self.i_events_stream.start()

        self._reset_state_vars()
        self._reset_buffer()

    def _reset_state_vars(self):
        # reset state variables
        self.done = False
        self._decode_done = False
        self._seek_time = -1
        self._seek_event = -1
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
        # if self.delta_t:
        #     assert final_time % self.delta_t == 0
        self._advance(delta_t=final_time - self.current_time, drop_events=True)
        if self._seek_time > 0 and len(self._event_buffer):
            for index, (ts, evs) in enumerate(self._event_buffer):
                if ts - self.delta_t < self._seek_time:
                    self._event_buffer[index] = (ts, evs[evs['t'] >= self._seek_time])
        self.current_time = final_time

    def seek_event(self, n_events):
        """
        Advance n_events into the RAW file

        Args:
            n_events (int): number of events to skip (only multiples of n_events are supported.).
        """
        assert self.ev_count and n_events % self.ev_count
        self._advance(n_events=n_events, drop_events=True)

    def _last_loaded_ts(self):
        """returns the timestamp of the last loaded event and -1 if None are in the buffer"""
        if self._event_buffer:
            return self._event_buffer[-1][0]
        else:
            return 0

    def _count_ev_loaded(self):
        """return the number of events in buffer"""
        n_events_loaded = 0
        for buf in self._event_buffer:
            n_events_loaded += len(buf)
        return n_events_loaded

    def _load_next_buffer(self):
        """
        Loads a batch of events from the queue.

        Returns:
            events
        """

        while not (self._decode_done or len(self._event_buffer)):
            self._run()
        # if enough events have been decoded, we take the first buffer in the queue.
        if len(self._event_buffer):
            current_ts, events = self._event_buffer.popleft()
            # update variables describing the object state.
            self._current_event_index += events.size
        else:
            events = np.zeros(0, dtype=EventCD)

        increase_time_by_delta_t = self.delta_t != 0 and (not len(events) or len(events) != self.ev_count)
        if increase_time_by_delta_t:
            self.current_time += self.buffer_producer.get_processing_n_us()
        else:
            if events.size:
                self.current_time = events[-1]['t']

        return events

    def load_n_events(self, n_events):
        """
        Loads a batch of *n_events* events.

        Args:
            n_events (int): Number of events to load

        Returns:
            events (numpy array): structured numpy array containing the events.
        """

        return self._load_next_buffer()

    def load_delta_t(self, delta_t):
        """
        Loads all the events contained in the next *delta_t* microseconds.

        Args:
            delta_t (int): Interval of time in us since last loading, within which events are loaded

        Returns:
            events (numpy array): structured numpy array containing the events.
        """

        return self._load_next_buffer()

    def load_mixed(self, n_events, delta_t):
        """Loads batch of n events or delta_t microseconds, whichever comes first.

        Args:
            n_events (int): Maximum number of events that will be loaded.
            delta_t (int): Maximum allowed slice duration (in us).

        Returns:
            events (numpy array): structured numpy array containing the events.

        Note that current time will be incremented to reach the timestamp of the first event not loaded yet Unless
        the maximal time slice duration is reached in which case current time will be increased by delta_t instead.
        """
        return self._load_next_buffer()


class RawReader(RawReaderBase):
    """
    RawReader loads events from a RAW file.

    RawReader allows to read a file of events while maintaining a position of the cursor.
    Further manipulations like advancing the cursor in time are possible.

    Attributes:
        path (string): Path to the file being read. If `path` is an empty string or a camera serial number it will try to open
            that camera instead.
        current_time (int): Indicating the position of the cursor in the file in us.
        do_time_shifting (bool): If True the origin of time is a few us from the first events.
            Otherwise it is when the camera was started.

    Args:
        record_base (string): Path to the record being read.
        do_time_shifting (bool): If True the origin of time is a few us from the first events.
            Otherwise it is when the camera was started.
        use_external_triggers (int List): list of integer values corresponding to the channels of external trigger
            to be activated (only relevant for a live camera).
    """

    def __init__(self, record_base, max_events=int(1e7), do_time_shifting=True,
                 device=None, initiate_device=True, use_external_triggers=[]):
        super().__init__(record_base, device=device, do_time_shifting=do_time_shifting,
                         initiate_device=initiate_device, use_external_triggers=use_external_triggers)
        self._event_buffer = np.empty(max_events, dtype=EventCD)

    @classmethod
    def from_device(cls, device, max_events=int(1e7)):
        """
        Alternate way of constructing an RawReader from an already initialized HAL device.

                Note that it is not recommended to leave a device in the global scope, so either create the HAL device
                in a function or, delete explicitly afterwards. In some cameras this could result in an undefined
                behaviour.

        Args:
            device (device): Hal device object initialized independently.

        Examples:
            >>> device = initiate_device(path=args.input_path)
            >>> # call any methods on device
            >>> reader = RawReader.from_device(device=device)
            >>> del device  # do not leave the device variable in the global scope
        """
        return cls("", device=device, max_events=max_events, initiate_device=False)

    # callbacks
    def _process_batch(self, ts, batch):
        length = len(batch)

        # in case we received empty buffer
        if (length == 0):
            return

        # in case of fast forward incoming events are discarded
        if self._seek_time > batch[-1]['t']:
            self._current_event_index += (self._end_buffer - self._begin_buffer) + length
            self._begin_buffer = self._end_buffer
            return

        if self._seek_event >= length:
            self._current_event_index += (self._end_buffer - self._begin_buffer) + length
            self._begin_buffer = self._end_buffer
            self._seek_event -= length
            self.current_time = ts
            return

        # otherwise the rolling buffer parameters are updated
        begin_buffer = self._end_buffer
        self._end_buffer += length
        # and events are copied in one go or two goes depending on whether we need to "roll" around the buffer
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

    def _last_loaded_ts(self):
        """returns the timestamp of the last loaded event and -1 if None are in the buffer"""
        if self._end_buffer == self._begin_buffer:
            return -1
        return int(self._event_buffer[self._end_buffer - 1]['t'])

    def _reset_buffer(self):
        # resets memory buffer "pointers"
        self._begin_buffer, self._end_buffer = 0, 0

    def load_n_events(self, n_events):
        """
        Loads a batch of *n_events* events.

        Args:
            n_events (int): Number of events to load

        Returns:
            events (numpy array): structured numpy array containing the events.
        """
        n_events = int(n_events)
        self._advance(n_events)

        # if all events are decoded, there is only this classes buffer left.
        if self._decode_done:
            n_events = min(n_events, self._count_ev_loaded())

        if self._begin_buffer + n_events < self._event_buffer.size:
            events = self._event_buffer[self._begin_buffer:self._begin_buffer + n_events]
            self._begin_buffer += n_events
        else:
            ev_dtype = self._event_buffer[self._begin_buffer].dtype
            events = np.concatenate(
                (self._event_buffer[self._begin_buffer:],
                 self._event_buffer[:n_events - self._event_buffer.size + self._begin_buffer])).astype(ev_dtype)
            self._begin_buffer = n_events - self._event_buffer.size + self._begin_buffer
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
            events (numpy array): structured numpy array containing the events.
        """

        delta_t = int(delta_t)
        final_time = self.current_time + delta_t

        self._advance(delta_t=delta_t, drop_events=False)

        # return events that have timestamps between [current_time, current_time+dt[
        n_events = self._count_ev_loaded()

        if self._begin_buffer + n_events < self._event_buffer.size:
            index = np.searchsorted(
                self._event_buffer[self._begin_buffer:self._begin_buffer + n_events]['t'], final_time)
            events = self._event_buffer[self._begin_buffer:self._begin_buffer + index]
            self._begin_buffer += index
        else:
            ev_dtype = self._event_buffer[self._begin_buffer].dtype
            events = np.concatenate(
                (self._event_buffer[self._begin_buffer:],
                 self._event_buffer[:n_events - self._event_buffer.size + self._begin_buffer])).astype(ev_dtype)
            index = np.searchsorted(events[:]['t'], final_time)
            if self._begin_buffer + index < self._event_buffer.size:
                self._begin_buffer += index
            else:
                self._begin_buffer = self._end_buffer - (n_events - index)
        # update variables describing the Class state
        self._current_event_index += events[:index].size
        self.current_time = final_time
        self.is_done()
        return events[:index]

    def load_mixed(self, n_events, delta_t):
        """Loads batch of n events or delta_t microseconds, whichever comes first.

        Args:
            n_events (int): Maximum number of events that will be loaded.
            delta_t (int): Maximum allowed slice duration (in us).

        Returns:
            events (numpy array): structured numpy array containing the events.

        Note that current time will be incremented to reach the timestamp of the first event not loaded yet unless
        the maximal time slice duration is reached in which case current time will be increased by delta_t instead.
        """
        n_events = int(n_events)
        delta_t = int(delta_t)
        self._advance(n_events=n_events, delta_t=delta_t)

        # if all events are decoded, there is only this classes buffer left.
        if self._decode_done:
            n_events = min(n_events, self._count_ev_loaded())

        if self._begin_buffer + n_events < self._event_buffer.size:
            if self._last_loaded_ts() >= (self.current_time + delta_t):
                # we search by delta_t to limit how many events are loaded
                n_events = np.searchsorted(self._event_buffer[self._begin_buffer:self._end_buffer]['t'],
                                           self.current_time + delta_t)

            # we simply load n events in one go from the round buffer
            events = self._event_buffer[self._begin_buffer:self._begin_buffer + n_events]
            self._begin_buffer += n_events

        else:
            # in this case we need to "go around the buffer": to read till the end of the buffer and some from the
            # beginning and concatenate their result.
            if self._last_loaded_ts() >= (self.current_time + delta_t):

                index = np.searchsorted(self._event_buffer[self._begin_buffer:]['t'],
                                        self.current_time + delta_t)
                if index == len(self._event_buffer[self._begin_buffer:]):
                    # we need the whole end of the actual buffer and some extra event from the beginning.
                    second_buffer_part = self._event_buffer[:n_events - self._event_buffer.size + self._begin_buffer]
                    index = np.searchsorted(second_buffer_part['t'], self.current_time + delta_t)
                    ev_dtype = self._event_buffer[self._begin_buffer].dtype
                    events = np.concatenate(
                        (self._event_buffer[self._begin_buffer:], self._event_buffer[:index])).astype(ev_dtype)
                    self._begin_buffer = index

                else:
                    # here all events are gathered before the actual end of the round buffer, no need to read twice.
                    events = self._event_buffer[self._begin_buffer:self._begin_buffer + index]
                    self._begin_buffer += index

            else:
                # we need the whole end of the actual buffer and some extra event from the beginning.
                ev_dtype = self._event_buffer[self._begin_buffer].dtype
                events = np.concatenate(
                    (self._event_buffer[self._begin_buffer:],
                     self._event_buffer[:n_events - self._event_buffer.size + self._begin_buffer])).astype(ev_dtype)
                self._begin_buffer = n_events - self._event_buffer.size + self._begin_buffer

        # update variables describing the Class state
        self._current_event_index += events.size
        if self._last_loaded_ts() >= (self.current_time + delta_t):
            self.current_time += delta_t
        else:
            self.current_time = self._event_buffer[self._begin_buffer]['t']
        self.is_done()
        return events

    def seek_time(self, final_time):
        """
        seeks into the RAW file until current_time >= final_time.

        Args:
            final_time (int): Timestamp in us at which the search stops.
        """
        final_time = int(final_time)

        self._advance(delta_t=final_time - self.current_time, drop_events=True)

        # adjust the beginning of the buffer to the right final_time
        n_events = self._count_ev_loaded()
        begin_buffer_mem = self._begin_buffer

        if self._begin_buffer + n_events <= self._event_buffer.size:
            self._begin_buffer += np.searchsorted(
                self._event_buffer[self._begin_buffer:self._begin_buffer + n_events]['t'], final_time)
        else:
            ev_dtype = self._event_buffer[self._begin_buffer].dtype
            events = np.concatenate(
                (self._event_buffer[self._begin_buffer:],
                 self._event_buffer[:n_events - self._event_buffer.size + self._begin_buffer])).astype(ev_dtype)
            index = np.searchsorted(events[:]['t'], final_time)
            if self._begin_buffer + index < self._event_buffer.size:
                self._begin_buffer += index
            else:
                self._begin_buffer = self._end_buffer - (n_events - index)
        # update variables describing the Class state
        self._current_event_index += self._begin_buffer - begin_buffer_mem
        self.current_time = final_time
        self.is_done()

    def seek_event(self, n_events):
        """
        Advance n_events into the RAW file. The decoded events are dropped.

        Args:
            n_events (int): number of events to skip.
        """
        assert n_events >= 0, "Error: cannot seek in the past"
        if n_events == 0:
            return

        # advance loads buffer until the last one which contains the last buffer we want to move to
        self._advance(n_events=int(n_events), drop_events=True)

        # if there are still event to drop we advance the buffer pointers and class state variables
        self._current_event_index += self._seek_event
        if self._seek_event:
            # first we advance towards the end of the buffer
            adv = min(self._seek_event, self._event_buffer.size - self._begin_buffer)
            self._begin_buffer = self._begin_buffer + adv % self._event_buffer.size
            self._seek_event -= adv
            # if necessary we advance toward the beginning
            if self._seek_event:
                self._begin_buffer = self._seek_event

            self.current_time = self._event_buffer['t'][self._begin_buffer]
        self.is_done()

    def is_done(self):
        """
        Indicates if all events have been already read.
        """
        # we check if all event have been loaded and if the rolling buffer is empty.
        self.done = self._decode_done and (self._begin_buffer >= self._end_buffer)
        return self.done
