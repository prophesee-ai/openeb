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

import os
import numpy as np
from metavision_sdk_base import EventCD
from metavision_core_ml.video_to_event.simulator import EventSimulator
from metavision_core_ml.data.video_stream import TimedVideoStream
from metavision_sdk_core import SharedCdEventsBufferProducer as EventsBufferProducer
from collections import deque
import skvideo.io
import cv2


class SimulatedEventsIterator(object):
    """
    SimulatedEventsIterator is a small convenience class to generate an iterator of events from any video

    Attributes:
        reader : class handling the video (iterator of the frames and their timestamps).
        delta_t (int): Duration of served event slice in us.
        max_duration (int): If not None, maximal duration of the iteration in us.
        end_ts (int): If max_duration is not None, last time_stamp to consider.
        relative_timestamps (boolean): Whether the timestamps of served events are relative to the current
            reader timestamp, or since the beginning of the recording.

    Args:
        input_path (str): Path to the file to read.
        start_ts (int): First timestamp to consider (in us).
        mode (string): Load by timeslice or number of events. Either "delta_t" or "n_events"
        delta_t (int): Duration of served event slice in us.
        n_events (int): Number of events in the timeslice.
        max_duration (int): If not None, maximal duration of the iteration in us.
        relative_timestamps (boolean): Whether the timestamps of served events are relative to the current
            reader timestamp, or since the beginning of the recording.
        Cp (float): mean for ON threshold
        Cn (float): mean for OFF threshold
        refractory_period (float): min time between 2 events / pixel
        sigma_threshold (float): standard deviation for threshold array
        cutoff_hz (float): cutoff frequency for photodiode latency simulation
        leak_rate_hz (float): frequency of reference value leakage
        shot_noise_rate_hz (float): frequency for shot noise events
        override_fps (int): override fps of the input video.

    Examples:
            >>> for ev in SimulatedEventsIterator("beautiful_record.mp4", delta_t=1000000, max_duration=1e6*60):
            >>>     print("Rate : {:.2f}Mev/s".format(ev.size * 1e-6))
    """

    def __init__(self, input_path, start_ts=0, mode="delta_t", delta_t=10000, n_events=10000, max_duration=None,
                 relative_timestamps=False, height=-1, width=-1, Cp=0.11, Cn=0.1, refractory_period=1e-3,
                 sigma_threshold=0.0, cutoff_hz=0, leak_rate_hz=0, shot_noise_rate_hz=0, override_fps=0):

        # Build simulator
        # initialise a buffer of events
        # Read metadata from videos
        self.mode = mode
        self.path = input_path
        self.relative_timestamps = relative_timestamps
        self.height, self.width = height, width

        # Time attributes
        self.start_ts = start_ts
        self.delta_t = delta_t
        self.n_events = n_events
        self.current_time = start_ts
        self.override_fps = override_fps

        # Metadata
        metadata = skvideo.io.ffprobe(input_path)
        self.num_frames = int(metadata["video"]["@nb_frames"])
        self.original_height, self.original_width = int(metadata["video"]["@height"]), int(metadata["video"]["@width"])
        self.freq = eval(metadata["video"]["@avg_frame_rate"]) * 1e-6
        self.length = float(metadata["video"]['@duration']) * 1e6
        self.nb_frames = int(metadata["video"]["@nb_frames"])

        if max_duration is None:
            self.max_frames = 0
        else:
            self.max_frames = int(max_duration * self.freq)

        if height is None or height < 0:
            self.height = self.original_height
        if width is None or width < 0:
            self.width = self.original_width

        # Simulator parameters
        self.Cp = Cp
        self.Cn = Cn
        self.refractory_period = refractory_period
        self.sigma_threshold = sigma_threshold
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.shot_noise_rate_hz = shot_noise_rate_hz

        # Initialize Simulator, video frame iterator and buffer
        self._initialize()
        self.end_ts = self.reader.duration_s * 1e6

    def _initialize(self):
        # Initializes Event buffer
        if self.mode == "delta_t":
            self.buffer_producer = EventsBufferProducer(self._process_batch, event_count=0, time_slice_us=self.delta_t)
        else:
            self.buffer_producer = EventsBufferProducer(
                self._process_batch, event_count=self.n_events, time_slice_us=0)
        self._event_buffer = deque()

        ts_path = os.path.splitext(self.path)[0] + '_ts.npy'
        if os.path.exists(ts_path):
            assert self.override_fps == 0, "Parameter override_fps should not be given if _ts.npy file is provided"
            ts_npy = np.load(ts_path)
            assert ts_npy.size == self.nb_frames, f"Error: Number of frames ({self.nb_frames}) and number of timestamps ({ts_npy.size}) are inconsistent"
            start_frame = np.searchsorted(1e6*ts_npy, self.start_ts, side="left")
        elif self.override_fps:
            start_frame = int(self.start_ts * self.override_fps)
        else:
            start_frame = int(self.start_ts * self.freq)

        # Initializes Video iterator
        self.reader = TimedVideoStream(
            self.path, self.height, self.width, start_frame=start_frame, max_frames=self.max_frames, rgb=False,
            override_fps=self.override_fps)
        # Initializes Simulator
        self.simu = EventSimulator(self.height, self.width, Cp=self.Cp, Cn=self.Cn,
                                   refractory_period=self.refractory_period, sigma_threshold=self.sigma_threshold,
                                   cutoff_hz=self.cutoff_hz, leak_rate_hz=self.leak_rate_hz,
                                   shot_noise_rate_hz=self.shot_noise_rate_hz)
        self.simu.last_event_timestamp[...] = self.start_ts

    def get_size(self):
        """Function returning the size of the imager which produced the events.

        Returns:
            Tuple of int (height, width) which might be (None, None)"""
        return int(self.original_height), int(self.original_width)

    def __repr__(self):
        string = "SimulatedEventsIterator({})\n".format(self.path)
        string += "delta_t {} us\n".format(self.delta_t)
        string += "starts_ts {} us end_ts {}".format(self.start_ts, self.end_ts)
        return string

    def __iter__(self):
        # reinitializes the simulator at start_ts
        self._initialize()
        self.current_time = self.start_ts + self.delta_t
        previous_time = self.start_ts
        for img, ts in self.reader:
            total = self.simu.image_callback(img, ts)
            if (self.mode == 'n_events' and total < self.n_events) or (
                    self.mode == 'delta_t' and ts < self.current_time):
                continue
            events = self.simu.get_events()
            self.simu.flush_events()
            self.buffer_producer.process_events(events)

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

    def _process_batch(self, ts, batch):
        self._event_buffer.append((ts, batch))

    def __del__(self):
        if hasattr(self, "reader"):
            del self.reader
        if hasattr(self, "simu"):
            self.simu.__del__()
