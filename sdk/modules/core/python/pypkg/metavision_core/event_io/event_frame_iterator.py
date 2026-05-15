# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Simple Iterator over event frame reader (diff and histo)
"""
from collections import deque
import os

from metavision_sdk_core import RawEventFrameConverter
from metavision_hal import DeviceDiscovery
from metavision_core.event_io.raw_info import is_event_frame_raw, is_event_raw, raw_file_header


class EventFrameReader(object):
    def __init__(self, input_path):
        """
        Reads a raw file of either DIFF3D or HISTO3D
        """
        assert os.path.isfile(input_path)
        if is_event_raw(input_path):
            raise RuntimeError(
                f"The file {input_path} is an event raw, not a event-frame raw. EventFrameReader should be used only on frames. Please use EventsIterator to read events")
        assert is_event_frame_raw(input_path), "Error: filename {input_path} is not a valid event-frame raw file"
        header_dic = raw_file_header(input_path)
        assert "format" in header_dic
        self.frame_type = header_dic["format"]
        assert self.frame_type in ["DIFF3D", "HISTO3D"], f"Unsupported frame type: {self.frame_type}"

        self._decode_done = False
        self.decoded_frames = deque()

        self.device = DeviceDiscovery.open_raw_file(input_path)
        if self.device is None:
            raise OSError(f"Incorrect raw input file : {input_path} !")

        i_geometry = self.device.get_i_geometry()
        if i_geometry is None:
            raise OSError("No I_Geometry facility")

        self.width = i_geometry.get_width()
        self.height = i_geometry.get_height()
        assert (self.height, self.width) == (320, 320), "Wrong size for event frames: only 320x320 is supported"

        self.raw_event_frame_converter = RawEventFrameConverter(self.height, self.width)

        def process_event_frame(frame):
            if self.frame_type == "DIFF3D":
                frame_np = self.raw_event_frame_converter.convert_diff(frame)
            elif self.frame_type == "HISTO3D":
                frame_np = self.raw_event_frame_converter.convert_histo(frame)
            else:
                raise NotImplementedError(f"Unsupported frame type: {self.frame_type}")
            self.decoded_frames.append(frame_np)

        self.i_events_stream = self.device.get_i_events_stream()

        if self.frame_type == "DIFF3D":
            self.frame_decoder = self.device.get_i_event_frame_diff_decoder()
        elif self.frame_type == "HISTO3D":
            self.frame_decoder = self.device.get_i_event_frame_histo_decoder()
        else:
            raise NotImplementedError(f"Unsupported type of event frame: {self.frame_type}")
        self.frame_decoder.add_event_frame_callback(process_event_frame)

        self.i_events_stream_decoder = self.device.get_i_events_stream_decoder()

        self.i_events_stream.start()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.__del__()

    def get_size(self):
        return (self.height, self.width)

    def _decode_next_frames(self):
        if self._decode_done:
            return
        if self.i_events_stream.poll_buffer() < 0:
            # This would only happen in a live camera or when the file is done
            self._decode_done = True
            return
        data = self.i_events_stream.get_latest_raw_data()
        self.frame_decoder.decode(data)

    def load_next_frame(self):
        while (not self._decode_done) and (len(self.decoded_frames) == 0):
            self._decode_next_frames()
        if self._decode_done and len(self.decoded_frames) == 0:
            raise StopIteration
        assert len(self.decoded_frames) > 0
        return self.decoded_frames.popleft()

    def is_done(self):
        self.done = self._decode_done and (len(self.decoded_frames) == 0)
        return self.done

    def __del__(self):
        if hasattr(self, "i_events_stream") and self.i_events_stream is not None:
            self.i_events_stream.stop()
            self.i_events_stream.stop_log_raw_data()
        if hasattr(self, "i_events_stream"):
            del self.i_events_stream
        if hasattr(self, "frame_decoder"):
            del self.frame_decoder
        if hasattr(self, "device"):
            del self.device
        if hasattr(self, "decoded_frames"):
            del self.decoded_frames


class EventFrameIterator(object):
    def __init__(self, input_path):
        """
        Iterates over a raw file of either DIFF3D or HISTO3D
        """
        self._ran = False
        self.reader = EventFrameReader(input_path=input_path)

    def get_size(self):
        """Function returning the size of the imager which produced the events.

        Returns:
            Tuple of int (height, width) which might be (None, None)"""
        return self.reader.get_size()

    def get_frame_type(self):
        """
        Returns the frame type. Will be either 'DIFF3D' or 'HISTO3D'
        """
        return self.reader.frame_type

    def __iter__(self):
        if self._ran:
            raise Exception('Can not iterate twice over the same EventIterator!')
        self._ran = True

        with self.reader as reader:
            while not reader.is_done():
                try:
                    frame = reader.load_next_frame()
                    yield frame
                except StopIteration:
                    break

    def __del__(self):
        if hasattr(self, "reader"):
            self.reader.__del__()
