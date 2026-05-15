# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Extended Metavision Iterator
We show here how to integrate your own format in an extension of the factory class EventsIterator
"""

from metavision_core.event_io.events_iterator import EventsIterator
from metavision_core.event_io.raw_reader import RawReaderBase
from metavision_core.event_io.py_reader import EventDatReader
from metavision_core.event_io.meta_event_producer import MetaEventBufferProducer
from csv_reader import CSVBaseReader

ROS = True
AEDAT = True

try:
    from ros_reader import RosBaseReader
except:
    ROS = False
try:
    from aedat_reader import AEDATBaseReader
except:
    AEDAT = False


class ExtendedEventsIterator(EventsIterator):
    """
    We extend the EventsIterator to more formats.
    It can handle the following:

    - raw
    - dat
    - aedat
    - zipped csv
    - rosbag

    Attributes:
        reader : class handling the file or camera.
        delta_t (int): Duration of served event slice in us.
        max_duration (int): If not None, maximal duration of the iteration in us.
        end_ts (int): If max_duration is not None, last timestamp to consider.
        relative_timestamps (boolean): Whether the timestamp of served events are relative to the current
            reader timestamp, or since the beginning of the recording.
    Args:
        input_path (str): Path to the file to read. If `path` is an empty string or a camera serial number it will try to open
            that camera instead.
        start_ts (int): First timestamp to consider.
        mode (string): Load by timeslice or number of events. Either "delta_t", "n_events" or "mixed",
            where mixed uses both delta_t and n_events and chooses the first met criterion.
        delta_t (int): Duration of served event slice in us.
        n_events (int): Number of events in the timeslice.
        max_duration (int): If not None, maximal duration of the iteration in us.
        relative_timestamps (boolean): Whether the timestamp of served events are relative to the current
            reader timestamp, or since the beginning of the recording.
        **kwargs: Arbitrary keyword arguments passed to the underlying RawReaderBase or
            EventDatReader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_readers(self, input_path, **kwargs):
        if isinstance(input_path, type("")):
            if input_path.endswith(".dat"):
                self.reader = EventDatReader(input_path, **kwargs)
            elif input_path.endswith(".bag") and ROS:
                ros_reader = RosBaseReader(input_path)
                self.reader = MetaEventBufferProducer(
                    ros_reader, mode=self.mode, delta_t=self.delta_t, n_events=self.n_events,
                    relative_timestamps=self.relative_timestamps)
            elif input_path.endswith(".zip") or input_path.endswith(".txt"):
                csv_reader = CSVBaseReader(input_path)
                self.reader = MetaEventBufferProducer(
                    csv_reader, mode=self.mode, delta_t=self.delta_t, n_events=self.n_events,
                    relative_timestamps=self.relative_timestamps)
            elif input_path.endswith(".aedat4") and AEDAT:
                aedat_reader = AEDATBaseReader(input_path)
                self.reader = MetaEventBufferProducer(
                    aedat_reader, mode=self.mode, delta_t=self.delta_t, n_events=self.n_events,
                    relative_timestamps=self.relative_timestamps)
            elif input_path.endswith(".raw"):
                self.reader = RawReaderBase(input_path, delta_t=self.delta_t, ev_count=self.n_events, **kwargs)
            else:
                if input_path.endswith(".bag") and not ROS:
                    print('rosbag is not installed!')
                if input_path.endswith(".aedat4") and not AEDAT:
                    print('aedat is not installed!')

                raise BaseException("format not handled!")
        else:
            # we assume input_path is an actual device
            self.reader = RawReaderBase.from_device(input_path, delta_t=self.delta_t, ev_count=self.n_events, **kwargs)
