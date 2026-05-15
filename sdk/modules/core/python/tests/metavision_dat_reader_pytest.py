# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for EventDatReader class
"""
import os
import math
import numpy as np

from metavision_core.event_io.py_reader import EventDatReader
from metavision_core.event_io import DatWriter, load_events
from metavision_core.event_io.dat_tools import EV_TYPES, EV_STRINGS


def pytestcase_check_ev_dtypes():
    # check that EV_TYPES and EV_STRINGS describe the same events
    assert set(EV_TYPES.keys()) == set(EV_STRINGS.keys())
    for ev_type in EV_TYPES:
        # check that the dtype are correct
        np.empty(10, dtype=EV_TYPES[ev_type])


def pytestcase_init(dataset_dir):
    """Tests initialization of all member variables after creation of EventDatReader object from a file"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    record = EventDatReader(filename)
    # check that the representation strings Works
    print(record)
    assert record.ev_type == 12
    assert record._ev_size == 8
    assert record.get_size() == [480, 640]
    assert EV_TYPES[record.ev_type] == [('t', 'u4'), ('_', 'i4')]

    assert record.event_count() == 667855
    assert record.done is False
    assert record.current_time == 0
    assert record.current_event_index() == 0
    assert math.isclose(record.duration_s, 7.702821)
    assert record.load_n_events(1).dtype == np.dtype({'names': ['x', 'y', 'p', 't'],
                                                      'formats': ['<u2', '<u2', '<i2', '<i8'],
                                                      'offsets': [0, 2, 4, 8], 'itemsize': 16})


def pytestcase_load_n_events(dataset_dir):
    """Tests loading a define number of events"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    record = EventDatReader(filename)
    assert record.current_event_index() == 0
    events = record.load_n_events(12)
    reference = np.array([(62,  50, 0, 24), (408, 191, 0, 25), (105, 344, 1, 39), (190, 154, 1, 40),
                          (487, 101, 1, 42), (410, 259, 1, 45), (347, 34, 0, 53), (478, 283, 0, 64),
                          (535, 111, 1, 65), (426, 110, 0, 78), (466, 101, 1, 80), (252, 227, 1, 87)],
                         dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'],
                                'offsets': [0, 2, 4, 8], 'itemsize': 16})
    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])
    # we loaded 10 events, so cursor should have been shifted by 10
    assert record.current_event_index() == 12
    assert record.done is False
    assert record.current_time == 87


def pytestcase_load_n_events_second_test(dataset_dir):
    """Tests loading events of a file by n events with different values"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    record = EventDatReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 667855
    events = record.load_n_events(824)
    assert record.current_event_index() == 824
    assert record.done is False
    assert record.current_time == 9812
    events = record.load_n_events(2)
    reference = np.array([(364, 97, 0, 9814), (254, 463, 0, 9824)],
                         dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'],
                                'offsets': [0, 2, 4, 8], 'itemsize': 16})

    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])
    assert record.current_event_index() == 826
    assert not record.done
    assert record.current_time == 9824


def pytestcase_reset(dataset_dir):
    """Tests resetting EventDatReader object after loading some events"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    record = EventDatReader(filename)
    events = record.load_n_events(10)
    record.reset()
    assert record.current_event_index() == 0
    assert record.done is False
    assert record.current_time == 0


def pytestcase_load_event_plus_delta_t(dataset_dir):
    """Tests loading a define number of events and consecutively a time window"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    record = EventDatReader(filename)
    events = record.load_n_events(13)
    # Now we should be after event (t,x,y,p) : (88, 27, 153, 0)
    test = np.array(events[12])
    reference = np.array((419, 356, 1, 88),
                         dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'],
                                'offsets': [0, 2, 4, 8], 'itemsize': 16})
    assert all([np.allclose(test[name], reference[name]) for name in events.dtype.names])
    # current time should be the timestamp of the event that will be loaded next
    assert record.current_time == 88
    events = record.load_delta_t(87)
    # current time will be 88 + 87 = 175 but event with ts 175 will not be loaded
    reference = np.array([(401, 367, 1,  93), (421, 332, 1, 118), (569, 360, 1, 121), (385, 220, 1, 135),
                          (182, 213, 1, 138), (354, 340, 1, 138), (390, 331, 0, 163)],
                         dtype={'names': ['x', 'y', 'p', 't'], 'formats': ['<u2', '<u2', '<i2', '<i8'],
                                'offsets': [0, 2, 4, 8], 'itemsize': 16})
    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])
    assert record.current_event_index() == 13 + 7
    assert record.done is False
    assert record.current_time == 175


def pytestcase_seek_event_future(dataset_dir):
    """Tests seeking at a define position (after n events)"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    record = EventDatReader(filename)
    assert record.current_event_index() == 0
    assert record.done is False
    assert record.current_time == 0
    record.seek_event(13)
    assert record.current_event_index() == 13
    assert record.done is False
    # current_time should be the timestamp of the thirteenth event
    assert record.current_time == 88


def pytestcase_seek_event_past(dataset_dir):
    """Tests seeking at a define position (after n events)"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    record = EventDatReader(filename)
    assert record.current_event_index() == 0
    assert record.done is False
    assert record.current_time == 0
    record.load_n_events(18)
    assert record.current_event_index() == 18
    assert record.done is False
    # current_time should be the timestamp of the eighteenth event
    assert record.current_time == 138
    record.seek_event(15)
    assert record.current_event_index() == 15
    assert record.done is False
    # current_time should be the timestamp of the fifteenth event
    assert record.current_time == 118


def pytestcase_seek_time_with_numerous_events(dataset_dir):
    """Tests seeking in a file containing not so many events at a position defined by a timestamp.The fact
    that there are numerous events implies that position should be found using dichotomy plus numpy searchsort"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    record = EventDatReader(filename)
    record.seek_time(100)
    assert record.current_event_index() == 14
    assert record.done is False
    assert record.current_time == 100


def pytestcase_cycle_consistency_read_write(tmpdir, dataset_dir):
    """Tests reading and writing DAT files and that the read write read cycle is consistent"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    record = EventDatReader(filename)
    # height, width = record.get_size()
    # it is a gen1 recording with no height or width
    tmp_filename = str(tmpdir.join("tmp_td.dat"))

    writer = DatWriter(tmp_filename, height=480, width=640)

    event_buffers = []

    for i in range(20):
        events = record.load_n_events(1000)
        event_buffers.append(events)
        writer.write(events)
    writer.close()

    new_record = EventDatReader(tmp_filename)
    assert new_record.ev_type == 0
    assert new_record.get_size() == [480, 640]
    assert record.current_time >= new_record.duration_s * 1e6
    assert math.isclose(events['t'][-1] - event_buffers[0]['t'][0], new_record.duration_s * 1e6)
    assert new_record.event_count() == 20 * 1000

    for i, events in enumerate(event_buffers):
        new_events = new_record.load_n_events(1000)
        assert all([np.allclose(events[name], new_events[name]) for name in events.dtype.names])

    assert record.current_event_index() == new_record.current_event_index()


def pytestcase_consistency(dataset_dir):
    """Tests reading and DAT files with the function and the class to check equivalence"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    record = EventDatReader(filename)

    event_buffers = []

    for i in range(7):
        events = record.load_n_events(1000)
        new_events = load_events(filename, ev_count=1000, ev_start=i * 1000)
        assert all([np.allclose(events[name], new_events[name]) for name in events.dtype.names])
