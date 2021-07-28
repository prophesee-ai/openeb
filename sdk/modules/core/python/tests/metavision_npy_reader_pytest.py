# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for EventNpyReader class loading npy
"""
import os
import numpy as np

from metavision_core.event_io.py_reader import EventNpyReader


def convert_to_array(lines):
    lines = lines.split('\n')[:-1]
    bbox_dtype = [('t', 'u8'), ('x', 'f4'), ('y', 'f4'), ('w', 'f4'), ('h', 'f4'),
                  ('class_id', 'u1'), ('class_confidence', 'f4')]
    array = np.zeros((len(lines)), dtype=bbox_dtype)
    cnt = 0
    for i, line in enumerate(lines):
        if "BB_DELETE" in line:
            continue
        line_content = line.split(" ")
        array['t'][cnt] = int(line_content[0])
        array['class_id'][cnt] = int(line_content[3])
        array['x'][cnt] = float(line_content[4])
        array['y'][cnt] = float(line_content[5])
        array['w'][cnt] = float(line_content[6])
        array['h'][cnt] = float(line_content[7])
        cnt += 1
    return array[:cnt]


def create_temporary_npy_file(tmpdir, name, lines):
    """
    Create a numpy binary file out of the given lines converted to bboxes
    """
    filename = str(tmpdir.join(name + '.npy'))
    bboxes = convert_to_array(lines)
    np.save(filename, bboxes)
    return filename, bboxes


def pytestcase_load_n_events_npy(tmpdir):
    """Tests loading a defined number of events"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "load_n_events",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "199999 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "199999 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199999 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    n_events = 4
    events = record.load_n_events(n_events)
    assert all([np.allclose(events[name], bboxes[name][:n_events]) for name in events.dtype.names])
    # we loaded n_events events, so cursor should have been shifted by 10*8
    assert record.current_event_index() == 0 + n_events
    assert record.done is False
    # current time should be the timestamp of the event that will be loaded next
    assert record.current_time == 199999


def pytestcase_load_n_events_all_npy(tmpdir):
    """Tests loading all of events"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "load_n_events_all",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "199999 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "199999 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199999 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    n_events = record.event_count()
    events = record.load_n_events(n_events)
    assert all([np.allclose(events[name], bboxes[name][:n_events]) for name in events.dtype.names])
    # we loaded n_events events, so cursor should have been shifted by 10*8
    assert record.current_event_index() == 0 + n_events
    assert record.done
    # current time should be the timestamp of the last event + 1
    assert record.current_time == 199999 + 1


def pytestcase_load_n_events_too_much_npy(tmpdir):
    """Tests loading too many events"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "load_n_events_too_much",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "199999 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "199999 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199999 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    events = record.load_n_events(3)
    assert record.done is False
    events = record.load_n_events(10)
    assert all([np.allclose(events[name], bboxes[name][3:]) for name in events.dtype.names])
    # we loaded n_events events, so cursor should have been shifted by 10*8
    assert record.current_event_index() == 0 + record.event_count()
    assert record.done
    # current time should be the timestamp of the last event + 1
    assert record.current_time == 199999 + 1


def pytestcase_reset_npy(tmpdir):
    """Tests resetting EventNpyReader object after loading some events"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "reset",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "199999 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "199999 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199999 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    events = record.load_n_events(10)
    record.reset()
    assert record.current_event_index() == 0
    assert record.done is False
    assert record.current_time == 0


def pytestcase_load_delta_t_npy(tmpdir):
    """Tests loading events inside a time window"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "delta_t",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100000 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199999 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    events = record.load_delta_t(100000)
    assert record.done is False
    assert all([np.allclose(events[name], bboxes[name][:3]) for name in events.dtype.names])
    assert record.current_event_index() == 0 + 3
    # current time should be 0 + 100000
    assert record.current_time == 100000


def pytestcase_load_delta_t_too_much_npy(tmpdir):
    """Tests loading events in a time window larger than total file duration"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "delta_t_too_much",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100000 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199999 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    events = record.load_delta_t(100000)
    assert record.done is False
    events = record.load_delta_t(10 * 100000)
    assert record.done
    assert all([np.allclose(events[name], bboxes[name][3:]) for name in events.dtype.names])
    # current_time should be last event timestamp + 1
    assert record.current_time == 199999 + 1


def pytestcase_load_event_plus_delta_t_npy(tmpdir):
    """Tests loading a define number of events and consecutively a time window"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "load_event_plus_delta_t",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100000 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199999 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    events = record.load_n_events(3)
    assert record.done is False
    # current time should be the timestamp of the event that will be loaded next
    assert record.current_time == 99999 + 1
    events = record.load_delta_t(15000)
    assert record.current_event_index() == 0 + 3 + 2
    assert all([np.allclose(events[name], bboxes[name][3:5]) for name in events.dtype.names])
    assert record.done is False
    assert record.current_time == 99999 + 1 + 15000
    assert record.done is False


def pytestcase_seek_event_future_npy(tmpdir):
    """Tests seeking at a define position (after n events)"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "seek_event_future",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100001 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199998 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    record.seek_event(3)
    assert record.current_event_index() == 0 + 3
    assert record.done is False
    # current_time should be the timestamp of the fourth event
    assert record.current_time == 100000


def pytestcase_seek_event_past_npy(tmpdir):
    """Tests seeking at a define position (after n events)"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "seek_event_past",
                                                 "100 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "200 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100001 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199998 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    record.seek_event(3)
    assert record.current_event_index() == 0 + 3
    assert record.done is False
    # current_time should be the timestamp of the fourth event
    assert record.current_time == 100000
    record.seek_event(1)
    assert record.current_event_index() == 0 + 1
    assert record.done is False
    # current_time should be the timestamp of the second event
    assert record.current_time == 200


def pytestcase_seek_event_zero_npy(tmpdir):
    """Tests seeking in the file after 0 event"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "seek_event_zero",
                                                 "100 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "200 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100001 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199998 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    record.load_n_events(6)
    assert record.current_event_index() == 0 + 6
    assert record.done is False
    # current_time should be the timestamp of the seventh event
    assert record.current_time == 199999
    events = record.seek_event(0)
    assert record.current_event_index() == 0 + 0
    assert record.done is False
    assert record.current_time == 0


def pytestcase_seek_event_negative_npy(tmpdir):
    """Tests seeking in the file after a negative number of events"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "seek_event_negative",
                                                 "100 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "200 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100001 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199998 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    record.load_n_events(6)
    assert record.current_event_index() == 0 + 6
    assert record.done is False
    # current_time should be the timestamp of the seventh event
    assert record.current_time == 199999
    events = record.seek_event(-4)
    assert record.current_event_index() == 0 + 0
    assert record.done is False
    assert record.current_time == 0


def pytestcase_seek_event_too_large_npy(tmpdir):
    """Tests seeking in the file after a number of events superior to the
    number of events in the whole file"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "seek_event_too_large",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100001 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199998 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    record.seek_event(900)
    assert record.current_event_index() == 0 + 7
    assert record.done
    # current_time should be last event timestamp + 1
    assert record.current_time == 199999 + 1


def pytestcase_seek_time_with_few_events_npy(tmpdir):
    """Tests seeking in a file containing not so many events at a position defined by a timestamp.
    The fact that there are few events implies that position should only be found using numpy searchsort"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "seek_time_with_few_events",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100001 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "199998 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    record.seek_time(150000)
    assert record.current_event_index() == 0 + 5
    assert record.done is False
    assert record.current_time == 150000


def pytestcase_seek_time_with_numerous_events_npy(tmpdir):
    """Tests seeking in a file containing with many events at a position defined by a timestamp. The fact
    that there are numerous events implies that position should be found using dichotomy plus numpy searchsort"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "seek_time_with_numerous_events",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100000 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "100000 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    record.seek_time(100000)
    assert record.current_event_index() == 0 + 3
    assert record.done is False
    assert record.current_time == 100000


def pytestcase_seek_time_before_first_event_npy(tmpdir):
    """Tests seeking in a file at a position in time lower than first event timestamp"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "seek_time_before first_event",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100000 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "100000 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    record.seek_time(9000)
    assert record.current_event_index() == 0 + 0
    assert record.done is False
    assert record.current_time == 9000


def pytestcase_seek_time_after_last_event_npy(tmpdir):
    """Tests seeking in a file at a position in time higher than last event timestamp"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "seek_time_after_last_event",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100000 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "100000 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    record.seek_time(20000000)
    assert record.current_event_index() == 0 + record.event_count()
    assert record.done
    # current_time should be last event timestamp + 1
    assert record.current_time == 199999 + 1


def pytestcase_seek_time_with_negative_time_npy(tmpdir):
    """Tests seeking in a file at a position with negative time"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "seek_time_with_negative_time",
                                                 "99999 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99999 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100000 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "100000 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    record.seek_time(-15)
    assert record.current_event_index() == 0 + 0
    assert record.done is False
    # current_time should be last event timestamp + 1
    assert record.current_time == 0


def pytestcase_seek_time_exactly_on_an_event_npy(tmpdir):
    """Tests seeking in a file at an exact event timestamp position"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "seek_time_exactly_on_an_event",
                                                 "99997 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99998 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100001 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "100002 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    record.seek_time(100001)
    # event with timestamp 131200 is the 5th event, it should not be loaded
    assert record.current_event_index() == 0 + 4
    assert record.done is False
    # current_time should be the 5th event timestamp
    assert record.current_time == 100001


def pytestcase_total_time_npy(tmpdir):
    """Tests accessing file total time without changing position in the file"""
    filename, bboxes = create_temporary_npy_file(tmpdir, "total_time",
                                                 "99997 0 BB_CREATE 7 96 102 22 21 0.3\n"
                                                 "99998 2 BB_CREATE 7 127 87 53 53 0.3\n"
                                                 "99999 3 BB_CREATE 7 141 81 119 78 0.7\n"
                                                 "100000 4 BB_CREATE 7 97 102 22 21 0.3\n"
                                                 "100001 5 BB_CREATE 6 8 69 51 53 0.6\n"
                                                 "100002 6 BB_CREATE 7 127 87 53 53 0.4\n"
                                                 "199999 7 BB_CREATE 7 141 81 119 78 0.7\n")
    record = EventNpyReader(filename)
    assert record.current_event_index() == 0
    assert record.event_count() == 7
    assert record.current_time == 0
    time = record.total_time()
    assert time == 199999


def pytestcase_equivalency(dataset_dir):
    """loading boxes through numpy and EventNpyReader should be equivalent"""
    # GIVEN
    box_file = os.path.join(dataset_dir, "metavision_core", "event_io", "bbox.npy")
    boxes = np.load(box_file)
    record = EventNpyReader(box_file)

    # WHEN
    boxes2 = []
    while not record.is_done():
        boxes2.append(record.load_delta_t(1500000))
    boxes2 = np.concatenate(boxes2)

    # THEN
    for name in boxes.dtype.names:
        assert np.allclose(boxes2[name], boxes[name])

    # WHEN
    record.seek_time(0)
    boxes2 = []
    while not record.is_done():
        boxes2.append(record.load_n_events(500))
    boxes2 = np.concatenate(boxes2)

    # THEN
    for name in boxes.dtype.names:
        assert np.allclose(boxes2[name], boxes[name])
