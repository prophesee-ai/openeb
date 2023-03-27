# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# pylint: disable=E1101

"""
Unit tests for RawReader class
"""
import os
import numpy as np

from metavision_core.event_io import RawReader
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import load_events, EventDatReader
from metavision_core.event_io import EventsIterator


def pytestcase_rawreader_init(tmpdir, dataset_dir):
    """Tests initialization of all member variables after creation of RawReader object from a file"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")

    video = RawReader(filename, do_time_shifting=False)
    # WHEN
    video.reset()
    # THEN
    assert video.width == 640
    assert video.height == 480
    assert video.is_done() == False
    assert sum(
        [key in video._event_buffer.dtype.names for key in ('x', 'y', 'p', 't')]) == 4
    assert video._event_buffer.size == 1e7
    assert video.current_time == 0


def pytestcase_rawreader_init_from_device(tmpdir, dataset_dir):
    """Tests initialization of all member variables after creation of RawReader object from a file"""
    # GIVEN
    filename = os.path.join(dataset_dir, "openeb", "core", "event_io", "recording.raw")
    device = initiate_device(filename, do_time_shifting=True)
    # WHEN
    video = RawReader.from_device(device)

    # THEN
    assert video.width == 640
    assert video.height == 480
    assert video.is_done() == False
    assert sum(
        [key in video._event_buffer.dtype.names for key in ('x', 'y', 'p', 't')]) == 4
    assert video._event_buffer.size == int(1e7)
    assert video.current_time == 0

    dat_evs = load_events(filename.replace(".raw", "_td.dat"))
    raw_evs = video.load_n_events(1e8)
    # THEN
    assert len(dat_evs) == 667855
    assert len(raw_evs) == 667855
    assert all([np.allclose(dat_evs[name], raw_evs[name]) for name in ("x", "y", "p")])
    assert np.allclose(dat_evs['t'], raw_evs['t'])


def pytestcase_rawreader_load_n_events(tmpdir, dataset_dir):
    """Tests loading a define number of events"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)
    # WHEN
    events = video.load_n_events(12)
    # THEN
    reference = np.array([(8720024, 62, 50, 0), (8720025, 408, 191, 0), (8720039, 105, 344, 1),
                          (8720040, 190, 154, 1), (8720042, 487, 101, 1), (8720045, 410, 259, 1),
                          (8720053, 347, 34, 0), (8720064, 478, 283, 0), (8720065, 535, 111, 1),
                          (8720078, 426, 110, 0), (8720080, 466, 101, 1), (8720087, 252, 227, 1)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name])
                for name in events.dtype.names])


def pytestcase_rawreader_seek_n_events(tmpdir, dataset_dir):
    """Tests seeking a precise number of events"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)
    # WHEN
    video.seek_event(11)
    events = video.load_n_events(1)
    # THEN
    reference = np.array([(8720087, 252, 227, 1)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name])
                for name in events.dtype.names])
    assert video.current_event_index() == 12

    # WHEN
    video.seek_event(1e7)
    # THEN
    assert video.is_done()


def pytestcase_rawreader_load_n_events_all(tmpdir, dataset_dir):
    """Tests loading all the events of a file"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)
    # WHEN
    events = video.load_n_events(667850)
    # THEN
    assert video.current_event_index() == 667850
    assert video.done == False
    # WHEN
    events = video.load_n_events(5)  # loading the 6 last events
    # THEN
    reference = np.array([(16422810, 324, 146, 1), (16422839, 190, 232, 1), (16422840, 13, 340, 0),
                          (16422842, 418, 289, 1), (16422845, 454, 358, 1)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name])
                for name in events.dtype.names])

    evs = video.load_n_events(1)
    assert evs.size == 0
    assert video.current_event_index() == 667855
    assert video.done  # now it is done


def pytestcase_rawreader_load_n_events_too_much(tmpdir, dataset_dir):
    """Tests loading more events than the number in the file"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)

    events = video.load_n_events(667850)
    assert video.done == False
    events = video.load_n_events(10)
    reference = np.array([(16422810, 324, 146, 1), (16422839, 190, 232, 1), (16422840, 13, 340, 0),
                          (16422842, 418, 289, 1), (16422845, 454, 358, 1)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4],
                                'names': ['t', 'x', 'y', 'p'], 'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name])
                for name in events.dtype.names])
    assert video.done
    assert video.current_event_index() == 667855


def pytestcase_rawreader_load_delta_t(tmpdir, dataset_dir):
    """Tests loading events inside a time window"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)
    assert video.current_time == 0
    _ = video.load_n_events(1)  # we are in no time shifting mode
    events = video.load_delta_t(100)

    reference = np.array([(8720025, 408, 191, 0), (8720039, 105, 344, 1), (8720040, 190, 154, 1),
                          (8720042, 487, 101, 1), (8720045, 410, 259, 1), (8720053, 347, 34, 0),
                          (8720064, 478, 283, 0), (8720065, 535, 111, 1), (8720078, 426, 110, 0),
                          (8720080, 466, 101, 1), (8720087, 252, 227, 1), (8720088, 419, 356, 1),
                          (8720093, 401, 367, 1), (8720118, 421, 332, 1), (8720121, 569, 360, 1)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})

    assert all([np.allclose(events[name], reference[name])
                for name in events.dtype.names])
    assert video.done == False
    assert video.current_time == 8720125


def pytestcase_rawreader_load_small_delta_t(tmpdir, dataset_dir):
    """Tests loading events inside a time window smaller than the controller timeslice"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)
    assert video.current_time == 0
    _ = video.load_n_events(1)  # we are in no time shifting mode
    events = video.load_delta_t(100)

    reference = np.array([(8720025, 408, 191, 0), (8720039, 105, 344, 1), (8720040, 190, 154, 1),
                          (8720042, 487, 101, 1), (8720045, 410, 259, 1), (8720053, 347, 34, 0),
                          (8720064, 478, 283, 0), (8720065, 535, 111, 1), (8720078, 426, 110, 0),
                          (8720080, 466, 101, 1), (8720087, 252, 227, 1), (8720088, 419, 356, 1),
                          (8720093, 401, 367, 1), (8720118, 421, 332, 1), (8720121, 569, 360, 1)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])
    assert video.done == False

    assert video.current_time == 8720125
    # load again
    events = video.load_delta_t(200)
    reference = np.array([(8720135, 385, 220, 1), (8720138, 182, 213, 1),
                          (8720138, 354, 340, 1), (8720163, 390, 331, 0),
                          (8720182, 59, 278, 1), (8720212, 413, 302, 0),
                          (8720218, 532, 217, 1), (8720261, 455, 359, 1),
                          (8720271, 78, 191, 1), (8720272, 72, 186, 0),
                          (8720274, 626, 386, 1), (8720276, 545, 455, 0),
                          (8720305, 489, 282, 0), (8720308, 534, 115, 1),
                          (8720315, 631, 258, 1), (8720315, 500, 384, 0)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])
    assert video.done == False
    assert video.current_time == 8720325
    events = video.load_delta_t(300)
    reference = np.array([(8720328, 139, 4, 0), (8720344, 293, 299, 0),
                          (8720345, 469, 413, 1), (8720346, 149, 445, 0),
                          (8720349, 432, 232, 1), (8720357, 635, 152, 1),
                          (8720372, 76, 252, 0), (8720381, 391, 461, 0),
                          (8720394, 445, 330, 1), (8720416, 603, 345, 0),
                          (8720418, 152, 67, 0), (8720420, 335, 384, 0),
                          (8720444, 0, 100, 1), (8720446, 473, 424, 1),
                          (8720456, 456, 309, 1), (8720458, 466, 450, 1),
                          (8720470, 296, 456, 0), (8720473, 398, 148, 0),
                          (8720477, 311, 278, 0), (8720492, 519, 422, 0),
                          (8720500, 443, 267, 1), (8720510, 381, 189, 0),
                          (8720515, 621, 25, 1), (8720519, 85, 74, 1),
                          (8720521, 545, 359, 1), (8720525, 437, 80, 1),
                          (8720536, 143, 120, 0), (8720536, 343, 283, 0),
                          (8720553, 77, 43, 0), (8720570, 325, 448, 0),
                          (8720586, 623, 443, 1), (8720590, 183, 345, 0)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])
    assert video.current_time == 8720625
    assert video.done == False


def pytestcase_rawreader_load_delta_t_too_much(tmpdir, dataset_dir):
    """Tests loading events in a time window larger than total file duration"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)
    assert video.current_time == 0
    while video.current_time < 16420000:
        _ = video.load_delta_t(10000)
    assert video.done == False
    assert video.current_time == 16420000

    events = video.load_delta_t(10000)
    assert video.done
    assert video.load_delta_t(10000).size == 0
    assert events.size == 273
    events = events[-10:]  # we only check the last 10
    reference = np.array([(16422737, 283, 363, 1), (16422758, 50, 373, 1), (16422758, 3, 303, 1),
                          (16422763, 33, 48, 0), (16422799, 164, 226, 0), (16422810, 324, 146, 1),
                          (16422839, 190, 232, 1), (16422840, 13, 340, 0), (16422842, 418, 289, 1),
                          (16422845, 454, 358, 1)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})

    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])


def pytestcase_rawreader_seek_time(tmpdir, dataset_dir):
    """Tests loading events after a call to seek_time()"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)
    assert video.current_time == 0

    video.seek_time(16420000)
    assert video.done == False
    assert video.current_time == 16420000

    events = video.load_delta_t(10000)
    assert video.done

    assert events.size == 273
    events = events[-10:]  # we only check the last 10
    reference = np.array([(16422737, 283, 363, 1), (16422758, 50, 373, 1), (16422758, 3, 303, 1),
                          (16422763, 33, 48, 0), (16422799, 164, 226, 0), (16422810, 324, 146, 1),
                          (16422839, 190, 232, 1), (16422840, 13, 340, 0), (16422842, 418, 289, 1),
                          (16422845, 454, 358, 1)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})

    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])


def pytestcase_rawreader_exotic_seek_time_and_delta_t(tmpdir, dataset_dir):
    """Tests loading events after a call to seek_time() using a variety delta_t values"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)
    assert video.current_time == 0

    video.seek_time(16420458)
    assert video.done == False
    assert video.current_time == 16420458

    events = video.load_delta_t(2511)
    assert video.current_time == 16422969
    assert video.done
    assert events.size == 230

    events = events[:10]  # we only check the first 10
    reference = np.array([(16420459, 126, 356, 1), (16420485, 278, 142, 1),
                          (16420500, 602, 88, 1), (16420508, 548, 120, 0),
                          (16420513, 518, 195, 0), (16420536, 95, 149, 0),
                          (16420544, 147, 108, 0), (16420546, 477, 18, 1),
                          (16420561, 92, 288, 0), (16420573, 551, 39, 0)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])


def pytestcase_rawreader_exotic_mix_seek_time_load_n_and_delta_t(tmpdir, dataset_dir):
    """Tests loading events after a call to seek_time() using a variety delta_t values"""
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)
    assert video.current_time == 0

    video.seek_time(16420000)
    assert video.done == False
    assert video.current_time == 16420000
    assert video.current_event_index() == 667582

    events = video.load_n_events(3)
    assert video.current_time == 16420046  # timestamp of next available event
    assert video.current_event_index() == 667585
    reference = np.array([(16420019, 311, 334, 0), (16420020, 322, 215, 0), (16420032, 328, 383, 1)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])
    assert not video.done

    events = video.load_delta_t(10)
    assert video.current_time == 16420056
    assert video.current_event_index() == 667586
    reference = np.array([(16420046, 56, 10, 0)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])
    assert not video.done

    events = video.load_delta_t(10)
    assert video.current_time == 16420066
    assert video.current_event_index() == 667586
    assert len(events) == 0
    assert not video.done

    events = video.load_n_events(19)
    assert video.current_event_index() == 667605
    assert video.current_time == 16420250  # timestamp of next available event
    reference = np.array([(16420066, 433, 308, 1), (16420072, 23, 374, 0),
                          (16420122, 196, 356, 0), (16420128, 469, 413, 0),
                          (16420152, 246, 77, 0), (16420160, 522, 69, 1),
                          (16420163, 148, 250, 1), (16420168, 613, 78, 1),
                          (16420169, 18, 163, 1), (16420170, 485, 117, 1),
                          (16420174, 520, 160, 1), (16420185, 627, 376, 1),
                          (16420195, 267, 429, 0), (16420206, 444, 267, 0),
                          (16420211, 408, 1, 0), (16420238, 294, 426, 0),
                          (16420241, 536, 217, 0), (16420242, 481, 353, 0),
                          (16420248, 537, 252, 0)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])

    events = video.load_delta_t(828)
    assert video.current_event_index() == 667685
    assert video.current_time == 16421078
    assert not video.done
    assert len(events) == 80

    events = events[:50:5]  # we select some events

    reference = np.array([(16420250, 107, 221, 0), (16420311, 268, 390, 0),
                          (16420362, 101, 457, 0), (16420395, 184, 21, 0),
                          (16420459, 126, 356, 1), (16420536, 95, 149, 0),
                          (16420585, 464, 385, 0), (16420622, 419, 212, 1),
                          (16420656, 230, 65, 1), (16420711, 510, 12, 0)],
                         dtype={'itemsize': 16, 'offsets': [8, 0, 2, 4], 'names': ['t', 'x', 'y', 'p'],
                                'formats': ['i8', 'u2', 'u2', 'i2']})
    assert all([np.allclose(events[name], reference[name]) for name in events.dtype.names])


def pytestcase_rawreader_equivalence(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from a RAW file and its RAW to DAT equivalent"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)
    # WHEN
    dat_evs = load_events(filename.replace(".raw", "_td.dat"))
    raw_evs = video.load_n_events(1e8)
    # THEN
    assert len(dat_evs) == 667855
    assert len(raw_evs) == 667855
    assert all([np.allclose(dat_evs[name], raw_evs[name]) for name in ("x", "y", "p")])
    assert np.allclose(np.diff(dat_evs['t']), np.diff(raw_evs['t']))


def pytestcase_rawreader_pyreader_equivalence_mixed(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from a RAW file and its RAW to DAT equivalent in mixed mode"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=True)
    # WHEN
    dat_video = EventDatReader(filename.replace(".raw", "_td.dat"))

    while not video.is_done():
        # THEN
        raw_evs = video.load_mixed(10000, 2000)
        dat_evs = dat_video.load_mixed(10000, 2000)

        assert len(dat_evs) == len(raw_evs)

        assert all([np.allclose(dat_evs[name], raw_evs[name]) for name in ("x", "y", "p")])
        assert np.allclose(np.diff(dat_evs['t']), np.diff(raw_evs['t']))


def pytestcase_rawreader_ext_triggerevent(tmpdir, dataset_dir):
    """Tries to read external trigger events in a RAW file"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=False)
    trigger_evt_gt = np.loadtxt(os.path.join(dataset_dir,
                                             "openeb", "core", "event_io", "triggerevt.csv"), delimiter=",")
    # WHEN
    video.load_delta_t(int(1e8))  # read everything in this small file
    ext_trigger_evt = video.get_ext_trigger_events()
    # THEN
    assert np.allclose(ext_trigger_evt['p'], trigger_evt_gt[:, 0])
    assert np.allclose(ext_trigger_evt['id'], trigger_evt_gt[:, 1])
    # np.diff due to time_shifting being False
    assert np.allclose(np.diff(ext_trigger_evt['t']), np.diff(trigger_evt_gt[:, 2]))

    # check now the clearing of events
    video.clear_ext_trigger_events()
    assert not len(video.get_ext_trigger_events())


def pytestcase_rawreader_time_shifting(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from a RAW file and its RAW to DAT equivalent"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    video = RawReader(filename, do_time_shifting=True)
    # WHEN
    dat_evs = load_events(filename.replace(".raw", "_td.dat"))
    raw_evs = video.load_n_events(1E8)
    # THEN
    assert len(dat_evs) == 667855
    assert len(raw_evs) == 667855
    assert all([np.allclose(dat_evs[name], raw_evs[name]) for name in ("x", "y", "p")])
    assert np.allclose(dat_evs['t'], raw_evs['t'])


def pytestcase_rawreader_seek_compare(tmpdir, dataset_dir):
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    assert os.path.isfile(filename)

    for delta_t in [35000, 50000, 65000]:
        iterator = EventsIterator(filename, mode="delta_t", delta_t=delta_t)
        reader = RawReader(filename)
        offset_nb_events = 0
        offset_time = 0

        for i, events_gt in enumerate(iterator):
            n_events = len(events_gt)

            # seek event, load_n_events
            reader.reset()
            reader.seek_event(offset_nb_events)
            events = reader.load_n_events(n_events)
            assert events.size == n_events
            assert (events == events_gt).all()

            # seek time, load_n_events
            reader.reset()
            reader.seek_time(offset_time)
            events = reader.load_n_events(n_events)
            assert events.size == n_events
            assert (events == events_gt).all()

            # update offsets
            offset_nb_events += n_events
            offset_time += delta_t

            if i >= 50:
                break
