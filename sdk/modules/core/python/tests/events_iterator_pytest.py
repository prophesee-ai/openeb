# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for EventsIterator class
"""
import os
import math
import numpy as np

from metavision_core.event_io import EventsIterator
from metavision_core.event_io import load_events

from metavision_core.event_io.raw_reader import initiate_device


def pytestcase_iterator_init(tmpdir, dataset_dir):
    """Tests initialization of all member variables after creation of RawReader object from a file"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=10000,
                                 max_duration=1e7, relative_timestamps=True)
    # WHEN
    height, width = mv_iterator.get_size()
    # THEN
    assert width == 640
    assert height == 480


def pytestcase_iterator_float_init(tmpdir, dataset_dir):
    """Tests initialization with float values !"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    # WHEN
    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=1e4,
                                 max_duration=2E4, relative_timestamps=True)

    # THEN
    for _ in mv_iterator:
        pass
    # WHEN
    mv_iterator = EventsIterator(filename, start_ts=0, mode="n_events", n_events=10.,
                                 max_duration=2E4, relative_timestamps=True)
    # THEN
    for _ in mv_iterator:
        pass


def pytestcase_rawiterator_max_duration(tmpdir, dataset_dir):
    """Tests max_duration parameter effect"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    delta_t = 100000
    for max_duration in (10000, 100000, 400000):
        mv_iterator = EventsIterator(filename, start_ts=0, delta_t=delta_t,
                                     max_duration=max_duration, relative_timestamps=True)
        # WHEN
        counter = 0
        for evs in mv_iterator:
            # relative timestamps is True so all timestamp should be less than delta t
            if evs.size:
                assert evs['t'][-1] < delta_t  # relative
            counter += 1
        # THEN
        assert counter == int(math.ceil(float(max_duration) / delta_t))


def pytestcase_rawiterator_max_duration_not_round(tmpdir, dataset_dir):
    """Tests max_duration parameter effect with a not round value"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    delta_t = 100000
    max_duration = 120000
    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=delta_t,
                                 max_duration=max_duration, relative_timestamps=False)
    # WHEN
    counter = 0
    for evs in mv_iterator:
        counter += 1
    # THEN
    assert counter == int(math.ceil(float(max_duration) / delta_t))
    assert evs[-1]['t'] < max_duration


def pytestcase_rawiterator_start_ts(tmpdir, dataset_dir):
    """Tests start ts parameter effect"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    max_duration = 100000
    start_ts = 230000
    delta_t = 10000
    mv_iterator = EventsIterator(filename, start_ts=start_ts, delta_t=delta_t,
                                 max_duration=max_duration, relative_timestamps=True)
    # WHEN
    counter = 0
    for evs in mv_iterator:
        assert mv_iterator.get_current_time() >= start_ts
        # relative timestamps is True so all timestamp should be less than delta t
        if evs.size:
            assert evs['t'][-1] < delta_t  # relative
        counter += 1
    # THEN
    assert counter == int(math.ceil(max_duration / delta_t))


def pytestcase_rawiterator_absolute_timestamps(tmpdir, dataset_dir):
    """Tests that the timestamp are increasing from slice to slice"""
    # GIVEN
    timeslice = 100000
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=timeslice,
                                 max_duration=1e6, relative_timestamps=False)
    # WHEN & THEN
    current_time = 0
    for evs in mv_iterator:
        # with absolute timestamp event timestamps are increasing.
        if evs.size:
            assert evs['t'][0] >= current_time  # abs
        current_time += timeslice
        if evs.size:
            assert evs['t'][-1] < current_time  # absolute


def pytestcase_iterator_equivalence(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from a RAW file and its RAW to DAT equivalent"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=2000000,
                                 max_duration=1e6, relative_timestamps=False)

    # WHEN
    evs = np.concatenate([ev for ev in mv_iterator])

    dat_evs = load_events(filename.replace(".raw", "_td.dat"))
    dat_evs = dat_evs[dat_evs['t'] < 1e6]
    # THEN
    assert len(dat_evs) == len(evs)
    assert all([np.allclose(dat_evs[name], evs[name]) for name in ("t", "x", "y", "p")])


def pytestcase_iterator_equivalence_mixed_mode(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from a RAW file and its RAW to DAT equivalent when runs
    from two iterators in mixed mode."""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    filename_dat = filename.replace(".raw", "_td.dat")
    mv_it = EventsIterator(filename, start_ts=0, n_events=50000, delta_t=200000, mode="mixed",
                           max_duration=2e6, relative_timestamps=False)
    mv_it_dat = EventsIterator(filename_dat, start_ts=0, n_events=50000, delta_t=200000, mode="mixed",
                               max_duration=2e6, relative_timestamps=False)

    # WHEN
    for ev, ev_dat in zip(mv_it, mv_it_dat):
        # THEN
        assert len(ev_dat) == len(ev)
        assert all([np.allclose(ev_dat[name], ev[name]) for name in ("t", "x", "y", "p")])


def pytestcase_iterator_from_device(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from a RAW file and its RAW to DAT equivalent when intiliased
    from a device"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    device = initiate_device(filename, do_time_shifting=True)
    mv_iterator = EventsIterator.from_device(device, start_ts=0, delta_t=2000000, max_duration=1e6,
                                             relative_timestamps=False)

    # WHEN
    evs = np.concatenate([ev for ev in mv_iterator])

    dat_evs = load_events(filename.replace(".raw", "_td.dat"))
    dat_evs = dat_evs[dat_evs['t'] < 1e6]
    # THEN
    assert len(dat_evs) == len(evs)
    assert all([np.allclose(dat_evs[name], evs[name]) for name in ("t", "x", "y", "p")])


def pytestcase_iterator_equivalence_large_delta_t(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from a RAW file and its RAW to DAT equivalent
    using a large delta_t"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=20000000,
                                 max_duration=1e6, relative_timestamps=False)

    # WHEN
    evs = np.concatenate([ev for ev in mv_iterator])

    dat_evs = load_events(filename.replace(".raw", "_td.dat"))
    dat_evs = dat_evs[dat_evs['t'] < 1e6]
    # THEN
    assert len(dat_evs) == len(evs)
    assert all([np.allclose(dat_evs[name], evs[name]) for name in ("t", "x", "y", "p")])


def pytestcase_iterator_run_twice(tmpdir, dataset_dir):
    """Ensures the equivalence between two consecutive runs of the iterator"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    mv_iterator = EventsIterator(filename, start_ts=1e5, delta_t=20000,
                                 max_duration=1e6)

    # WHEN
    evs = [ev for ev in mv_iterator]
    evs2 = [ev for ev in mv_iterator]

    # THEN
    assert len(evs) == len(evs2)
    for ev, ev2 in zip(evs, evs2):
        assert all([np.allclose(ev2[name], ev[name]) for name in ("t", "x", "y", "p")])


def pytestcase_dat_iterator_equivalence(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from a DAT file with the Iterator and the function are equivalent
    """
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=20000000,
                                 max_duration=1e6, relative_timestamps=False)

    # WHEN
    evs = np.concatenate([ev for ev in mv_iterator])

    dat_evs = load_events(filename)
    dat_evs = dat_evs[dat_evs['t'] < 1e6]
    # THEN
    assert len(dat_evs) == len(evs)
    assert all([np.allclose(dat_evs[name], evs[name]) for name in ('t', "x", "y", "p")])


def pytestcase_iterator_large_shared_pointer_nb(tmpdir, dataset_dir):
    """Ensures that using a large number of sharedpointers is possible """
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=500, relative_timestamps=False)

    # WHEN & THEN
    evs = [ev for ev in mv_iterator]
    # given the small delta_t a large number of buffers are passed from cpp the code should not block


def pytestcase_iterator_n_event_equivalence(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from a RAW file and its RAW to DAT equivalent
    using n_event argument"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    mv_iterator = EventsIterator(filename, start_ts=0, mode="n_events", n_events=0,
                                 max_duration=1e6, relative_timestamps=False)

    # WHEN
    evs = np.concatenate([ev for ev in mv_iterator])

    dat_evs = load_events(filename.replace(".raw", "_td.dat"))
    dat_evs = dat_evs[dat_evs['t'] < 1e6]
    # THEN
    assert len(dat_evs) == len(evs)
    assert all([np.allclose(dat_evs[name], evs[name]) for name in ("x", "y", "p")])


def pytestcase_iterator_n_event_correctness(tmpdir, dataset_dir):
    """Ensures that events indeed come into slice of n events"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    mv_iterator = EventsIterator(filename, start_ts=0, mode="n_events", n_events=10000,
                                 max_duration=1e6, relative_timestamps=False)

    # WHEN
    for i, ev in enumerate(mv_iterator):
        # THEN
        assert ev.size == 10000 or i == 8


def pytestcase_rawiterator_dont_do_time_shifting(tmpdir, dataset_dir):
    """Tests that you can pass an argument to the RawReader ctor"""
    # GIVEN
    timeslice = 100000
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording.raw")
    # first event's timestamp of this sequence (without time_shifting is: 8720024)

    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=timeslice,
                                 max_duration=1e6, relative_timestamps=False, do_time_shifting=False)

    # WHEN
    nb_non_empty_frames = 0
    for evs in mv_iterator:
        if evs.size > 0:
            nb_non_empty_frames += 1
    # THEN
    assert nb_non_empty_frames == 0  # first event arrives too late (after 1e6 max duration)

    # here there should be 3 valid frames:
    # 8'700'000 -> 8'800'000
    # 8'800'000 -> 8'900'000
    # 8'900'000 -> 9'000'000
    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=timeslice,
                                 max_duration=9e6, relative_timestamps=False, do_time_shifting=False)
    # WHEN
    list_non_empty_frames = []
    for evs in mv_iterator:
        if evs.size > 0:
            list_non_empty_frames.append(evs)
    # THEN
    assert len(list_non_empty_frames) == 3
    assert list_non_empty_frames[0]['t'][0] >= 8700000 and list_non_empty_frames[0]['t'][-1] < 8800000
    assert list_non_empty_frames[1]['t'][0] >= 8800000 and list_non_empty_frames[1]['t'][-1] < 8900000
    assert list_non_empty_frames[2]['t'][0] >= 8900000 and list_non_empty_frames[2]['t'][-1] < 9000000
