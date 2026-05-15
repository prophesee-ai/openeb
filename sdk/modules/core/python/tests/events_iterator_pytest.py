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
        assert mv_it.current_time == mv_it_dat.current_time
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

    # WHEN
    evs = [ev for ev in EventsIterator(filename, start_ts=1e5, delta_t=20000, max_duration=1e6)]
    evs2 = [ev for ev in EventsIterator(filename, start_ts=1e5, delta_t=20000, max_duration=1e6)]

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


def pytestcase_triggers_in_raw(dataset_dir):
    filename = os.path.join(dataset_dir, "openeb", "blinking_gen4_with_ext_triggers.raw")
    assert os.path.isfile(filename)
    mv_iterator = EventsIterator(filename, mode="delta_t", delta_t=100000)

    # check attempt to retrieve trigger events before the begining of the loop raises an exception
    exception_raised = False
    try:
        trigger_events = mv_iterator.get_ext_trigger_events()
    except AssertionError as e:
        exception_raised = True
        assert "retrieve trigger events before first iteration" in str(e)
    assert exception_raised

    for idx, events in enumerate(mv_iterator):
        if (idx + 1) % 5 == 0:
            # check we can access trigger events during execution of the loop
            trigger_events = mv_iterator.get_ext_trigger_events()
            assert len(trigger_events) > 0

    # check we can access trigger events after the loop is over
    all_trigger_events = mv_iterator.get_ext_trigger_events()
    assert len(all_trigger_events) == 82


def pytestcase_triggers_in_dat(dataset_dir):
    filename = os.path.join(dataset_dir,
                            "openeb", "core", "event_io", "recording_td.dat")
    assert os.path.isfile(filename)
    mv_iterator = EventsIterator(filename, mode="delta_t", delta_t=100000)
    for idx, events in enumerate(mv_iterator):
        # check we can iterate on this sequence without failure
        pass

    # check an exception is raised when trying to access trigger events on a _cd.dat file after the loop
    exception_raised = False
    try:
        trigger_events = mv_iterator.get_ext_trigger_events()
    except RuntimeError as e:
        exception_raised = True
        assert "does not handle ext_trigger events" in str(e)
    assert exception_raised

    exception_raised = False
    mv_iterator = EventsIterator(filename, mode="delta_t", delta_t=100000)
    for idx, events in enumerate(mv_iterator):
        # check an exception is raised if trying to access the trigger events on a _cd.dat file during the loop
        if (idx+1) % 10 != 0:
            try:
                trigger_events = mv_iterator.get_ext_trigger_events()
            except RuntimeError as e:
                exception_raised = True
                assert "does not handle ext_trigger events" in str(e)
                break
    assert exception_raised


def pytestcase_hdf5_getsize(tmpdir, dataset_dir):
    """Tests initialization of all member variables after creation of HDF5Reader object from a file"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "blinking_gen4_with_ext_triggers.hdf5")
    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=10000,
                                 max_duration=1e7, relative_timestamps=True)
    # WHEN
    height, width = mv_iterator.get_size()
    # THEN
    assert width == 1280
    assert height == 720


def pytestcase_hdf5_float_init(tmpdir, dataset_dir):
    """Tests initialization with float values !"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "blinking_gen4_with_ext_triggers.hdf5")
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


def pytestcase_hdf5_max_duration(tmpdir, dataset_dir):
    """Tests max_duration parameter effect"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "blinking_gen4_with_ext_triggers.hdf5")
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


def pytestcase_hdf5_max_duration_not_round(tmpdir, dataset_dir):
    """Tests max_duration parameter effect with a not round value"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "blinking_gen4_with_ext_triggers.hdf5")
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


def pytestcase_hdf5_start_ts(tmpdir, dataset_dir):
    """Tests start ts parameter effect"""
    # GIVEN
    filename = os.path.join(dataset_dir,
                            "openeb", "blinking_gen4_with_ext_triggers.hdf5")
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


def pytestcase_hdf5_absolute_timestamps(tmpdir, dataset_dir):
    """Tests that the timestamp are increasing from slice to slice"""
    # GIVEN
    timeslice = 100000
    filename = os.path.join(dataset_dir,
                            "openeb", "blinking_gen4_with_ext_triggers.hdf5")
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


def pytestcase_hdf5_iterator_run_twice(tmpdir, dataset_dir):
    """Ensures the equivalence between two consecutive runs of the iterator"""
    filename = os.path.join(dataset_dir,
                            "openeb", "blinking_gen4_with_ext_triggers.hdf5")

    # WHEN
    evs = [ev for ev in EventsIterator(filename, start_ts=1e5, delta_t=20000, max_duration=1e6)]
    evs2 = [ev for ev in EventsIterator(filename, start_ts=1e5, delta_t=20000, max_duration=1e6)]

    # THEN
    assert len(evs) == len(evs2)
    for ev, ev2 in zip(evs, evs2):
        assert all([np.allclose(ev2[name], ev[name]) for name in ("t", "x", "y", "p")])


def pytestcase_hdf5_raw_iterator_equivalence(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from RAW and HDF5 files"""
    raw_filename = os.path.join(dataset_dir,
                                "openeb", "blinking_gen4_with_ext_triggers.raw")
    raw_iterator = EventsIterator(raw_filename, start_ts=0, delta_t=2000000,
                                  max_duration=1e6, relative_timestamps=False)

    raw_evs = np.concatenate([ev for ev in raw_iterator])

    hdf5_filename = os.path.join(dataset_dir,
                                 "openeb", "blinking_gen4_with_ext_triggers.hdf5")
    hdf5_iterator = EventsIterator(hdf5_filename, start_ts=0, delta_t=2000000,
                                   max_duration=1e6, relative_timestamps=False)

    hdf5_evs = np.concatenate([ev for ev in hdf5_iterator])
    # THEN
    assert len(raw_evs) == len(hdf5_evs)
    assert all([np.allclose(raw_evs[name], hdf5_evs[name]) for name in ("t", "x", "y", "p")])


def pytestcase_hdf5_raw_iterator_equivalence_mixed_mode(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from RAW and HDF5 files when runs
    from two iterators in mixed mode."""
    raw_filename = os.path.join(dataset_dir,
                                "openeb", "blinking_gen4_with_ext_triggers.raw")
    hdf5_filename = os.path.join(dataset_dir,
                                 "openeb", "blinking_gen4_with_ext_triggers.hdf5")
    raw_iterator = EventsIterator(raw_filename, start_ts=0, n_events=50000, delta_t=200000, mode="mixed",
                                  max_duration=2e6, relative_timestamps=False)
    hdf5_iterator = EventsIterator(hdf5_filename, start_ts=0, n_events=50000, delta_t=200000, mode="mixed",
                                   max_duration=2e6, relative_timestamps=False)

    for raw_evs, hdf5_evs in zip(raw_iterator, hdf5_iterator):
        assert raw_iterator.current_time == hdf5_iterator.current_time
        assert len(raw_evs) == len(hdf5_evs)
        assert all([np.allclose(raw_evs[name], hdf5_evs[name]) for name in ("t", "x", "y", "p")])


def pytestcase_hdf5_raw_iterator_n_event_equivalence(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from RAW and HDF5 files
    using n_event argument"""
    raw_filename = os.path.join(dataset_dir,
                                "openeb", "blinking_gen4_with_ext_triggers.raw")
    hdf5_filename = os.path.join(dataset_dir,
                                 "openeb", "blinking_gen4_with_ext_triggers.hdf5")
    raw_iterator = EventsIterator(raw_filename, start_ts=0, mode="n_events", n_events=1000,
                                  max_duration=1e6, relative_timestamps=False)
    hdf5_iterator = EventsIterator(hdf5_filename, start_ts=0, mode="n_events", n_events=1000,
                                   max_duration=1e6, relative_timestamps=False)

    for raw_evs, hdf5_evs in zip(raw_iterator, hdf5_iterator):
        assert raw_iterator.current_time == hdf5_iterator.current_time
        assert len(raw_evs) == len(hdf5_evs)
        assert all([np.allclose(raw_evs[name], hdf5_evs[name]) for name in ("x", "y", "p")])


def pytestcase_hdf5_raw_iterator_delta_t_equivalence(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from RAW and HDF5 files
    using delta_t argument"""
    raw_filename = os.path.join(dataset_dir,
                                "openeb", "blinking_gen4_with_ext_triggers.raw")
    hdf5_filename = os.path.join(dataset_dir,
                                 "openeb", "blinking_gen4_with_ext_triggers.hdf5")
    raw_iterator = EventsIterator(raw_filename, start_ts=2e5, mode="delta_t", delta_t=1e5,
                                  relative_timestamps=False)
    hdf5_iterator = EventsIterator(hdf5_filename, start_ts=2e5, mode="delta_t", delta_t=1e5,
                                   relative_timestamps=False)

    for raw_evs, hdf5_evs in zip(raw_iterator, hdf5_iterator):
        assert raw_iterator.current_time == hdf5_iterator.current_time
        assert len(raw_evs) == len(hdf5_evs)
        assert all([np.allclose(raw_evs[name], hdf5_evs[name]) for name in ("x", "y", "p")])


def pytestcase_hdf5_iterator_n_event_correctness(tmpdir, dataset_dir):
    """Ensures that events indeed come into slice of n events"""
    filename = os.path.join(dataset_dir,
                            "openeb", "blinking_gen4_with_ext_triggers.hdf5")
    mv_iterator = EventsIterator(filename, start_ts=0, mode="n_events", n_events=100000,
                                 max_duration=1e6, relative_timestamps=False)

    # WHEN
    for i, ev in enumerate(mv_iterator):
        # THEN
        assert ev.size == 100000 or i == 4


def pytestcase_triggers_in_hdf5_load_delta_t(dataset_dir):
    filename = os.path.join(dataset_dir, "openeb", "blinking_gen4_with_ext_triggers.hdf5")
    assert os.path.isfile(filename)
    mv_iterator = EventsIterator(filename, mode="delta_t", delta_t=100000)

    # check attempt to retrieve trigger events before the begining of the loop raises an exception
    exception_raised = False
    try:
        trigger_events = mv_iterator.get_ext_trigger_events()
    except AssertionError as e:
        exception_raised = True
        assert "retrieve trigger events before first iteration" in str(e)
    assert exception_raised

    for idx, events in enumerate(mv_iterator):
        if (idx + 1) % 5 == 0:
            # check we can access trigger events during execution of the loop
            trigger_events = mv_iterator.get_ext_trigger_events()
            assert len(trigger_events) > 0

    # check we can access trigger events after the loop is over
    all_trigger_events = mv_iterator.get_ext_trigger_events()
    assert len(all_trigger_events) == 82


def pytestcase_triggers_in_hdf5_load_n_events(dataset_dir):
    filename = os.path.join(dataset_dir, "openeb", "blinking_gen4_with_ext_triggers.hdf5")
    assert os.path.isfile(filename)
    mv_iterator = EventsIterator(filename, mode="n_events", n_events=30000)

    # check attempt to retrieve trigger events before the begining of the loop raises an exception
    exception_raised = False
    try:
        trigger_events = mv_iterator.get_ext_trigger_events()
    except AssertionError as e:
        exception_raised = True
        assert "retrieve trigger events before first iteration" in str(e)
    assert exception_raised

    for idx, events in enumerate(mv_iterator):
        if (idx + 1) % 5 == 0:
            # check we can access trigger events during execution of the loop
            trigger_events = mv_iterator.get_ext_trigger_events()
            assert len(trigger_events) > 0

    # check we can access trigger events after the loop is over
    all_trigger_events = mv_iterator.get_ext_trigger_events()
    assert len(all_trigger_events) == 82


def pytestcase_hdf5_ts_offset(dataset_dir):
    """Test dataset which has ts_offset attribute."""
    filename = os.path.join(dataset_dir, "openeb", "ts_offset_test.hdf5")

    mv_iterator = EventsIterator(filename, start_ts=0, delta_t=1e4)
    for idx, events in enumerate(mv_iterator):
        if idx < 100:
            assert len(events) == 0
        else:
            assert len(events) > 0
    assert mv_iterator.current_time == mv_iterator.reader.events_CD["t"][-1]

    mv_iterator = EventsIterator(filename, start_ts=1E6, delta_t=1e4, relative_timestamps=True)
    for idx, events in enumerate(mv_iterator):
        assert len(events) > 0

    # There are 10 events in the file and we load 4 events each time
    mv_iterator = EventsIterator(filename, start_ts=0, mode="n_events", n_events=4)
    ts_offset = mv_iterator.reader.ts_offset
    for idx, events in enumerate(mv_iterator):
        if idx < 2:
            assert int(mv_iterator.current_time + ts_offset + 2000) == (idx + 1) * 2000 * 4
    assert mv_iterator.current_time == mv_iterator.reader.events_CD["t"][-1]

    mv_iterator = EventsIterator(filename, start_ts=0, n_events=4, delta_t=6000, mode="mixed")
    for idx, events in enumerate(mv_iterator):
        if idx < 166:
            assert len(events) == 0
        elif idx == 166:
            assert len(events) == 1
        elif idx == 167:
            assert len(events) == 4
        elif idx == 168:
            assert len(events) == 2
        elif idx == 169:
            assert len(events) == 3

    mv_iterator = EventsIterator(filename, start_ts=0, n_events=4, delta_t=5000, mode="mixed")
    for idx, events in enumerate(mv_iterator):
        if idx < 200:
            assert len(events) == 0
        elif idx == 200:
            assert len(events) == 3
        elif idx == 201:
            assert len(events) == 2
        elif idx == 202:
            assert len(events) == 3
        elif idx == 204:
            assert len(events) == 2


def pytestcase_hdf5_raw_dat_iterator_delta_t_equivalence_corner_cases(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from RAW and HDF5 files
    using delta_t argument. The test files contain three groups of events,
    there is a huge time gap between each group. And all the events in the 
    third group have the same timestamp.
    """
    raw_filename = os.path.join(dataset_dir, "openeb", "events_big_time_gap_repeated_ts.raw")
    dat_filename = os.path.join(dataset_dir, "openeb", "events_big_time_gap_repeated_ts.dat")
    raw_iterator = EventsIterator(raw_filename, start_ts=0, mode="delta_t", delta_t=12345)
    dat_iterator = EventsIterator(dat_filename, start_ts=0, mode="delta_t", delta_t=12345)

    for raw_evs, dat_evs in zip(raw_iterator, dat_iterator):
        assert raw_iterator.current_time == dat_iterator.current_time
        assert len(raw_evs) == len(dat_evs)
        assert all([np.allclose(raw_evs[name], dat_evs[name]) for name in ("x", "y", "p")])


def pytestcase_hdf5_raw_dat_iterator_n_events_equivalence_corner_cases(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from RAW and HDF5 files
    using n_events argument. The test files contain three groups of events,
    there is a huge time gap between each group. And all the events in the 
    third group have the same timestamp."""
    raw_filename = os.path.join(dataset_dir, "openeb", "events_big_time_gap_repeated_ts.raw")
    dat_filename = os.path.join(dataset_dir, "openeb", "events_big_time_gap_repeated_ts.dat")
    raw_iterator = EventsIterator(raw_filename, start_ts=0, mode="n_events", n_events=12345)
    dat_iterator = EventsIterator(dat_filename, start_ts=0, mode="n_events", n_events=12345)

    for raw_evs, dat_evs in zip(raw_iterator, dat_iterator):
        assert raw_iterator.current_time == dat_iterator.current_time
        assert len(raw_evs) == len(dat_evs)
        assert all([np.allclose(raw_evs[name], dat_evs[name]) for name in ("x", "y", "p")])


def pytestcase_hdf5_raw_dat_iterator_mixed_equivalence_corner_cases(tmpdir, dataset_dir):
    """Ensures the equivalence of events coming from RAW and HDF5 files
    using mixed mode. The test files contain three groups of events,
    there is a huge time gap between each group. And all the events in the 
    third group have the same timestamp."""
    raw_filename = os.path.join(dataset_dir, "openeb", "events_big_time_gap_repeated_ts.raw")
    dat_filename = os.path.join(dataset_dir, "openeb", "events_big_time_gap_repeated_ts.dat")
    raw_iterator = EventsIterator(raw_filename, start_ts=0, n_events=1234, delta_t=1234, mode="mixed")
    dat_iterator = EventsIterator(dat_filename, start_ts=0, n_events=1234, delta_t=1234, mode="mixed")

    for raw_evs, dat_evs in zip(raw_iterator, dat_iterator):
        assert raw_iterator.current_time == dat_iterator.current_time
        assert len(raw_evs) == len(dat_evs)
        assert all([np.allclose(raw_evs[name], dat_evs[name]) for name in ("x", "y", "p")])
