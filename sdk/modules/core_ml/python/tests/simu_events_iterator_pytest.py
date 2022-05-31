# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Unit tests for the Event Simulator Interface
"""
import os
import pytest
import skvideo.io
import numpy as np
from metavision_core_ml.video_to_event.simu_events_iterator import SimulatedEventsIterator
from metavision_core_ml.data.video_stream import TimedVideoStream


def pytestcase_all_events_in_frame(tmpdir):
    """checks that all events returned are in the frame (and resize works)."""
    tmp_res_filename = str(tmpdir.join("video_0.mp4"))
    video_frames = np.random.random(size=(100, 12, 12, 3)) * 255
    video_frames = video_frames.astype(np.uint8)

    skvideo.io.vwrite(tmp_res_filename, video_frames)
    simulation = SimulatedEventsIterator(tmp_res_filename, height=4, width=4)

    for events in simulation:
        assert (0 <= events["x"]).all()
        assert (events["x"] <= 3).all()
        assert (0 <= events["y"]).all()
        assert (events["y"] <= 3).all()
    simulation.__del__()


def pytestcase_no_events_before_start_dt(tmpdir):
    """checks that no events are before start_ts for mode delta_t"""
    tmp_res_filename = str(tmpdir.join("video_0.mp4"))
    start_ts = 3000000
    video_frames = np.random.random(size=(100, 12, 12, 3)) * 255
    video_frames = video_frames.astype(np.uint8)

    skvideo.io.vwrite(tmp_res_filename, video_frames)
    simulation = SimulatedEventsIterator(tmp_res_filename, start_ts=start_ts)
    for events in simulation:
        assert (events["t"] >= start_ts).all()
    simulation.__del__()


def pytestcase_no_events_before_start_ne(tmpdir):
    """checks that no events are before start_ts for mode n_events"""
    tmp_res_filename = str(tmpdir.join("video_0.mp4"))
    start_ts = 3000000
    video_frames = np.random.random(size=(100, 12, 12, 3)) * 255
    video_frames = video_frames.astype(np.uint8)

    skvideo.io.vwrite(tmp_res_filename, video_frames)
    simulation = SimulatedEventsIterator(tmp_res_filename, start_ts=start_ts, mode='n_events')
    for events in simulation:
        assert (events["t"] >= start_ts).all()
    simulation.__del__()


def pytestcase_correct_n_events(tmpdir):
    """checks that the buffers returned have n_events"""
    tmp_res_filename = str(tmpdir.join("video_0.mp4"))
    n_events = 10000
    video_frames = np.random.random(size=(100, 12, 12, 3)) * 255
    video_frames = video_frames.astype(np.uint8)

    skvideo.io.vwrite(tmp_res_filename, video_frames)
    simulation = SimulatedEventsIterator(tmp_res_filename, mode='n_events', n_events=n_events)

    for events in simulation:
        assert len(events) == n_events
    simulation.__del__()


def pytestcase_buffers_dt_intra(tmpdir):
    """checks that the range of the buffers is smaller than delta_t"""
    tmp_res_filename = str(tmpdir.join("video_0.mp4"))
    start_ts = 3000000
    delta_t = 15000
    video_frames = np.random.random(size=(100, 12, 12, 3)) * 255
    video_frames = video_frames.astype(np.uint8)

    skvideo.io.vwrite(tmp_res_filename, video_frames)
    simulation = SimulatedEventsIterator(tmp_res_filename, start_ts=start_ts, delta_t=delta_t)

    for events in simulation:
        if len(events):
            assert events['t'][-1]-events['t'][0] <= delta_t
    simulation.__del__()


def pytestcase_buffers_dt_extra(tmpdir):
    """checks that the first event of each buffer is in the correct buffer"""
    tmp_res_filename = str(tmpdir.join("video_0.mp4"))
    start_ts = 3000000
    delta_t = 15000
    video_frames = np.random.random(size=(100, 12, 12, 3)) * 255
    video_frames = video_frames.astype(np.uint8)

    skvideo.io.vwrite(tmp_res_filename, video_frames)
    simulation = SimulatedEventsIterator(tmp_res_filename, start_ts=start_ts, delta_t=delta_t)
    begin_buffer_ts = start_ts
    for events in simulation:
        if len(events):
            assert events['t'][0] >= begin_buffer_ts
        begin_buffer_ts += delta_t
    simulation.__del__()


def pytestcase_buffers_dt_extra_smaller(tmpdir):
    """checks that the period between two consecutive buffers is smaller than 2 delta_t"""
    tmp_res_filename = str(tmpdir.join("video_0.mp4"))
    start_ts = 3000000
    delta_t = 15000
    video_frames = np.random.random(size=(100, 12, 12, 3)) * 255
    video_frames = video_frames.astype(np.uint8)

    skvideo.io.vwrite(tmp_res_filename, video_frames)
    simulation = SimulatedEventsIterator(tmp_res_filename, start_ts=start_ts, delta_t=delta_t)
    last_buffer = start_ts
    simulation = iter(simulation)
    first_buffer = next(simulation)
    for events in simulation:
        if len(events):
            assert events['t'][0]-last_buffer <= 2*delta_t
        last_buffer += delta_t
    simulation.__del__()


def pytestcase_relative_timestamp(tmpdir):
    """checks that the relative timestamps are between 0 and delta_t"""
    tmp_res_filename = str(tmpdir.join("video_0.mp4"))
    video_frames = np.random.random(size=(100, 12, 12, 3)) * 255
    video_frames = video_frames.astype(np.uint8)
    start_ts = 3000000
    delta_t = 15000
    skvideo.io.vwrite(tmp_res_filename, video_frames)

    simulation = SimulatedEventsIterator(tmp_res_filename, start_ts=start_ts,
                                         delta_t=delta_t, relative_timestamps=True)
    for events in simulation:
        if len(events):
            assert events['t'][0] >= 0
            assert events['t'][0] <= delta_t
            assert events['t'][-1] >= 0
            assert events['t'][-1] <= delta_t
    simulation.__del__()


def pytestcase_timed_videostream_fps(tmpdir):
    """checks that the override_fps argument is correctly applied"""
    tmp_res_filename = str(tmpdir.join("video_0.mp4"))
    video_frames = np.random.random(size=(241, 12, 12, 3)) * 255
    video_frames = video_frames.astype(np.uint8)

    skvideo.io.vwrite(tmp_res_filename, video_frames)

    video_stream_iterator = iter(TimedVideoStream(tmp_res_filename, override_fps=240))
    for img, ts in video_stream_iterator:
        timestamp = ts
    assert int(timestamp) == int(1e6)  # We ran 241 frames timestamp should be 1sec (timestamp of first frame is 0)


def pytestcase_timed_videostream_max_frames(tmpdir):
    """checks that TimedVideoStream does not generate more than max frames"""
    tmp_res_filename = str(tmpdir.join("video_0.mp4"))
    video_frames = np.random.random(size=(240, 12, 12, 3)) * 255
    video_frames = video_frames.astype(np.uint8)
    max_frames = 34
    skvideo.io.vwrite(tmp_res_filename, video_frames)

    video_stream_iterator = TimedVideoStream(tmp_res_filename, max_frames=max_frames)
    assert len(video_stream_iterator) == max_frames


def pytestcase_can_be_run_twice(tmpdir):
    """Checks if the iter function works, meaning if the iterator can be run twice"""
    tmp_res_filename = str(tmpdir.join("video_0.mp4"))
    video_frames = np.random.random(size=(100, 12, 12, 3)) * 255
    video_frames = video_frames.astype(np.uint8)
    start_ts = 3000000
    delta_t = 15000
    skvideo.io.vwrite(tmp_res_filename, video_frames)

    simulation = SimulatedEventsIterator(tmp_res_filename, start_ts=start_ts,
                                         delta_t=delta_t, relative_timestamps=True)
    for events in simulation:
        last_event = events
    for events in simulation:
        last_event2 = events
    assert (last_event == last_event2).all()
    simulation.__del__()
