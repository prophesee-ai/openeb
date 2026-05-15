# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import os
import pytest

import numpy as np
import metavision_sdk_base
import metavision_sdk_core
from metavision_core.event_io import EventsIterator, EventDatReader

# pylint: disable=no-member


def get_cd_events_filename(dataset_dir):
    filename_cd_events = os.path.join(
        dataset_dir,
        "openeb", "core", "event_io", "recording_td.dat")
    assert os.path.isfile(filename_cd_events)
    return filename_cd_events


def pytestcase_EventRescalerAlgorithm():
    events = np.zeros(2, dtype=metavision_sdk_base.EventCD)
    events["x"] = [0, 9]
    events["y"] = [0, 10]
    events["t"] = [0, 10]

    evt_rescaler = metavision_sdk_core.EventRescalerAlgorithm(0.5, 0.5)

    rescaled_evts_array = evt_rescaler.get_empty_output_buffer()
    evt_rescaler.process_events(events, rescaled_evts_array)

    assert(rescaled_evts_array.numpy()["x"].tolist() == [0, 4])
    assert(rescaled_evts_array.numpy()["y"].tolist() == [0, 5])
    assert(rescaled_evts_array.numpy()["t"].tolist() == [0, 10])


def pytestcase_EventPreprocessor(dataset_dir):
    dat_reader = EventDatReader(get_cd_events_filename(dataset_dir))
    height, width = dat_reader.get_size()

    network_input_width = int(width // 2)
    network_input_height = int(height // 2)
    evt_rescaler = metavision_sdk_core.EventRescalerAlgorithm(
        network_input_width/float(width), network_input_height/float(height))
    evt_preproc = metavision_sdk_core.EventPreprocessor.create_HistoProcessor(input_event_width=network_input_width,
                                                                              input_event_height=network_input_height,
                                                                              max_incr_per_pixel=5,
                                                                              clip_value_after_normalization=1.)
    assert evt_preproc.get_frame_shape() == [2, 240, 320]
    ev = dat_reader.load_delta_t(50000)
    rescaled_evts_array = evt_rescaler.get_empty_output_buffer()
    evt_rescaler.process_events(ev, rescaled_evts_array)
    frame_array = evt_preproc.init_output_tensor()
    evt_preproc.process_events(0, rescaled_evts_array, frame_array)
    assert frame_array.shape == (2, 240, 320)
    assert len(frame_array[frame_array != 0]) != 0
    assert np.max(frame_array) <= 1.
    ev = dat_reader.load_delta_t(50000)

    evt_rescaler.process_events(ev, rescaled_evts_array)
    frame_array.fill(0)
    evt_preproc.process_events(50000, rescaled_evts_array, frame_array)
    assert (frame_array.shape == (2, 240, 320))
    assert (len(frame_array[frame_array != 0]) != 0)
    assert (np.max(frame_array) <= 1.)

    delta_t = 50000
    mv_it = EventsIterator(get_cd_events_filename(dataset_dir), delta_t=delta_t)
    height, width = mv_it.get_size()

    network_input_width = int(width // 2)
    network_input_height = int(height // 2)
    evt_rescaler = metavision_sdk_core.EventRescalerAlgorithm(
        network_input_width/float(width), network_input_height/float(height))
    evt_preproc = metavision_sdk_core.EventPreprocessor.create_HistoProcessor(input_event_width=network_input_width,
                                                                              input_event_height=network_input_height,
                                                                              max_incr_per_pixel=5,
                                                                              clip_value_after_normalization=1.)

    rescaled_evts_array = evt_rescaler.get_empty_output_buffer()
    frame_array = evt_preproc.init_output_tensor()
    for idx, ev in enumerate(mv_it):
        cur_frame_start_ts = idx * delta_t
        if idx % 5 == 0:
            frame_array.fill(0)
        evt_rescaler.process_events(ev, rescaled_evts_array)
        evt_preproc.process_events(cur_frame_start_ts, rescaled_evts_array, frame_array)
        if idx >= 20:
            break

    # Check with wrong parameters
    frame_array_double = frame_array.astype(np.double)
    with pytest.raises(RuntimeError):
        mv_it = EventsIterator(get_cd_events_filename(dataset_dir), delta_t=delta_t)
        for idx, ev in enumerate(mv_it):
            cur_frame_start_ts = idx * delta_t
            if idx % 5 == 0:
                frame_array_double.fill(0)
            evt_rescaler.process_events(ev, rescaled_evts_array)
            evt_preproc.process_events(cur_frame_start_ts, rescaled_evts_array,
                                       frame_array_double)  # KO: frame_array should be double
            if idx >= 20:
                break

    frame_wrong_size = np.zeros((evt_preproc.get_frame_width(),
                                 evt_preproc.get_frame_height() + 1,
                                 evt_preproc.get_frame_channels()), dtype=np.float32)

    with pytest.raises(RuntimeError):
        mv_it = EventsIterator(get_cd_events_filename(dataset_dir), delta_t=delta_t)
        for idx, ev in enumerate(mv_it):
            cur_frame_start_ts = idx * delta_t
            if idx % 5 == 0:
                frame_wrong_size.fill(0)
            evt_rescaler.process_events(ev, rescaled_evts_array)
            evt_preproc.process_events(cur_frame_start_ts, rescaled_evts_array,
                                       frame_wrong_size)  # KO: height is not correct
            if idx >= 20:
                break

    assert evt_preproc.get_frame_width() % 2 == 0
    frame_wrong_shape = np.zeros((evt_preproc.get_frame_width() // 2,
                                  evt_preproc.get_frame_height() * 2,
                                  evt_preproc.get_frame_channels()), dtype=np.float32)
    assert frame_wrong_shape.size == evt_preproc.get_frame_size()

    with pytest.raises(RuntimeError):
        mv_it = EventsIterator(get_cd_events_filename(dataset_dir), delta_t=delta_t)
        for idx, ev in enumerate(mv_it):
            cur_frame_start_ts = idx * delta_t
            if idx % 5 == 0:
                frame_wrong_shape.fill(0)
            evt_rescaler.process_events(ev, rescaled_evts_array)
            evt_preproc.process_events(cur_frame_start_ts, rescaled_evts_array,
                                       frame_wrong_shape)  # KO: shape is not correct
            if idx >= 20:
                break

    if evt_preproc.is_CHW():
        frame_wrong_dim_order = np.zeros((evt_preproc.get_frame_height(),
                                          evt_preproc.get_frame_width(),
                                          evt_preproc.get_frame_channels()), dtype=np.float32)
    else:
        frame_wrong_dim_order = np.zeros((evt_preproc.get_frame_channels(),
                                          evt_preproc.get_frame_height(),
                                          evt_preproc.get_frame_width()), dtype=np.float32)

    with pytest.raises(RuntimeError):
        mv_it = EventsIterator(get_cd_events_filename(dataset_dir), delta_t=delta_t)
        for idx, ev in enumerate(mv_it):
            cur_frame_start_ts = idx * delta_t
            if idx % 5 == 0:
                frame_wrong_dim_order.fill(0)
                evt_rescaler.process_events(ev, rescaled_evts_array)
            evt_preproc.process_events(cur_frame_start_ts, rescaled_evts_array,
                                       frame_wrong_dim_order)  # KO: dimension order is not correct
            if idx >= 20:
                break


def pytestcase_RoiFilterAlgorithm():
    events = np.zeros(5, dtype=metavision_sdk_base.EventCD)
    events["x"] = range(10, 60, 10)
    events["y"] = range(110, 160, 10)
    events["t"] = range(1000, 6000, 1000)

    roi_filter_relative_coord = metavision_sdk_core.RoiFilterAlgorithm(x0=25, y0=125, x1=45, y1=145,
                                                                       output_relative_coordinates=True)
    assert roi_filter_relative_coord.x0 == 25
    assert roi_filter_relative_coord.y0 == 125
    assert roi_filter_relative_coord.x1 == 45
    assert roi_filter_relative_coord.y1 == 145
    assert roi_filter_relative_coord.is_resetting() == True
    filtered_events_buffer = roi_filter_relative_coord.get_empty_output_buffer()
    assert filtered_events_buffer.numpy().dtype == metavision_sdk_base.EventCD
    roi_filter_relative_coord.process_events(events, filtered_events_buffer)
    assert filtered_events_buffer.numpy()["x"].tolist() == [5, 15]
    assert filtered_events_buffer.numpy()["y"].tolist() == [5, 15]

    roi = metavision_sdk_core.RoiFilterAlgorithm(x0=25, y0=125, x1=45, y1=145)
    assert roi.x0 == 25
    assert roi.y0 == 125
    assert roi.x1 == 45
    assert roi.y1 == 145
    assert roi.is_resetting() == False
    filtered_events_buffer = roi.get_empty_output_buffer()
    assert filtered_events_buffer.numpy().dtype == metavision_sdk_base.EventCD
    roi.process_events(events, filtered_events_buffer)
    assert filtered_events_buffer.numpy()["x"].tolist() == [30, 40]
    assert filtered_events_buffer.numpy()["y"].tolist() == [130, 140]


def pytestcase_FlipXAlgorithm():
    events = np.zeros(3, dtype=metavision_sdk_base.EventCD)
    events["x"] = [10, 20, 30]

    flip_x = metavision_sdk_core.FlipXAlgorithm(639)
    assert flip_x.width_minus_one == 639

    flip_buffer = flip_x.get_empty_output_buffer()
    assert flip_buffer.numpy().dtype == metavision_sdk_base.EventCD

    flip_x.process_events(events, flip_buffer)
    assert metavision_sdk_base._buffer_info(events).ptr_hex() != flip_buffer._buffer_info().ptr_hex()

    assert flip_buffer.numpy()["x"].tolist() == [629, 619, 609]

    flip_x.process_events_(flip_buffer)
    assert flip_buffer.numpy()["x"].tolist() == [10, 20, 30]

    flip_x.process_events_(events)
    assert events["x"].tolist() == [629, 619, 609]

    ptr_hex_before = flip_buffer._buffer_info().ptr_hex()
    flip_x.process_events_(flip_buffer)
    assert flip_buffer._buffer_info().ptr_hex() == ptr_hex_before
    assert metavision_sdk_base._buffer_info(flip_buffer.numpy()).ptr_hex() == ptr_hex_before
    assert flip_buffer.numpy()["x"].tolist() == [629, 619, 609]


def pytestcase_FlipYAlgorithm():
    events = np.zeros(3, dtype=metavision_sdk_base.EventCD)
    events["y"] = [10, 20, 30]

    flip_y = metavision_sdk_core.FlipYAlgorithm(639)
    assert flip_y.height_minus_one == 639

    flip_buffer = flip_y.get_empty_output_buffer()
    assert flip_buffer.numpy().dtype == metavision_sdk_base.EventCD

    flip_y.process_events(events, flip_buffer)
    assert metavision_sdk_base._buffer_info(events).ptr_hex() != flip_buffer._buffer_info().ptr_hex()

    assert flip_buffer.numpy()["y"].tolist() == [629, 619, 609]

    flip_y.process_events_(flip_buffer)
    assert flip_buffer.numpy()["y"].tolist() == [10, 20, 30]

    flip_y.process_events_(events)
    assert events["y"].tolist() == [629, 619, 609]

    ptr_hex_before = flip_buffer._buffer_info().ptr_hex()
    flip_y.process_events_(flip_buffer)
    assert flip_buffer._buffer_info().ptr_hex() == ptr_hex_before
    assert metavision_sdk_base._buffer_info(flip_buffer.numpy()).ptr_hex() == ptr_hex_before
    assert flip_buffer.numpy()["y"].tolist() == [629, 619, 609]


def pytestcase_PeriodicFrameGenerationAlgorithm():
    events = np.zeros(5, dtype=metavision_sdk_base.EventCD)
    events["x"] = [1,   2,  3,  1,     4]
    events["p"] = [0,   1,  0,  1,     1]
    events["t"] = [10, 20, 30, 40, 10002]

    last_processed_timestamp = 0
    frame = np.zeros((5, 5, 3), np.uint8)

    frame_generator = metavision_sdk_core.PeriodicFrameGenerationAlgorithm(5, 5, 10000)

    def callback_frame_generator(ts, cv_frame):
        nonlocal last_processed_timestamp
        nonlocal frame
        last_processed_timestamp = ts
        frame[...] = cv_frame[...]
    frame_generator.set_output_callback(callback_frame_generator)

    # last event is not processed because its timestamp is above 10000
    frame_generator.process_events(events)
    assert last_processed_timestamp == 10000
    assert (frame[0, 1] == frame_generator.on_color_default()).all()
    assert (frame[0, 2] == frame_generator.on_color_default()).all()
    assert (frame[0, 3] == frame_generator.off_color_default()).all()
    assert (frame[0, 4] == frame_generator.bg_color_default()).all()
    assert (frame[1, :] == frame_generator.bg_color_default()).all()


def pytestcase_PeriodicFrameGenerationAlgorithmGray():
    events = np.zeros(5, dtype=metavision_sdk_base.EventCD)
    events["x"] = [1,   2,  3,  1,     4]
    events["p"] = [0,   1,  0,  1,     1]
    events["t"] = [10, 20, 30, 40, 10002]

    last_processed_timestamp = 0
    frame = np.zeros((5, 5), np.uint8)

    frame_generator = metavision_sdk_core.PeriodicFrameGenerationAlgorithm(5, 5, 10000)

    frame_generator.set_colors(background_color=[128], on_color=[255], off_color=[0],
                               colored=False)

    def callback_frame_generator(ts, cv_frame):
        nonlocal last_processed_timestamp
        nonlocal frame
        last_processed_timestamp = ts
        frame[...] = cv_frame[...]
    frame_generator.set_output_callback(callback_frame_generator)

    # last event is not processed because its timestamp is above 10000
    frame_generator.process_events(events)
    assert last_processed_timestamp == 10000
    assert (frame[0, 1] == 255)
    assert (frame[0, 2] == 255)
    assert (frame[0, 3] == 0)
    assert (frame[0, 4] == 128)
    assert (frame[1, :] == 128).all()


def pytestcase_OnDemandFrameGenerationAlgorithm():
    events = np.zeros(5, dtype=metavision_sdk_base.EventCD)
    events["x"] = [1,   2,  3,  1,     4]
    events["p"] = [0,   1,  0,  1,     1]
    events["t"] = [10, 20, 30, 40, 10002]

    frame = np.zeros((5, 5, 3), np.uint8)

    frame_generator = metavision_sdk_core.OnDemandFrameGenerationAlgorithm(5, 5)

    frame_generator.process_events(events)
    frame_generator.generate(10000, frame)

    assert (frame[0, 1] == frame_generator.on_color_default()).all()
    assert (frame[0, 2] == frame_generator.on_color_default()).all()
    assert (frame[0, 3] == frame_generator.off_color_default()).all()
    assert (frame[0, 4] == frame_generator.bg_color_default()).all()  # Bg color since it hasn't been processed
    assert (frame[1, :] == frame_generator.bg_color_default()).all()

    # set new colors and check the frame has changed accordingly
    frame_generator.set_colors(background_color=(255, 0, 0), on_color=(0, 255, 0), off_color=(0, 0, 255),
                               colored=True)
    frame_generator.generate(10000, frame)

    assert (frame[0, 1] == (0, 255, 0)).all()
    assert (frame[0, 2] == (0, 255, 0)).all()
    assert (frame[0, 3] == (0, 0, 255)).all()
    assert (frame[0, 4] == (255, 0, 0)).all()
    assert (frame[1, :] == (255, 0, 0)).all()


def pytestcase_BaseFrameGenerationAlgorithmStaticGeneration():
    events = np.zeros(5, dtype=metavision_sdk_base.EventCD)
    events["x"] = [1,   2,  3,  1,     4]
    events["p"] = [0,   1,  0,  1,     1]
    events["t"] = [10, 20, 30, 40, 10002]

    frame = np.zeros((5, 5, 3), np.uint8)
    metavision_sdk_core.BaseFrameGenerationAlgorithm.generate_frame(events, frame)

    assert (frame[0, 1] == metavision_sdk_core.BaseFrameGenerationAlgorithm.on_color_default()).all()
    assert (frame[0, 2] == metavision_sdk_core.BaseFrameGenerationAlgorithm.on_color_default()).all()
    assert (frame[0, 3] == metavision_sdk_core.BaseFrameGenerationAlgorithm.off_color_default()).all()
    assert (frame[0, 4] == metavision_sdk_core.BaseFrameGenerationAlgorithm.on_color_default()).all()
    assert (frame[1, :] == metavision_sdk_core.BaseFrameGenerationAlgorithm.bg_color_default()).all()


def pytestcase_PolarityFilterAlgorithm():
    events = np.zeros(5, dtype=metavision_sdk_base.EventCD)
    events["x"] = range(10, 15)
    events["y"] = range(110, 115)
    events["t"] = range(10000, 15000, 1000)
    events["p"] = [1, 0, 0, 1, 1]

    polarity_filter = metavision_sdk_core.PolarityFilterAlgorithm(1)
    events_buf = polarity_filter.get_empty_output_buffer()

    polarity_filter.process_events(events, events_buf)
    assert events_buf.numpy().size == 3
    assert events_buf.numpy()["x"].tolist() == [10, 13, 14]

    polarity_filter.polarity = 0
    polarity_filter.process_events(events, events_buf)
    assert events_buf.numpy().size == 2
    assert events_buf.numpy()["x"].tolist() == [11, 12]


def pytestcase_PolarityInverterAlgorithm():
    events = np.zeros(5, dtype=metavision_sdk_base.EventCD)
    events["x"] = range(10, 15)
    events["y"] = range(110, 115)
    events["t"] = range(10000, 15000, 1000)
    events["p"] = [1, 0, 0, 1, 1]

    polarity_inverter = metavision_sdk_core.PolarityInverterAlgorithm()
    events_buf = polarity_inverter.get_empty_output_buffer()

    polarity_inverter.process_events(events, events_buf)
    assert events_buf.numpy()["p"].tolist() == [0, 1, 1, 0, 0]

    polarity_inverter.process_events_(events_buf)
    assert events_buf.numpy()["p"].tolist() == [1, 0, 0, 1, 1]

    polarity_inverter.process_events_(events)
    assert events["p"].tolist() == [0, 1, 1, 0, 0]


def pytestcase_TimeSurfaceProducerAlgoritm():
    events = np.zeros(5, dtype=metavision_sdk_base.EventCD)
    events["x"] = [1, 2, 3, 2, 1]
    events["p"] = [0, 1, 0, 1, 1]
    events["t"] = [1, 2, 3, 4, 5]

    # Single channel (merge polarities)
    last_processed_timestamp = 0
    time_surface_single_channel = metavision_sdk_core.MostRecentTimestampBuffer(5, 5, 1)

    ts_prod_single_channel = metavision_sdk_core.EventPreprocessor.create_TimeSurfaceProcessor(5, 5, False)

    ts_prod_single_channel.process_events(last_processed_timestamp, events, time_surface_single_channel.numpy())
    assert (time_surface_single_channel.numpy()[1:, :] == 0).all()
    assert time_surface_single_channel.numpy()[0, 0] == 0
    assert time_surface_single_channel.numpy()[0, 1] == 5
    assert time_surface_single_channel.numpy()[0, 2] == 4
    assert time_surface_single_channel.numpy()[0, 3] == 3
    assert time_surface_single_channel.numpy()[0, 4] == 0

    # Two channels (split polarities)
    last_processed_timestamp = 0
    time_surface_double_channel = metavision_sdk_core.MostRecentTimestampBuffer(5, 5, 2)

    ts_prod_double_channel = metavision_sdk_core.EventPreprocessor.create_TimeSurfaceProcessor(5, 5, True)

    ts_prod_double_channel.process_events(last_processed_timestamp, events, time_surface_double_channel.numpy())
    assert (time_surface_double_channel.numpy()[1:, :] == 0).all()
    assert (time_surface_double_channel.numpy()[0, 0] == 0).all()
    assert time_surface_double_channel.numpy()[0, 1].tolist() == [1, 5]
    assert time_surface_double_channel.numpy()[0, 2].tolist() == [0, 4]
    assert time_surface_double_channel.numpy()[0, 3].tolist() == [3, 0]
    assert (time_surface_double_channel.numpy()[0, 4] == 0).all()


def pytestcase_RoiMaskAlgorithm(dataset_dir):
    im = np.zeros((480, 640), dtype=np.uint8)
    algo = metavision_sdk_core.RoiMaskAlgorithm(im)

    output = algo.get_empty_output_buffer()

    ev = np.zeros(3, metavision_sdk_base.EventCD)
    ev["x"] = (10, 11, 100)
    ev["y"] = (20, 21, 200)

    algo.process_events(ev, output)
    assert output.numpy().size == 0

    im[21, 11] = 1
    algo.set_pixel_mask(im)
    algo.process_events(ev, output)
    assert output.numpy().size == 1
    assert output.numpy().tolist() == [(11, 21, 0, 0)]

    algo.enable_rectangle(x0=99, y0=199, x1=101, y1=201)
    algo.process_events(ev, output)
    assert output.numpy().size == 2
    assert output.numpy().tolist() == [(11, 21, 0, 0), (100, 200, 0, 0)]


def pytestcase_RotateEventsAlgorithm(dataset_dir):
    import math

    W, H = 640, 480
    algo = metavision_sdk_core.RotateEventsAlgorithm(
        width_minus_one=W-1, height_minus_one=H-1, rotation=math.pi/2)

    output = algo.get_empty_output_buffer()

    ev = np.zeros(1, metavision_sdk_base.EventCD)
    ev["x"] = W/2 - 10
    ev["y"] = H/2

    algo.process_events(ev, output)
    output_np = output.numpy()
    assert output_np.size == 1
    assert output_np["x"] == W/2
    assert output_np["y"] == H/2 - 10


def pytestcase_TransposeEventsAlgorithm(dataset_dir):
    filename_raw_events = os.path.join(dataset_dir, "openeb", "gen31_timer.raw")
    mv_it = EventsIterator(filename_raw_events, start_ts=0, delta_t=1000,
                           relative_timestamps=False)
    transpose_algo = metavision_sdk_core.TransposeEventsAlgorithm()
    transposed_buffer = transpose_algo.get_empty_output_buffer()
    for ev in mv_it:
        transpose_algo.process_events(ev, transposed_buffer)
        ev_transposed = transposed_buffer.numpy()
        assert (len(ev_transposed) == len(ev))
        assert (ev_transposed["x"] == ev["y"]).all()
        assert (ev_transposed["y"] == ev["x"]).all()
        transpose_algo.process_events_(ev)
        assert (ev_transposed == ev).all()
        break


def pytestcase_ContrastMapGenerationAlgorithm():
    events = np.zeros(5, dtype=metavision_sdk_base.EventCD)
    events["x"] = [1, 2, 3, 2, 1]
    events["y"] = [0, 0, 0, 0, 0]
    events["p"] = [0, 1, 0, 1, 1]
    events["t"] = [1, 2, 3, 4, 5]

    contrast_map_generator = metavision_sdk_core.ContrastMapGenerationAlgorithm(5, 5, 1.2)
    contrast_map_generator.process_events(events)
    contrast_map_32f = np.zeros((5, 5), np.float32)
    contrast_map_generator.generate(contrast_map_32f)
    contrast_map_8u = np.zeros((5, 5), np.uint8)
    contrast_map_generator.generate(contrast_map_8u, 64, 128)


def pytestcase_EventsIntegrationAlgorithm():
    events = np.zeros(5, dtype=metavision_sdk_base.EventCD)
    events["x"] = [1, 2, 3, 2, 1]
    events["y"] = [0, 0, 0, 0, 0]
    events["p"] = [0, 1, 0, 1, 1]
    events["t"] = [1, 2, 3, 4, 5]

    ev_integrator = metavision_sdk_core.EventsIntegrationAlgorithm(5, 5, 10, 1.2)
    ev_integrator.process_events(events)
    integrated_map = np.zeros((5, 5), np.uint8)
    ev_integrator.generate(integrated_map)
