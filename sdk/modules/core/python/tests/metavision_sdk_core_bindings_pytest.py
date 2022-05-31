# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import numpy as np
import metavision_sdk_base
import metavision_sdk_core

# pylint: disable=no-member


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

    ts_prod_single_channel = metavision_sdk_core.TimeSurfaceProducerAlgorithmMergePolarities(5, 5)

    def callback_single_channel(ts, time_surface):
        nonlocal last_processed_timestamp
        nonlocal time_surface_single_channel
        last_processed_timestamp = ts
        time_surface_single_channel.numpy()[...] = time_surface.numpy()[...]
    ts_prod_single_channel.set_output_callback(callback_single_channel)

    ts_prod_single_channel.process_events(events)
    assert last_processed_timestamp == 6
    assert (time_surface_single_channel.numpy()[1:, :] == 0).all()
    assert time_surface_single_channel.numpy()[0, 0] == 0
    assert time_surface_single_channel.numpy()[0, 1] == 5
    assert time_surface_single_channel.numpy()[0, 2] == 4
    assert time_surface_single_channel.numpy()[0, 3] == 3
    assert time_surface_single_channel.numpy()[0, 4] == 0

    # Two channels (split polarities)
    last_processed_timestamp = 0
    time_surface_double_channel = metavision_sdk_core.MostRecentTimestampBuffer(5, 5, 2)

    ts_prod_double_channel = metavision_sdk_core.TimeSurfaceProducerAlgorithmSplitPolarities(5, 5)

    def callback_double_channel(ts, time_surface):
        nonlocal last_processed_timestamp
        nonlocal time_surface_double_channel
        last_processed_timestamp = ts
        time_surface_double_channel.numpy()[...] = time_surface.numpy()[...]
    ts_prod_double_channel.set_output_callback(callback_double_channel)

    ts_prod_double_channel.process_events(events)
    assert last_processed_timestamp == 6
    assert (time_surface_double_channel.numpy()[1:, :] == 0).all()
    assert (time_surface_double_channel.numpy()[0, 0] == 0).all()
    assert time_surface_double_channel.numpy()[0, 1].tolist() == [1, 5]
    assert time_surface_double_channel.numpy()[0, 2].tolist() == [0, 4]
    assert time_surface_double_channel.numpy()[0, 3].tolist() == [3, 0]
    assert (time_surface_double_channel.numpy()[0, 4] == 0).all()
