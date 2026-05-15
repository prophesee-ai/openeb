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

import metavision_sdk_base
import metavision_sdk_stream

# pylint: disable=no-member


def pytestcase_camera_from_raw_file(dataset_dir):
    camera = metavision_sdk_stream.Camera.from_file(os.path.join(dataset_dir,
                                                                 "openeb", "gen4_evt3_hand.raw"))
    assert (camera is not None)


def pytestcase_camera_from_raw_file_pathlib(dataset_dir):
    import pathlib
    camera = metavision_sdk_stream.Camera.from_file(pathlib.Path(dataset_dir,
                                                                 "openeb", "gen4_evt3_hand.raw"))
    assert(camera is not None)


def pytestcase_camera_from_hdf5_file(dataset_dir):
    camera = metavision_sdk_stream.Camera.from_file(os.path.join(dataset_dir,
                                                                 "openeb", "gen4_evt3_hand.hdf5"))
    assert (camera is not None)


def pytestcase_camera_access_facility(dataset_dir):
    camera = metavision_sdk_stream.Camera.from_file(os.path.join(dataset_dir,
                                                                 "openeb", "gen4_evt3_hand.raw"))
    assert (camera is not None)
    geom = camera.get_device().get_i_geometry()
    assert (geom is not None)
    assert (geom.get_width() == 1280 and geom.get_height() == 720)


def pytestcase_camera_check_geometry(dataset_dir):
    camera = metavision_sdk_stream.Camera.from_file(os.path.join(dataset_dir,
                                                                 "openeb", "gen4_evt3_hand.hdf5"))
    assert (camera is not None)
    assert (camera.width() == 1280 and camera.height() == 720)


def pytestcase_camera_check_geometry(dataset_dir):
    camera = metavision_sdk_stream.Camera.from_file(os.path.join(dataset_dir,
                                                                 "openeb", "gen4_evt3_hand.hdf5"))
    assert (camera is not None)
    assert (camera.width() == 1280 and camera.height() == 720)


def pytestcase_camera_from_file_config_hints(dataset_dir):
    fch = metavision_sdk_stream.FileConfigHints(
    ).real_time_playback(False).max_memory(1024 * 1024)

    assert (fch.max_memory() == 1024 * 1024)
    assert (not fch.real_time_playback())
    camera = metavision_sdk_stream.Camera.from_file(os.path.join(dataset_dir,
                                                                 "openeb", "gen4_evt3_hand.raw"),
                                                    fch)
    assert (camera is not None)


def pytestcase_hdf5_file_writer_test(tmpdir, dataset_dir):
    camera = metavision_sdk_stream.Camera.from_file(os.path.join(dataset_dir,
                                                                 "openeb", "gen4_evt3_hand.raw"))
    file_writer = metavision_sdk_stream.HDF5EventFileWriter()
    file_writer.open(os.path.join(tmpdir, "hdf5_written_file.hdf5"))
    file_writer.add_metadata_map_from_camera(camera)
    buf = metavision_sdk_base.EventCDBuffer(10)
    np_from_buf = buf.numpy()
    for i in range(0, 10):
        np_from_buf[i]["x"] = 1
        np_from_buf[i]["y"] = 1
        np_from_buf[i]["p"] = 1
        np_from_buf[i]["t"] = 1

    file_writer.add_cd_events(np_from_buf)

    buf = metavision_sdk_base.EventExtTriggerBuffer(1)
    np_from_buf = buf.numpy()
    np_from_buf[0]["p"] = 0
    file_writer.add_ext_trigger_events(np_from_buf)

    file_writer.flush()
    file_writer.close()


def pytestcase_raw_evt2_file_writer_test(tmpdir, dataset_dir):
    camera = metavision_sdk_stream.Camera.from_file(os.path.join(dataset_dir,
                                                                 "openeb", "gen4_evt3_hand.raw"))
    file_writer = metavision_sdk_stream.RAWEvt2EventFileWriter(
        camera.width(), camera.height(), "", True)
    file_writer.open(os.path.join(tmpdir, "rawevt2_written_file.raw"))
    buf = metavision_sdk_base.EventCDBuffer(10)
    np_from_buf = buf.numpy()
    for i in range(0, 10):
        np_from_buf[i]["x"] = 1
        np_from_buf[i]["y"] = 1
        np_from_buf[i]["p"] = 1
        np_from_buf[i]["t"] = i

    file_writer.add_cd_events(np_from_buf)

    buf = metavision_sdk_base.EventExtTriggerBuffer(1)
    np_from_buf = buf.numpy()
    np_from_buf[0]["p"] = 0
    np_from_buf[0]["id"] = 6
    np_from_buf[0]["t"] = 5
    file_writer.add_ext_trigger_events(np_from_buf)

    file_writer.flush()
    file_writer.close()


def pytestcase_hdf5_file_writer_pathlib_test(tmpdir, dataset_dir):
    import pathlib
    camera = metavision_sdk_stream.Camera.from_file(pathlib.Path(dataset_dir,
                                                                 "openeb", "gen4_evt3_hand.raw"))
    file_writer = metavision_sdk_stream.HDF5EventFileWriter(pathlib.Path(tmpdir,
                                                                         "hdf5_written_file.hdf5"))
    file_writer.add_metadata_map_from_camera(camera)
    buf = metavision_sdk_base.EventCDBuffer(10)
    np_from_buf = buf.numpy()
    for i in range(0, 10):
        np_from_buf[i]["x"] = 1
        np_from_buf[i]["y"] = 1
        np_from_buf[i]["p"] = 1
        np_from_buf[i]["t"] = 1

    file_writer.add_cd_events(np_from_buf)

    buf = metavision_sdk_base.EventExtTriggerBuffer(1)
    np_from_buf = buf.numpy()
    np_from_buf[0]["p"] = 0
    file_writer.add_ext_trigger_events(np_from_buf)

    file_writer.flush()
    file_writer.close()
