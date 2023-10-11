#!/usr/bin/env python

# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import pytest
import os
import re
import cv2
from metavision_utils import os_tools, pytest_tools


def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])


def generate_video(filename_full, accumulation_time, slow_motion_factor, expected_video_info):

    # Before launching the app, check the dataset file exists
    assert os.path.exists(filename_full)

    tmp_dir = os_tools.TemporaryDirectoryHandler()
    output_video = os.path.join(tmp_dir.temporary_directory(), "out.avi")

    cmd = "./metavision_file_to_video -i \"{}\" -o {} -a {} --fourcc MJPG".format(filename_full, output_video,
                                                                                  accumulation_time)
    if slow_motion_factor != 1:
        cmd += " -s {}".format(slow_motion_factor)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Before opening the video, verify it has been written
    assert os.path.exists(output_video)

    # Now check basic information on the video
    cap = cv2.VideoCapture(output_video)

    # Check size
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == expected_video_info['width']
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == expected_video_info['height']

    # Check frame rate
    assert int(cap.get(cv2.CAP_PROP_FPS)) == 30

    # Check frame count
    cap_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cap_frame_count > 0:
        assert cap_frame_count == expected_video_info['frame-count']

    # Check codec
    cap_4CC = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    if cap_4CC != "\0\0\0\0":
        assert cap_4CC == "MJPG"


def generate_image_sequence(filename_full, expected_imseq_info):

    # Before launching the app, check the dataset file exists
    assert os.path.exists(filename_full)

    tmp_dir = os_tools.TemporaryDirectoryHandler()
    output_video = os.path.join(tmp_dir.temporary_directory(), "out%04d.png")

    cmd = "./metavision_file_to_video -i \"{}\" -o {}".format(filename_full, output_video)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Count the number of valid images
    stride_checked_imgs = expected_imseq_info['frame-count']/10
    n_images = 0
    while(os.path.exists(os.path.join(tmp_dir.temporary_directory(), "out{:04d}.png".format(n_images)))):
        if n_images % stride_checked_imgs == 0:
            img = cv2.imread(os.path.join(tmp_dir.temporary_directory(), "out{:04d}.png".format(n_images)))
            assert int(img.shape[1]) == expected_imseq_info['width']
            assert int(img.shape[0]) == expected_imseq_info['height']
        n_images += 1

    # Check frame count
    if n_images > 0:
        assert n_images == expected_imseq_info['frame-count']


def pytestcase_test_metavision_file_to_video_show_help():
    """
    Checks output of metavision_file_to_video when displaying help message
    """

    cmd = "./metavision_file_to_video --help"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check that the options showed in the output
    assert "Options:" in output, "******\nMissing options display in output :{}\n******".format(output)


def pytestcase_test_metavision_file_to_video_non_existing_input_file():
    """
    Checks that metavision_file_to_video returns an error when passing an input file that doesn't exist
    """

    # Create a file path that we are sure does not exist
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_file = os.path.join(tmp_dir.temporary_directory(), "nonexistent.raw")

    cmd = "./metavision_file_to_video -i {}".format(input_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file could not be read
    assert "not an existing file" in output


def pytestcase_test_metavision_file_to_video_missing_input_args():
    """
    Checks that metavision_file_to_video returns an error when not passing required input args
    """

    cmd = "./metavision_file_to_video"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file arg is missing
    assert re.search("Parsing error: the option (.+) is required but missing", output)


def pytestcase_test_metavision_file_to_video_on_raw_gen31_recording_default_args(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 640, 'height': 480, 'frame-count': 392}
    generate_video(filename_full, 10000, 1, expected_video_info)


def pytestcase_test_metavision_file_to_video_on_raw_gen31_recording_default_args_imseq(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 640, 'height': 480, 'frame-count': 392}
    generate_image_sequence(filename_full, expected_video_info)


def pytestcase_test_metavision_file_to_video_on_raw_gen4_evt2_recording_default_args(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 1280, 'height': 720, 'frame-count': 314}
    generate_video(filename_full, 10000, 1, expected_video_info)


def pytestcase_test_metavision_file_to_video_on_raw_gen4_evt3_recording_default_args(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 1280, 'height': 720, 'frame-count': 451}
    generate_video(filename_full, 10000, 1, expected_video_info)


def pytestcase_test_metavision_file_to_video_on_raw_gen31_recording_slow_motion_2(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 640, 'height': 480, 'frame-count': 783}
    generate_video(filename_full, 10000, 2, expected_video_info)


def pytestcase_test_metavision_file_to_video_on_raw_gen4_evt2_recording_slow_motion_0_5_acc_time_200(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 1280, 'height': 720, 'frame-count': 157}
    generate_video(filename_full, 200, 0.5, expected_video_info)


def pytestcase_test_metavision_file_to_video_on_raw_gen4_evt3_recording_slow_motion_3_acc_time_20000(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 1280, 'height': 720, 'frame-count': 1351}
    generate_video(filename_full, 20000, 3, expected_video_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_to_video_on_hdf5_gen31_recording_default_args(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen31_timer.hdf5"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 640, 'height': 480, 'frame-count': 392}
    generate_video(filename_full, 10000, 1, expected_video_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_to_video_on_hdf5_gen31_recording_default_args_imseq(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen31_timer.hdf5"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 640, 'height': 480, 'frame-count': 392}
    generate_image_sequence(filename_full, expected_video_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_to_video_on_hdf5_gen4_evt2_recording_default_args(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen4_evt2_hand.hdf5"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 1280, 'height': 720, 'frame-count': 314}
    generate_video(filename_full, 10000, 1, expected_video_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_to_video_on_hdf5_gen4_evt3_recording_default_args(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen4_evt3_hand.hdf5"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 1280, 'height': 720, 'frame-count': 451}
    generate_video(filename_full, 10000, 1, expected_video_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_to_video_on_hdf5_gen31_recording_slow_motion_2(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen31_timer.hdf5"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 640, 'height': 480, 'frame-count': 783}
    generate_video(filename_full, 10000, 2, expected_video_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_to_video_on_hdf5_gen4_evt2_recording_slow_motion_0_5_acc_time_200(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen4_evt2_hand.hdf5"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 1280, 'height': 720, 'frame-count': 157}
    generate_video(filename_full, 200, 0.5, expected_video_info)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_to_video_on_hdf5_gen4_evt3_recording_slow_motion_3_acc_time_20000(dataset_dir):
    """
    Checks output of metavision_file_to_video application
    """

    filename = "gen4_evt3_hand.hdf5"
    filename_full = os.path.join(dataset_dir, "openeb", filename)
    expected_video_info = {'width': 1280, 'height': 720, 'frame-count': 1351}
    generate_video(filename_full, 20000, 3, expected_video_info)
