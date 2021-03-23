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


def generate_video(filename_full, fps, expected_video_infos):

    # Before launching the app, check the dataset file exists
    assert os.path.exists(filename_full)

    tmp_dir = os_tools.TemporaryDirectoryHandler()
    output_video = os.path.join(tmp_dir.temporary_directory(), "out.avi")

    cmd = "./metavision_raw_to_video -i \"{}\" --fps {} -o {} --fourcc MJPG".format(filename_full, fps, output_video)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Before opening the video, verify it has been written :
    assert os.path.exists(output_video)

    # Now check basic information on the video
    cap = cv2.VideoCapture(output_video)

    # Check size :
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == expected_video_infos['width']
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == expected_video_infos['height']

    # Check frame count and frame rate
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_video_infos['frame-count']
    assert int(cap.get(cv2.CAP_PROP_FPS)) == fps

    # Check codec
    assert decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)) == "MJPG"


def pytestcase_test_metavision_raw_to_video_show_help():
    """
    Checks output of metavision_raw_to_video when displaying help message
    """

    cmd = "./metavision_raw_to_video --help"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check that the options showed in the output
    assert "Options:" in output, "******\nMissing options display in output :{}\n******".format(output)


def pytestcase_test_metavision_raw_to_video_non_existing_input_file():
    """
    Checks that metavision_raw_to_video returns an error when passing an input file that doesn't exist
    """

    # Create a filepath that we are sure it does not exist
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_rawfile = os.path.join(tmp_dir.temporary_directory(), "data_in.raw")

    cmd = "./metavision_raw_to_video -i {}".format(input_rawfile)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file could not be read
    assert "not an existing file" in output


def pytestcase_test_metavision_raw_to_video_missing_input_args():
    """
    Checks that metavision_raw_to_video returns an error when not passing required input args
    """

    cmd = "./metavision_raw_to_video"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file arg is missing
    assert re.search("Parsing error: the option (.+) is required but missing", output)


def pytestcase_test_metavision_raw_to_video_on_gen31_recording_30fps(dataset_dir):
    """
    Checks output of metavision_raw_to_video application
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.join(dataset_dir, filename)
    expected_video_infos = {'width': 640, 'height': 480, 'frame-count': 391}
    generate_video(filename_full, 30, expected_video_infos)


def pytestcase_test_metavision_raw_to_video_on_gen4_evt2_recording_25fps(dataset_dir):
    """
    Checks output of metavision_raw_to_video application
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.join(dataset_dir, filename)
    expected_video_infos = {'width': 1280, 'height': 720, 'frame-count': 261}
    generate_video(filename_full, 25, expected_video_infos)


def pytestcase_test_metavision_raw_to_video_on_gen4_evt3_recording_66fps(dataset_dir):
    """
    Checks output of metavision_raw_to_video application
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.join(dataset_dir, filename)
    expected_video_infos = {'width': 1280, 'height': 720, 'frame-count': 1019}
    generate_video(filename_full, 66, expected_video_infos)
