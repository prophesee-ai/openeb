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
import numpy as np
import h5py
from metavision_utils import os_tools, pytest_tools


def run_file_to_hdf5_on_recording_and_check_result(
        filename_full,
        width_expected,
        height_expected,
        number_cd_expected,
        first_10_cd_events_expected,
        middle_10_cd_events_expected,
        last_10_cd_events_expected,
        number_trigger_expected=None,
        first_10_ext_trigger_events_expected=None,
        middle_10_ext_trigger_events_expected=None,
        last_10_ext_trigger_events_expected=None):

    # Before launching the app, check the dataset file exists
    assert os.path.exists(filename_full)

    # Since the application metavision_file_to_dat writes the output file in the same directory
    # as the input file, in order not to pollute the git status of the repository (input dataset
    # is committed), copy input file in temporary directory and run the app on that

    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_file = tmp_dir.copy_file_in_tmp_dir(filename_full)
    assert input_file  # i.e. assert input_file != None, to verify the copy was successful

    expected_generated_file = input_file.replace(".raw", ".hdf5")
    # Just to be sure, check that the DAT file does not already exist, otherwise the test could be misleading
    assert not os.path.exists(expected_generated_file)

    cmd = "./metavision_file_to_hdf5 -i {}".format(input_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check DAT file has been written
    assert os.path.exists(expected_generated_file)

    # read HDF5 and extract relevant information
    with h5py.File(expected_generated_file, "r") as f:
        # Check groups
        keys = f.keys()
        for key in ["CD", "EXT_TRIGGER"]:
            assert key in keys
            assert "events" in f[key]
            assert "indexes" in f[key]

        # Check attributes
        attrs = f.attrs
        assert "geometry" in attrs.keys()
        geometry = attrs["geometry"]
        width, height = geometry.split("x")

        # Verify expected size
        assert int(width) == width_expected
        assert int(height) == height_expected

        # Get data
        cds = f["CD"]["events"]
        assert cds.size == number_cd_expected

        # Check first 10 events :
        for idx in range(0, 10):
            ev = {'x': cds[idx]["x"], 'y': cds[idx]["y"], 'p': cds[idx]["p"], 't': cds[idx]["t"]}
            assert ev == first_10_cd_events_expected[idx], "Error on event nr {}".format(idx)

        # Check the 10 events in the middle:
        idx_ev = number_cd_expected // 2 - 5
        for idx in range(0, 10):
            ev = {'x': cds[idx_ev]["x"], 'y': cds[idx_ev]["y"], 'p': cds[idx_ev]["p"], 't': cds[idx_ev]["t"]}
            assert ev == middle_10_cd_events_expected[idx], "Error on event nr {}".format(idx_ev)
            idx_ev += 1

        # Check last 10 events :
        for idx in range(0, 10):
            idx_ev = -(10 - idx)
            ev = {'x': cds[idx_ev]["x"], 'y': cds[idx_ev]["y"], 'p': cds[idx_ev]["p"], 't': cds[idx_ev]["t"]}
            assert ev == last_10_cd_events_expected[idx], "Error on event nr {}".format(idx_ev)

        # Check indexes
        indexes = f["CD"]["indexes"]
        offset = int(indexes.attrs["offset"])
        i = 0
        for index in indexes:
            if index["ts"] >= 0:
                assert index["ts"] == cds[index["id"]
                                          ]["t"] + offset, "Error on index nr {}, timestamp does not correspond to event".format(i)
            i = i+1

        if not(number_trigger_expected and first_10_ext_trigger_events_expected and
               middle_10_ext_trigger_events_expected and last_10_ext_trigger_events_expected):
            return

        # Get data
        triggers = f["EXT_TRIGGER"]["events"]

        assert triggers.size == number_trigger_expected

        # Check first 10 events :
        for idx in range(0, 10):
            ev = {'p': triggers[idx]["p"], 'id': triggers[idx]["id"], 't': triggers[idx]["t"]}
            assert ev == first_10_ext_trigger_events_expected[idx], "Error on event nr {}".format(idx)

        # Check the 10 events in the middle:
        idx_ev = number_trigger_expected // 2 - 5
        for idx in range(0, 10):
            ev = {'p': triggers[idx_ev]["p"], 'id': triggers[idx_ev]["id"], 't': triggers[idx_ev]["t"]}
            assert ev == middle_10_ext_trigger_events_expected[idx], "Error on event nr {}".format(idx_ev)
            idx_ev += 1

        # Check last 10 events :
        for idx in range(0, 10):
            idx_ev = -(10 - idx)
            ev = {'p': triggers[idx_ev]["p"], 'id': triggers[idx_ev]["id"], 't': triggers[idx_ev]["t"]}
            assert ev == last_10_ext_trigger_events_expected[idx], "Error on event nr {}".format(idx_ev)

        # Check indexes
        indexes = f["EXT_TRIGGER"]["indexes"]
        offset = int(indexes.attrs["offset"])
        i = 0
        for index in indexes:
            if index["ts"] >= 0:
                assert index["ts"] == triggers[index["id"]
                                               ]["t"] + offset, "Error on index nr {}, timestamp does not correspond to event".format(i)
            i = i+1


def pytestcase_test_metavision_file_to_hdf5_show_help():
    """
    Checks output of metavision_file_to_hdf5 when displaying help message
    """

    cmd = "./metavision_file_to_hdf5 --help"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check that the options showed in the output
    assert "Options:" in output, "******\nMissing options display in output :{}\n******".format(output)


def pytestcase_test_metavision_file_to_hdf5_non_existing_input_file():
    """
    Checks that metavision_file_to_hdf5 returns an error when passing an input file that doesn't exist
    """

    # Create a filepath that we are sure does not exist
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_rawfile = os.path.join(tmp_dir.temporary_directory(), "nonexistent.raw")

    cmd = "./metavision_file_to_hdf5 -i {}".format(input_rawfile)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file could not be read
    assert "not an existing file" in output


def pytestcase_test_metavision_file_to_hdf5_recursive_mode_no_pattern(dataset_dir):
    """
    Checks that metavision_file_to_hdf5 returns an error when using recursive mode and not specifying a filename pattern
    """

    filename = "gen31_timer.hdf5"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    cmd = "./metavision_file_to_hdf5 -i {} -r".format(filename_full)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the filename pattern is missing
    assert "Error: please specify a file pattern for the recursive conversion" in output


def pytestcase_test_metavision_file_to_hdf5_missing_input_args():
    """
    Checks that metavision_file_to_hdf5 returns an error when not passing required input args
    """

    cmd = "./metavision_file_to_hdf5"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file arg is missing
    assert re.search("Parsing error: the option (.+) is required but missing", output)


def pytestcase_test_metavision_file_to_hdf5_input_file_already_hdf5(dataset_dir):
    """
    Checks that metavision_file_to_hdf5 returns an error when passing an HDF5 file as input
    """

    filename = "gen31_timer.hdf5"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    cmd = "./metavision_file_to_hdf5 -i {}".format(filename_full)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file could not be read
    assert "Error: output file is the same as input file" in output


def pytestcase_test_metavision_file_to_hdf5_on_gen31_raw_recording(dataset_dir):
    """
    Checks result of metavision_file_to_hdf5 application on dataset gen31_timer.raw
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    width_expected = 640
    height_expected = 480
    number_cd_expected = 29450906

    first_10events_expected = [{'x': 487, 'y': 316, 'p': 1, 't': 16}, {'x': 6, 'y': 310, 'p': 0, 't': 16},
                               {'x': 579, 'y': 433, 'p': 1, 't': 16}, {'x': 467, 'y': 62, 'p': 1, 't': 16},
                               {'x': 255, 'y': 6, 'p': 0, 't': 17}, {'x': 264, 'y': 6, 'p': 0, 't': 17},
                               {'x': 424, 'y': 107, 'p': 0, 't': 17}, {'x': 217, 'y': 99, 'p': 0, 't': 17},
                               {'x': 50, 'y': 125, 'p': 1, 't': 18}, {'x': 331, 'y': 216, 'p': 1, 't': 18}
                               ]

    middle_10events_expected = [{'x': 77, 'y': 95, 'p': 0, 't': 6357174}, {'x': 6, 'y': 432, 'p': 0, 't': 6357174},
                                {'x': 487, 'y': 286, 'p': 0, 't': 6357175}, {'x': 98, 'y': 176, 'p': 0, 't': 6357175},
                                {'x': 463, 'y': 53, 'p': 1, 't': 6357176}, {'x': 198, 'y': 92, 'p': 0, 't': 6357176},
                                {'x': 296, 'y': 273, 'p': 0, 't': 6357177}, {'x': 447, 'y': 149, 'p': 1, 't': 6357178},
                                {'x': 228, 'y': 406, 'p': 0, 't': 6357178}, {'x': 1, 'y': 202, 'p': 0, 't': 6357178}
                                ]

    last_10events_expected = [{'x': 247, 'y': 77, 'p': 0, 't': 13043027}, {'x': 192, 'y': 452, 'p': 0, 't': 13043027},
                              {'x': 433, 'y': 78, 'p': 1, 't': 13043031}, {'x': 575, 'y': 316, 'p': 1, 't': 13043031},
                              {'x': 12, 'y': 273, 'p': 1, 't': 13043032}, {'x': 241, 'y': 63, 'p': 1, 't': 13043032},
                              {'x': 422, 'y': 102, 'p': 0, 't': 13043032}, {'x': 407, 'y': 198, 'p': 1, 't': 13043032},
                              {'x': 392, 'y': 240, 'p': 0, 't': 13043033}, {'x': 349, 'y': 225, 'p': 0, 't': 13043033}
                              ]

    run_file_to_hdf5_on_recording_and_check_result(
        filename_full, width_expected, height_expected, number_cd_expected,
        first_10events_expected, middle_10events_expected, last_10events_expected)


def pytestcase_test_metavision_file_to_hdf5_on_gen4_evt2_raw_recording(dataset_dir):
    """
    Checks result of metavision_file_to_hdf5 application on dataset gen4_evt2_hand.raw
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    width_expected = 1280
    height_expected = 720
    number_cd_expected = 17025195

    first_10events_expected = [{'x': 369, 'y': 477, 'p': 1, 't': 49}, {'x': 1181, 'y': 480, 'p': 1, 't': 50},
                               {'x': 1181, 'y': 480, 'p': 1, 't': 53}, {'x': 922, 'y': 61, 'p': 0, 't': 53},
                               {'x': 1278, 'y': 634, 'p': 1, 't': 55}, {'x': 1181, 'y': 480, 'p': 1, 't': 57},
                               {'x': 478, 'y': 74, 'p': 0, 't': 59}, {'x': 922, 'y': 61, 'p': 0, 't': 60},
                               {'x': 1181, 'y': 480, 'p': 1, 't': 60}, {'x': 283, 'y': 99, 'p': 0, 't': 61}
                               ]

    middle_10events_expected = [{'x': 762, 'y': 342, 'p': 0, 't': 4655829}, {'x': 636, 'y': 182, 'p': 1, 't': 4655829},
                                {'x': 1181, 'y': 480, 'p': 1, 't': 4655829}, {'x': 509, 'y': 76, 'p': 0, 't': 4655830},
                                {'x': 872, 'y': 223, 'p': 0, 't': 4655830}, {'x': 484, 'y': 491, 'p': 0, 't': 4655830},
                                {'x': 922, 'y': 61, 'p': 0, 't': 4655831}, {'x': 992, 'y': 369, 'p': 1, 't': 4655832},
                                {'x': 883, 'y': 88, 'p': 0, 't': 4655832}, {'x': 944, 'y': 217, 'p': 0, 't': 4655833}
                                ]

    last_10events_expected = [
        {'x': 752, 'y': 261, 'p': 0, 't': 10442727}, {'x': 1181, 'y': 480, 'p': 1, 't': 10442729},
        {'x': 758, 'y': 319, 'p': 0, 't': 10442730}, {'x': 1181, 'y': 480, 'p': 1, 't': 10442732},
        {'x': 922, 'y': 61, 'p': 0, 't': 10442733}, {'x': 1181, 'y': 480, 'p': 1, 't': 10442736},
        {'x': 922, 'y': 61, 'p': 0, 't': 10442739}, {'x': 1181, 'y': 480, 'p': 1, 't': 10442740},
        {'x': 369, 'y': 477, 'p': 1, 't': 10442740}, {'x': 1181, 'y': 480, 'p': 1, 't': 10442743}
    ]

    run_file_to_hdf5_on_recording_and_check_result(
        filename_full, width_expected, height_expected, number_cd_expected,
        first_10events_expected, middle_10events_expected, last_10events_expected)


def pytestcase_test_metavision_file_to_hdf5_on_gen4_evt3_raw_recording(dataset_dir):
    """
    Checks result of metavision_file_to_hdf5 application on dataset gen4_evt3_hand.raw
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    width_expected = 1280
    height_expected = 720
    number_cd_expected = 18094969

    first_10events_expected = [{'x': 922, 'y': 61, 'p': 0, 't': 5714}, {'x': 1181, 'y': 480, 'p': 1, 't': 5714},
                               {'x': 869, 'y': 702, 'p': 1, 't': 5715}, {'x': 1181, 'y': 480, 'p': 1, 't': 5717},
                               {'x': 540, 'y': 443, 'p': 0, 't': 5718}, {'x': 922, 'y': 61, 'p': 0, 't': 5720},
                               {'x': 1181, 'y': 480, 'p': 1, 't': 5720}, {'x': 1005, 'y': 248, 'p': 0, 't': 5721},
                               {'x': 1181, 'y': 480, 'p': 1, 't': 5723}, {'x': 922, 'y': 61, 'p': 0, 't': 5725}
                               ]

    mid_10events_expected = [
        {'x': 417, 'y': 91, 'p': 0, 't': 7171644}, {'x': 608, 'y': 351, 'p': 1, 't': 7171644},
        {'x': 407, 'y': 167, 'p': 0, 't': 7171644}, {'x': 922, 'y': 61, 'p': 0, 't': 7171645},
        {'x': 864, 'y': 152, 'p': 0, 't': 7171645}, {'x': 1181, 'y': 480, 'p': 1, 't': 7171645},
        {'x': 375, 'y': 252, 'p': 0, 't': 7171646}, {'x': 547, 'y': 158, 'p': 0, 't': 7171647},
        {'x': 521, 'y': 336, 'p': 0, 't': 7171647}, {'x': 1099, 'y': 20, 'p': 0, 't': 7171647}
    ]

    last_10events_expected = [
        {'x': 510, 'y': 562, 'p': 1, 't': 15000115}, {'x': 922, 'y': 61, 'p': 0, 't': 15000117},
        {'x': 1181, 'y': 480, 'p': 1, 't': 15000117}, {'x': 369, 'y': 477, 'p': 1, 't': 15000119},
        {'x': 1175, 'y': 381, 'p': 0, 't': 15000119}, {'x': 1181, 'y': 480, 'p': 1, 't': 15000120},
        {'x': 442, 'y': 507, 'p': 1, 't': 15000121}, {'x': 922, 'y': 61, 'p': 0, 't': 15000122},
        {'x': 1181, 'y': 480, 'p': 1, 't': 15000123}, {'x': 315, 'y': 408, 'p': 1, 't': 15000125}
    ]

    run_file_to_hdf5_on_recording_and_check_result(
        filename_full, width_expected, height_expected, number_cd_expected,
        first_10events_expected, mid_10events_expected, last_10events_expected)


def pytestcase_test_metavision_file_to_hdf5_on_gen4_evt3_with_triggers_raw_recording(dataset_dir):
    """
    Checks result of metavision_file_to_hdf5 application on dataset blinking_gen4_with_ext_triggers.raw
    """

    filename = "blinking_gen4_with_ext_triggers.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    width_expected = 1280
    height_expected = 720
    number_cd_expected = 2003016
    number_trigger_expected = 82

    first_10_cd_events_expected = [
        {'x': 1203, 'y': 301, 'p': 0, 't': 19},
        {'x': 912, 'y': 186, 'p': 0, 't': 21}, {'x': 687, 'y': 680, 'p': 0, 't': 23},
        {'x': 396, 'y': 298, 'p': 0, 't': 26}, {'x': 63, 'y': 150, 'p': 0, 't': 30},
        {'x': 719, 'y': 505, 'p': 1, 't': 30}, {'x': 930, 'y': 646, 'p': 0, 't': 31},
        {'x': 375, 'y': 291, 'p': 0, 't': 33},
        {'x': 722, 'y': 314, 'p': 0, 't': 34},
        {'x': 321, 'y': 111, 'p': 0, 't': 39},
    ]

    mid_10_cd_events_expected = [
        {'x': 596, 'y': 271, 'p': 1, 't': 2172617},
        {'x': 940, 'y': 392, 'p': 0, 't': 2172619},
        {'x': 969, 'y': 471, 'p': 1, 't': 2172619},
        {'x': 780, 'y': 134, 'p': 0, 't': 2172619},
        {'x': 24, 'y': 306, 'p': 0, 't': 2172621},
        {'x': 960, 'y': 225, 'p': 0, 't': 2172623},
        {'x': 35, 'y': 704, 'p': 0, 't': 2172624},
        {'x': 300, 'y': 615, 'p': 0, 't': 2172629},
        {'x': 227, 'y': 385, 'p': 0, 't': 2172630},
        {'x': 749, 'y': 308, 'p': 0, 't': 2172633},
    ]

    last_10_cd_events_expected = [
        {'x': 1234, 'y': 237, 'p': 0, 't': 4194887},
        {'x': 554, 'y': 453, 'p': 0, 't': 4194888},
        {'x': 677, 'y': 548, 'p': 1, 't': 4194889},
        {'x': 114, 'y': 128, 'p': 0, 't': 4194889},
        {'x': 125, 'y': 146, 'p': 0, 't': 4194891},
        {'x': 388, 'y': 498, 'p': 0, 't': 4194892},
        {'x': 1151, 'y': 330, 'p': 0, 't': 4194894},
        {'x': 809, 'y': 598, 'p': 0, 't': 4194894},
        {'x': 311, 'y': 687, 'p': 0, 't': 4194896},
        {'x': 757, 'y': 401, 'p': 0, 't': 4194897},
    ]

    first_10_trigger_events_expected = [
        {'p': 1, 'id': 6, 't': 127081},
        {'p': 0, 'id': 6, 't': 177081},
        {'p': 1, 'id': 6, 't': 227081},
        {'p': 0, 'id': 6, 't': 277081},
        {'p': 1, 'id': 6, 't': 327081},
        {'p': 0, 'id': 6, 't': 377081},
        {'p': 1, 'id': 6, 't': 427081},
        {'p': 0, 'id': 6, 't': 477081},
        {'p': 1, 'id': 6, 't': 527081},
        {'p': 0, 'id': 6, 't': 577081},
    ]

    mid_10_trigger_events_expected = [
        {'p': 1, 'id': 6, 't': 1927081},
        {'p': 0, 'id': 6, 't': 1977081},
        {'p': 1, 'id': 6, 't': 2027081},
        {'p': 0, 'id': 6, 't': 2077081},
        {'p': 1, 'id': 6, 't': 2127081},
        {'p': 0, 'id': 6, 't': 2177081},
        {'p': 1, 'id': 6, 't': 2227081},
        {'p': 0, 'id': 6, 't': 2277081},
        {'p': 1, 'id': 6, 't': 2327081},
        {'p': 0, 'id': 6, 't': 2377081},

    ]

    last_10_trigger_events_expected = [
        {'p': 1, 'id': 6, 't': 3727081},
        {'p': 0, 'id': 6, 't': 3777081},
        {'p': 1, 'id': 6, 't': 3827081},
        {'p': 0, 'id': 6, 't': 3877081},
        {'p': 1, 'id': 6, 't': 3927081},
        {'p': 0, 'id': 6, 't': 3977081},
        {'p': 1, 'id': 6, 't': 4027081},
        {'p': 0, 'id': 6, 't': 4077081},
        {'p': 1, 'id': 6, 't': 4127081},
        {'p': 0, 'id': 6, 't': 4177081},
    ]

    run_file_to_hdf5_on_recording_and_check_result(
        filename_full, width_expected, height_expected, number_cd_expected, first_10_cd_events_expected,
        mid_10_cd_events_expected, last_10_cd_events_expected, number_trigger_expected,
        first_10_trigger_events_expected, mid_10_trigger_events_expected, last_10_trigger_events_expected)
