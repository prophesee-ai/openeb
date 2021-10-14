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
from metavision_utils import os_tools, pytest_tools

CD_X_MASK = 2**14 - 1  # 18 zeros followed by 14 ones when formulated as a binary number.
CD_Y_MASK = 2**28 - 2**14  # 4 zeros, 14 ones and then 14 zeros.
CD_P_MASK = 2 ** 29 - 2**28  # 3 zeros, a one and 28 zeros.


def run_raw_to_dat_on_cd_recording_and_check_result(
        filename_full,
        width_expected,
        height_expected,
        number_cd_expected,
        first_10_events_expected,
        middle_10_events_expected,
        last_10_events_expected):

    # Before launching the app, check the dataset file exists
    assert os.path.exists(filename_full)

    # Since the application metavision_raw_to_dat writes the output file in the same directory
    # as the input file, in order not to pollute the git status of the repository (input dataset
    # is committed), copy input file in temporary directory and run the app on that

    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_file = tmp_dir.copy_file_in_tmp_dir(filename_full)
    assert input_file  # i.e. assert input_file != None, to verify the copy was successful

    expected_generated_file = input_file.replace(".raw", "_cd.dat")
    # Just to be sure, check that the DAT file does not already exist, otherwise the test could be misleading
    assert not os.path.exists(expected_generated_file)

    cmd = "./metavision_raw_to_dat -i {}".format(input_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check DAT file has been written
    assert os.path.exists(expected_generated_file)

    # Now open the file and check for its contents
    with open(expected_generated_file, 'rb') as f:
        # Parse header
        width = -1
        height = -1
        begin_events_pos = 0
        ev_type = -1
        ev_size = -1
        while True:
            begin_events_pos = f.tell()
            line = f.readline().decode("latin-1")
            first_char = line[0]
            if first_char == '%':
                # Look for width and height :
                res_width = re.match(r"% Width (\d+)", line)
                if res_width:
                    width = int(res_width.group(1))
                else:
                    res_height = re.match(r"% Height (\d+)", line)
                    if res_height:
                        height = int(res_height.group(1))
            else:
                # Position cursor after header and exit loop
                f.seek(begin_events_pos, os.SEEK_SET)
                # Read event type
                ev_type = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                # Read event size
                ev_size = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                break

        # Verify expected size
        assert width == width_expected
        assert height == height_expected

        # Assert event type and size
        assert ev_type == 12  # CD
        assert ev_size == 8

        # Now check total number of CD events and time of first and last
        data = np.fromfile(f, dtype=[('t', 'u4'), ('xyp', 'i4')])

        x = np.bitwise_and(data["xyp"], CD_X_MASK)
        y = np.right_shift(np.bitwise_and(data["xyp"], CD_Y_MASK), 14)
        p = np.right_shift(np.bitwise_and(data["xyp"], CD_P_MASK), 28)

        nr_cd = len(data)
        assert nr_cd == number_cd_expected

        # Check first 10 events :
        for idx in range(0, 10):
            ev = {'x': x[idx], 'y': y[idx], 'p': p[idx], 't': data["t"][idx]}
            assert ev == first_10_events_expected[idx], "Error on event nr {}".format(idx)

        # Check the 10 events in the middle:
        idx_ev = nr_cd // 2 - 5
        for idx in range(0, 10):
            ev = {'x': x[idx_ev], 'y': y[idx_ev], 'p': p[idx_ev], 't': data["t"][idx_ev]}
            assert ev == middle_10_events_expected[idx], "Error on event nr {}".format(idx_ev)
            idx_ev += 1

        # Check last 10 events :
        for idx in range(0, 10):
            idx_ev = -(10 - idx)
            ev = {'x': x[idx_ev], 'y': y[idx_ev], 'p': p[idx_ev], 't': data["t"][idx_ev]}
            assert ev == last_10_events_expected[idx], "Error on event nr {}".format(idx_ev)


def pytestcase_test_metavision_raw_to_dat_show_help():
    """
    Checks output of metavision_raw_to_dat when displaying help message
    """

    cmd = "./metavision_raw_to_dat --help"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check that the options showed in the output
    assert "Options:" in output, "******\nMissing options display in output :{}\n******".format(output)


def pytestcase_test_metavision_raw_to_dat_non_existing_input_file():
    """
    Checks that metavision_raw_to_dat returns an error when passing an input file that doesn't exist
    """

    # Create a filepath that we are sure does not exist
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_rawfile = os.path.join(tmp_dir.temporary_directory(), "nonexistent.raw")

    cmd = "./metavision_raw_to_dat -i {}".format(input_rawfile)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file could not be read
    assert "not an existing file" in output


def pytestcase_test_metavision_raw_to_dat_missing_input_args():
    """
    Checks that metavision_raw_to_dat returns an error when not passing required input args
    """

    cmd = "./metavision_raw_to_dat"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file arg is missing
    assert re.search("Parsing error: the option (.+) is required but missing", output)


def pytestcase_test_metavision_raw_to_dat_on_gen31_recording(dataset_dir):
    """
    Checks result of metavision_raw_to_dat application on dataset gen31_timer.raw
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

    run_raw_to_dat_on_cd_recording_and_check_result(
        filename_full, width_expected, height_expected, number_cd_expected,
        first_10events_expected, middle_10events_expected, last_10events_expected)


def pytestcase_test_metavision_raw_to_dat_on_gen4_evt2_recording(dataset_dir):
    """
    Checks result of metavision_raw_to_dat application on dataset gen4_evt2_hand.raw
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

    run_raw_to_dat_on_cd_recording_and_check_result(
        filename_full, width_expected, height_expected, number_cd_expected,
        first_10events_expected, middle_10events_expected, last_10events_expected)


def pytestcase_test_metavision_raw_to_dat_on_gen4_evt3_recording(dataset_dir):
    """
    Checks result of metavision_raw_to_dat application on dataset gen4_evt3_hand.raw
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    width_expected = 1280
    height_expected = 720
    number_cd_expected = 18453063

    first_10events_expected = [{'x': 922, 'y': 61, 'p': 0, 't': 5714}, {'x': 1181, 'y': 480, 'p': 1, 't': 5714},
                               {'x': 869, 'y': 702, 'p': 1, 't': 5715}, {'x': 1181, 'y': 480, 'p': 1, 't': 5717},
                               {'x': 540, 'y': 443, 'p': 0, 't': 5718}, {'x': 922, 'y': 61, 'p': 0, 't': 5720},
                               {'x': 1181, 'y': 480, 'p': 1, 't': 5720}, {'x': 1005, 'y': 248, 'p': 0, 't': 5721},
                               {'x': 1181, 'y': 480, 'p': 1, 't': 5723}, {'x': 922, 'y': 61, 'p': 0, 't': 5725}
                               ]

    mid_10events_expected = [{'x': 546, 'y': 303, 'p': 1, 't': 7266906}, {'x': 1181, 'y': 480, 'p': 1, 't': 7266906},
                             {'x': 865, 'y': 100, 'p': 1, 't': 7266907}, {'x': 922, 'y': 61, 'p': 0, 't': 7266907},
                             {'x': 667, 'y': 328, 'p': 1, 't': 7266909}, {'x': 1181, 'y': 480, 'p': 1, 't': 7266909},
                             {'x': 461, 'y': 243, 'p': 1, 't': 7266912}, {'x': 522, 'y': 325, 'p': 1, 't': 7266912},
                             {'x': 1181, 'y': 480, 'p': 1, 't': 7266913}, {'x': 922, 'y': 61, 'p': 0, 't': 7266913}
                             ]

    last_10events_expected = [
        {'x': 1181, 'y': 480, 'p': 1, 't': 15445493}, {'x': 785, 'y': 323, 'p': 1, 't': 15445494},
        {'x': 922, 'y': 61, 'p': 0, 't': 15445495}, {'x': 840, 'y': 392, 'p': 0, 't': 15445495},
        {'x': 1181, 'y': 480, 'p': 1, 't': 15445496}, {'x': 1181, 'y': 480, 'p': 1, 't': 15445499},
        {'x': 922, 'y': 61, 'p': 0, 't': 15445501}, {'x': 369, 'y': 477, 'p': 1, 't': 15445502},
        {'x': 687, 'y': 248, 'p': 0, 't': 15445502}, {'x': 1181, 'y': 480, 'p': 1, 't': 15445502}]

    run_raw_to_dat_on_cd_recording_and_check_result(
        filename_full, width_expected, height_expected, number_cd_expected,
        first_10events_expected, mid_10events_expected, last_10events_expected)
