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
from metavision_utils import os_tools, pytest_tools


def get_ev_from_line(line):
    items = line.strip().split(",")
    assert len(items) == 4
    return {'x': int(items[0]), 'y': int(items[1]), 'p': int(items[2]), 't': int(items[3])}


def run_raw_to_csv_and_check_result(filename_full, number_events_expected, first_10_events_expected,
                                    middle_10_events_expected, last_10_events_expected):

    # Before launching the sample, check the dataset file exists
    assert os.path.exists(filename_full)

    # Since the sample metavision_raw_to_csv writes the output file in the same directory
    # the sample is launched from, create a tmp directory from where we can run the sample
    tmp_dir = os_tools.TemporaryDirectoryHandler()

    # The pytest is run from the build/bin dir (cf CMakeLists.txt), but since we'll run the command
    # from the temporary directory created above, we need to get the full path to the sample
    sample_full_path = os.path.join(os.getcwd(), "metavision_raw_to_csv")

    cmd = "\"{}\" -i \"{}\"".format(sample_full_path, filename_full)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd, working_directory=tmp_dir.temporary_directory())

    # Check sample exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check CSV file has been written
    expected_generated_file = os.path.join(tmp_dir.temporary_directory(), "cd.csv")
    assert os.path.exists(expected_generated_file)

    def parse_lines(expected_generated_file, buffer_size):
        """
        Parse and yield each lines of file 'expected_generated_file'.
        Internal reading of the file is done by chunks of 'buffer_size' bytes.
        """

        with open(expected_generated_file, 'r') as f:
            remaining_buff = ""
            buff = f.read(buffer_size)
            while buff:
                last_new_line = buff.rfind("\n")
                lines_buff = buff[:last_new_line]
                remaining_buff = buff[last_new_line+1:]

                for line in lines_buff.split("\n"):
                    yield line

                buff = remaining_buff + f.read(buffer_size)

    idx = 0
    idx_ev = number_events_expected // 2 - 5
    _1MB = 1 * 1024 * 1024
    for line in parse_lines(expected_generated_file, _1MB):
        # Check first 10 events :
        if 0 <= idx < 10:
            ev = get_ev_from_line(line)
            assert ev == first_10_events_expected[idx], "Error on event nr {}".format(idx)

        # Check the 10 events in the middle:
        if idx_ev <= idx < idx_ev+10:
            ev = get_ev_from_line(line)
            assert ev == middle_10_events_expected[idx-idx_ev], "Error on event nr {}".format(idx)

        # Check last 10 events :
        if number_events_expected-10 <= idx < number_events_expected:
            ev = get_ev_from_line(line)
            assert ev == last_10_events_expected[idx-number_events_expected], "Error on event nr {}".format(idx)

        idx += 1

    # Check number of events
    assert idx == number_events_expected


def pytestcase_test_metavision_raw_to_csv_show_help():
    """
    Checks output of metavision_raw_to_csv when displaying help message
    """

    cmd = "./metavision_raw_to_csv --help"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check sample exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check that the options showed in the output
    assert "Options:" in output, "******\nMissing options display in output :{}\n******".format(output)


def pytestcase_test_metavision_raw_to_csv_non_existing_input_file():
    """
    Checks that metavision_raw_to_csv returns an error when passing an input file that doesn't exist
    """

    # Create a file path that we are sure does not exist
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_raw_file = os.path.join(tmp_dir.temporary_directory(), "nonexistent.raw")

    cmd = "./metavision_raw_to_csv -i {}".format(input_raw_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert sample returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file could not be read
    assert "not an existing file" in output


def pytestcase_test_metavision_raw_to_csv_missing_input_args():
    """
    Checks that metavision_raw_to_csv returns an error when not passing required input args
    """

    cmd = "./metavision_raw_to_csv"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert sample returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file arg is missing
    assert re.search("Parsing error: the option (.+) is required but missing", output)


def pytestcase_test_metavision_raw_to_csv_on_gen31_recording(dataset_dir):
    """
    Checks result of metavision_raw_to_csv sample on dataset gen31_timer.raw
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    number_events_expected = 29450906

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

    run_raw_to_csv_and_check_result(filename_full, number_events_expected, first_10events_expected,
                                    middle_10events_expected, last_10events_expected)


def pytestcase_test_metavision_raw_to_csv_on_gen4_evt2_recording(dataset_dir):
    """
    Checks result of metavision_raw_to_csv sample on dataset gen4_evt2_hand.raw
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    number_events_expected = 17025195

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

    run_raw_to_csv_and_check_result(filename_full, number_events_expected, first_10events_expected,
                                    middle_10events_expected, last_10events_expected)


def pytestcase_test_metavision_raw_to_csv_on_gen4_evt3_recording(dataset_dir):
    """
    Checks result of metavision_raw_to_csv sample on dataset gen4_evt3_hand.raw
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    number_events_expected = 18453063

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
        {'x': 687, 'y': 248, 'p': 0, 't': 15445502}, {'x': 1181, 'y': 480, 'p': 1, 't': 15445502}
    ]

    run_raw_to_csv_and_check_result(filename_full, number_events_expected, first_10events_expected,
                                    mid_10events_expected, last_10events_expected)
