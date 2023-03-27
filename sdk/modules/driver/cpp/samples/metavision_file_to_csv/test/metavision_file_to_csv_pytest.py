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


def run_file_to_csv_and_check_result(filename_full, number_events_expected, first_10_events_expected,
                                     middle_10_events_expected, last_10_events_expected):

    # Before launching the sample, check the dataset file exists
    assert os.path.exists(filename_full)

    # Since the sample metavision_file_to_csv writes the output file in the same directory
    # the sample is launched from, create a tmp directory from where we can run the sample
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    output_filename = os.path.join(
        tmp_dir.temporary_directory(),
        os.path.splitext(os.path.basename(filename_full))[0] + ".csv")

    # The pytest is run from the build/bin dir (cf CMakeLists.txt), but since we'll run the command
    # from the temporary directory created above, we need to get the full path to the sample
    sample_full_path = os.path.join(os.getcwd(), "metavision_file_to_csv")

    cmd = "\"{}\" -i \"{}\" -o \"{}\"".format(sample_full_path, filename_full, output_filename)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd, working_directory=tmp_dir.temporary_directory())

    # Check sample exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check CSV file has been written
    assert os.path.exists(output_filename)

    def parse_lines(filename, buffer_size):
        """
        Parse and yield each lines of file 'filename'.
        Internal reading of the file is done by chunks of 'buffer_size' bytes.
        """

        with open(filename, 'r') as f:
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
    for line in parse_lines(output_filename, _1MB):
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


def pytestcase_test_metavision_file_to_csv_show_help():
    """
    Checks output of metavision_file_to_csv when displaying help message
    """

    cmd = "./metavision_file_to_csv --help"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check sample exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check that the options showed in the output
    assert "Options:" in output, "******\nMissing options display in output :{}\n******".format(output)


def pytestcase_test_metavision_file_to_csv_non_existing_input_file():
    """
    Checks that metavision_file_to_csv returns an error when passing an input file that doesn't exist
    """

    # Create a file path that we are sure does not exist
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_file = os.path.join(tmp_dir.temporary_directory(), "nonexistent.raw")

    cmd = "./metavision_file_to_csv -i {}".format(input_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert sample returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file could not be read
    assert "not an existing file" in output


def pytestcase_test_metavision_file_to_csv_missing_input_args():
    """
    Checks that metavision_file_to_csv returns an error when not passing required input args
    """

    cmd = "./metavision_file_to_csv"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert sample returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file arg is missing
    assert re.search("Parsing error: the option (.+) is required but missing", output)


@pytest.mark.skipif(
    "ENABLE_EXHAUSTIVE_TESTING" not in os.environ or os.environ["ENABLE_EXHAUSTIVE_TESTING"] != "TRUE",
    reason="ENABLE_EXHAUSTIVE_TESTING is disabled")
def pytestcase_test_metavision_file_to_csv_on_raw_gen31_recording(dataset_dir):
    """
    Checks result of metavision_file_to_csv sample on dataset gen31_timer.raw
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

    run_file_to_csv_and_check_result(filename_full, number_events_expected, first_10events_expected,
                                     middle_10events_expected, last_10events_expected)


def pytestcase_test_metavision_file_to_csv_on_raw_gen4_evt2_recording(dataset_dir):
    """
    Checks result of metavision_file_to_csv sample on dataset gen4_evt2_hand.raw
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

    run_file_to_csv_and_check_result(filename_full, number_events_expected, first_10events_expected,
                                     middle_10events_expected, last_10events_expected)


@pytest.mark.skipif(
    "ENABLE_EXHAUSTIVE_TESTING" not in os.environ or os.environ["ENABLE_EXHAUSTIVE_TESTING"] != "TRUE",
    reason="ENABLE_EXHAUSTIVE_TESTING is disabled")
def pytestcase_test_metavision_file_to_csv_on_raw_gen4_evt3_recording(dataset_dir):
    """
    Checks result of metavision_file_to_csv sample on dataset gen4_evt3_hand.raw
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    number_events_expected = 18094969

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

    run_file_to_csv_and_check_result(filename_full, number_events_expected, first_10events_expected,
                                     mid_10events_expected, last_10events_expected)


@pytest.mark.skipif(
    "ENABLE_EXHAUSTIVE_TESTING" not in os.environ or os.environ["ENABLE_EXHAUSTIVE_TESTING"] != "TRUE",
    reason="ENABLE_EXHAUSTIVE_TESTING is disabled")
@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_to_csv_on_hdf5_gen31_recording(dataset_dir):
    """
    Checks result of metavision_file_to_csv sample on dataset gen31_timer.hdf5
    """

    filename = "gen31_timer.hdf5"
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

    run_file_to_csv_and_check_result(filename_full, number_events_expected, first_10events_expected,
                                     middle_10events_expected, last_10events_expected)


@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_to_csv_on_hdf5_gen4_evt2_recording(dataset_dir):
    """
    Checks result of metavision_file_to_csv sample on dataset gen4_evt2_hand.hdf5
    """

    filename = "gen4_evt2_hand.hdf5"
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

    run_file_to_csv_and_check_result(filename_full, number_events_expected, first_10events_expected,
                                     middle_10events_expected, last_10events_expected)


@pytest.mark.skipif(
    "ENABLE_EXHAUSTIVE_TESTING" not in os.environ or os.environ["ENABLE_EXHAUSTIVE_TESTING"] != "TRUE",
    reason="ENABLE_EXHAUSTIVE_TESTING is disabled")
@pytest.mark.skipif("HAS_HDF5" not in os.environ or os.environ["HAS_HDF5"] != "TRUE", reason="hdf5 not available")
def pytestcase_test_metavision_file_to_csv_on_hdf5_gen4_evt3_recording(dataset_dir):
    """
    Checks result of metavision_file_to_csv sample on dataset gen4_evt3_hand.hdf5
    """

    filename = "gen4_evt3_hand.hdf5"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    number_events_expected = 18094969

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

    run_file_to_csv_and_check_result(filename_full, number_events_expected, first_10events_expected,
                                     mid_10events_expected, last_10events_expected)
