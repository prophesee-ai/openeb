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


def cut_and_check_info(input_raw, start, end, expected_output_info=None):
    """"Runs metavision_hal_raw_cutter on input file and checks the output

    Args:
        input_raw (str): path of the input file
        start, end : cut range [s]
        expected_output_info : expected output on running metavision_file_info on the output file
                               If none is provided, we assume that we have to get the same info as
                               the input file (which is the case when the range covers all the input file)
    """

    # Before launching the app, check the dataset file exists
    assert os.path.exists(input_raw)

    # Create temporary directory, where we'll put the output
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    output_file_name = "raw_cut_{}_{}.raw".format(start, end)
    output_file_path = os.path.join(tmp_dir.temporary_directory(), output_file_name)

    cmd = "./metavision_hal_raw_cutter -i \"{}\" --start {} --end {} -o {}".format(input_raw, start, end,
                                                                                   output_file_path)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check output file has been written
    assert os.path.exists(output_file_path)

    # Now, with the app metavision_file_info we check the information
    cmd = "./metavision_file_info -i {}".format(output_file_path)
    info_cut_file, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0

    # Now check output
    if not expected_output_info:
        # Then run metavision_file_info on input file, since we expect the same output
        cmd = "./metavision_file_info -i \"{}\"".format(input_raw)
        info_input_file, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

        # Check no error occurred
        assert error_code == 0

        # Need to format the output
        expected_output_info = info_input_file.replace(input_raw, "{}").replace(os.path.basename(input_raw), "{}")

    expected_output_info_formatted = expected_output_info.format(
        output_file_name, re.escape(os.path.realpath(output_file_path)))

    output_strip = pytest_tools.get_mv_info_stripped_output(info_cut_file)
    expected_output_strip = pytest_tools.get_mv_info_stripped_output(expected_output_info_formatted)
    # Do not check plugin name, it may differ if the original plugin does not exist anymore
    output_strip = re.sub("Plugin name.*\n", "", output_strip)
    expected_output_strip = re.sub("Plugin name.*\n", "", expected_output_strip)
    assert re.search(expected_output_strip, output_strip)


def pytestcase_test_metavision_hal_raw_cutter_show_help():
    """
    Checks output of metavision_hal_raw_cutter when displaying help message
    """

    cmd = "./metavision_hal_raw_cutter --help"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check that the options shows in the output
    assert "Options:" in output, "******\nMissing options display in output :{}\n******".format(output)


def pytestcase_test_metavision_hal_raw_cutter_nonexistent_input_file():
    """
    Checks that metavision_hal_raw_cutter returns an error when passing an input file that doesn't exist
    """

    # Create temporary directory for nonexistent RAW file
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_raw_file = os.path.join(tmp_dir.temporary_directory(), "nonexistent_in.raw")
    output_raw_file = os.path.join(tmp_dir.temporary_directory(), "nonexistent_out.raw")

    cmd = "./metavision_hal_raw_cutter -i {} --start {} --end {} -o {}".format(input_raw_file, 0, 2, output_raw_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited with error
    assert error_code != 0

    # And now check that the error came from the fact that the input file could not be read
    assert "Unable to open RAW file" in output


def pytestcase_test_metavision_hal_raw_cutter_missing_input_args():
    """
    Checks that metavision_hal_raw_cutter returns an error when not passing required input args
    """

    cmd = "./metavision_hal_raw_cutter"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited with error
    assert error_code != 0

    # And now check that the error came from the fact that the input file arg is missing
    assert re.search("Parsing error: the option (.+) is required but missing", output)


def pytestcase_test_metavision_hal_raw_cutter_invalid_range(dataset_dir):
    """
    Checks that metavision_hal_raw_cutter returns an error when passing inconsistent values for start and stop
    """

    # To be sure the error isn't thrown because the input file doesn't exist, use one from the datasets
    input_raw_file = os.path.join(dataset_dir, "openeb", "gen31_timer.raw")
    assert os.path.exists(input_raw_file)

    # Create temporary directory for output RAW file
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    output_raw_file = os.path.join(tmp_dir.temporary_directory(), "data_out.raw")

    start = 4
    end = 2
    cmd = "./metavision_hal_raw_cutter -i {} --start {} --end {} -o {}".format(
        input_raw_file, start, end, output_raw_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited with error
    assert error_code != 0

    # And now check that the error came from the fact that the input range was invalid
    assert re.search("end time {} is less than or equal to start {}".format(end, start), output)


def pytestcase_test_metavision_hal_raw_cutter_on_gen31_recording_full_cut(dataset_dir):
    """
    Checks output of metavision_hal_raw_cutter application when the range given spans through the whole file
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 15  # This recording is ~13s, so 15 is well after its end

    cut_and_check_info(filename_full, start, end)


def pytestcase_test_metavision_hal_raw_cutter_on_gen31_recording_from_0s_to_6s(dataset_dir):
    """
    Checks output of metavision_hal_raw_cutter on dataset gen31_timer.raw, cutting from 0s to 6s
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 6

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            6s 0ms 187us
Integrator          Prophesee
Plugin name         hal_plugin_gen31_fx3
Data encoding       EVT2
Camera generation   3.1
Camera serial       00001621

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  14067447            16                  6000187             2.3 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_hal_raw_cutter_on_gen31_recording_from_8s_to_11s(dataset_dir):
    """
    Checks output of metavision_hal_raw_cutter on dataset gen31_timer.raw, cutting from 8s to 11s
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 8
    end = 11

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            3s 0ms 210us
Integrator          Prophesee
Plugin name         hal_plugin_gen31_fx3
Data encoding       EVT2
Camera generation   3.1
Camera serial       00001621

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  5590919             16                  3000210             1.9 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_hal_raw_cutter_on_gen4_evt2_recording_full_cut(dataset_dir):
    """
    Checks output of metavision_hal_raw_cutter application when the range given spans throws all the file
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 11

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            10s 442ms 743us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT2
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  17025195            49                  10442743            1.6 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_hal_raw_cutter_on_gen4_evt2_recording_from_2s_to_3s(dataset_dir):
    """
    Checks output of metavision_hal_raw_cutter on dataset gen4_evt2_hand.raw, cutting from 2s to 3s
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 2
    end = 3

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            999ms 977us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT2
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  1985451             16                  999977              2.0 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_hal_raw_cutter_on_gen4_evt2_recording_from_4s_to_10s(dataset_dir):
    """
    Checks output of metavision_hal_raw_cutter on dataset gen4_evt2_hand.raw, cutting from 4s to 10s
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 4
    end = 10

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            6s 0ms 53us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT2
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  9468423             0                   6000053             1.6 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_hal_raw_cutter_on_gen4_evt3_recording_full_cut(dataset_dir):
    """
    Checks output of metavision_hal_raw_cutter application when the range given spans throws all the file
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 0
    end = 16

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            15s 441ms 920us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT3
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  18094969            5714                15000125            1.2 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_hal_raw_cutter_on_gen4_evt3_recording_from_3s_to_7s(dataset_dir):
    """
    Checks output of metavision_hal_raw_cutter on dataset gen4_evt3_hand.raw, cutting from 3s to 7s
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 3
    end = 7

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            4s 6ms 217us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT3
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  4884793             5841                4006217             1.2 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_hal_raw_cutter_on_gen4_evt3_recording_from_8s_to_9s(dataset_dir):
    """
    Checks output of metavision_hal_raw_cutter on dataset gen4_evt3_hand.raw, cutting from 8s to 9s
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 8
    end = 9

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            1s 4ms 655us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT3
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  1319855             4753                1004655             1.3 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)


def pytestcase_test_metavision_hal_raw_cutter_on_gen4_evt3_recording_from_4s_to_15s(dataset_dir):
    """
    Checks output of metavision_hal_raw_cutter on dataset gen4_evt3_hand.raw, cutting from 4s to 15s
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    start = 4
    end = 15

    expected_output_info = r"""
====================================================================================================

Name                {}
Path                {}
Duration            11s 6ms 525us
Integrator          Prophesee
Plugin name         hal_plugin_gen41_evk3
Data encoding       EVT3
Camera generation   4.0
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  12759106            6464                11006525            1.2 Mev/s
"""
    cut_and_check_info(filename_full, start, end, expected_output_info)
