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


def check_file_information(filename_full, expected_output):

    # Before launching the app, check the dataset file exists
    assert os.path.exists(filename_full)

    cmd = "./metavision_raw_info -i \"{}\"".format(filename_full)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0

    # Now check output, after stripping them for trailing whitespaces
    output_strip = "\n".join([line.strip() for line in output.splitlines()])
    expected_output_strip = "\n".join([line.strip() for line in expected_output.splitlines()])
    assert output_strip.find(expected_output_strip) >= 0


def pytestcase_test_metavision_raw_info_show_help():
    """
    Checks output of metavision_raw_info when displaying help message
    """

    cmd = "./metavision_raw_info --help"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check that the options showed in the output
    assert "Options:" in output, "******\nMissing options display in output :{}\n******".format(output)


def pytestcase_test_metavision_raw_info_non_existing_input_file():
    """
    Checks that metavision_raw_info returns an error when passing an input file that doesn't exist
    """

    # Create a file path that we are sure does not exist
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_raw_file = os.path.join(tmp_dir.temporary_directory(), "nonexistent.raw")

    cmd = "./metavision_raw_info -i {}".format(input_raw_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file could not be read
    assert "Unable to open RAW file" in output


def pytestcase_test_metavision_raw_info_missing_input_args():
    """
    Checks that metavision_raw_info returns an error when not passing required input args
    """

    cmd = "./metavision_raw_info"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Assert app returned error
    assert error_code != 0

    # And now check that the error came from the fact that the input file arg is missing
    assert re.search("Parsing error: the option (.+) is required but missing", output)


def pytestcase_test_metavision_raw_info_on_gen31_recording(dataset_dir):
    """
    Checks output of metavision_raw_info application
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    expected_output = """
====================================================================================================

Name                {}
Path                {}
Duration            13s 43ms 33us
Integrator          Prophesee
Plugin name         hal_plugin_gen31_fx3
Event encoding      EVT2
Camera generation   3.1
Camera systemID     28
Camera serial       00001621

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  29450906            16                  13043033            2.3 Mev/s
""".format(filename, filename_full)
    check_file_information(filename_full, expected_output)


def pytestcase_test_metavision_raw_info_on_gen4_evt2_recording(dataset_dir):
    """
    Checks output of metavision_raw_info application
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    expected_output = """
====================================================================================================

Name                {}
Path                {}
Duration            10s 442ms 743us
Integrator          Prophesee
Plugin name         hal_plugin_gen4_fx3
Event encoding      EVT2
Camera generation   4.0
Camera systemID     26
Camera subsystemID  537921537
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  17025195            49                  10442743            1.6 Mev/s
""".format(filename, filename_full)
    check_file_information(filename_full, expected_output)


def pytestcase_test_metavision_raw_info_on_gen4_evt3_recording(dataset_dir):
    """
    Checks output of metavision_raw_info application
    """

    filename = "gen4_evt3_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    expected_output = """
====================================================================================================

Name                {}
Path                {}
Duration            15s 445ms 502us
Integrator          Prophesee
Plugin name         hal_plugin_gen4_fx3
Event encoding      EVT3
Camera generation   4.0
Camera systemID     26
Camera subsystemID  537921537
Camera serial       00001495

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  18453063            5714                15445502            1.2 Mev/s
""".format(filename, filename_full)
    check_file_information(filename_full, expected_output)
