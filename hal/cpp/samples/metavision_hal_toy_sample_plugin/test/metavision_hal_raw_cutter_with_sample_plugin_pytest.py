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
from metavision_utils import os_tools, pytest_tools


def cut_and_check_infos(dataset_dir, start, end, expected_output_info):
    """"Runs metavision_hal_raw_cutter on sample_plugin_recording.raw and checks the output

    Args:
        dataset_dir (str): path of the directory with the datasets
        start, end : cut range [s]
        expected_output_info : expected output on running metavision_file_info on the output file
                               If none is provided, we assume that we have to get the same info as
                               the input file (which is the case when the range covers all the input file)
    """

    filename = "sample_plugin_recording.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    # Before launching the app, check the dataset file exists
    assert os.path.exists(filename_full)

    # Create temporary directory, where we'll put the output
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    output_file_name = "sample_plugin_raw_cut_{}_{}.raw".format(start, end)
    output_file_path = os.path.join(tmp_dir.temporary_directory(), output_file_name)

    cmd = "./metavision_hal_raw_cutter -i \"{}\" --start {} --end {} -o {}".format(filename_full, start, end,
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
    expected_output_info_formatted = expected_output_info.format(output_file_name, os.path.realpath(output_file_path))

    output_strip = pytest_tools.get_mv_info_stripped_output(info_cut_file)
    expected_output_strip = pytest_tools.get_mv_info_stripped_output(expected_output_info_formatted)

    assert output_strip.find(expected_output_strip) >= 0


def pytestcase_test_metavision_hal_raw_cutter_with_sample_plugin_full_cut(dataset_dir):
    '''
    Checks output of metavision_hal_raw_cutter application when using sample plugin
    '''

    expected_output = """
====================================================================================================

Name                {}
Path                {}
Duration            4s 805ms 980us
Integrator          SampleIntegratorName
Plugin name         hal_toy_sample_plugin
Data encoding       SAMPLE-FORMAT-1.0
Camera generation   1.0
Camera serial       000000

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  790840              0                   4805980             164.6 Kev/s
"""

    start = 0
    end = 5

    cut_and_check_infos(dataset_dir, start, end, expected_output)


def pytestcase_test_metavision_hal_raw_cutter_with_sample_plugin_cut_from_0s_to_3s(dataset_dir):
    '''
    Checks output of metavision_hal_raw_cutter application when using sample plugin
    '''

    expected_output = """
====================================================================================================

Name                {}
Path                {}
Duration            3s 5ms 490us
Integrator          SampleIntegratorName
Plugin name         hal_toy_sample_plugin
Data encoding       SAMPLE-FORMAT-1.0
Camera generation   1.0
Camera serial       000000

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  494592              0                   3005490             164.6 Kev/s
"""

    start = 0
    end = 3

    cut_and_check_infos(dataset_dir, start, end, expected_output)


def pytestcase_test_metavision_hal_raw_cutter_with_sample_plugin_cut_from_2s_to_4s(dataset_dir):
    '''
    Checks output of metavision_hal_raw_cutter application when using sample plugin
    '''

    expected_output = """
====================================================================================================

Name                {}
Path                {}
Duration            2s 4ms 995us
Integrator          SampleIntegratorName
Plugin name         hal_toy_sample_plugin
Data encoding       SAMPLE-FORMAT-1.0
Camera generation   1.0
Camera serial       000000

====================================================================================================

Type of event       Number of events    First timestamp     Last timestamp      Average event rate
----------------------------------------------------------------------------------------------------
CD                  329728              0                   2004995             164.5 Kev/s
"""

    start = 2
    end = 4

    cut_and_check_infos(dataset_dir, start, end, expected_output)
