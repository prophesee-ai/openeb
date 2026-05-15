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
from metavision_utils import pytest_tools


def pytestcase_test_metavision_file_info_with_sample_plugin(dataset_dir):
    '''
    Checks output of metavision_file_info application when using sample plugin
    '''

    filename = "sample_plugin_recording.raw"
    filename_full = os.path.realpath(os.path.join(dataset_dir, "openeb", filename))

    # Before launching the app, check the dataset file exists
    assert os.path.exists(filename_full)

    cmd = "./metavision_file_info -i \"{}\"".format(filename_full)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    expected_output = r"""
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
""".format(filename, re.escape(filename_full))

    # Now check output ,after stripping them for trailing white spaces
    output_strip = pytest_tools.get_mv_info_stripped_output(output)
    expected_output_strip = pytest_tools.get_mv_info_stripped_output(expected_output)
    assert re.search(expected_output_strip, output_strip)
