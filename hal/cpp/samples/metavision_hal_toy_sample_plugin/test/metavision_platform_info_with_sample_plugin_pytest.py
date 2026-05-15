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
from metavision_utils import pytest_tools


def pytestcase_test_metavision_platform_info_with_sample_plugin_system():
    '''
    Checks output of metavision_platform_info application when using sample plugin
    '''

    cmd = "./metavision_platform_info --system"
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0

    # Check expected output
    expected_output_contains = """## SampleIntegratorName Gen1.0 600x500 ##

# System information
Available Data Encoding Formats                   SAMPLE-FORMAT-1.0
Connection                                        USB
Current Data Encoding Format                      SAMPLE-FORMAT-1.0
Integrator                                        SampleIntegratorName
Sensor Name                                       Gen1.0
Serial                                            000000

# Available device config options
"""
    output_strip = "\n".join([line.strip() for line in output.splitlines()])
    expected_output_strip = "\n".join([line.strip() for line in expected_output_contains.splitlines()])
    assert output_strip.find(expected_output_strip) >= 0
