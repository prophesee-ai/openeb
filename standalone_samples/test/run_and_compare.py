#!/usr/bin/env python

# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import os
from metavision_utils import shell_tools, os_tools, pytest_tools


def run_standalone_decoder_and_compare_to_hal_implementation(filename_full, format_event):
    """"Run standalone apps and compare its output with the app implemented with hal

    Args:
        filename_full (str): path of the file
        format_event (int): format event of the file (2 or 3)
    """

    # Before launching the app, check the dataset file exists
    assert os.path.exists(filename_full)

    # Create a temporary directory for the output files that will be generated
    tmp_dir = os_tools.TemporaryDirectoryHandler()

    # Run the standalone decoder application
    standalone_output_file = os.path.join(tmp_dir.temporary_directory(), "standalone_output_file.csv")
    cmd = "./metavision_evt{}_raw_file_decoder \"{}\" \"{}\"".format(format_event, filename_full,
                                                                     standalone_output_file)
    output, error, err_code = shell_tools.execute_cmd(cmd)
    assert err_code == 0, "******\nError while executing cmd '{}':{}\n{}\n******".format(cmd, output, error)
    assert os.path.exists(standalone_output_file)

    # Now run the application implemented with hal
    hal_output_file = os.path.join(tmp_dir.temporary_directory(), "hal_output_file.csv")
    cmd = "./raw_file_decoder_with_hal \"{}\" \"{}\"".format(filename_full, hal_output_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)
    assert os.path.exists(hal_output_file)

    # Now open the files and store their content :
    with open(standalone_output_file, 'r') as f:
        standalone_output = f.read()

    with open(hal_output_file, 'r') as f:
        hal_output = f.read()

    assert standalone_output == hal_output
