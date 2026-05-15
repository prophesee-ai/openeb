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

def load_cd_csv_contents_ignoring_header(input_cd_csv_path):
    """Loads the contents of a CD CSV file, ignoring header lines starting with a '%' character"""
    # Count number of header lines to skip
    n_header_lines = 0
    with open(input_cd_csv_path, 'r') as f:
        while True:
            line = f.readline()
            if not line.startswith("%"):
                break
            n_header_lines += 1
    # Now open the files, skip the headers and return the rest of their content
    with open(input_cd_csv_path, 'r') as f:
        for i in range(0,n_header_lines):
            f.readline()
        cd_csv_contents = f.read()
    return cd_csv_contents

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

    # Now load the files contents, ignoring headers
    standalone_output = load_cd_csv_contents_ignoring_header(standalone_output_file)
    hal_output = load_cd_csv_contents_ignoring_header(hal_output_file)

    assert standalone_output == hal_output
