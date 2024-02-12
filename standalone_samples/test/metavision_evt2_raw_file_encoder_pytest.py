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
from metavision_utils import shell_tools, os_tools

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

def decode_encode_decode_and_compare(input_raw_path, contains_triggers=False):
    """"Decodes file with standalone sample and encode it back. Then decodes the encoded file and compare this output
        with the output obtained by decoding the original file

    Args:
        input_raw_path (str): path of the input RAW file
        contains_triggers (bool): True if input file contains triggers, False otherwise
    """

    # Before launching the app, check the input file exists
    assert os.path.exists(input_raw_path)

    # Create a temporary directory for the output files that will be generated
    tmp_dir = os_tools.TemporaryDirectoryHandler()

    # STEP 1: run the standalone decoder application on input file
    cd_csv_from_input = os.path.join(tmp_dir.temporary_directory(), "cd_csv_from_input.csv")
    triggers_csv_from_input = os.path.join(tmp_dir.temporary_directory(), "triggers_csv_from_input.csv")
    cmd = "./metavision_evt2_raw_file_decoder \"{}\" \"{}\"".format(input_raw_path, cd_csv_from_input)
    if (contains_triggers):
        cmd += " \"{}\"".format(triggers_csv_from_input)
    output, error, err_code = shell_tools.execute_cmd(cmd)
    assert err_code == 0, "******\nError while executing cmd '{}':{}\n{}\n******".format(cmd, output, error)
    assert os.path.exists(cd_csv_from_input)
    if (contains_triggers):
        assert os.path.exists(triggers_csv_from_input)

    # STEP 2: run the standalone encoder application on output of step 1
    input_raw_path_encoded = os.path.join(tmp_dir.temporary_directory(), "encoded_raw.raw")
    cmd = "./metavision_evt2_raw_file_encoder \"{}\" \"{}\"".format(input_raw_path_encoded, cd_csv_from_input)
    if (contains_triggers):
        cmd += " \"{}\"".format(triggers_csv_from_input)
    output, error, err_code = shell_tools.execute_cmd(cmd)
    assert err_code == 0, "******\nError while executing cmd '{}':{}\n{}\n******".format(cmd, output, error)
    assert os.path.exists(input_raw_path_encoded)

    # STEP 3: run the standalone decoder application on encoded file (i.e. output of step 2)
    cd_csv_from_encoded = os.path.join(tmp_dir.temporary_directory(), "cd_csv_from_encoded.csv")
    triggers_csv_from_encoded = os.path.join(tmp_dir.temporary_directory(), "triggers_csv_from_encoded.csv")
    cmd = "./metavision_evt2_raw_file_decoder \"{}\" \"{}\"".format(input_raw_path_encoded, cd_csv_from_encoded)
    if (contains_triggers):
        cmd += " \"{}\"".format(triggers_csv_from_encoded)
    output, error, err_code = shell_tools.execute_cmd(cmd)
    assert err_code == 0, "******\nError while executing cmd '{}':{}\n{}\n******".format(cmd, output, error)
    assert os.path.exists(cd_csv_from_encoded)
    if (contains_triggers):
        assert os.path.exists(triggers_csv_from_encoded)

    # STEP 4: compare the 2 outputs
    cd_csv_from_input_contents = load_cd_csv_contents_ignoring_header(cd_csv_from_input)
    cd_csv_from_encoded_contents = load_cd_csv_contents_ignoring_header(cd_csv_from_encoded)
    assert cd_csv_from_input_contents != ""  # To make sure to actually test something
    assert cd_csv_from_input_contents == cd_csv_from_encoded_contents
    if (contains_triggers):
        with open(triggers_csv_from_input, 'r') as f:
            triggers_csv_from_input_contents = f.read()
        with open(triggers_csv_from_encoded, 'r') as f:
            triggers_csv_from_encoded_contents = f.read()
        assert triggers_csv_from_input_contents != ""  # To make sure to actually test something
        assert triggers_csv_from_input_contents == triggers_csv_from_encoded_contents


def pytestcase_evt2_rawfile_encoder_on_gen31_recording(dataset_dir):
    """
    Checks result of metavision_evt2_raw_file_encoder application on dataset gen31_timer.raw
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    decode_encode_decode_and_compare(filename_full)


def pytestcase_evt2_rawfile_decoder_on_gen4_evt2_recording(dataset_dir):
    """
    Checks result of metavision_evt2_raw_file_encoder application on dataset gen4_evt2_hand.raw
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    decode_encode_decode_and_compare(filename_full)


def pytestcase_evt2_rawfile_encoder_on_gen31_recording_with_triggers(dataset_dir):
    """
    Checks result of metavision_evt2_raw_file_encoder application on dataset openeb/event_io/recording.raw
    """

    filename_full = os.path.join(dataset_dir, "openeb", "core", "event_io", "recording.raw")

    decode_encode_decode_and_compare(filename_full, True)
