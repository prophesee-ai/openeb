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
from run_and_compare import run_standalone_decoder_and_compare_to_hal_implementation


def pytestcase_evt2_rawfile_decoder_on_gen31_recording(dataset_dir):
    """
    Checks result of evt2_rawfile_decoder application on dataset gen31_timer.raw
    """

    filename = "gen31_timer.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    run_standalone_decoder_and_compare_to_hal_implementation(filename_full, 2)


def pytestcase_evt2_rawfile_decoder_on_gen4_evt2_recording(dataset_dir):
    """
    Checks result of evt2_rawfile_decoder application on dataset gen4_evt2_hand.raw
    """

    filename = "gen4_evt2_hand.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    run_standalone_decoder_and_compare_to_hal_implementation(filename_full, 2)
