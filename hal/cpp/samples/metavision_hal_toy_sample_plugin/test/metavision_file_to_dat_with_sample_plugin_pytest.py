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
import numpy as np
from metavision_utils import os_tools, pytest_tools

CD_X_MASK = 2**14 - 1  # 18 zeros followed by 14 ones when formulated as a binary number.
CD_Y_MASK = 2**28 - 2**14  # 4 zeros, 14 ones and then 14 zeros.
CD_P_MASK = 2 ** 29 - 2**28  # 3 zeros a one and 28 zeros.


def pytestcase_test_metavision_file_to_dat_with_sample_plugin(dataset_dir):
    '''
    Checks output of metavision_file_to_dat application when using sample plugin
    '''

    filename = "sample_plugin_recording.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    # Before launching the app, check the dataset file exists
    assert os.path.exists(filename_full)

    # Since the application metavision_file_to_dat writes the output file in the same directory
    # as the input file, in order not to pollute the git status of the repository (input dataset
    # is committed), copy input file in temporary directory and run the app on that

    tmp_dir = os_tools.TemporaryDirectoryHandler()
    input_file = tmp_dir.copy_file_in_tmp_dir(filename_full)
    assert input_file  # i.e. assert input_file != None, to verify the copy was successful

    expected_generated_file = input_file.replace(".raw", "_cd.dat")
    # Just to be sure, check that the DAT file does not already exist, otherwise the test could be misleading
    assert not os.path.exists(expected_generated_file)

    cmd = "./metavision_file_to_dat -i {}".format(input_file)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd)

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check DAT file has been written
    assert os.path.exists(expected_generated_file)

    # Checking all events would take long, so we just test a subset
    idx_to_check = [1, 2114, 20185, 50001, 104618, 116783, 130045, 159141, 171864, 177868, 181003, 198623, 207518,
                    213796, 216581, 245937, 250340, 280508, 285551, 370256, 373888, 380633, 381277, 389611, 396206,
                    406399, 409008, 411388, 417972, 472989, 474184, 491985, 505700, 527520, 531732, 535567, 577006,
                    582677, 587597, 619476, 632649, 640135, 641796, 662353, 669284, 684812, 701566, 709096, 757240,
                    767908, 773760, 790839]
    events_expected = [{'x': 294, 'y': 284, 'p': 1, 't': 0}, {'x': 285, 'y': 270, 'p': 0, 't': 14030},
                       {'x': 350, 'y': 364, 'p': 1, 't': 122305}, {'x': 446, 'y': 436, 'p': 1, 't': 304755},
                       {'x': 587, 'y': 366, 'p': 0, 't': 635585}, {'x': 533, 'y': 301, 'p': 1, 't': 709770},
                       {'x': 475, 'y': 301, 'p': 0, 't': 789970}, {'x': 382, 'y': 194, 'p': 1, 't': 966410},
                       {'x': 343, 'y': 168, 'p': 1, 't': 1044600}, {'x': 365, 'y': 134, 'p': 0, 't': 1080695},
                       {'x': 355, 'y': 119, 'p': 0, 't': 1100740}, {'x': 285, 'y': 53, 'p': 1, 't': 1207010},
                       {'x': 243, 'y': 26, 'p': 1, 't': 1261145}, {'x': 232, 'y': 7, 'p': 1, 't': 1299240},
                       {'x': 111, 'y': 143, 'p': 0, 't': 1315680}, {'x': 119, 'y': 117, 'p': 1, 't': 1493725},
                       {'x': 115, 'y': 144, 'p': 1, 't': 1521790}, {'x': 54, 'y': 223, 'p': 0, 't': 1704250},
                       {'x': 563, 'y': 473, 'p': 0, 't': 1734725}, {'x': 298, 'y': 467, 'p': 1, 't': 2249610},
                       {'x': 309, 'y': 458, 'p': 1, 't': 2271665}, {'x': 323, 'y': 419, 'p': 1, 't': 2313765},
                       {'x': 309, 'y': 417, 'p': 1, 't': 2317775}, {'x': 338, 'y': 432, 'p': 0, 't': 2367900},
                       {'x': 353, 'y': 412, 'p': 0, 't': 2408000}, {'x': 408, 'y': 360, 'p': 1, 't': 2470155},
                       {'x': 404, 'y': 333, 'p': 1, 't': 2486195}, {'x': 401, 'y': 366, 'p': 0, 't': 2500230},
                       {'x': 405, 'y': 346, 'p': 0, 't': 2540330}, {'x': 549, 'y': 179, 'p': 1, 't': 2873165},
                       {'x': 579, 'y': 136, 'p': 1, 't': 2881185}, {'x': 506, 'y': 82, 'p': 1, 't': 2989455},
                       {'x': 479, 'y': 40, 'p': 1, 't': 3073660}, {'x': 423, 'y': 56, 'p': 0, 't': 3205990},
                       {'x': 580, 'y': 217, 'p': 1, 't': 3230655}, {'x': 396, 'y': 50, 'p': 0, 't': 3254115},
                       {'x': 273, 'y': 192, 'p': 0, 't': 3506740}, {'x': 253, 'y': 233, 'p': 1, 't': 3540825},
                       {'x': 208, 'y': 248, 'p': 1, 't': 3570900}, {'x': 105, 'y': 310, 'p': 1, 't': 3763385},
                       {'x': 84, 'y': 384, 'p': 1, 't': 3843585}, {'x': 42, 'y': 402, 'p': 1, 't': 3889695},
                       {'x': 43, 'y': 372, 'p': 0, 't': 3899725}, {'x': 48, 'y': 474, 'p': 1, 't': 4024035},
                       {'x': 70, 'y': 495, 'p': 1, 't': 4066140}, {'x': 93, 'y': 418, 'p': 0, 't': 4160375},
                       {'x': 184, 'y': 371, 'p': 1, 't': 4262630}, {'x': 167, 'y': 368, 'p': 0, 't': 4308740},
                       {'x': 343, 'y': 236, 'p': 0, 't': 4601470}, {'x': 385, 'y': 182, 'p': 1, 't': 4665635},
                       {'x': 363, 'y': 156, 'p': 0, 't': 4701720}, {'x': 455, 'y': 103, 'p': 1, 't': 4805980}]
    # Now open the file and check for its contents
    with open(expected_generated_file, 'rb') as f:
        # Parse header
        width = -1
        height = -1
        begin_events_pos = 0
        ev_type = -1
        ev_size = -1
        while(True):
            begin_events_pos = f.tell()
            line = f.readline().decode("latin-1")
            first_char = line[0]
            if first_char == '%':
                # Look for width and height :
                res_width = re.match(r"% Width (\d+)", line)
                if res_width:
                    width = int(res_width.group(1))
                else:
                    res_height = re.match(r"% Height (\d+)", line)
                    if res_height:
                        height = int(res_height.group(1))
            else:
                # Position cursor after header and exit loop
                f.seek(begin_events_pos, os.SEEK_SET)
                # Read event type
                ev_type = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                # Read event size
                ev_size = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                break

        # Verify expected size
        assert width == 600
        assert height == 500

        # Assert event type and size
        assert ev_type == 12  # CD
        assert ev_size == 8

        # Now check total number of CD events and time of first and last
        data = np.fromfile(f, dtype=[('t', 'u4'), ('xyp', 'i4')])

        x = np.bitwise_and(data["xyp"], CD_X_MASK)
        y = np.right_shift(np.bitwise_and(data["xyp"], CD_Y_MASK), 14)
        p = np.right_shift(np.bitwise_and(data["xyp"], CD_P_MASK), 28)

        nr_cd = len(data)
        assert nr_cd == 790840

        # Check events :
        for idx in range(0, len(idx_to_check)):
            event_number = idx_to_check[idx]
            ev = {'x': x[event_number], 'y': y[event_number], 'p': p[event_number], 't': data["t"][event_number]}
            assert ev == events_expected[idx], "Error on event nr {}".format(event_number)
