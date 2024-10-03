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


def get_ev_from_line(line):
    items = line.strip().split(",")
    assert len(items) == 4
    return {'x': int(items[0]), 'y': int(items[1]), 'p': int(items[2]), 't': int(items[3])}


def pytestcase_test_metavision_file_to_csv_with_sample_plugin(dataset_dir):
    '''
    Checks output of metavision_file_to_csv application when using sample plugin
    '''

    filename = "sample_plugin_recording.raw"
    filename_full = os.path.join(dataset_dir, "openeb", filename)

    # Before launching the app, check the dataset file exists
    assert os.path.exists(filename_full)

    # Since the application metavision_file_to_csv writes the output file in the same directory
    # the app is launched from, create a tmp directory from where we can run the app
    tmp_dir = os_tools.TemporaryDirectoryHandler()
    output_filename = os.path.join(
        tmp_dir.temporary_directory(),
        os.path.splitext(filename)[0] + ".csv")

    # The pytest is run from the build/bin dir (cf CMakeLists.txt), but since we'll run the command
    # from the temporary directory created above, we need to get the full path to the application
    application_full_path = os.path.join(os.getcwd(), "metavision_file_to_csv")

    cmd = "\"{}\" -i \"{}\" -o \"{}\"".format(application_full_path, filename_full, output_filename)
    output, error_code = pytest_tools.run_cmd_setting_mv_log_file(cmd, working_directory=tmp_dir.temporary_directory())

    # Check app exited without error
    assert error_code == 0, "******\nError while executing cmd '{}':{}\n******".format(cmd, output)

    # Check CSV file has been written
    assert os.path.exists(output_filename)

    # Checking all events would take long, so we just test a subset
    idx_to_check = [1, 2647, 15098, 31393, 55056, 60392, 126838, 160953, 161756, 239542, 255380, 263120, 286172,
                    305025, 306674, 313755, 337452, 340242, 348143, 362732, 367017, 367516, 411226, 411721, 414272,
                    469949, 478421, 487793, 494067, 502675, 510576, 518535, 519037, 541733, 542226, 595092, 595579,
                    599828, 615304, 653587, 655153, 663832, 665051, 665486, 671977, 676311, 689468, 697722, 700298,
                    745706, 760479, 790839]
    events_expected = [{'x': 294, 'y': 284, 'p': 1, 't': 0}, {'x': 302, 'y': 298, 'p': 1, 't': 16040},
                       {'x': 300, 'y': 327, 'p': 0, 't': 92225}, {'x': 372, 'y': 398, 'p': 1, 't': 190475},
                       {'x': 427, 'y': 430, 'p': 0, 't': 334835}, {'x': 477, 'y': 468, 'p': 1, 't': 366915},
                       {'x': 487, 'y': 270, 'p': 1, 't': 771920}, {'x': 376, 'y': 180, 'p': 1, 't': 978435},
                       {'x': 410, 'y': 205, 'p': 0, 't': 982450}, {'x': 160, 'y': 71, 'p': 0, 't': 1455630},
                       {'x': 110, 'y': 119, 'p': 0, 't': 1551870}, {'x': 66, 'y': 143, 'p': 0, 't': 1599985},
                       {'x': 3, 'y': 214, 'p': 0, 't': 1738335}, {'x': 474, 'y': 293, 'p': 1, 't': 1853820},
                       {'x': 400, 'y': 497, 'p': 0, 't': 1863645}, {'x': 87, 'y': 321, 'p': 0, 't': 1906750},
                       {'x': 161, 'y': 408, 'p': 1, 't': 2051110}, {'x': 207, 'y': 408, 'p': 1, 't': 2067155},
                       {'x': 231, 'y': 413, 'p': 1, 't': 2115275}, {'x': 235, 'y': 446, 'p': 0, 't': 2203495},
                       {'x': 285, 'y': 497, 'p': 1, 't': 2229560}, {'x': 286, 'y': 499, 'p': 1, 't': 2233565},
                       {'x': 408, 'y': 367, 'p': 0, 't': 2498230}, {'x': 405, 'y': 365, 'p': 0, 't': 2502235},
                       {'x': 404, 'y': 317, 'p': 1, 't': 2518275}, {'x': 567, 'y': 189, 'p': 0, 't': 2855120},
                       {'x': 532, 'y': 144, 'p': 1, 't': 2907245}, {'x': 537, 'y': 135, 'p': 0, 't': 2963390},
                       {'x': 522, 'y': 116, 'p': 0, 't': 3001485}, {'x': 459, 'y': 75, 'p': 1, 't': 3053615},
                       {'x': 435, 'y': 32, 'p': 1, 't': 3101735}, {'x': 74, 'y': 192, 'p': 0, 't': 3151055},
                       {'x': 416, 'y': 0, 'p': 1, 't': 3153865}, {'x': 353, 'y': 109, 'p': 1, 't': 3292205},
                       {'x': 355, 'y': 110, 'p': 1, 't': 3294215}, {'x': 31, 'y': 307, 'p': 1, 't': 3615615},
                       {'x': 177, 'y': 261, 'p': 1, 't': 3619020}, {'x': 192, 'y': 285, 'p': 1, 't': 3645085},
                       {'x': 157, 'y': 326, 'p': 0, 't': 3739320}, {'x': 41, 'y': 445, 'p': 0, 't': 3971900},
                       {'x': 357, 'y': 491, 'p': 0, 't': 3980725}, {'x': 42, 'y': 479, 'p': 1, 't': 4034055},
                       {'x': 76, 'y': 216, 'p': 0, 't': 4040475}, {'x': 51, 'y': 484, 'p': 1, 't': 4044080},
                       {'x': 54, 'y': 492, 'p': 0, 't': 4082180}, {'x': 99, 'y': 441, 'p': 1, 't': 4110245},
                       {'x': 107, 'y': 440, 'p': 0, 't': 4188445}, {'x': 119, 'y': 398, 'p': 0, 't': 4239170},
                       {'x': 158, 'y': 409, 'p': 0, 't': 4254610}, {'x': 278, 'y': 237, 'p': 0, 't': 4531295},
                       {'x': 363, 'y': 195, 'p': 1, 't': 4621520}, {'x': 455, 'y': 103, 'p': 1, 't': 4805980}]

    # Now open the file and check for its contents
    with open(output_filename, 'r') as f:
        lines = f.readlines()

        # Check number of events
        nr_events = len(lines)
        assert nr_events == 790840

        # Check events :
        for idx in range(0, len(idx_to_check)):
            ev = get_ev_from_line(lines[idx_to_check[idx]])
            assert ev == events_expected[idx], "Error on event nr {}".format(idx_to_check[idx])
