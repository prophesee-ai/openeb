# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.import inspect

import os
import pytest
import inspect

import metavision_hal


def pytestcase_hal_psee_plugin_can_open_raw_string_path(dataset_dir):
    device = metavision_hal.DeviceDiscovery.open_raw_file(os.path.join(dataset_dir,
                                                                       "openeb", "gen4_evt3_hand.raw"))
    assert device is not None


def pytestcase_hal_psee_plugin_can_open_raw_pathlib(dataset_dir):
    import pathlib
    device = metavision_hal.DeviceDiscovery.open_raw_file(pathlib.Path(dataset_dir,
                                                                       "openeb", "gen4_evt3_hand.raw"))
    assert device is not None
