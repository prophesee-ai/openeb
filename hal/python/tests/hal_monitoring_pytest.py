# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.import inspect

import inspect
import metavision_hal
import pytest


def get_class_member_names(class_type):
    return [m[0] for m in inspect.getmembers(class_type)]


def pytestcase_metavision_hal_has_monitoring_interface():
    i_monitoring_member_names = get_class_member_names(metavision_hal.I_Monitoring)

    assert "get_temperature" in i_monitoring_member_names
    assert "get_illumination" in i_monitoring_member_names
    assert "get_pixel_dead_time" in i_monitoring_member_names


@pytest.fixture
def i_monitoring():
    dev = metavision_hal.DeviceDiscovery.open("__DummyTest__")
    assert dev
    i_monitoring = dev.get_i_monitoring()
    assert i_monitoring
    return i_monitoring


def pytestcase_should_get_monitoring_facilities(i_monitoring):
    assert i_monitoring


def pytestcase_should_get_temperature_from_monitoring(i_monitoring):
    assert i_monitoring.get_temperature() == 12


def pytestcase_should_get_illuminiation_from_monitoring(i_monitoring):
    assert i_monitoring.get_illumination() == 34


def pytestcase_should_get_pixel_dead_time_from_monitoring(i_monitoring):
    assert i_monitoring.get_pixel_dead_time() == 56
