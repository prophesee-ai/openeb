# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.import inspect

import metavision_hal
import pytest
import inspect


def get_class_member_names(class_type):
    return [m[0] for m in inspect.getmembers(class_type)]


def pytestcase_metavision_hal_has_digital_crop_module():
    metavision_hal_members_name = get_class_member_names(metavision_hal)
    assert "I_DigitalCrop" in metavision_hal_members_name


def pytestcase_metavision_hal_has_digital_crop_interface():
    digital_crop_member_names = get_class_member_names(metavision_hal.I_DigitalCrop)

    assert "enable" in digital_crop_member_names
    assert "is_enabled" in digital_crop_member_names
    assert "set_window_region" in digital_crop_member_names
    assert "get_window_region" in digital_crop_member_names


@pytest.fixture
def i_digital_crop():
    dev = metavision_hal.DeviceDiscovery.open("__DummyTest__")
    assert dev
    return dev.get_i_digital_crop()


def pytestcase_should_get_digital_crop_facility(i_digital_crop):
    assert i_digital_crop


def pytestcase_should_disable_digital_crop_by_default(i_digital_crop):
    assert i_digital_crop.is_enabled() == False


def pytestcase_should_enable_digital_crop(i_digital_crop):
    assert i_digital_crop.enable(True)
    assert i_digital_crop.is_enabled() == True

    assert i_digital_crop.enable(False)
    assert i_digital_crop.is_enabled() == False


def pytestcase_should_set_digital_crop_region(i_digital_crop):
    assert i_digital_crop.set_window_region((1, 2, 3, 4), True)
    assert i_digital_crop.get_window_region() == (1, 2, 3, 4)


def pytestcase_should_raise_on_wrong_crop_region(i_digital_crop):
    with pytest.raises(RuntimeError):
        i_digital_crop.set_window_region((100, 100, 0, 0), True) == False
