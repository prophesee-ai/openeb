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


def pytestcase_metavision_hal_has_i_trigger_in_module():
    metavision_hal_members_name = get_class_member_names(metavision_hal)
    assert "I_TriggerIn" in metavision_hal_members_name


def pytestcase_metavision_hal_has_digital_crop_interface():
    trigger_in_member_names = get_class_member_names(metavision_hal.I_TriggerIn)

    assert "enable" in trigger_in_member_names
    assert "disable" in trigger_in_member_names
    assert "is_enabled" in trigger_in_member_names
    assert "get_available_channels" in trigger_in_member_names
    assert "Channel" in trigger_in_member_names


def pytestcase_metavision_hal_i_trigger_in_module_has_channel_enum():
    trigger_in_channel_member_names = get_class_member_names(metavision_hal.I_TriggerIn.Channel)

    assert "MAIN" in trigger_in_channel_member_names
    assert "AUX" in trigger_in_channel_member_names
    assert "LOOPBACK" in trigger_in_channel_member_names


@pytest.fixture
def i_trigger_in():
    dev = metavision_hal.DeviceDiscovery.open("__DummyTest__")
    assert dev
    return dev.get_i_trigger_in()


def pytestcase_should_get_trigger_in_facility(i_trigger_in):
    assert i_trigger_in


def pytestcase_should_have_all_channels_disabled_by_default(i_trigger_in):
    assert i_trigger_in.is_enabled(metavision_hal.I_TriggerIn.Channel.MAIN) == False
    assert i_trigger_in.is_enabled(metavision_hal.I_TriggerIn.Channel.AUX) == False
    assert i_trigger_in.is_enabled(metavision_hal.I_TriggerIn.Channel.LOOPBACK) == False


def pytestcase_should_enable_main_channel(i_trigger_in):
    assert i_trigger_in.enable(metavision_hal.I_TriggerIn.Channel.MAIN)
    assert i_trigger_in.is_enabled(metavision_hal.I_TriggerIn.Channel.MAIN) == True


def pytestcase_should_disable_main_channel(i_trigger_in):
    assert i_trigger_in.disable(metavision_hal.I_TriggerIn.Channel.MAIN)
    assert i_trigger_in.is_enabled(metavision_hal.I_TriggerIn.Channel.MAIN) == False


def pytestcase_should_have_correct_channel_map(i_trigger_in):
    assert i_trigger_in.get_available_channels() == {
        metavision_hal.I_TriggerIn.Channel.MAIN: 0, metavision_hal.I_TriggerIn.Channel.AUX: 1, metavision_hal.
        I_TriggerIn.Channel.LOOPBACK: 2}
