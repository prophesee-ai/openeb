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


def pytestcase_metavision_hal_has_event_trail_filter_module():
    metavision_hal_members_name = get_class_member_names(metavision_hal)
    assert "I_EventTrailFilterModule" in metavision_hal_members_name


def pytestcase_metavision_hal_has_event_trail_filter_interface():
    event_trail_filter_module_member_names = get_class_member_names(metavision_hal.I_EventTrailFilterModule)

    assert "enable" in event_trail_filter_module_member_names
    assert "is_enabled" in event_trail_filter_module_member_names
    assert "get_available_types" in event_trail_filter_module_member_names
    assert "get_type" in event_trail_filter_module_member_names
    assert "set_type" in event_trail_filter_module_member_names
    assert "set_threshold" in event_trail_filter_module_member_names
    assert "get_threshold" in event_trail_filter_module_member_names
    assert "get_min_supported_threshold" in event_trail_filter_module_member_names
    assert "get_max_supported_threshold" in event_trail_filter_module_member_names
    assert "Type" in event_trail_filter_module_member_names


def pytestcase_metavision_hal_i_event_trail_filter_module_has_type_enum():
    event_trail_filter_module_type_member_names = get_class_member_names(metavision_hal.I_EventTrailFilterModule.Type)

    assert "TRAIL" in event_trail_filter_module_type_member_names
    assert "STC_CUT_TRAIL" in event_trail_filter_module_type_member_names
    assert "STC_KEEP_TRAIL" in event_trail_filter_module_type_member_names


@pytest.fixture
def i_event_trail_filter_module():
    dev = metavision_hal.DeviceDiscovery.open("__DummyTest__")
    assert dev
    return dev.get_i_event_trail_filter_module()


def pytestcase_should_get_event_trail_filter_module_facility(i_event_trail_filter_module):
    assert i_event_trail_filter_module


def pytestcase_event_trail_filter_module_should_be_disabled_by_default(i_event_trail_filter_module):
    assert i_event_trail_filter_module.is_enabled() == False


def pytestcase_event_trail_filter_module_should_enable(i_event_trail_filter_module):
    assert i_event_trail_filter_module.enable(True)
    assert i_event_trail_filter_module.is_enabled() == True


def pytestcase_event_trail_filter_module_should_disable(i_event_trail_filter_module):
    assert i_event_trail_filter_module.enable(False)
    assert i_event_trail_filter_module.is_enabled() == False


def pytestcase_event_trail_filter_module_should_have_correct_available_types(i_event_trail_filter_module):
    assert i_event_trail_filter_module.get_available_types() == {
        metavision_hal.I_EventTrailFilterModule.Type.TRAIL, metavision_hal.I_EventTrailFilterModule.Type.STC_CUT_TRAIL,
        metavision_hal.I_EventTrailFilterModule.Type.STC_KEEP_TRAIL}


def pytestcase_event_trail_filter_module_should_have_correct_type_by_default(i_event_trail_filter_module):
    assert i_event_trail_filter_module.get_type() == metavision_hal.I_EventTrailFilterModule.Type.TRAIL


def pytestcase_event_trail_filter_module_should_set_type(i_event_trail_filter_module):
    assert i_event_trail_filter_module.set_type(metavision_hal.I_EventTrailFilterModule.Type.STC_CUT_TRAIL)
    assert i_event_trail_filter_module.get_type() == metavision_hal.I_EventTrailFilterModule.Type.STC_CUT_TRAIL
    assert i_event_trail_filter_module.set_type(metavision_hal.I_EventTrailFilterModule.Type.STC_KEEP_TRAIL)
    assert i_event_trail_filter_module.get_type() == metavision_hal.I_EventTrailFilterModule.Type.STC_KEEP_TRAIL
    assert i_event_trail_filter_module.set_type(metavision_hal.I_EventTrailFilterModule.Type.TRAIL)
    assert i_event_trail_filter_module.get_type() == metavision_hal.I_EventTrailFilterModule.Type.TRAIL


def pytestcase_event_trail_filter_module_should_have_correct_threshold_by_default(i_event_trail_filter_module):
    assert i_event_trail_filter_module.get_threshold() == 1


def pytestcase_event_trail_filter_module_should_set_threshold(i_event_trail_filter_module):
    assert i_event_trail_filter_module.set_threshold(2)
    assert i_event_trail_filter_module.get_threshold() == 2


def pytestcase_event_trail_filter_module_should_have_correct_min_supported_threshold(i_event_trail_filter_module):
    assert i_event_trail_filter_module.get_min_supported_threshold() == 0


def pytestcase_event_trail_filter_module_should_have_correct_max_supported_threshold(i_event_trail_filter_module):
    assert i_event_trail_filter_module.get_max_supported_threshold() == 1000
