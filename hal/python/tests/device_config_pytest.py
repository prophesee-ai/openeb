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


def pytestcase_metavision_hal_has_device_config_class():
    metavision_hal_members_name = get_class_member_names(metavision_hal)
    assert "DeviceConfig" in metavision_hal_members_name


def pytestcase_metavision_hal_has_device_config_interface():
    device_config_member_names = get_class_member_names(metavision_hal.DeviceConfig)

    assert "get_format_key" in device_config_member_names
    assert "format" in device_config_member_names
    assert "set_format" in device_config_member_names
    assert "get_biases_range_check_bypass_key" in device_config_member_names
    assert "biases_range_check_bypass" in device_config_member_names
    assert "enable_biases_range_check_bypass" in device_config_member_names
    assert "get" in device_config_member_names
    assert "get_bool" in device_config_member_names
    assert "get_int" in device_config_member_names
    assert "get_double" in device_config_member_names
    assert "set" in device_config_member_names


def pytestcase_metavision_hal_has_device_config_option_class():
    metavision_hal_members_name = get_class_member_names(metavision_hal)
    assert "DeviceConfigOption" in metavision_hal_members_name

    device_config_option_member_names = get_class_member_names(metavision_hal.DeviceConfigOption)
    assert "Type" in device_config_option_member_names


def pytestcase_metavision_hal_has_device_config_option_interface():
    device_config_option_member_names = get_class_member_names(metavision_hal.DeviceConfigOption)

    assert "get_range" in device_config_option_member_names
    assert "get_values" in device_config_option_member_names
    assert "get_default_value" in device_config_option_member_names
    assert "type" in device_config_option_member_names

    device_config_option_type_member_names = get_class_member_names(metavision_hal.DeviceConfigOption.Type)

    assert "INVALID" in device_config_option_type_member_names
    assert "BOOLEAN" in device_config_option_type_member_names
    assert "INT" in device_config_option_type_member_names
    assert "DOUBLE" in device_config_option_type_member_names
    assert "STRING" in device_config_option_type_member_names


def pytestcase_metavision_hal_device_config_option_should_have_invalid_type():
    assert metavision_hal.DeviceConfigOption().type() == metavision_hal.DeviceConfigOption.Type.INVALID


def pytestcase_metavision_hal_device_config_option_should_have_boolean_type():
    assert metavision_hal.DeviceConfigOption(True).type() == metavision_hal.DeviceConfigOption.Type.BOOLEAN
    assert metavision_hal.DeviceConfigOption(False).type() == metavision_hal.DeviceConfigOption.Type.BOOLEAN


def pytestcase_metavision_hal_device_config_option_should_get_default_boolean_value():
    assert metavision_hal.DeviceConfigOption(True).get_default_value() == True
    assert metavision_hal.DeviceConfigOption(False).get_default_value() == False


def pytestcase_metavision_hal_device_config_option_should_raise_error_on_get_boolean_range():
    with pytest.raises(RuntimeError):
        metavision_hal.DeviceConfigOption(True).get_range()
        metavision_hal.DeviceConfigOption(False).get_range()


def pytestcase_metavision_hal_device_config_option_should_raise_error_on_get_boolean_values():
    with pytest.raises(RuntimeError):
        metavision_hal.DeviceConfigOption(True).get_values()
        metavision_hal.DeviceConfigOption(False).get_values()


def pytestcase_metavision_hal_device_config_option_should_have_int_type():
    assert metavision_hal.DeviceConfigOption(3, 5, 4).type() == metavision_hal.DeviceConfigOption.Type.INT


def pytestcase_metavision_hal_device_config_option_should_raise_error_on_wrong_int_init():
    with pytest.raises(RuntimeError):
        metavision_hal.DeviceConfigOption(3, 5, 12)


def pytestcase_metavision_hal_device_config_option_should_get_default_int_type():
    assert metavision_hal.DeviceConfigOption(3, 5, 4).get_default_value() == 4


def pytestcase_metavision_hal_device_config_option_should_get_int_range():
    assert metavision_hal.DeviceConfigOption(3, 5, 4).get_range() == (3, 5)


def pytestcase_metavision_hal_device_config_option_should_raise_error_on_get_int_values():
    with pytest.raises(RuntimeError):
        metavision_hal.DeviceConfigOption(3, 5, 4).get_values()


def pytestcase_metavision_hal_device_config_option_should_have_double_type():
    assert metavision_hal.DeviceConfigOption(3.0, 5.0, 4.0).type() == metavision_hal.DeviceConfigOption.Type.DOUBLE


def pytestcase_metavision_hal_device_config_option_should_raise_error_on_wrong_double_init():
    with pytest.raises(RuntimeError):
        metavision_hal.DeviceConfigOption(3.0, 5.0, 12.0)


def pytestcase_metavision_hal_device_config_option_should_get_default_double_type():
    assert metavision_hal.DeviceConfigOption(3.0, 5.0, 4.0).get_default_value() == 4.0


def pytestcase_metavision_hal_device_config_option_should_get_double_range():
    assert metavision_hal.DeviceConfigOption(3.0, 5.0, 4.0).get_range() == (3.0, 5.0)


def pytestcase_metavision_hal_device_config_option_should_raise_error_on_get_double_values():
    with pytest.raises(RuntimeError):
        metavision_hal.DeviceConfigOption(3.0, 5.0, 4.0).get_values()


def pytestcase_metavision_hal_device_config_option_should_have_string_type():
    assert metavision_hal.DeviceConfigOption(["a", "b"], "a").type() == metavision_hal.DeviceConfigOption.Type.STRING


def pytestcase_metavision_hal_device_config_option_should_raise_error_on_wrong_string_init():
    with pytest.raises(RuntimeError):
        metavision_hal.DeviceConfigOption(["a", "b"], "c")


def pytestcase_metavision_hal_device_config_option_should_get_default_string_value():
    assert metavision_hal.DeviceConfigOption(["a", "b"], "a").get_default_value() == "a"


def pytestcase_metavision_hal_device_config_option_should_raise_error_on_get_range():
    with pytest.raises(RuntimeError):
        metavision_hal.DeviceConfigOption(["a", "b"], "a").get_range()


def pytestcase_metavision_hal_device_config_option_should_get_string_values():
    assert metavision_hal.DeviceConfigOption(["a", "b"], "a").get_values() == ["a", "b"]


def pytestcase_metavision_hal_device_config_should_set_format():
    opt = metavision_hal.DeviceConfig()
    opt.set_format("blub")
    assert opt.format() == "blub"
    assert opt.get(metavision_hal.DeviceConfig.get_format_key()) == "blub"


def pytestcase_metavision_hal_device_config_should_set_format_via_key():
    opt = metavision_hal.DeviceConfig()
    opt.set(metavision_hal.DeviceConfig.get_format_key(), "blub")
    assert opt.format() == "blub"
    assert opt.get(metavision_hal.DeviceConfig.get_format_key()) == "blub"


def pytestcase_metavision_hal_device_config_should_enable_biases_range_check_bypass():
    opt = metavision_hal.DeviceConfig()
    opt.enable_biases_range_check_bypass(True)
    assert opt.biases_range_check_bypass() == True
    assert opt.get_bool(metavision_hal.DeviceConfig.get_biases_range_check_bypass_key()) == True


def pytestcase_metavision_hal_device_config_should_enable_biases_range_check_bypass_via_key():
    opt = metavision_hal.DeviceConfig()
    opt.set(metavision_hal.DeviceConfig.get_biases_range_check_bypass_key(), True)
    assert opt.biases_range_check_bypass() == True
    assert opt.get_bool(metavision_hal.DeviceConfig.get_biases_range_check_bypass_key()) == True
    opt.set(metavision_hal.DeviceConfig.get_biases_range_check_bypass_key(), "true")
    assert opt.biases_range_check_bypass() == True
    assert opt.get_bool(metavision_hal.DeviceConfig.get_biases_range_check_bypass_key()) == True
