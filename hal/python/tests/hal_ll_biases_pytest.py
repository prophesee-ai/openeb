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


def pytestcase_metavision_hal_has_ll_biases_module():
    metavision_hal_members_name = get_class_member_names(metavision_hal)
    assert "I_LL_Biases" in metavision_hal_members_name


def pytestcase_metavision_hal_has_ll_biases_interface():
    ll_biases_member_names = get_class_member_names(metavision_hal.I_LL_Biases)

    assert "set" in ll_biases_member_names
    assert "get" in ll_biases_member_names
    assert "get_bias_info" in ll_biases_member_names
    assert "get_all_biases" in ll_biases_member_names


def pytestcase_metavision_hal_has_ll_bias_info_class():
    metavision_hal_members_name = get_class_member_names(metavision_hal)
    assert "LL_Bias_Info" in metavision_hal_members_name


def pytestcase_metavision_hal_has_ll_bias_info_interface():
    bias_info_member_names = get_class_member_names(metavision_hal.LL_Bias_Info)

    assert "get_description" in bias_info_member_names
    assert "get_category" in bias_info_member_names
    assert "get_bias_range" in bias_info_member_names
    assert "get_bias_recommended_range" in bias_info_member_names
    assert "get_bias_allowed_range" in bias_info_member_names
    assert "is_modifiable" in bias_info_member_names


@pytest.fixture
def i_ll_biases():
    dev = metavision_hal.DeviceDiscovery.open("__DummyTest__")
    assert dev
    return dev.get_i_ll_biases()


def pytestcase_should_get_i_ll_biases(i_ll_biases):
    assert i_ll_biases


def pytestcase_should_get_default_bias(i_ll_biases):
    assert i_ll_biases.get("dummy") == 1


def pytestcase_should_set_default_bias(i_ll_biases):
    assert i_ll_biases.set("dummy", 3) == True
    assert i_ll_biases.get("dummy") == 3


def pytestcase_should_raise_on_get_unknown_bias(i_ll_biases):
    with pytest.raises(RuntimeError):
        i_ll_biases.get("unknown")


def pytestcase_should_not_set_unknown_bias(i_ll_biases):
    with pytest.raises(RuntimeError):
        i_ll_biases.set("unknown", 3)


def pytestcase_should_get_default_bias_info(i_ll_biases):
    bias = i_ll_biases.get_bias_info("dummy")
    assert bias.get_description() == "dummy desc"
    assert bias.get_category() == "dummy category"
    assert bias.get_bias_range()[0] == -10
    assert bias.get_bias_range()[1] == 10
    assert bias.get_bias_recommended_range()[0] == -10
    assert bias.get_bias_recommended_range()[1] == 10
    assert bias.get_bias_allowed_range()[0] == -10
    assert bias.get_bias_allowed_range()[1] == 10


def pytestcase_should_raise_on_get_unknown_bias_info(i_ll_biases):
    with pytest.raises(RuntimeError):
        bias = i_ll_biases.get_bias_info("unknown")
