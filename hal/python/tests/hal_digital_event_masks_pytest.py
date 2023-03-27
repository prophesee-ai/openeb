# Copyright (c) Prophesee S.A.
#
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


def pytestcase_metavision_hal_has_digital_event_mask_interface():
    digital_event_mask_member_names = get_class_member_names(metavision_hal.I_DigitalEventMask)

    assert "get_pixel_masks" in digital_event_mask_member_names


def pytestcase_metavision_hal_has_pixel_mask_interface():
    pixel_mask_members = get_class_member_names(metavision_hal.I_PixelMask)

    assert "set_mask" in pixel_mask_members
    assert "get_mask" in pixel_mask_members


@pytest.fixture
def digital_event_mask():
    dev = metavision_hal.DeviceDiscovery.open("__DummyTest__")
    assert dev
    return dev.get_i_digital_event_mask()


def pytestcase_should_get_facilities(digital_event_mask):
    assert digital_event_mask


def pytestcase_should_get_pixel_masks_from_facilities(digital_event_mask):
    masks = digital_event_mask.get_pixel_masks()
    assert len(masks) > 0


def pytestcase_should_get_and_set_pixel_mask(digital_event_mask):
    pixel_mask = digital_event_mask.get_pixel_masks()[0]
    pixel_mask.set_mask(12, 34, True)
    assert pixel_mask.get_mask() == (12, 34, True)
