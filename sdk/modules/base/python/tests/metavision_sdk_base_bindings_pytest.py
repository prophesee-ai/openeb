# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import numpy as np
import metavision_sdk_base


def pytestcase_EventCD():
    ev = np.zeros(3, dtype=metavision_sdk_base.EventCD)
    buf = metavision_sdk_base.EventCDBuffer(2)
    np_from_buf = buf.numpy()
    assert np_from_buf.size == 2
    assert ev.dtype == np_from_buf.dtype


def pytestcase_EventExtTrigger():
    ev = np.zeros(3, dtype=metavision_sdk_base.EventExtTrigger)
    buf = metavision_sdk_base.EventExtTriggerBuffer(2)
    np_from_buf = buf.numpy()
    assert np_from_buf.size == 2
    assert ev.dtype == np_from_buf.dtype


def pytestcase_check_SoftwareInfo_exists():
    metavision_sdk_base.SoftwareInfo


def pytestcase_BufferInfo():
    ev_buff = metavision_sdk_base.EventCDBuffer(10)
    buff_info_dict = ev_buff._buffer_info().to_dict()

    assert hex(buff_info_dict["ptr"]) == buff_info_dict["ptr_hex"]
    assert ev_buff._buffer_info().ptr_hex() == buff_info_dict["ptr_hex"]

    assert ev_buff._buffer_info().itemsize == ev_buff.numpy().itemsize
    assert ev_buff._buffer_info().size == ev_buff.numpy().size
    assert ev_buff._buffer_info().ndim == 1
    assert ev_buff._buffer_info().ndim == ev_buff.numpy().ndim
    assert ev_buff._buffer_info().shape[0] == ev_buff.numpy().shape[0]
    assert ev_buff._buffer_info().strides[0] == ev_buff.numpy().strides[0]

    assert ev_buff._buffer_info().ptr_hex() == metavision_sdk_base._buffer_info(ev_buff.numpy()).ptr_hex()
    assert ev_buff._buffer_info().ptr_hex() != metavision_sdk_base._buffer_info(ev_buff.numpy(copy=True)).ptr_hex()


def pytestcase_GenericHeader():
    header = metavision_sdk_base.GenericHeader()
    assert header.empty()
    header.set_field("Test", "Hello")
    assert not header.empty()
    assert header.get_header_map() == {"Test": "Hello"}
    header.set_field("Test", "Hella")
    assert header.get_header_map() == {"Test": "Hella"}
    header.set_field("Hello", "There")
    assert header.get_header_map() == {"Hello": "There", "Test": "Hella"}
    header.remove_field("Hello")
    header.remove_field("Test")
    assert header.empty()
    assert header.get_field("Hello") == ""

    dictio = {"Darth": "Vader", 10: 20, "Yolo": 123}
    header = metavision_sdk_base.GenericHeader(dictio)
    assert header.get_header_map() == {"Darth": "Vader"}

    header.add_date()
    assert len(header.get_header_map()["date"]) > 0
    header.remove_date()
    assert header.get_date() == ""
