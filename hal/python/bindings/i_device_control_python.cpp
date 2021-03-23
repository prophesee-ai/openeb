/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#include <pybind11/pybind11.h>

#include "hal_python_binder.h"
#include "metavision/hal/facilities/i_device_control.h"
#include "pb_doc_hal.h"

namespace Metavision {

static DeviceFacilityGetter<I_DeviceControl> getter("get_i_device_control");

static HALFacilityPythonBinder<I_DeviceControl> bind(
    [](auto &module, auto &class_binding) {
        class_binding.def("reset", &I_DeviceControl::reset, pybind_doc_hal["Metavision::I_DeviceControl::reset"])
            .def("start", &I_DeviceControl::start, pybind_doc_hal["Metavision::I_DeviceControl::start"])
            .def("stop", &I_DeviceControl::stop, pybind_doc_hal["Metavision::I_DeviceControl::stop"])
            .def("set_mode_standalone", &I_DeviceControl::set_mode_standalone,
                 pybind_doc_hal["Metavision::I_DeviceControl::set_mode_standalone"])
            .def("set_mode_master", &I_DeviceControl::set_mode_master,
                 pybind_doc_hal["Metavision::I_DeviceControl::set_mode_master"])
            .def("set_mode_slave", &I_DeviceControl::set_mode_slave,
                 pybind_doc_hal["Metavision::I_DeviceControl::set_mode_slave"]);
    },
    "I_DeviceControl", pybind_doc_hal["Metavision::I_DeviceControl"]);

} // namespace Metavision
