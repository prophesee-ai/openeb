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

#include "hal_python_binder.h"
#include "metavision/hal/facilities/i_antiflicker_module.h"
#include "pb_doc_hal.h"

namespace Metavision {

static DeviceFacilityGetter<I_AntiFlickerModule> getter("get_i_antiflicker_module");

static HALFacilityPythonBinder<I_AntiFlickerModule> bind(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("enable", &I_AntiFlickerModule::enable, pybind_doc_hal["Metavision::I_AntiFlickerModule::enable"])
            .def("disable", &I_AntiFlickerModule::disable, pybind_doc_hal["Metavision::I_AntiFlickerModule::disable"])
            .def("set_frequency", &I_AntiFlickerModule::set_frequency, py::arg("frequency_center"),
                 py::arg("bandwidth"), py::arg("stop"),
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::set_frequency"])
            .def("set_frequency_band", &I_AntiFlickerModule::set_frequency_band, py::arg("min_freq"),
                 py::arg("max_freq"), py::arg("stop"),
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::set_frequency_band"]);
    },
    "I_AntiFlickerModule", pybind_doc_hal["Metavision::I_AntiFlickerModule"]);
} // namespace Metavision
