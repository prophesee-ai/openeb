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
#include "metavision/hal/facilities/i_event_rate_noise_filter_module.h"
#include "pb_doc_hal.h"

namespace Metavision {

static DeviceFacilityGetter<I_EventRateNoiseFilterModule> getter("get_i_event_rate");

static HALFacilityPythonBinder<I_EventRateNoiseFilterModule> bind(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("enable", &I_EventRateNoiseFilterModule::enable, py::arg("enable_filter"),
                 pybind_doc_hal["Metavision::I_EventRateNoiseFilterModule::enable"])
            .def("set_event_rate_threshold", &I_EventRateNoiseFilterModule::set_event_rate_threshold,
                 py::arg("threshold_Kev_s"),
                 pybind_doc_hal["Metavision::I_EventRateNoiseFilterModule::set_event_rate_threshold"])
            .def("get_event_rate_threshold", &I_EventRateNoiseFilterModule::get_event_rate_threshold,
                 pybind_doc_hal["Metavision::I_EventRateNoiseFilterModule::get_event_rate_threshold"]);
    },
    "I_EventRateNoiseFilterModule", pybind_doc_hal["Metavision::I_EventRateNoiseFilterModule"]);

} // namespace Metavision
