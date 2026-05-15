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
#include "metavision/hal/facilities/i_trigger_out.h"
#include "metavision/utils/pybind/deprecation_warning_exception.h"
#include "pb_doc_hal.h"

namespace Metavision {

static DeviceFacilityGetter<I_TriggerOut> getter("get_i_trigger_out");

static HALFacilityPythonBinder<I_TriggerOut> bind(
    [](auto &module, auto &class_binding) {
        class_binding.def("enable", &I_TriggerOut::enable, pybind_doc_hal["Metavision::I_TriggerOut::enable"])
            .def("disable", &I_TriggerOut::disable, pybind_doc_hal["Metavision::I_TriggerOut::disable"])
            .def("set_period", &I_TriggerOut::set_period, py::arg("period_us"),
                 pybind_doc_hal["Metavision::I_TriggerOut::set_period"])
            .def("set_duty_cycle", &I_TriggerOut::set_duty_cycle, py::arg("period_ratio"),
                 pybind_doc_hal["Metavision::I_TriggerOut::set_duty_cycle"]);
    },
    "I_TriggerOut", pybind_doc_hal["Metavision::I_TriggerOut"]);

} // namespace Metavision
