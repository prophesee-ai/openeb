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
#include "metavision/hal/facilities/i_event_rate_activity_filter_module.h"
#include "pb_doc_hal.h"

namespace Metavision {

static DeviceFacilityGetter<I_EventRateActivityFilterModule> getter("get_i_event_rate");

static HALFacilityPythonBinder<I_EventRateActivityFilterModule> bind(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("enable", &I_EventRateActivityFilterModule::enable, py::arg("enable_filter"),
                 pybind_doc_hal["Metavision::I_EventRateActivityFilterModule::enable"])
            .def("is_enabled", &I_EventRateActivityFilterModule::is_enabled,
                 pybind_doc_hal["Metavision::I_EventRateActivityFilterModule::is_enabled"])
            .def("get_thresholds", &I_EventRateActivityFilterModule::get_thresholds,
                 pybind_doc_hal["Metavision::I_EventRateActivityFilterModule::get_thresholds"])

            .def("set_thresholds", &I_EventRateActivityFilterModule::set_thresholds, py::arg("threshold_ev_s"),
                 pybind_doc_hal["Metavision::I_EventRateActivityFilterModule::set_thresholds"])

            .def("is_thresholds_supported", &I_EventRateActivityFilterModule::is_thresholds_supported,
                 pybind_doc_hal["Metavision::I_EventRateActivityFilterModule::is_thresholds_supported"])

            .def("get_min_supported_thresholds", &I_EventRateActivityFilterModule::get_min_supported_thresholds,
                 pybind_doc_hal["Metavision::I_EventRateActivityFilterModule::get_min_supported_thresholds"])

            .def("get_max_supported_thresholds", &I_EventRateActivityFilterModule::get_max_supported_thresholds,
                 pybind_doc_hal["Metavision::I_EventRateActivityFilterModule::get_max_supported_thresholds"]);

        py::class_<I_EventRateActivityFilterModule::thresholds>(
            class_binding, "thresholds", pybind_doc_hal["Metavision::I_EventRateActivityFilterModule::thresholds"])
            .def_readwrite("lower_bound_start", &I_EventRateActivityFilterModule::thresholds::lower_bound_start)
            .def_readwrite("lower_bound_stop", &I_EventRateActivityFilterModule::thresholds::lower_bound_stop)
            .def_readwrite("upper_bound_start", &I_EventRateActivityFilterModule::thresholds::upper_bound_start)
            .def_readwrite("upper_bound_stop", &I_EventRateActivityFilterModule::thresholds::upper_bound_stop);
    },
    "I_EventRateActivityFilterModule", pybind_doc_hal["Metavision::I_EventRateActivityFilterModule"]);

} // namespace Metavision
