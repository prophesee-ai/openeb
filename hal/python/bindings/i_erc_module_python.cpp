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
#include "metavision/hal/facilities/i_erc_module.h"
#include "metavision/utils/pybind/deprecation_warning_exception.h"
#include "pb_doc_hal.h"

namespace Metavision {

static DeviceFacilityGetter<I_ErcModule> getter("get_i_erc_module");

static HALFacilityPythonBinder<I_ErcModule> bind(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("enable", &I_ErcModule::enable, py::arg("b"), pybind_doc_hal["Metavision::I_ErcModule::enable"])
            .def("is_enabled", &I_ErcModule::is_enabled, pybind_doc_hal["Metavision::I_ErcModule::is_enabled"])
            .def("erc_from_file", &I_ErcModule::erc_from_file)
            .def("set_cd_event_rate", &I_ErcModule::set_cd_event_rate, py::arg("events_per_sec"),
                 pybind_doc_hal["Metavision::I_ErcModule::set_cd_event_rate"])
            .def("get_cd_event_rate", &I_ErcModule::get_cd_event_rate,
                 pybind_doc_hal["Metavision::I_ErcModule::get_cd_event_rate"])
            .def("get_min_supported_cd_event_rate", &I_ErcModule::get_min_supported_cd_event_rate,
                 pybind_doc_hal["Metavision::I_ErcModule::get_min_supported_cd_event_rate"])
            .def("get_max_supported_cd_event_rate", &I_ErcModule::get_max_supported_cd_event_rate,
                 pybind_doc_hal["Metavision::I_ErcModule::get_max_supported_cd_event_rate"])
            .def("get_count_period", &I_ErcModule::get_count_period,
                 pybind_doc_hal["Metavision::I_ErcModule::get_count_period"])
            .def("set_cd_event_count", &I_ErcModule::set_cd_event_count, py::arg("event_count"),
                 pybind_doc_hal["Metavision::I_ErcModule::set_cd_event_count"])
            .def("get_cd_event_count", &I_ErcModule::get_cd_event_count,
                 pybind_doc_hal["Metavision::I_ErcModule::get_cd_event_count"])
            .def("get_min_supported_cd_event_count", &I_ErcModule::get_min_supported_cd_event_count,
                 pybind_doc_hal["Metavision::I_ErcModule::get_min_supported_cd_event_count"])
            .def("get_max_supported_cd_event_count", &I_ErcModule::get_max_supported_cd_event_count,
                 pybind_doc_hal["Metavision::I_ErcModule::get_max_supported_cd_event_count"]);
    },
    "I_ErcModule", pybind_doc_hal["Metavision::I_ErcModule"]);

} // namespace Metavision
