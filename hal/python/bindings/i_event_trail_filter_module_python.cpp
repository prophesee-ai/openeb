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
#include "metavision/hal/facilities/i_event_trail_filter_module.h"
#include "pb_doc_hal.h"

namespace Metavision {

static DeviceFacilityGetter<I_EventTrailFilterModule> getter("get_i_event_trail_filter_module");

static HALFacilityPythonBinder<I_EventTrailFilterModule> bind(
    [](auto &module, auto &class_binding) {
        py::enum_<I_EventTrailFilterModule::Type>(class_binding, "Type", py::arithmetic())
            .value("TRAIL", I_EventTrailFilterModule::Type::TRAIL)
            .value("STC_CUT_TRAIL", I_EventTrailFilterModule::Type::STC_CUT_TRAIL)
            .value("STC_KEEP_TRAIL", I_EventTrailFilterModule::Type::STC_KEEP_TRAIL);

        class_binding
            .def("enable", &I_EventTrailFilterModule::enable, py::arg("state"),
                 pybind_doc_hal["Metavision::I_EventTrailFilterModule::enable"])
            .def("is_enabled", &I_EventTrailFilterModule::is_enabled,
                 pybind_doc_hal["Metavision::I_EventTrailFilterModule::is_enabled"])
            .def("get_available_types", &I_EventTrailFilterModule::get_available_types,
                 pybind_doc_hal["Metavision::I_EventTrailFilterModule::get_available_types"])
            .def("get_type", &I_EventTrailFilterModule::get_type,
                 pybind_doc_hal["Metavision::I_EventTrailFilterModule::get_type"])
            .def("set_type", &I_EventTrailFilterModule::set_type, py::arg("type"),
                 pybind_doc_hal["Metavision::I_EventTrailFilterModule::set_type"])
            .def("set_threshold", &I_EventTrailFilterModule::set_threshold, py::arg("threshold"),
                 pybind_doc_hal["Metavision::I_EventTrailFilterModule::set_threshold"])
            .def("get_threshold", &I_EventTrailFilterModule::get_threshold,
                 pybind_doc_hal["Metavision::I_EventTrailFilterModule::get_threshold"])
            .def("get_min_supported_threshold", &I_EventTrailFilterModule::get_min_supported_threshold,
                 pybind_doc_hal["Metavision::I_EventTrailFilterModule::get_min_supported_threshold"])
            .def("get_max_supported_threshold", &I_EventTrailFilterModule::get_max_supported_threshold,
                 pybind_doc_hal["Metavision::I_EventTrailFilterModule::get_max_supported_threshold"]);
    },
    "I_EventTrailFilterModule", pybind_doc_hal["Metavision::I_EventTrailFilterModule"]);
} // namespace Metavision
