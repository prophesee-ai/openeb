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

#include <pybind11/operators.h>

#include "hal_python_binder.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/utils/pybind/deprecation_warning_exception.h"
#include "pb_doc_hal.h"

namespace Metavision {

py::dict get_system_info_wrapper(I_HW_Identification &self) {
    auto system_info = self.get_system_info();
    py::dict dictionary;
    for (auto it = system_info.begin(), it_end = system_info.end(); it != it_end; ++it) {
        dictionary[py::str(it->first)] = it->second;
    }
    return dictionary;
}

static DeviceFacilityGetter<I_HW_Identification> getter("get_i_hw_identification");

static HALFacilityPythonBinder<I_HW_Identification> bind(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("get_serial", &I_HW_Identification::get_serial,
                 pybind_doc_hal["Metavision::I_HW_Identification::get_serial"])
            .def("get_sensor_info", &I_HW_Identification::get_sensor_info,
                 pybind_doc_hal["Metavision::I_HW_Identification::get_sensor_info"])
            .def("get_available_data_encoding_formats", &I_HW_Identification::get_available_data_encoding_formats,
                 pybind_doc_hal["Metavision::I_HW_Identification::get_available_data_encoding_formats"])
            .def("get_current_data_encoding_format", &I_HW_Identification::get_current_data_encoding_format,
                 pybind_doc_hal["Metavision::I_HW_Identification::get_current_data_encoding_format"])
            .def("get_integrator", &I_HW_Identification::get_integrator,
                 pybind_doc_hal["Metavision::I_HW_Identification::get_integrator"])
            .def("get_system_info", &get_system_info_wrapper,
                 pybind_doc_hal["Metavision::I_HW_Identification::get_system_info"])
            .def("get_connection_type", &I_HW_Identification::get_connection_type,
                 pybind_doc_hal["Metavision::I_HW_Identification::get_connection_type"])
            .def("get_header", &I_HW_Identification::get_header,
                 pybind_doc_hal["Metavision::I_HW_Identification::get_header"])
            .def("get_device_config_options", &I_HW_Identification::get_device_config_options,
                 pybind_doc_hal["Metavision::I_HW_Identification::get_device_config_options"]);

        py::class_<I_HW_Identification::SensorInfo>(module, "SensorInfo",
                                                    pybind_doc_hal["Metavision::I_HW_Identification::SensorInfo"])
            .def_readonly("major_version", &I_HW_Identification::SensorInfo::major_version_,
                          pybind_doc_hal["Metavision::I_HW_Identification::SensorInfo::major_version_"])
            .def_readonly("minor_version", &I_HW_Identification::SensorInfo::minor_version_,
                          pybind_doc_hal["Metavision::I_HW_Identification::SensorInfo::minor_version_"])
            .def_readonly("name", &I_HW_Identification::SensorInfo::name_,
                          pybind_doc_hal["Metavision::I_HW_Identification::SensorInfo::name_"]);
    },
    "I_HW_Identification", pybind_doc_hal["Metavision::I_HW_Identification"]);

} // namespace Metavision
