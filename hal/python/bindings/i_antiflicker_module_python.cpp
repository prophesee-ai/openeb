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
            .def("enable", &I_AntiFlickerModule::enable, py::arg("b"),
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::enable"])
            .def("is_enabled", &I_AntiFlickerModule::is_enabled,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::is_enabled"])
            .def("set_frequency_band", &I_AntiFlickerModule::set_frequency_band, py::arg("min_freq"),
                 py::arg("max_freq"), pybind_doc_hal["Metavision::I_AntiFlickerModule::set_frequency_band"])
            .def("get_min_supported_frequency", &I_AntiFlickerModule::get_min_supported_frequency,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_min_supported_frequency"])
            .def("get_max_supported_frequency", &I_AntiFlickerModule::get_max_supported_frequency,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_max_supported_frequency"])
            .def("get_band_low_frequency", &I_AntiFlickerModule::get_band_low_frequency,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_band_low_frequency"])
            .def("get_band_high_frequency", &I_AntiFlickerModule::get_band_high_frequency,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_band_high_frequency"])
            .def("get_frequency_band", &I_AntiFlickerModule::get_frequency_band,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_frequency_band"])
            .def("set_filtering_mode", &I_AntiFlickerModule::set_filtering_mode,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::set_filtering_mode"])
            .def("get_filtering_mode", &I_AntiFlickerModule::get_filtering_mode,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_filtering_mode"])
            .def("set_duty_cycle", &I_AntiFlickerModule::set_duty_cycle, py::arg("percent_activity"),
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::set_duty_cycle"])
            .def("get_duty_cycle", &I_AntiFlickerModule::get_duty_cycle,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_duty_cycle"])
            .def("get_min_supported_duty_cycle", &I_AntiFlickerModule::get_min_supported_duty_cycle,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_min_supported_duty_cycle"])
            .def("get_max_supported_duty_cycle", &I_AntiFlickerModule::get_max_supported_duty_cycle,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_max_supported_duty_cycle"])
            .def("set_start_threshold", &I_AntiFlickerModule::set_start_threshold, py::arg("threshold"),
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::set_start_threshold"])
            .def("get_start_threshold", &I_AntiFlickerModule::get_start_threshold,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_start_threshold"])
            .def("get_min_supported_start_threshold", &I_AntiFlickerModule::get_min_supported_start_threshold,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_min_supported_start_threshold"])
            .def("get_max_supported_start_threshold", &I_AntiFlickerModule::get_max_supported_start_threshold,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_max_supported_start_threshold"])
            .def("set_stop_threshold", &I_AntiFlickerModule::set_stop_threshold, py::arg("threshold"),
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::set_stop_threshold"])
            .def("get_stop_threshold", &I_AntiFlickerModule::get_stop_threshold,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_stop_threshold"])
            .def("get_min_supported_stop_threshold", &I_AntiFlickerModule::get_min_supported_stop_threshold,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_min_supported_stop_threshold"])
            .def("get_max_supported_stop_threshold", &I_AntiFlickerModule::get_max_supported_stop_threshold,
                 pybind_doc_hal["Metavision::I_AntiFlickerModule::get_max_supported_stop_threshold"]);

        py::enum_<I_AntiFlickerModule::AntiFlickerMode>(class_binding, "AntiFlickerMode", py::module_local())
            .value("BandStop", I_AntiFlickerModule::AntiFlickerMode::BAND_STOP)
            .value("BandPass", I_AntiFlickerModule::AntiFlickerMode::BAND_PASS);
    },
    "I_AntiFlickerModule", pybind_doc_hal["Metavision::I_AntiFlickerModule"]);
} // namespace Metavision
