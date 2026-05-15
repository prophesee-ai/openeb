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
#include "metavision/hal/facilities/i_digital_crop.h"
#include "pb_doc_hal.h"

namespace Metavision {

static DeviceFacilityGetter<I_DigitalCrop> getter("get_i_digital_crop");

static HALFacilityPythonBinder<I_DigitalCrop> bind_digital_crop(
    [](auto &module, auto &class_binding) {
        class_binding.def("enable", &I_DigitalCrop::enable, pybind_doc_hal["Metavision::I_DigitalCrop::enable"])
            .def("is_enabled", &I_DigitalCrop::is_enabled, pybind_doc_hal["Metavision::I_DigitalCrop::is_enabled"])
            .def("set_window_region", &I_DigitalCrop::set_window_region,
                 pybind_doc_hal["Metavision::I_DigitalCrop::set_window_region"])
            .def("get_window_region", &I_DigitalCrop::get_window_region,
                 pybind_doc_hal["Metavision::I_DigitalCrop::get_window_region"]);
    },
    "I_DigitalCrop", pybind_doc_hal["Metavision::I_DigitalCrop"]);
} // namespace Metavision
