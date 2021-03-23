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
#include "metavision/hal/facilities/i_noise_filter_module.h"
#include "pb_doc_hal.h"

namespace Metavision {

namespace {
void enable_stc_wrapper(I_NoiseFilterModule &self, int threshold) {
    self.enable(I_NoiseFilterModule::Type::STC, threshold);
}

void enable_trail_wrapper(I_NoiseFilterModule &self, int threshold) {
    self.enable(I_NoiseFilterModule::Type::TRAIL, threshold);
}
} // namespace

static DeviceFacilityGetter<I_NoiseFilterModule> getter("get_i_noisefilter_module");

static HALFacilityPythonBinder<I_NoiseFilterModule> bind(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("disable", &I_NoiseFilterModule::disable, pybind_doc_hal["Metavision::I_NoiseFilterModule::disable"])
            .def("enable_stc", &enable_stc_wrapper, "Enables the NoiseFilterModule in the mode STC")
            .def("enable_trail", &enable_trail_wrapper, "Enables the NoiseFilterModule in the mode Trail");
    },
    "I_NoiseFilterModule", pybind_doc_hal["Metavision::I_NoiseFilterModule"]);
} // namespace Metavision
