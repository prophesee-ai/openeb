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
#include "metavision/hal//facilities/i_digital_event_mask.h"
#include "pb_doc_hal.h"

namespace Metavision {

static DeviceFacilityGetter<I_DigitalEventMask> getter("get_i_digital_event_mask");

static HALFacilityPythonBinder<I_DigitalEventMask> bind_digital_event_mask(
    [](auto &module, auto &class_binding) {
        class_binding.def("get_pixel_masks", &I_DigitalEventMask::get_pixel_masks,
                          pybind_doc_hal["Metavision::I_DigitalEventMask::get_pixel_masks"]);
    },
    "I_DigitalEventMask", pybind_doc_hal["Metavision::I_DigitalEventMask"]);

static HALFacilityPythonBinder<I_DigitalEventMask::I_PixelMask> bind_pixel_mask(
    [](auto &module, auto &class_binding) {
        class_binding
            .def("set_mask", &I_DigitalEventMask::I_PixelMask::set_mask,
                 pybind_doc_hal["Metavision::I_DigitalEventMask::I_PixelMask::set_mask"])
            .def("get_mask", &I_DigitalEventMask::I_PixelMask::get_mask,
                 pybind_doc_hal["Metavision::I_DigitalEventMask::I_PixelMask::get_mask"]);
    },
    "I_PixelMask", pybind_doc_hal["Metavision::I_DigitalEventMask::I_PixelMask"]);
} // namespace Metavision
