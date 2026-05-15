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

#include "sample_digital_event_mask.h"
#include "internal/sample_register_access.h"

// This functionality is currently not available. A placeholder implementation has been provided for now..
// To implement this functionality, one could use the implementation from the official Prophesee plugin
// as a starting point. However, it requires modifications: replacing the use of the regmap with a similar concept
// tailored specifically to handle the Digital Event Mask case.

bool SampleDigitalEventMask::SamplePixelMask::set_mask(uint32_t x, uint32_t y, bool enabled) {
    return true;
};

std::tuple<uint32_t, uint32_t, bool> SampleDigitalEventMask::SamplePixelMask::get_mask() const {
    return std::make_tuple(0, 0, false);
};

const std::vector<SampleDigitalEventMask::I_PixelMaskPtr> &SampleDigitalEventMask::get_pixel_masks() const {
    return pixel_masks_;
};
