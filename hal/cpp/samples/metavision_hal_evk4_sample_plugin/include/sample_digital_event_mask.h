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

#ifndef METAVISION_HAL_SAMPLE_DIGITAL_EVENT_MASK_H
#define METAVISION_HAL_SAMPLE_DIGITAL_EVENT_MASK_H

#include <metavision/hal/facilities/i_digital_event_mask.h>


/// @brief Interface for Digital Event Mask commands.
///
/// This class is the implementation of HAL's facility @ref Metavision::I_DigitalEventMask
class SampleDigitalEventMask : public Metavision::I_DigitalEventMask {
public:
    class SamplePixelMask : public Metavision::I_DigitalEventMask::I_PixelMask {
    public:
        bool set_mask(uint32_t x, uint32_t y, bool enabled) override final;
        std::tuple<uint32_t, uint32_t, bool> get_mask() const override final;
    };

    std::vector<I_PixelMaskPtr> pixel_masks_;
    const std::vector<I_PixelMaskPtr> &get_pixel_masks() const override final;
};

#endif // METAVISION_HAL_SAMPLE_DIGITAL_EVENT_MASK_H
