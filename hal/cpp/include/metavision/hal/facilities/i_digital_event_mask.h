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

#ifndef METAVISION_HAL_FACILITY_DIGITAL_EVENT_MASK
#define METAVISION_HAL_FACILITY_DIGITAL_EVENT_MASK

#include <vector>
#include <tuple>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Interface for Digital Event Mask commands.
class I_DigitalEventMask : public I_RegistrableFacility<I_DigitalEventMask> {
public:
    /// @brief Interface for Digital Pixel Mask commands.
    /// @note Each mask can be associated with a pixel and enabled/disabled on demand
    class I_PixelMask {
    public:
        /// @brief default destructor
        virtual ~I_PixelMask() = default;

        /// @brief Set coordinate for the pixel to be masked
        /// @param x the pixel horizontal coordinate
        /// @param y the pixel vertical coordinate
        /// @param enabled when true, the mask will prevent pixel from generating CD event
        /// @return true on success
        virtual bool set_mask(uint32_t x, uint32_t y, bool enabled) = 0;

        /// @brief Get the current mask settings
        /// @return a tuple that packs the X and Y pixel coordinates and a boolean defining whether the mask is enabled
        virtual std::tuple<uint32_t, uint32_t, bool> get_mask() const = 0;

    protected:
        /// @brief default constructor
        I_PixelMask() = default;
    };

    /// @brief Type of a pointer to a pixel mask
    using I_PixelMaskPtr = std::shared_ptr<I_PixelMask>;

    /// @brief default destructor
    virtual ~I_DigitalEventMask() = default;

    /// @brief Get all available masks.
    /// @return a list of pixel masks pointer
    virtual const std::vector<I_PixelMaskPtr> &get_pixel_masks() const = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_FACILITY_DIGITAL_EVENT_MASK
