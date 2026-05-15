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

#ifndef METAVISION_HAL_I_ROI_PIXEL_MASK_H
#define METAVISION_HAL_I_ROI_PIXEL_MASK_H

#include <cstdint>
#include <string>
#include <vector>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Interface facility for ROI (Region Of Interest) pixel mask
class I_RoiPixelMask : public I_RegistrableFacility<I_RoiPixelMask> {
public:
    /// @brief Set individual pixel ROI mask
    /// The pixel coordinates are stored in the driver and will be sent to the sensor when apply_pixels function is
    /// called.
    /// @param column Pixel x coordinate
    /// @param row Pixel y coordinate
    /// @param enable Pixel will be masked when true
    /// @return true on success
    virtual bool set_pixel(const unsigned int &column, const unsigned int &row, const bool &enable) = 0;

    /// @brief Apply pixels configuration to sensor
    /// Pixels selected to be masked with the set_pixel function are applied to the sensor hardware.
    virtual void apply_pixels() = 0;

    /// @brief Reset pixels configuration applied on the sensor
    /// All previous pixels masked on the sensor will be enabled back. Pixels configuration in the driver is cleared.
    virtual void reset_pixels() = 0;

    /// @brief Get list of pixels selected for masking
    /// @return list of x, y coordinates
    virtual std::vector<std::pair<unsigned int, unsigned int>> get_pixels() const = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_ROI_PIXEL_MASK_H
