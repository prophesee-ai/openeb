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

#ifndef METAVISION_HAL_FACILITY_DIGITAL_CROP_H
#define METAVISION_HAL_FACILITY_DIGITAL_CROP_H

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Digital Crop feature
/// All pixels outside of the cropping region will be dropped by the sensor
class I_DigitalCrop : public I_RegistrableFacility<I_DigitalCrop> {
public:
    ///  @brief Structure that defines a Region with 4 values :
    ///     1. X (horizontal) position of the start pixel of the top left region
    ///     2. Y (vertical) position of the start pixel of the top left region
    ///     3. X (horizontal) position of the end pixel of the bottom right region
    ///     4. Y (vertical) position of the end pixel of the bottom right region
    using Region = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;

    virtual ~I_DigitalCrop() = default;

    /// @brief Applies Digital Crop
    /// @param state If true, enables Digital Crop. If false, disables it
    /// @return true on success
    virtual bool enable(bool state) = 0;

    /// @brief Returns Digital Crop activation state
    /// @return The Digital Crop state
    virtual bool is_enabled() = 0;

    /// @brief Defines digital crop window region
    /// @param region The region of pixels that should be cropped
    /// @param reset_origin If true, the origin of the event output coordinates will shift to the Crop Window start
    /// @warning When reset_origin is true, start_x must be a multiple of 32 and end_x a multiple of 31,
    ///        if not then the function will fail and return false.
    /// @return true on success
    virtual bool set_window_region(const Region &region, bool reset_origin = false) = 0;

    /// @brief Gets the digital crop window region currently defined
    /// @return digital crop window region currently defined
    virtual Region get_window_region() = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_FACILITY_DIGITAL_CROP_H
