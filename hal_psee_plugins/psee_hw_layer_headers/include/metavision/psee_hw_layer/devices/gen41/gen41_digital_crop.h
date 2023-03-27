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

#ifndef METAVISION_HAL_PSEE_PLUGINS_DEVICES_GEN41_DIGITAL_CROP_H
#define METAVISION_HAL_PSEE_PLUGINS_DEVICES_GEN41_DIGITAL_CROP_H

#include "metavision/hal/facilities/i_digital_crop.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

/// @brief Digital Crop Gen4.1 implementation
/// All pixels outside of the cropping region will be dropped by the sensor
class Gen41DigitalCrop : public I_DigitalCrop {
    RegisterMap::FieldAccess enable_;
    RegisterMap::FieldAccess reset_orig_;
    RegisterMap::FieldAccess start_x_;
    RegisterMap::FieldAccess start_y_;
    RegisterMap::FieldAccess end_x_;
    RegisterMap::FieldAccess end_y_;

public:
    /// @brief Gen41DigitalCrop constructor
    /// @param regmap the Register map for the sensor
    /// @param prefix the prefix path to retrieve registers in the regmap
    Gen41DigitalCrop(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix);

    /// @brief Applies Digital Crop
    /// @param state If true, enables Digital Crop. If false, disables it
    /// @return true on success
    bool enable(bool state) override;

    /// @brief Returns Digital Crop activation state
    /// @return The Digital Crop state
    bool is_enabled() override;

    /// @brief Defines digital crop window region
    /// @param region The region of pixels that should be cropped
    /// @param reset_origin If true, the origin of the event output coordinates will shift to the Crop Window start
    /// @warning When reset_origin is true, start_x must be a multiple of 32 and end_x a multiple of 31,
    bool set_window_region(const Region &region, bool reset_origin) override;

    /// @brief Gets the digital crop window region currently defined
    /// @return digital crop window region currently defined
    Region get_window_region() override;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_DEVICES_GEN41_DIGITAL_CROP_H
