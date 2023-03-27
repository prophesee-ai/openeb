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

#ifndef METAVISION_HAL_PSEE_PLUGINS_DEVICES_GEN41_DIGITAL_EVENT_MASK
#define METAVISION_HAL_PSEE_PLUGINS_DEVICES_GEN41_DIGITAL_EVENT_MASK

#include "metavision/hal/facilities/i_digital_event_mask.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

/// @brief Digital Event Mask implementation for Gen 4.1
class Gen41DigitalEventMask : public I_DigitalEventMask {
private:
    /// @brief Pixel Mask implementation for Gen 4.1
    class Gen41PixelMask : public I_PixelMask {
        using Register = RegisterMap::RegisterAccess;
        Register reg_;

    public:
        /// @brief Pixel Mask Constructor
        /// @param reg the register access of the pixel mask
        Gen41PixelMask(const Register &reg);

        /// @brief Sets mask pixel coordinate and activation flag
        /// @param x the pixel horizontal coordinate
        /// @param y the pixel vertical coordinate
        /// @param enable when true, the mask will prevent pixel from generating CD event
        /// @return true on success
        virtual bool set_mask(uint32_t x, uint32_t y, bool enabled) override;

        /// @brief Get the current mask settings
        /// @return a tuple that packs the X and Y coordinates and a boolean defining if the mask is enabled
        virtual std::tuple<uint32_t, uint32_t, bool> get_mask() const override;
    };

    constexpr static size_t NUM_MASK_REGISTERS_ = 64;

    std::shared_ptr<RegisterMap> regmap_;
    std::string prefix_;
    std::vector<I_PixelMaskPtr> pixel_masks_;

public:
    /// @brief Gen41DigitalEventMask constructor
    /// @param regmap the Register map associated with the sensor
    /// @param prefix the path to the digital pixel mask registers in the regmap
    Gen41DigitalEventMask(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix);

    /// @brief Get all available masks.
    /// @return a list of pixel masks pointer
    virtual const std::vector<I_PixelMaskPtr> &get_pixel_masks() const override;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_DEVICES_GEN41_DIGITAL_EVENT_MASK
