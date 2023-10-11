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

#ifndef METAVISION_HAL_PSEE_PLUGINS_DEVICES_GENX320_DEM_INTERFACE
#define METAVISION_HAL_PSEE_PLUGINS_DEVICES_GENX320_DEM_INTERFACE

#include "metavision/hal/facilities/i_digital_event_mask.h"
#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_dem_driver.h"

namespace Metavision {

/// @brief Digital Event Mask implementation for GenX320
class GenX320DemInterface : public I_DigitalEventMask {
private:
    /// @brief Pixel Mask implementation for GenX320
    class GenX320PixelMask : public I_PixelMask {
    public:
        /// @brief Pixel Mask Constructor
        GenX320PixelMask(const std::shared_ptr<GenX320DemDriver> &driver, const unsigned int id);

        /// @brief Sets mask pixel coordinate and activation flag
        /// @param x the pixel horizontal coordinate
        /// @param y the pixel vertical coordinate
        /// @param enable when true, the mask will prevent pixel from generating CD event
        /// @return true on success
        virtual bool set_mask(uint32_t x, uint32_t y, bool enabled) override;

        /// @brief Get the current mask settings
        /// @return a tuple that packs the X and Y coordinates and a boolean defining if the mask is enabled
        virtual std::tuple<uint32_t, uint32_t, bool> get_mask() const override;

    private:
        std::shared_ptr<GenX320DemDriver> driver_;
        unsigned int id_;
    };

    constexpr static size_t NUM_MASK_REGISTERS_ = 16;

    std::vector<I_PixelMaskPtr> pixel_masks_;
    std::shared_ptr<GenX320DemDriver> mask_ctrl_;

public:
    /// @brief GenX320DemInterface constructor
    /// @param regmap the Register map associated with the sensor
    /// @param prefix the path to the digital pixel mask registers in the regmap
    GenX320DemInterface(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix);

    /// @brief Get all available masks.
    /// @return a list of pixel masks pointer
    virtual const std::vector<I_PixelMaskPtr> &get_pixel_masks() const override;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_DEVICES_GENX320_DEM_INTERFACE
