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

#ifndef METAVISION_HAL_GENX320_ROI_INTERFACE_H
#define METAVISION_HAL_GENX320_ROI_INTERFACE_H

#include "metavision/hal/facilities/i_roi.h"
#include "metavision/psee_hw_layer/devices/genx320/genx320_roi_driver.h"

namespace Metavision {

class GenX320RoiInterface : public I_ROI {
public:
    GenX320RoiInterface(const std::shared_ptr<GenX320RoiDriver> &driver);

    /// @brief Returns the default device width
    ///
    /// This values is obtained from the default Device passed in the constructor of the class.
    int device_width() const;

    /// @brief Returns the default device height
    ///
    /// This values is obtained from the default Device passed in the constructor of the class.
    int device_height() const;

    /// @brief Applies ROI
    /// @param state If true, enables ROI. If false, disables it
    /// @warning At least one ROI should have been set before calling this function
    /// @return true on success
    bool enable(bool state);

    bool is_enabled() const;

    /// @brief Sets the window mode
    /// @param mode window mode to set (ROI or RONI)
    /// @return true on success
    bool set_mode(const Mode &mode);

    I_ROI::Mode get_mode() const;

    /// @brief Gets the maximum number of windows
    /// @return the maximum number of windows that can be set via @ref set_windows
    size_t get_max_supported_windows_count() const;

    /// @brief Sets multiple ROIs from row and column binary maps
    ///
    /// The binary maps (std::vector<bool>) arguments must have the sensor's dimension
    ///
    /// @param cols Vector of boolean of size sensor's width representing the binary map of the columns to
    /// enable
    /// @param rows Vector of boolean of size sensor's height representing the binary map of the rows to
    /// enable
    /// @return true if input have the correct dimension and the ROI is set correctly, false otherwise
    /// @warning For a pixel to be enabled, it must be enabled on both its row and column
    bool set_lines(const std::vector<bool> &cols, const std::vector<bool> &rows);

    std::vector<I_ROI::Window> get_windows() const;

private:
    /// @brief Implementation of `set_windows`
    /// @param windows A vector of windows to set
    /// @return true on success
    /// @throw an exception if the size of the vector is higher than the maximum supported number
    ///        of windows (see @ref get_max_supported_windows_count)
    bool set_windows_impl(const std::vector<Window> &windows);

    std::shared_ptr<GenX320RoiDriver> driver_;
    bool enabled_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_ROI_INTERFACE_H
