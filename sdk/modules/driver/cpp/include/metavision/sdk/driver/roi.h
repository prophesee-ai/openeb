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

#ifndef METAVISION_SDK_DRIVER_ROI_H
#define METAVISION_SDK_DRIVER_ROI_H

#include <vector>

#include "metavision/hal/facilities/i_roi.h"

namespace Metavision {

/// @brief Facility class to handle a Region Of Interest (ROI)
class Roi {
public:
    /// @brief Basic struct used to set a hardware region of interest (ROI) - a window - on the sensor
    ///
    /// This struct defines an ROI of with*height pixels going from [x, y] to [x + width -1, y + height -1]
    struct Window {
        /// ROI top left column coordinate of the ROI
        int x = 0;

        /// ROI top left row coordinate of the ROI
        int y = 0;

        /// Width of the ROI
        int width = 0;

        /// Height of the ROI
        int height = 0;
    };

    /// @brief Constructor
    Roi(I_ROI *roi);

    /// @brief Destructor
    ~Roi();

    /// @brief Sets an hardware ROI on the sensor
    ///
    /// When an ROI is set successfully, no events will be output by the sensor outside of this ROI.
    /// Since this is a hardware ROI, there is no processing added to filter the events.
    /// Setting an ROI will unset any previously set ROI.
    ///
    /// @param roi The ROI to set on the sensor
    void set(Window roi);

    /// @brief Sets multiple ROIs from row and column binary maps
    ///
    /// The binary maps (std::vector<bool>) arguments must have the sensor's dimension
    ///
    /// @param cols Vector of boolean of size sensor's width representing the binary map of the columns to
    /// enable
    /// @param rows Vector of boolean of size sensor's height representing the binary map of the rows to
    /// enable
    /// @throw an exception if the size of either @p cols or @p rows arguments do not match the sensor
    ///        geometry
    /// @warning For a pixel to be enabled, it must be enabled on both its row and column
    void set(const std::vector<bool> &cols, const std::vector<bool> &rows);

    /// @brief Sets multiple hardware ROI from @ref Roi::Window vector
    /// @param windows A vector of ROIs to set
    /// @throw an exception if the size of the vector is higher than the maximum supported number
    ///        of windows (see @ref I_ROI::get_max_supported_windows_count)
    void set(const std::vector<Window> &windows);

    /// @brief Unsets any set ROI on the sensor
    void unset();

    /// @brief Gets corresponding facility in HAL library
    I_ROI *get_facility() const;

private:
    I_ROI *pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_ROI_H
