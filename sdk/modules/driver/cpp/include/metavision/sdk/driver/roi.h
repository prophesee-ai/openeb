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
    /// @brief Basic struct used to set a hardware region of interest (ROI) - a rectangle - on the sensor
    ///
    /// This struct defines an ROI of with*height pixels going from [x, y] to [x + width -1, y + height -1]
    struct Rectangle {
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
    void set(Rectangle roi);

    /// @brief Sets several rectangular hardware ROI from binary maps of lines and columns
    ///
    /// The binary map arguments must have the sensor's dimension.
    /// Throws a @ref CameraException if input doesn't have the correct size.
    ///
    /// @param cols_to_enable vector of boolean of size sensor's width representing the binary map of the columns
    ///        to disable (0) or to enable (1)
    /// @param rows_to_enable vector of boolean of size sensor's height representing the binary map of the rows
    ///        to disable (0) or to enable (1)
    void set(const std::vector<bool> &cols_to_enable, const std::vector<bool> &rows_to_enable);

    /// @brief Sets several rectangular hardware ROI from @ref Roi::Rectangle vector
    ///
    /// Any line or column enabled by a single ROI is also enabled for all the other.
    /// Example: Input vector is composed with 2 rois (0, 0, 50, 50), (100, 100, 50, 50).
    /// In the sensor, it will result in 4 regions:
    ///     - (0, 0, 50, 50)
    ///     - (0, 100, 50, 50)
    ///     - (100, 0, 50, 50)
    ///     - (100, 100, 50, 50)
    /// Indeed, rows and columns from 0 to 50 and 100 to 150 are enabled.
    ///
    /// Example: Input vector is composed with 2 ROIs: (0, 0, 50, 50), (25, 25, 50, 50).
    /// In the sensor, it will result in a single region: (0, 0, 75, 75)
    ///
    /// @param to_set a vector of @ref Roi::Rectangle
    void set(const std::vector<Rectangle> &to_set);

    /// @brief Unsets any set ROI on the sensor
    void unset();

    /// @brief Gets corresponding facility in HAL library
    I_ROI *get_facility() const;

private:
    I_ROI *pimpl_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_ROI_H
