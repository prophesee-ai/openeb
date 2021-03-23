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

#ifndef METAVISION_HAL_I_ROI_H
#define METAVISION_HAL_I_ROI_H

#include <cstdint>
#include <string>
#include <vector>

#include "metavision/hal/utils/device_roi.h"
#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Interface facility for ROI (Region Of Interest)
class I_ROI : public I_RegistrableFacility<I_ROI> {
public:
    /// @brief Applies ROI
    /// @param state If true, enables ROI. If false, disables it
    /// @warning At least one ROI should have been set before calling this function
    virtual void enable(bool state) = 0;

    /// @brief Sets an ROI from DeviceRoi geometry
    /// The ROI input is translated into a bitword (register format) and then programmed in the sensor
    /// @param roi ROI to set
    /// @param enable If true, applies ROI
    void set_ROI(const DeviceRoi &roi, bool enable = true);

    /// @brief Sets an ROI from bitword (register format)
    /// @param vroiparams ROI to set
    /// @param enable If true, applies ROI. If false, disables it
    virtual void set_ROIs_from_bitword(const std::vector<uint32_t> &vroiparams, bool enable = true) = 0;

    /// @brief Sets multiple ROIs.
    ///
    /// The input ROIs are translated into a single bitword (register format) and then programmed in the sensor.
    /// Due to the sensor format, the final ROI is one or a set of regions (i.e. a grid).
    ///
    /// If the input vector is composed of 2 ROIs: (0, 0, 50, 50), (100, 100, 50, 50), then in the sensor,
    /// it will give 4 regions: (0, 0, 50, 50), (0, 100, 50, 50), (100, 0, 50, 50) and (100, 100, 50, 50).
    ///
    /// If the input vector is composed of 2 ROIs: (0, 0, 50, 50), (25, 25, 50, 50), then in the sensor
    /// it will give 1 region: (0, 0, 75, 75)
    ///
    /// @param vroi A vector of ROIs
    /// @param enable If true, applies ROI. If false, disables it
    void set_ROIs(const std::vector<DeviceRoi> &vroi, bool enable = true);

    /// @brief Sets union of several ROIs from a file with CSV format "x y width height"
    /// @param file_path Path to the CSV file with ROIs
    /// @param enable If true, applies ROI. If false, disables it
    void set_ROIs_from_file(std::string const &file_path, bool enable = true);

    /// @brief Sets multiple rectangular ROIs in bitword register format from two binary maps (for rows and a columns)
    ///
    /// The binary maps (std::vector<bool>) arguments must have the sensor's dimension
    ///
    /// @param cols_to_enable Vector of boolean of size sensor's width representing the binary map of the columns to
    /// disable (0) or to enable (1)
    /// @param rows_to_enable Vector of boolean of size sensor's height representing the binary map of the rows to
    /// disable (0) or to enable (1)
    /// @param enable If true, applies ROI
    /// @return true if input have the correct dimension and thus the ROI is set correctly, false otherwise
    /// @warning For a pixel to be enabled, it must be enabled on both its row and column
    virtual bool set_ROIs(const std::vector<bool> &cols_to_enable, const std::vector<bool> &rows_to_enable,
                          bool enable = true) = 0;

    /// @brief Creates a rectangular ROI in bitword register format
    /// @param roi ROI's geometry to transform to bitword register format
    /// @return The ROI in bitword register format
    std::vector<uint32_t> create_ROI(const DeviceRoi &roi);

    /// @brief Creates several rectangular ROIs in bitword register format
    /// @param vroi Vector of ROI to transform to bitword register format
    /// @return The ROIs in bitword register format
    virtual std::vector<uint32_t> create_ROIs(const std::vector<DeviceRoi> &vroi) = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_ROI_H
