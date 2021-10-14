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

#ifndef RAW_HW_IDENTIFICATION_H
#define RAW_HW_IDENTIFICATION_H

#include <metavision/hal/utils/raw_file_header.h>
#include <metavision/hal/facilities/i_hw_identification.h>

namespace Metavision {

/// @brief Facility to provide information about the available system
///
/// This class is the implementation of HAL's facility @ref Metavision::I_HW_Identification
class RawHWIdentification : public Metavision::I_HW_Identification {
public:
    /// @brief Constructor
    ///
    /// @param plugin_sw_info Information about the plugin software version
    RawHWIdentification(const std::shared_ptr<Metavision::I_PluginSoftwareInfo> &plugin_sw_info,
                        const std::string serial, const Metavision::I_HW_Identification::SensorInfo &sensor_info,
                        const std::string &evt_version);

    /// @brief Returns the serial number of the camera
    ///
    /// @return Serial number as a string
    std::string get_serial() const override final;

    /// @brief Returns the system ID of the camera
    ///
    /// @return The system id as an integer
    long get_system_id() const override final;

    /// @brief Returns the detail about the available sensor
    ///
    /// @return The sensor information
    I_HW_Identification::SensorInfo get_sensor_info() const override final;

    /// @brief Returns the version number for this system
    ///
    /// @return System version as an integer
    long get_system_version() const override final;

    /// @brief Returns the name of the available RAW format
    ///
    /// @return The available format
    std::vector<std::string> get_available_raw_format() const override final;

    /// @brief Returns the integrator name
    ///
    /// @return Name of the integrator
    std::string get_integrator() const override final;

    /// @brief Returns the connection with the camera as a string
    ///
    /// @return A string providing the type of connection with the available camera
    std::string get_connection_type() const override final;

private:
    Metavision::RawFileHeader get_header_impl() const override final;

    std::string serial_;
    Metavision::I_HW_Identification::SensorInfo sensor_info_;
    std::string evt_format_;
};

} // namespace Metavision

#endif // RAW_HW_IDENTIFICATION_H
