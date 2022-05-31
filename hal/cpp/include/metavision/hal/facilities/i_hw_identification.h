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

#ifndef METAVISION_HAL_I_HW_IDENTIFICATION_H
#define METAVISION_HAL_I_HW_IDENTIFICATION_H

#include <string>
#include <vector>
#include <map>

#include "metavision/hal/utils/raw_file_header.h"
#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

class I_PluginSoftwareInfo;

/// @brief Facility to provide information about the available system
class I_HW_Identification : public I_RegistrableFacility<I_HW_Identification> {
public:
    /// @brief Alias to provide any useful information for the end-user
    ///
    /// Example of possible key-value:
    ///  - "Connection" : "USB 3.0"
    ///  - "System Build Date" : "YYYY-MM-DD h:m:s (example : 2017-03-08 13:36:44 ) (UTC time)"
    using SystemInfo = std::map<std::string, std::string>;

    /// @brief Information about the type of sensor available
    ///
    /// Examples for Gen3.1 Sensor:
    ///  - major_version = 3
    ///  - minor_version = 1
    struct SensorInfo {
        /// Sensor Generation
        uint16_t major_version_;

        /// Sensor Revision
        uint16_t minor_version_;

        /// Returns the sensor's information
        /// @return The sensor's version as a string
        std::string as_string() const {
            return std::to_string(major_version_) + "." + std::to_string(minor_version_);
        }
    };

    /// @brief Constructor
    /// @param plugin_sw_info Information about the plugin software version
    I_HW_Identification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info);

    /// @brief Returns the serial number of the camera
    /// @return Serial number as a string
    virtual std::string get_serial() const = 0;

    /// @brief Returns the system id of the camera
    /// @return The system id as an integer
    /// @note This number can be used to check the compatibility of biases file
    virtual long get_system_id() const = 0;

    /// @brief Returns the detail about the available sensor
    /// @return The sensor information
    virtual SensorInfo get_sensor_info() const = 0;

    /// @brief Returns the version number for this system
    /// @return System version as an integer
    virtual long get_system_version() const = 0;

    /// @brief Returns the name of the available RAW format
    /// @return The available format
    /// @note Currently the available formats are:
    ///      - EVT2
    ///      - EVT3
    virtual std::vector<std::string> get_available_raw_format() const = 0;

    /// @brief Returns the integrator name
    /// @return Name of the integrator
    virtual std::string get_integrator() const = 0;

    /// @brief Returns all available information
    /// @return Map of key-value
    /// @note The purpose of this function is mainly for debug and display system information
    /// The class provides a basic implementation that can be enriched by inherit class
    virtual SystemInfo get_system_info() const;

    /// @brief Returns the connection with the camera as a string
    /// @return A string providing the type of connection with the available camera
    virtual std::string get_connection_type() const = 0;

    /// @brief Returns a header that can be used to log a RAW file
    /// @return A header that contains information compatible with this system
    RawFileHeader get_header() const;

protected:
    /// @brief Gets the plugin software info facility
    /// @return The plugin software info facility
    const std::shared_ptr<I_PluginSoftwareInfo> &get_plugin_software_info() const;

private:
    /// @brief Returns a header that can be used to log a RAW file
    /// @return A header that contains information compatible with this system
    virtual RawFileHeader get_header_impl() const;

    std::shared_ptr<I_PluginSoftwareInfo> plugin_sw_info_;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_HW_IDENTIFICATION_H
