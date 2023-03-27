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

#ifndef METAVISION_HAL_SAMPLE_HW_IDENTIFICATION_H
#define METAVISION_HAL_SAMPLE_HW_IDENTIFICATION_H

#include <metavision/hal/facilities/i_hw_identification.h>

/// @brief Facility to provide information about the available system
///
/// This class is the implementation of HAL's facility @ref Metavision::I_HW_Identification
class SampleHWIdentification : public Metavision::I_HW_Identification {
public:
    /// @brief Constructor
    ///
    /// @param plugin_sw_info Information about the plugin software version
    /// @param connection_type Type of connection with the device
    SampleHWIdentification(const std::shared_ptr<Metavision::I_PluginSoftwareInfo> &plugin_sw_info,
                           const std::string &connection_type);

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

    /// @brief Returns the name of the available data encoding formats
    ///
    /// @return The available data encoding formats
    std::vector<std::string> get_available_data_encoding_formats() const override final;

    /// @brief Returns the name of the currently used data encoding format
    ///
    /// @return The currently used data encoding format
    std::string get_current_data_encoding_format() const override;

    /// @brief Returns the integrator name
    ///
    /// @return Name of the integrator
    std::string get_integrator() const override final;

    /// @brief Returns the connection with the camera as a string
    ///
    /// @return A string providing the type of connection with the available camera
    std::string get_connection_type() const override final;

    /// @brief Lists device config options supported by the camera
    /// @return the map of (key,option) device config options
    Metavision::DeviceConfigOptionMap get_device_config_options_impl() const override final;

    static constexpr auto SAMPLE_SERIAL         = "000000";
    static constexpr long SAMPLE_SYSTEM_ID      = 42;
    static constexpr auto SAMPLE_INTEGRATOR     = "SampleIntegratorName";
    static constexpr auto SAMPLE_FORMAT         = "SAMPLE-FORMAT-1.0";

private:
    std::string connection_type_;
};

#endif // METAVISION_HAL_SAMPLE_HW_IDENTIFICATION_H
