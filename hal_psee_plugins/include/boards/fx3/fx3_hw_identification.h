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

#ifndef METAVISION_HAL_FX3_HW_IDENTIFICATION_H
#define METAVISION_HAL_FX3_HW_IDENTIFICATION_H

#include <string>
#include <vector>

#include "metavision/hal/facilities/i_hw_identification.h"

namespace Metavision {

class Fx3LibUSBBoardCommand;
class PseeDeviceControl;

/// @brief Facility to provide information about the available system
class Fx3HWIdentification : public Metavision::I_HW_Identification {
public:
    /// @brief Facility Constructor
    ///
    /// @param integrator Name of the system integrator
    Fx3HWIdentification(const std::shared_ptr<Metavision::I_PluginSoftwareInfo> &plugin_sw_info,
                        const std::shared_ptr<Fx3LibUSBBoardCommand> &board_cmd,
                        const std::shared_ptr<PseeDeviceControl> &device_ctrl,
                        const std::string &integrator = "Prophesee");

    /// @brief Returns the serial number of the camera
    ///
    /// @return Serial number as a string
    virtual std::string get_serial() const override;

    /// @brief Returns the detail about the sensor available
    ///
    /// @return The sensor information
    virtual Metavision::I_HW_Identification::SensorInfo get_sensor_info() const override final;

    /// @brief Returns the name of the available RAW format
    ///
    /// @return The available format
    ///
    /// @note currently the available formats are:
    ///      - EVT2
    ///      - EVT3
    virtual std::vector<std::string> get_available_data_encoding_formats() const override final;

    /// @brief Returns the name of the currently used data encoding format
    /// @return The currently used data encoding format
    /// @sa get_available_data_encoding_formats
    virtual std::string get_current_data_encoding_format() const override final;

    /// @brief Returns the integrator name
    ///
    /// @return Name of the Integrator
    virtual std::string get_integrator() const override final;

    /// @brief Returns all available information
    ///
    /// @return Map of key-value
    ///
    /// @note the purpose of this function is mainly for debug and display system information
    virtual SystemInfo get_system_info() const override final;

    /// @brief Returns the connection with the camera as a string
    ///
    /// @return A string providing the kind of connection with the available camera
    virtual std::string get_connection_type() const override final;

    /// @brief Lists device config options supported by the camera
    /// @return the map of (key,option) device config options
    virtual DeviceConfigOptionMap get_device_config_options_impl() const override final;

private:
    virtual Metavision::RawFileHeader get_header_impl() const override;

    std::shared_ptr<Fx3LibUSBBoardCommand> icmd_;
    Metavision::I_HW_Identification::SensorInfo sensor_info_;
    std::string integrator_;
    std::shared_ptr<PseeDeviceControl> dev_ctrl_;
};

} // namespace Metavision

#endif // METAVISION_HAL_FX3_HW_IDENTIFICATION_H
