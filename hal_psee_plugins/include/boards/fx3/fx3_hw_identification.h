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

class PseeLibUSBBoardCommand;

/// @brief Facility to provide information about the available system
class Fx3HWIdentification : public I_HW_Identification {
public:
    /// @brief Facility Constructor
    ///
    /// @param integrator Name of the system integrator
    Fx3HWIdentification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info,
                        const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd, bool is_EVT3, long subsystem_ID,
                        const std::string &integrator = "Prophesee");

    /// @brief Returns the serial number of the camera
    ///
    /// @return Serial number as a string
    virtual std::string get_serial() const override;

    /// @brief Returns the serial number of the camera
    ///
    /// @return The system id as a integer
    ///
    /// @note this number can be used to check the compatibility of biases file for example
    virtual long get_system_id() const override;

    /// @brief Returns the detail about the sensor available
    ///
    /// @return The sensor information
    virtual I_HW_Identification::SensorInfo get_sensor_info() const override final;

    /// @brief Returns the version number for this system
    ///
    /// @return System version as an integer
    virtual long get_system_version() const override final;

    /// @brief Returns the name of the available RAW format
    ///
    /// @return The available format
    ///
    /// @note currently the available formats are:
    ///      - EVT2
    ///      - EVT3
    virtual std::vector<std::string> get_available_raw_format() const override final;

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

private:
    virtual RawFileHeader get_header_impl() const override;

    std::shared_ptr<PseeLibUSBBoardCommand> icmd_;
    I_HW_Identification::SensorInfo sensor_info_;
    bool is_evt3_;
    long subsystem_ID_;
    std::string integrator_;
};

} // namespace Metavision

#endif // METAVISION_HAL_FX3_HW_IDENTIFICATION_H
