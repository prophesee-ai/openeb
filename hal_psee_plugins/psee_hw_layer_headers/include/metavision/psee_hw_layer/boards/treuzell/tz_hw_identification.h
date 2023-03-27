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

#ifndef METAVISION_HAL_TZ_HW_IDENTIFICATION_H
#define METAVISION_HAL_TZ_HW_IDENTIFICATION_H

#include <string>

#include "metavision/hal/facilities/i_hw_identification.h"

namespace Metavision {

class TzLibUSBBoardCommand;
class TzDevice;

class TzHWIdentification : public I_HW_Identification {
public:
    /** Facility Constructor
     */
    TzHWIdentification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info,
                       const std::shared_ptr<TzLibUSBBoardCommand> &cmd,
                       std::vector<std::shared_ptr<TzDevice>> &devices);

    /**
     * Returns the serial number of the camera
     *
     * @return serial number as a string
     */
    virtual std::string get_serial() const override;

    /**
     * Returns the serial number of the camera
     *
     * @return the system id as a integer
     *
     * @note this number can be used to check the compatibility
     *       of biases file for example
     */
    virtual long get_system_id() const override;

    /**
     * Returns the detail about the sensor available
     *
     * @return the sensor information
     */
    virtual I_HW_Identification::SensorInfo get_sensor_info() const override final;

    /**
     * Returns the name of the available RAW format
     *
     * @return the available format
     *
     * @note currently the available formats are:
     *      - EVT2
     *      - EVT21
     *      - EVT3
     */
    virtual std::vector<std::string> get_available_data_encoding_formats() const override final;

    /** @brief Returns the name of the currently used data encoding format
     *
     * @return The currently used data encoding format
     *
     * @sa get_available_data_encoding_formats
     */
    virtual std::string get_current_data_encoding_format() const override final;

    /**
     * Returns the integrator name
     *
     * @return name of the Integrator
     */
    virtual std::string get_integrator() const override final;

    /**
     * Returns all available information
     *
     * @return map of key-value
     *
     * @note the purpose of this function is mainly for debug
     *  and display system information
     */
    virtual SystemInfo get_system_info() const override final;

    /**
     * Returns the connection with the camera as a string
     *
     * @return a string providing the kind of connection with the available camera
     */
    virtual std::string get_connection_type() const override final;

    /**
     * Lists device config options supported by the camera
     *
     * @return the map of (key,option) device config options
     */
    virtual DeviceConfigOptionMap get_device_config_options_impl() const override final;

private:
    virtual RawFileHeader get_header_impl() const override;

    std::shared_ptr<TzLibUSBBoardCommand> icmd_;
    I_HW_Identification::SensorInfo sensor_info_;
    std::vector<std::shared_ptr<TzDevice>> devices_;
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_HW_IDENTIFICATION_H
