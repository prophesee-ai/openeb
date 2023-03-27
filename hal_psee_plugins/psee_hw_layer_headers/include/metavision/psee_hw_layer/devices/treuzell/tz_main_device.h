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

#ifndef TZ_MAIN_DEVICE_H
#define TZ_MAIN_DEVICE_H

#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/facilities/i_hw_identification.h"

namespace Metavision {

// This is an interface for the pipeline element that will implement sensor modes
class TzMainDevice {
public:
    /// @brief Get the system ID of the device.
    ///
    /// Prophesee FPGA designs have a system ID, on other system, use a hard-coded value.
    /// @return the system ID
    virtual long get_system_id() const {
        return 0;
    }

    /// @brief Sets the camera in standalone mode.
    ///
    /// The camera does not interact with other devices.
    /// @return true on success
    virtual bool set_mode_standalone() = 0;

    /// @brief Sets the camera as master
    ///
    /// The camera sends clock signal to another device
    /// @return true on success
    virtual bool set_mode_master() = 0;

    /// @brief Sets the camera as slave
    ///
    /// The camera receives the clock from another device
    /// @return true on success
    virtual bool set_mode_slave() = 0;

    /// @brief Retrieves Synchronization mode
    /// @return synchronization mode
    virtual I_CameraSynchronization::SyncMode get_mode() = 0;

    /// @brief Provides information on the sensor used in the system
    ///
    /// Allow to get hardware information such as the sensor generation
    /// @return a filled SensorInfo structure
    virtual I_HW_Identification::SensorInfo get_sensor_info() = 0;
};

} // namespace Metavision
#endif /* TZ_MAIN_DEVICE_H */
