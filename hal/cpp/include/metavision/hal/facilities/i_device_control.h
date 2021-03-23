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

#ifndef METAVISION_HAL_I_DEVICE_CONTROL_H
#define METAVISION_HAL_I_DEVICE_CONTROL_H

#include <cstdint>
#include <ostream>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Facility that controls camera mode and allows to start, reset and stop it.
class I_DeviceControl : public I_RegistrableFacility<I_DeviceControl> {
public:
    /// @brief Enumerate synchronization modes
    enum class SyncMode { STANDALONE = 0, MASTER = 1, SLAVE = 2 };

    /// @brief Restarts the device and the connection with it
    virtual void reset() = 0;

    /// @brief Starts the generation of events from the camera side
    /// @warning All triggers will be disabled at stop. User should re-enable required triggers before start.
    virtual void start() = 0;

    /// @brief Stops the generation of events from the camera side
    virtual void stop() = 0;

    /// @brief Sets the camera in standalone mode.
    ///
    /// The camera does not interact with other devices.
    /// @warning This function must be called before starting the camera
    /// @return true on success
    virtual bool set_mode_standalone() = 0;

    /// @brief Sets the camera as master
    ///
    /// The camera sends clock signal to another device
    /// @warning This function must be called before starting the camera
    /// @return true on success
    virtual bool set_mode_master() = 0;

    /// @brief Sets the camera as slave
    ///
    /// The camera receives the clock from another device
    /// @warning This function must be called before starting the camera
    /// @return true on success
    virtual bool set_mode_slave() = 0;

    /// @brief Retrieves Synchronization mode
    /// @return synchronization mode
    virtual SyncMode get_mode() = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_I_DEVICE_CONTROL_H
