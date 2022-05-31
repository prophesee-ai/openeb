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

#ifndef METAVISION_HAL_TZ_DEVICE_CONTROL_H
#define METAVISION_HAL_TZ_DEVICE_CONTROL_H

#include <memory>
#include <vector>

#include "metavision/hal/facilities/i_device_control.h"

namespace Metavision {

class TzDevice;

/// @brief Device Control facility controls camera mode and allows to start, reset and stop it.
class TzDeviceControl : public I_DeviceControl {
public:
    TzDeviceControl(std::vector<std::shared_ptr<TzDevice>> &devices);
    ~TzDeviceControl();

    /// @brief Restarts the device and the connection with it
    virtual void reset() override final;

    /// @brief Starts the generation of events from the camera side
    /// @warning All triggers will be disabled at stop. User should re-enable required triggers before start.
    virtual void start() override final;

    /// @brief Stops the generation of events from the camera side
    virtual void stop() override final;

    /// @brief Sets the camera in standalone mode.
    ///
    /// The camera does not interact with other devices.
    /// @return true on success
    virtual bool set_mode_standalone() override final;

    /// @brief Sets the camera as master
    ///
    /// The camera sends clock signal to another device
    /// @return true on success
    virtual bool set_mode_master() override final;

    /// @brief Sets the camera as slave
    ///
    /// The camera receives the clock from another device
    /// @return true on success
    virtual bool set_mode_slave() override final;

    /// @brief Retrieves Synchronization mode
    /// @return synchronization mode
    virtual SyncMode get_mode() override final;

private:
    bool streaming_;
    std::vector<std::shared_ptr<TzDevice>> devices_;
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_DEVICE_CONTROL_H
