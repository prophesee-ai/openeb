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

#ifndef METAVISION_HAL_TZ_CAMERA_SYNCHRONIZATION_H
#define METAVISION_HAL_TZ_CAMERA_SYNCHRONIZATION_H

#include <memory>
#include <vector>

#include "metavision/hal/facilities/i_camera_synchronization.h"

namespace Metavision {

class TzDevice;
class TzDeviceControl;

/// @brief Camera Synchronization facility controls camera mode
class TzCameraSynchronization : public I_CameraSynchronization {
public:
    TzCameraSynchronization(std::vector<std::shared_ptr<TzDevice>> &devices,
                            const std::shared_ptr<TzDeviceControl> &device_control);
    ~TzCameraSynchronization();

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
    std::vector<std::shared_ptr<TzDevice>> devices_;
    std::shared_ptr<TzDeviceControl> device_control_;
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_CAMERA_SYNCHRONIZATION_H
