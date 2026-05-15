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

#ifndef METAVISION_HAL_SAMPLE_CAMERA_SYNCHRONIZATION_H
#define METAVISION_HAL_SAMPLE_CAMERA_SYNCHRONIZATION_H

#include <memory>

#include <metavision/hal/facilities/i_camera_synchronization.h>
#include <metavision/hal/utils/device_control.h>

class SampleUSBConnection;

/// @brief Facility that controls the camera mode (standalone, master or slave)
///
/// This class is the implementation of HAL's facility @ref Metavision::I_CameraSynchronization.
/// In this sample is just an empty class, but for a real camera you'll need to implement
/// the methods that allows to set the camera mode.
///
/// This class is the implementation of HAL's facility @ref Metavision::I_CameraSynchronization
class SampleCameraSynchronization : public Metavision::I_CameraSynchronization {
public:
    SampleCameraSynchronization(std::shared_ptr<SampleUSBConnection> usb_connection);
    bool set_mode_standalone() override final;
    bool set_mode_master() override final;
    bool set_mode_slave() override final;
    SyncMode get_mode() const override final;

private:
    std::shared_ptr<SampleUSBConnection> usb_connection_;
};

#endif // METAVISION_HAL_SAMPLE_CAMERA_SYNCHRONIZATION_H
