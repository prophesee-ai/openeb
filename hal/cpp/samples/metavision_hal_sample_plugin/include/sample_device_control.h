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

#ifndef METAVISION_HAL_SAMPLE_DEVICE_CONTROL_H
#define METAVISION_HAL_SAMPLE_DEVICE_CONTROL_H

#include <metavision/hal/facilities/i_camera_synchronization.h>
#include <metavision/hal/utils/device_control.h>

class SampleDeviceControl : public Metavision::DeviceControl {
public:
    /// @brief Restarts the device and the connection with it
    void reset() override final;

    /// @brief Starts the generation of events from the camera side
    void start() override final;

    /// @brief Stops the generation of events from the camera side
    void stop() override final;
};

#endif // METAVISION_HAL_SAMPLE_DEVICE_CONTROL_H
