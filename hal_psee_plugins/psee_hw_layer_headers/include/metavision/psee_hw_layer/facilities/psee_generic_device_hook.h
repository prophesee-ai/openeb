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

#ifndef METAVISION_HAL_PSEE_GENERIC_DEVICE_HOOK_H
#define METAVISION_HAL_PSEE_GENERIC_DEVICE_HOOK_H

#include <functional>

#include "metavision/hal/facilities/i_registrable_facility.h"

namespace Metavision {

/// @brief Device Control facility controls camera mode and allows to start, reset and stop it.
class I_PseeGenericDeviceHook : public I_RegistrableFacility<I_PseeGenericDeviceHook> {
public:
    using type_callback = std::function<void(void)>;

    I_PseeGenericDeviceHook() {}
    virtual ~I_PseeGenericDeviceHook() {}

    /// @brief Register a callback to be called when the main device is required to start.
    virtual void set_start_cb(const type_callback &cb) = 0;

    /// @brief Register a callback to be called when the main device is required to stop.
    virtual void set_stop_cb(const type_callback &cb) = 0;

    /// @brief Sets the camera resolution
    virtual void set_sensor_resolution(int width, int height) = 0;

    /// @brief Simple hook to write a register to the main system device
    virtual void write_register(uint32_t addr, uint32_t val) = 0;

    /// @brief Simple hook to read a register from the main system device
    virtual uint32_t read_register(uint32_t addr) = 0;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_GENERIC_DEVICE_HOOK_H
