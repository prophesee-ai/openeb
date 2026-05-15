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

#include <memory>

#include "metavision/psee_hw_layer/facilities/tz_camera_synchronization.h"
#include "metavision/psee_hw_layer/utils/tz_device_control.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_main_device.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

TzCameraSynchronization::TzCameraSynchronization(std::vector<std::shared_ptr<TzDevice>> &devices,
                                                 const std::shared_ptr<TzDeviceControl> &device_control) :
    devices_(devices), device_control_(device_control) {}

TzCameraSynchronization::~TzCameraSynchronization() {}

bool TzCameraSynchronization::set_mode_standalone() {
    if (device_control_->is_streaming()) {
        return false;
    }
    for (auto dev : devices_) {
        if (auto main_dev = dynamic_cast<TzMainDevice *>(dev.get())) {
            return main_dev->set_mode_standalone();
        }
    }
    return false;
}

bool TzCameraSynchronization::set_mode_master() {
    if (device_control_->is_streaming()) {
        return false;
    }
    for (auto dev : devices_) {
        if (auto main_dev = dynamic_cast<TzMainDevice *>(dev.get())) {
            return main_dev->set_mode_master();
        }
    }
    return false;
}

bool TzCameraSynchronization::set_mode_slave() {
    if (device_control_->is_streaming()) {
        return false;
    }
    for (auto dev : devices_) {
        if (auto main_dev = dynamic_cast<TzMainDevice *>(dev.get())) {
            return main_dev->set_mode_slave();
        }
    }
    return false;
}

I_CameraSynchronization::SyncMode TzCameraSynchronization::get_mode() const {
    for (auto dev : devices_) {
        if (auto main_dev = dynamic_cast<TzMainDevice *>(dev.get())) {
            return main_dev->get_mode();
        }
    }
    return I_CameraSynchronization::SyncMode::STANDALONE;
}

} // namespace Metavision
