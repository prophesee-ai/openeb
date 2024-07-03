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

#include "metavision/psee_hw_layer/utils/tz_device_control.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_main_device.h"
#include "metavision/hal/utils/hal_connection_exception.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

TzDeviceControl::TzDeviceControl(std::vector<std::shared_ptr<TzDevice>> &devices) :
    devices_(devices), streaming_(false) {
    // Directly start every device, except the one actually producing data
    for (auto dev : devices_) {
        auto main_dev = dynamic_cast<TzMainDevice *>(dev.get());
        if (!main_dev) {
            try {
                dev.get()->start();
            } catch (const std::system_error &e) {
                MV_HAL_LOG_TRACE() << dev.get()->name() << "did not start:" << e.what();
            }
        }
    }
}

TzDeviceControl::~TzDeviceControl() {
    // Stop streaming if the caller program "forgot"
    if (streaming_) {
        try {
            stop();
        } catch (const std::system_error &e) {
            MV_HAL_LOG_WARNING() << "Treuzell Device Control destruction failed: " << e.what();
        }
    }
    // Stop intermediate blocks
    for (auto dev = devices_.rbegin(); dev != devices_.rend(); dev++) {
        auto main_dev = dynamic_cast<TzMainDevice *>((*dev).get());
        if (!main_dev) {
            try {
                (*dev).get()->stop();
            } catch (const std::system_error &e) {
                MV_HAL_LOG_TRACE() << (*dev).get()->name() << "did not stop:" << e.what();
            }
        }
    }
}

void TzDeviceControl::reset() {
    return;
}

void TzDeviceControl::start() {
    streaming_ = true;
    // Start only the main device, the others are always running
    for (auto dev : devices_)
        if (auto main_dev = dynamic_cast<TzMainDevice *>(dev.get()))
            dev.get()->start();
}

void TzDeviceControl::stop() {
    // HAL calls stop regardless of the current state
    if (!streaming_)
        return;
    // Stop only the main device, the others are always running
    for (auto dev = devices_.rbegin(); dev != devices_.rend(); dev++)
        if (auto main_dev = dynamic_cast<TzMainDevice *>((*dev).get())) {
            try {
                (*dev).get()->stop();
            } catch (const HalConnectionException &e) {
                MV_HAL_LOG_WARNING() << "Failed to properly stop TzDevice due do connection error";
                MV_HAL_LOG_WARNING() << e.what();
            }
        }
    streaming_ = false;
}

bool TzDeviceControl::is_streaming() const {
    return streaming_;
}

} // namespace Metavision
