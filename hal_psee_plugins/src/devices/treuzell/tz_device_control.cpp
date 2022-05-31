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

#include "devices/treuzell/tz_device_control.h"
#include "devices/treuzell/tz_device.h"
#include "devices/treuzell/tz_main_device.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

namespace Metavision {

TzDeviceControl::TzDeviceControl(std::vector<std::shared_ptr<TzDevice>> &devices) :
    devices_(devices), streaming_(false) {
    // Directly start every devices, except the one actually producing data
    for (auto dev : devices_) {
        auto main_dev = dynamic_cast<TzMainDevice *>(dev.get());
        if (!main_dev)
            dev.get()->start();
    }
}

TzDeviceControl::~TzDeviceControl() {
    // Stop streaming if the caller program "forgot"
    if (streaming_) {
        stop();
    }
    // Stop intermediate blocks
    for (auto dev = devices_.rbegin(); dev != devices_.rend(); dev++) {
        auto main_dev = dynamic_cast<TzMainDevice *>((*dev).get());
        if (!main_dev)
            (*dev).get()->stop();
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
    // Stop only the main device, the others are always running
    for (auto dev = devices_.rbegin(); dev != devices_.rend(); dev++)
        if (auto main_dev = dynamic_cast<TzMainDevice *>((*dev).get()))
            (*dev).get()->stop();
    streaming_ = false;
}

bool TzDeviceControl::set_mode_standalone() {
    if (streaming_) {
        return false;
    }
    for (auto dev : devices_) {
        if (auto main_dev = dynamic_cast<TzMainDevice *>(dev.get())) {
            return main_dev->set_mode_standalone();
        }
    }
    return false;
}

bool TzDeviceControl::set_mode_master() {
    if (streaming_) {
        return false;
    }
    for (auto dev : devices_) {
        if (auto main_dev = dynamic_cast<TzMainDevice *>(dev.get())) {
            return main_dev->set_mode_master();
        }
    }
    return false;
}

bool TzDeviceControl::set_mode_slave() {
    if (streaming_) {
        return false;
    }
    for (auto dev : devices_) {
        if (auto main_dev = dynamic_cast<TzMainDevice *>(dev.get())) {
            return main_dev->set_mode_slave();
        }
    }
    return false;
}

I_DeviceControl::SyncMode TzDeviceControl::get_mode() {
    for (auto dev : devices_) {
        if (auto main_dev = dynamic_cast<TzMainDevice *>(dev.get())) {
            return main_dev->get_mode();
        }
    }
    return I_DeviceControl::SyncMode::STANDALONE;
}

} // namespace Metavision
