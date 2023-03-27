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

#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "metavision/psee_hw_layer/facilities/tz_hw_register.h"
#include "metavision/psee_hw_layer/utils/register_map.h"
#include <system_error>

namespace Metavision {

TzHwRegister::TzHwRegister(std::vector<std::shared_ptr<TzDevice>> &devices) {
    for (auto device : devices) {
        auto dev_with_regmap = std::dynamic_pointer_cast<TzDeviceWithRegmap>(device);
        if (dev_with_regmap) {
            regmap_vec_.push_back(dev_with_regmap);
        }
    }
    if (regmap_vec_.empty()) {
        // Do not instantiate a facility if no device can be accessed
        throw std::system_error(ENODEV, std::generic_category(), "no device with a known regmap");
    }
}

uint32_t TzHwRegister::get_register_address(uint32_t address) {
    // Only keeping 28 LSbs
    const uint32_t BITS_TO_KEEP = 0xfffffff;
    // Strip off the device address from the address
    return address & BITS_TO_KEEP;
}

void TzHwRegister::write_register(uint32_t address, uint32_t v) {
    // Determine the device in the chain from the 4 MSbs
    auto vect_idx = ((address >> 28) & 0xf);

    auto base_address = get_register_address(address);

    (*regmap_vec_[vect_idx]->register_map).write(base_address, v);
}

uint32_t TzHwRegister::read_register(uint32_t address) {
    // Determine the device in the chain from the 4 MSbs
    auto vect_idx = ((address >> 28) & 0xf);

    auto base_address = get_register_address(address);

    return (*regmap_vec_[vect_idx]->register_map).read(base_address);
}

void TzHwRegister::write_register(const std::string &address, uint32_t v) {
    for (auto &dev : regmap_vec_) {
        // if the register named address starts with the device root
        if (address.rfind(dev->root_, 0) == 0) {
            // Then access the register, minus the root
            (*dev->register_map)[address.substr(dev->root_.size())].write_value(v);
            return;
        }
    }
    MV_HAL_LOG_ERROR() << "Write: Invalid register";
}

uint32_t TzHwRegister::read_register(const std::string &address) {
    for (auto &dev : regmap_vec_) {
        // if the register named address starts with the device root
        if (address.rfind(dev->root_, 0) == 0) {
            // Then access the register, minus the root
            return (*dev->register_map)[address.substr(dev->root_.size())].read_value();
        }
    }
    MV_HAL_LOG_ERROR() << "Read: Invalid register";
    return -1;
}

void TzHwRegister::write_register(const std::string &address, const std::string &bitfield, uint32_t v) {
    for (auto &dev : regmap_vec_) {
        // if the register named address starts with the device root
        if (address.rfind(dev->root_, 0) == 0) {
            // Then access the register, minus the root
            (*dev->register_map)[address.substr(dev->root_.size())][bitfield].write_value(v);
            return;
        }
    }
    MV_HAL_LOG_ERROR() << "Write: Invalid register";
}

uint32_t TzHwRegister::read_register(const std::string &address, const std::string &bitfield) {
    for (auto &dev : regmap_vec_) {
        // if the register named address starts with the device root
        if (address.rfind(dev->root_, 0) == 0) {
            // Then access the register, minus the root
            return (*dev->register_map)[address.substr(dev->root_.size())][bitfield].read_value();
        }
    }
    MV_HAL_LOG_ERROR() << "Read: Invalid register";
    return -1;
}

} // namespace Metavision
