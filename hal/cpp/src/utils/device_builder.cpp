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

#include <map>
#include <iostream>
#include <string>

#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_hal_software_info.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_ll_biases.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/device_builder.h"

namespace Metavision {

DeviceBuilder::DeviceBuilder(std::unique_ptr<I_HALSoftwareInfo> i_hal_sw_info,
                             std::unique_ptr<I_PluginSoftwareInfo> i_plugin_sw_info) {
    i_hal_sw_info_    = add_facility(std::move(i_hal_sw_info));
    i_plugin_sw_info_ = add_facility(std::move(i_plugin_sw_info));
}

DeviceBuilder::DeviceBuilder(DeviceBuilder &&) = default;

DeviceBuilder &DeviceBuilder::operator=(DeviceBuilder &&) = default;

const std::shared_ptr<I_HALSoftwareInfo> &DeviceBuilder::get_hal_software_info() const {
    return i_hal_sw_info_;
}

const std::shared_ptr<I_PluginSoftwareInfo> &DeviceBuilder::get_plugin_software_info() const {
    return i_plugin_sw_info_;
}

std::unique_ptr<Device> DeviceBuilder::operator()() {
    Device *dev = new Device(facilities_.begin(), facilities_.end());

    if (dev && dev->get_facility<I_LL_Biases>()) {
        auto hw_identification = dev->get_facility<I_HW_Identification>();

        if (hw_identification) {
            hw_identification->add_hal_device_config_option(DeviceConfig::get_biases_range_check_bypass_key(),
                                                            DeviceConfigOption(false));
        }
    }

    return std::unique_ptr<Device>(dev);
}

} // namespace Metavision