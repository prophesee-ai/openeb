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

#ifndef METAVISION_HAL_DEVICE_BUILDER_MAKER_H
#define METAVISION_HAL_DEVICE_BUILDER_MAKER_H

#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/hal_software_info.h"
#include "metavision/hal/facilities/i_hal_software_info.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"

namespace Metavision {

inline DeviceBuilder make_device_builder(const std::string &plugin_integrator = std::string(),
                                         const std::string &plugin_name       = std::string()) {
    return DeviceBuilder(
        std::make_unique<I_HALSoftwareInfo>(Metavision::get_hal_software_info()),
        std::make_unique<I_PluginSoftwareInfo>(plugin_integrator, plugin_name, Metavision::get_hal_software_info()));
}

} // namespace Metavision

#endif // METAVISION_HAL_DEVICE_BUILDER_MAKER_H
