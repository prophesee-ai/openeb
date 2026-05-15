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

#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/hal_log.h"
#include "utils/device_builder_factory.h"

namespace Metavision {

bool DeviceBuilderFactory::insert(long key, const DeviceBuilderCallback &device_builder_cb) {
    auto iter = builder_map_.find(key);
    if (iter != builder_map_.end()) {
        MV_HAL_LOG_ERROR() << "Trying to insert an existing key:" << std::to_string(key);
        return false;
    }
    builder_map_.insert(iter, {key, device_builder_cb});
    return true;
}

bool DeviceBuilderFactory::remove(long key) {
    auto iter = builder_map_.find(key);
    if (iter == builder_map_.end()) {
        MV_HAL_LOG_ERROR() << "Key was not registered";
        return false;
    }
    builder_map_.erase(iter);
    return true;
}

bool DeviceBuilderFactory::build(long key, DeviceBuilder &device_builder,
                                 const DeviceBuilderParameters &device_builder_params,
                                 const DeviceConfig &device_config) {
    auto iter = builder_map_.find(key);
    if (iter == builder_map_.end()) {
        MV_HAL_LOG_TRACE() << "Trying to build a device with a key that was not registered before";
        return {};
    }
    return iter->second(device_builder, device_builder_params, device_config);
}

bool DeviceBuilderFactory::contains(long key) const {
    auto iter = builder_map_.find(key);
    return iter != builder_map_.end();
}

DeviceBuilderFactory::BuilderMap &DeviceBuilderFactory::generic_map() {
    static BuilderMap static_map;
    return static_map;
}

} // namespace Metavision
