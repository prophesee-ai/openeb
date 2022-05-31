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

#ifndef METAVISION_HAL_DEVICE_BUILDER_FACTORY_H
#define METAVISION_HAL_DEVICE_BUILDER_FACTORY_H

#include <functional>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace Metavision {

class DeviceBuilder;
class DeviceBuilderParameters;
class DeviceConfig;
using DeviceBuilderCallback =
    std::function<bool(DeviceBuilder &, const DeviceBuilderParameters &, const DeviceConfig &)>;

class DeviceBuilderFactory {
public:
    bool build(long key, DeviceBuilder &device_builder, const DeviceBuilderParameters &device_builder_parameters,
               const DeviceConfig &device_config);
    bool insert(long key, const DeviceBuilderCallback &callback);
    bool remove(long key);
    bool contains(long key);

private:
    std::unordered_map<long, DeviceBuilderCallback> builder_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_DEVICE_BUILDER_FACTORY_H
