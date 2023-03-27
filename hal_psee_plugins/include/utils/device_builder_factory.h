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
class RegisterDeviceBuilder;
using DeviceBuilderCallback =
    std::function<bool(DeviceBuilder &, const DeviceBuilderParameters &, const DeviceConfig &)>;

class DeviceBuilderFactory {
    using BuilderMap = std::unordered_map<long, DeviceBuilderCallback>;

public:
    DeviceBuilderFactory() : builder_map_(generic_map()) {}

    bool build(long key, DeviceBuilder &device_builder, const DeviceBuilderParameters &device_builder_parameters,
               const DeviceConfig &device_config);
    bool insert(long key, const DeviceBuilderCallback &callback);
    bool remove(long key);
    bool contains(long key) const;

private:
    BuilderMap builder_map_;
    static BuilderMap &generic_map();
    friend RegisterDeviceBuilder;
};

class RegisterDeviceBuilder {
public:
    RegisterDeviceBuilder(long key, const DeviceBuilderCallback &callback) {
        if (!DeviceBuilderFactory::generic_map().insert({key, callback}).second)
            throw std::logic_error("Several default build methods are declared for " + std::to_string(key));
    }
};

} // namespace Metavision

#endif // METAVISION_HAL_DEVICE_BUILDER_FACTORY_H
