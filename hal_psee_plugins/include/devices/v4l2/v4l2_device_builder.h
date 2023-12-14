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

#ifndef V4L2_DEVICE_BUILDER_H
#define V4L2_DEVICE_BUILDER_H

#include <cstdint>
#include <memory>
#include <functional>
#include <stdexcept>
#include <unordered_map>

namespace Metavision {

class V4L2Device;

class V4L2RegisterBuildMethod;
class BoardCommand;
class DeviceConfig;
class DeviceBuilder;
class V4L2DeviceBuilder {
public:

    V4L2DeviceBuilder() {}
    bool build_device(std::shared_ptr<BoardCommand> cmd, DeviceBuilder &device_builder,
                       const DeviceConfig &config);
};


} // namespace Metavision
#endif /* V4L2_DEVICE_H */
