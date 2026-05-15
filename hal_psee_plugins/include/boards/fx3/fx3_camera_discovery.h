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

#ifndef METAVISION_HAL_FX3_CAMERA_DISCOVERY_H
#define METAVISION_HAL_FX3_CAMERA_DISCOVERY_H

#include <string>
#include <unordered_map>

#include "metavision/hal/utils/camera_discovery.h"
#include "metavision/hal/utils/device_builder.h"
#include "utils/device_builder_factory.h"

namespace Metavision {

class Plugin;
class DeviceConfig;

} // namespace Metavision

namespace Metavision {

class Fx3LibUSBBoardCommand;

class Fx3CameraDiscovery : public Metavision::CameraDiscovery {
public:
    struct DeviceBuilderParameters : public Metavision::DeviceBuilderParameters {
        DeviceBuilderParameters(const std::shared_ptr<Fx3LibUSBBoardCommand> &board_cmd) : board_cmd(board_cmd) {}
        std::shared_ptr<Fx3LibUSBBoardCommand> board_cmd;
    };

    virtual CameraDiscovery::SerialList list() override;
    virtual CameraDiscovery::SystemList list_available_sources() override;

    virtual bool discover(Metavision::DeviceBuilder &device_builder, const std::string &serial,
                          const Metavision::DeviceConfig &config) override;

    DeviceBuilderFactory &device_builder_factory();

    void register_device_builder(long system_id, const DeviceBuilderCallback &cb);

private:
    DeviceBuilderFactory factory_;
};

} // namespace Metavision

#endif // METAVISION_HAL_FX3_CAMERA_DISCOVERY_H
