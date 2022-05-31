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

#ifndef METAVISION_HAL_TZ_CAMERA_BOARD_DISCOVERY_H
#define METAVISION_HAL_TZ_CAMERA_BOARD_DISCOVERY_H

#include <string>

#include "boards/utils/psee_camera_discovery.h"
#include "devices/treuzell/tz_device.h"

namespace Metavision {

class LibUSBContext;
class TzLibUSBBoardCommand;

class TzCameraDiscovery : public PseeCameraDiscovery {
public:
    TzCameraDiscovery();

    struct DeviceBuilderParameters : public Metavision::DeviceBuilderParameters {
        DeviceBuilderParameters(std::shared_ptr<LibUSBContext> libusb_ctx,
                                const std::shared_ptr<TzLibUSBBoardCommand> &board_cmd) :
            board_cmd(board_cmd), libusb_ctx(libusb_ctx) {}
        std::shared_ptr<TzLibUSBBoardCommand> board_cmd;
        std::shared_ptr<LibUSBContext> libusb_ctx;
    };

    virtual CameraDiscovery::SerialList list() override;
    virtual CameraDiscovery::SystemList list_available_sources() override;
    virtual bool discover(DeviceBuilder &device_builder, const std::string &serial,
                          const DeviceConfig &config) override;
    TzDeviceBuilder &factory() {
        return *builder.get();
    }

private:
    std::vector<std::shared_ptr<TzLibUSBBoardCommand>> list_boards();
    std::shared_ptr<LibUSBContext> libusb_ctx;
    std::unique_ptr<TzDeviceBuilder> builder;
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_CAMERA_BOARD_DISCOVERY_H
