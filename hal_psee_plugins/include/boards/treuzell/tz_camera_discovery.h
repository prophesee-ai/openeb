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

#include "metavision/psee_hw_layer/boards/treuzell/board_command.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "metavision/hal/utils/camera_discovery.h"
#include "metavision/hal/utils/device_builder.h"
#include "devices/treuzell/tz_device_builder.h"

namespace Metavision {

class LibUSBContext;

class TzCameraDiscovery : public Metavision::CameraDiscovery {
public:
    TzCameraDiscovery();

    struct DeviceBuilderParameters : public Metavision::DeviceBuilderParameters {
        DeviceBuilderParameters(std::shared_ptr<LibUSBContext> libusb_ctx,
                                const std::shared_ptr<BoardCommand> &board_cmd) :
            board_cmd(board_cmd), libusb_ctx(libusb_ctx) {}
        std::shared_ptr<BoardCommand> board_cmd;
        std::shared_ptr<LibUSBContext> libusb_ctx;
    };

    virtual CameraDiscovery::SerialList list() override;
    virtual CameraDiscovery::SystemList list_available_sources() override;
    virtual bool discover(Metavision::DeviceBuilder &device_builder, const std::string &serial,
                          const Metavision::DeviceConfig &config) override;
    TzDeviceBuilder &factory() {
        return *builder.get();
    }

    void add_usb_id(uint16_t vid, uint16_t pid, uint8_t subclass);

private:
    std::vector<std::shared_ptr<BoardCommand>> list_boards() const;
    std::shared_ptr<LibUSBContext> libusb_ctx;
    std::unique_ptr<TzDeviceBuilder> builder;

    // By default, nothing is supported, because we want boards to be ignored by the plugins that can manage it, so that
    // only one open a given board
    std::vector<UsbInterfaceId> known_usb_ids;
};

} // namespace Metavision

#endif // METAVISION_HAL_TZ_CAMERA_BOARD_DISCOVERY_H
