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

#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <list>
#include <memory>

#include "boards/treuzell/tz_camera_discovery.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

TzCameraDiscovery::TzCameraDiscovery() :
    libusb_ctx(std::make_shared<LibUSBContext>()), builder(std::make_unique<TzDeviceBuilder>()) {}

std::vector<std::shared_ptr<BoardCommand>> TzCameraDiscovery::list_boards() const {
    std::vector<std::shared_ptr<BoardCommand>> boards;
    libusb_device **devs;

    ssize_t cnt = libusb_get_device_list(libusb_ctx->ctx(), &devs); // get the list of devices
    if (cnt <= 0) {
        MV_HAL_LOG_TRACE() << "libusb BC: USB Device list empty cnt=" << cnt;
        return boards;
    }

    MV_HAL_LOG_TRACE() << "libusb BC: libusb_get_device_list found" << cnt << "devices";

    for (ssize_t i = 0; i < cnt; i++) {
        libusb_device_descriptor desc;
        int r = libusb_get_device_descriptor(devs[i], &desc);
        if (r < 0) {
            MV_HAL_LOG_TRACE() << "Failed to get device descriptor";
            continue;
        }

        try {
            std::shared_ptr<BoardCommand> cmd =
                std::make_shared<TzLibUSBBoardCommand>(libusb_ctx, devs[i], desc, known_usb_ids);
            MV_HAL_LOG_TRACE() << "Create board command for" << cmd->get_name() << cmd->get_serial() << "(" << std::hex
                               << desc.idVendor << ":" << desc.idProduct << std::dec << ")";
            if (builder->can_build(cmd)) {
                boards.push_back(cmd);
                MV_HAL_LOG_TRACE() << "Register board command for" << cmd->get_name() << cmd->get_serial() << "("
                                   << std::hex << desc.idVendor << ":" << desc.idProduct << std::dec << ")";
            }
        } catch (const HalException &e) {
            // Don't trace the reason, it's way too verbose, and there is probably no Treuzell interface
            continue;
        }
    }
    libusb_free_device_list(devs, 1); // free the list, unref the devices in it

    return boards;
}

CameraDiscovery::SerialList TzCameraDiscovery::list() {
    CameraDiscovery::SerialList ret;
    auto boards = list_boards();
    for (auto board : boards) {
        ret.push_back(board->get_serial());
    }
    return ret;
}

CameraDiscovery::SystemList TzCameraDiscovery::list_available_sources() {
    CameraDiscovery::SystemList system_list;
    auto boards = list_boards();
    for (auto board : boards) {
        // Last argument is the system ID, but we can't know how many the board has before building the devices
        system_list.push_back({board->get_serial(), USB_LINK, 0});
    }
    return system_list;
}

bool TzCameraDiscovery::discover(DeviceBuilder &device_builder, const std::string &serial, const DeviceConfig &config) {
    auto boards = list_boards();
    for (auto board : boards) {
        if (serial != "" && (board->get_serial() != serial))
            continue;
        const long kLibUSBSpeedSuper = 5000;
        if (board->get_board_speed() < kLibUSBSpeedSuper) {
            MV_HAL_LOG_WARNING() << "Your EVK camera" << serial
                                 << "isn't connected in USB3. Please check your connection.";
        }
        return builder->build_devices(board, device_builder, config);
    }
    return false;
}

void TzCameraDiscovery::add_usb_id(uint16_t vid, uint16_t pid, uint8_t subclass) {
    known_usb_ids.push_back({vid, pid, 0xFF, subclass});
}

} // namespace Metavision
