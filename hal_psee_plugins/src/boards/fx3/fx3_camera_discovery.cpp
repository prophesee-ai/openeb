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

#include "metavision/psee_hw_layer/boards/rawfile/psee_raw_file_header.h"
#include "boards/fx3/fx3_camera_discovery.h"
#include "metavision/psee_hw_layer/facilities/psee_device_control.h"
#include "metavision/psee_hw_layer/boards/fx3/fx3_libusb_board_command.h"
#include "metavision/hal/utils/device_config.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/psee_hw_layer/boards/fx3/fx3_libusb_board_command.h"
#include "devices/utils/device_system_id.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "metavision/psee_hw_layer/boards/utils/psee_libusb_data_transfer.h"

namespace Metavision {

DeviceBuilderFactory &Fx3CameraDiscovery::device_builder_factory() {
    return factory_;
}

void Fx3CameraDiscovery::register_device_builder(long system_id, const DeviceBuilderCallback &cb) {
    factory_.insert(system_id, cb);
}

Metavision::CameraDiscovery::SerialList Fx3CameraDiscovery::list() {
    Metavision::CameraDiscovery::SerialList ret;
    auto serials = Fx3LibUSBBoardCommand::get_list_serial();
    for (auto serial : serials) {
        Fx3LibUSBBoardCommand cmd;
        cmd.open(serial);
        SystemId system_id = static_cast<SystemId>(cmd.get_system_id());
        if (device_builder_factory().contains(system_id) || device_builder_factory().contains(SYSTEM_FX3_UNKNOWN)) {
            ret.push_back(serial);
        }
    }
    return ret;
}

Metavision::CameraDiscovery::SystemList Fx3CameraDiscovery::list_available_sources() {
    Metavision::CameraDiscovery::SystemList system_list;
    Metavision::CameraDiscovery::SerialList ret = Fx3LibUSBBoardCommand::get_list_serial();

    for (auto serial : ret) {
        Fx3LibUSBBoardCommand cmd;
        cmd.open(serial);
        SystemId system_id = static_cast<SystemId>(cmd.get_system_id());
        if (device_builder_factory().contains(system_id) || device_builder_factory().contains(SYSTEM_FX3_UNKNOWN)) {
            system_list.push_back({serial, Metavision::USB_LINK});
        }
    }
    return system_list;
}

bool Fx3CameraDiscovery::discover(Metavision::DeviceBuilder &device_builder, const std::string &serial,
                                  const Metavision::DeviceConfig &config) {
    Metavision::CameraDiscovery::SerialList ret = list();
    if (ret.empty())
        return false;
    if (serial != "" && std::find(ret.begin(), ret.end(), serial) == ret.end())
        return false;

    auto fx3boardcmd = std::make_shared<Fx3LibUSBBoardCommand>();
    if (!fx3boardcmd->open(serial)) {
        return false;
    }

    long fx3_version = fx3boardcmd->get_board_version();
    long system_id   = fx3boardcmd->get_system_id();
    if (fx3_version == 2) {
        DeviceBuilderParameters params(fx3boardcmd);
        if (device_builder_factory().build(system_id, device_builder, params, config)) {
            return true;
        } else {
            return device_builder_factory().build(SYSTEM_FX3_UNKNOWN, device_builder, params, config);
        }
    } else {
        MV_HAL_LOG_ERROR() << "####### Fx3 Version != 2";
    }

    return false;
}

} // namespace Metavision
