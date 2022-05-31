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

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include "devices/treuzell/tz_device.h"
#include "boards/treuzell/tz_libusb_board_command.h"
#include "boards/treuzell/tz_control_frame.h"
#include "boards/treuzell/treuzell_command_definition.h"
#include "metavision/hal/utils/hal_log.h"

// For system build
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/device_config.h"
#include <functional>

// For board-wide facilities
#include "boards/treuzell/tz_hw_identification.h"
#include "decoders/evt2/evt2_decoder.h"
#include "decoders/evt3/evt3_decoder.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "boards/treuzell/tz_board_data_transfer.h"
#include "devices/treuzell/tz_device_control.h"
#include "facilities/tz_monitoring.h"

namespace Metavision {

TzDevice::TzDevice(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent) :
    cmd(cmd), tzID(dev_id), parent(parent) {}

TzDevice::~TzDevice() {}

std::string TzDevice::get_name() {
    TzDeviceStringsCtrlFrame name(TZ_PROP_DEVICE_NAME, tzID);
    cmd->transfer_tz_frame(name);
    return name.get_strings()[0];
}

std::vector<std::string> TzDevice::get_compatible() {
    TzDeviceStringsCtrlFrame compat(TZ_PROP_DEVICE_COMPATIBLE, tzID);
    cmd->transfer_tz_frame(compat);
    return compat.get_strings();
}

void TzDevice::get_device_info(I_HW_Identification::SystemInfo &infos, std::string prefix) {
    try {
        infos.insert({prefix + std::to_string(tzID) + " name", get_name()});
    } catch (const std::system_error &e) { MV_HAL_LOG_TRACE() << "Dev" << tzID << "got no name string:" << e.what(); }

    try {
        for (auto str : get_compatible())
            infos.insert({prefix + std::to_string(tzID) + " compatible", str});
    } catch (const std::system_error &e) { MV_HAL_LOG_TRACE() << "Dev" << tzID << "got no compat string:" << e.what(); }
}

bool TzDeviceBuilder::can_build(std::shared_ptr<TzLibUSBBoardCommand> cmd) {
    try {
        auto device_count = cmd->get_device_count();

        for (uint32_t i = 0; i < device_count; i++)
            if (!can_build_device(cmd, i))
                return false;
        return true;
    } catch (const std::system_error &e) {
        MV_HAL_LOG_WARNING() << "Could not enumerate devices on board" << cmd->get_name() << e.what();
        return false;
    }
}

bool TzDeviceBuilder::can_build_device(std::shared_ptr<TzLibUSBBoardCommand> cmd, uint32_t dev_id) {
    std::vector<std::string> compat_str;
    try {
        TzDeviceStringsCtrlFrame compat(TZ_PROP_DEVICE_COMPATIBLE, dev_id);
        cmd->transfer_tz_frame(compat);
        compat_str = compat.get_strings();
    } catch (const std::system_error &e) {
        // On some old devices (treuzell-kernel 1.4.0, cx3 3.0.0), a compatibility string was used as name
        try {
            TzDeviceStringsCtrlFrame name(TZ_PROP_DEVICE_NAME, dev_id);
            cmd->transfer_tz_frame(name);
            compat_str = name.get_strings();
        } catch (const std::system_error &e2) {
            MV_HAL_LOG_WARNING() << "Failed to get compatibility string from treuzell device" << dev_id << e.what();
            return false;
        }
    }

    for (auto str : compat_str) {
        auto build = map.find(str);
        if (build != map.end()) {
            // if the device has a specific can_build method in the <Build_Fun,Check_Fun> pair, call it
            if (!build->second.second || build->second.second(cmd, dev_id))
                return true;
        }
    }
    return false;
}

bool TzDeviceBuilder::build_devices(std::shared_ptr<TzLibUSBBoardCommand> cmd, DeviceBuilder &device_builder,
                                    const DeviceConfig &config) {
    auto device_count = cmd->get_device_count();
    std::vector<std::shared_ptr<TzDevice>> devices;
    std::shared_ptr<TzDevice> dev;

    for (uint32_t i = 0; i < device_count; i++) {
        std::vector<std::string> compat_str;
        std::string name;
        std::shared_ptr<TzDevice> next;

        try {
            TzDeviceStringsCtrlFrame name_frame(TZ_PROP_DEVICE_NAME, i);
            cmd->transfer_tz_frame(name_frame);
            name = name_frame.get_strings()[0];
        } catch (const std::system_error &e) {
            MV_HAL_LOG_INFO() << "Could not get name for treuzell device" << i << e.what();
        }

        try {
            TzDeviceStringsCtrlFrame compat(TZ_PROP_DEVICE_COMPATIBLE, i);
            cmd->transfer_tz_frame(compat);
            compat_str = compat.get_strings();
        } catch (const std::system_error &e) {
            // On some old devices (treuzell-kernel 1.4.0, cx3 3.0.0), a compatibility string was used as name
            compat_str.push_back(name);
        }

        for (auto str : compat_str) {
            MV_HAL_LOG_TRACE() << "Checking str" << str;
            auto build = map.find(str);
            if (build != map.end()) {
                // Call the first member of the pair <Build_Fun,Check_Fun>
                next = build->second.first(cmd, i, dev);
                if (next) {
                    devices.push_back(next);
                    if (dev)
                        dev->child = next;
                    dev = next;
                    dev->spawn_facilities(device_builder);
                    break;
                }
            } else {
                MV_HAL_LOG_TRACE() << "Found no build method for " << str;
            }
        }
    }
    if (devices.empty())
        return false;
    auto hw_identification = device_builder.add_facility(
        std::make_unique<TzHWIdentification>(device_builder.get_plugin_software_info(), cmd, devices));
    auto format = devices[0]->get_output_format();
    std::shared_ptr<I_Geometry> geometry;
    if (format.geometry)
        geometry = device_builder.add_facility(std::move(format.geometry));
    auto cd_event_decoder          = device_builder.add_facility(std::make_unique<I_EventDecoder<EventCD>>());
    auto ext_trigger_event_decoder = device_builder.add_facility(std::make_unique<I_EventDecoder<EventExtTrigger>>());
    std::shared_ptr<I_Decoder> decoder;
    if (format.name == "EVT3" && geometry) {
        MV_HAL_LOG_TRACE() << "Adding EVT3 decoder";
        decoder = device_builder.add_facility(
            std::make_unique<EVT3Decoder>(false, geometry->get_height(), cd_event_decoder, ext_trigger_event_decoder));
    } else if (format.name == "EVT2") {
        MV_HAL_LOG_TRACE() << "Adding EVT2 decoder";
        decoder = device_builder.add_facility(
            std::make_unique<EVT2Decoder>(false, cd_event_decoder, ext_trigger_event_decoder));
    } else {
        MV_HAL_LOG_WARNING() << "System is streaming unknown format" << format.name;
    }
    if (decoder)
        device_builder.add_facility(std::make_unique<I_EventsStream>(
            std::make_unique<TzBoardDataTransfer>(cmd, decoder->get_raw_event_size_bytes()), hw_identification));

    std::shared_ptr<TemperatureProvider> temp;
    std::shared_ptr<IlluminationProvider> illu;
    for (auto device : devices) {
        auto tdev = std::dynamic_pointer_cast<TemperatureProvider>(device);
        if (tdev)
            temp = tdev;
    }
    for (auto device : devices) {
        auto idev = std::dynamic_pointer_cast<IlluminationProvider>(device);
        if (idev)
            illu = idev;
    }
    if (temp || illu)
        device_builder.add_facility(std::make_unique<TzMonitoring>(temp, illu));

    device_builder.add_facility(std::make_unique<TzDeviceControl>(devices));
    return true;
}

} // namespace Metavision
