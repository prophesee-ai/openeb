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

#include "metavision/hal/facilities/i_events_stream.h"

#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_control_frame.h"
#include "metavision/psee_hw_layer/boards/treuzell/board_command.h"
#include "boards/treuzell/treuzell_command_definition.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_hw_identification.h"
#include "metavision/psee_hw_layer/boards/utils/psee_libusb_data_transfer.h"
#include "metavision/psee_hw_layer/facilities/tz_camera_synchronization.h"
#include "metavision/psee_hw_layer/facilities/tz_monitoring.h"
#include "metavision/psee_hw_layer/facilities/tz_hw_register.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/psee_hw_layer/utils/tz_device_control.h"
#include "devices/treuzell/tz_device_builder.h"
#include "utils/make_decoder.h"

namespace Metavision {

bool TzDeviceBuilder::can_build(std::shared_ptr<BoardCommand> cmd) {
    try {
        auto device_count = cmd->get_device_count();
        MV_HAL_LOG_TRACE() << cmd->get_name() << "has" << device_count << "Treuzell devices";

        for (uint32_t i = 0; i < device_count; i++)
            if (!can_build_device(cmd, i))
                return false;
        return true;
    } catch (const std::system_error &e) {
        MV_HAL_LOG_WARNING() << "Could not enumerate devices on board" << cmd->get_name() << e.what();
        return false;
    }
}

std::vector<TzDeviceBuilder::Build_Fun> TzDeviceBuilder::get_build_fun(std::shared_ptr<BoardCommand> cmd,
                                                                       uint32_t dev_id) const {
    std::vector<TzDeviceBuilder::Build_Fun> build_fun;
    std::vector<std::string> compat_str;
    std::string name_str;

    try {
        TzDeviceStringsCtrlFrame compat(TZ_PROP_DEVICE_COMPATIBLE, dev_id);
        cmd->transfer_tz_frame(compat);
        compat_str = compat.get_strings();

        // Get device name for nicer debug traces
        try {
            TzDeviceStringsCtrlFrame name(TZ_PROP_DEVICE_NAME, dev_id);
            cmd->transfer_tz_frame(name);
            name_str = name.get_strings()[0];
        } catch (const std::system_error &e2) { name_str = "device" + std::to_string(dev_id); }
    } catch (const std::system_error &e) {
        // On some old devices (treuzell-kernel 1.4.0, cx3 3.0.0), a compatibility string was used as name
        try {
            TzDeviceStringsCtrlFrame name(TZ_PROP_DEVICE_NAME, dev_id);
            cmd->transfer_tz_frame(name);
            compat_str = name.get_strings();
            name_str   = name.get_strings()[0];
        } catch (const std::system_error &e2) {
            MV_HAL_LOG_WARNING() << "Failed to get compatibility string from treuzell device" << dev_id << e.what();
            return build_fun;
        }
    }
    // This allow to have a fallback driver handling unknown devices with standard Treuzell commands
    compat_str.push_back("");

    for (auto str : compat_str) {
        auto build = map.find(str);
        if (build != map.end()) {
            // if the device has a specific can_build method in the <Build_Fun,Check_Fun> pair, call it
            if (!build->second.second || build->second.second(cmd, dev_id)) {
                if (str != "")
                    MV_HAL_LOG_TRACE() << name_str << "is compatible with" << str;
                build_fun.push_back(build->second.first);
                continue;
            }
            MV_HAL_LOG_TRACE() << "Driver compatible with" << str << "can't build" << name_str;
        } else {
            if (str != "")
                MV_HAL_LOG_TRACE() << "Found no driver compatible with" << str;
        }
    }
    MV_HAL_LOG_TRACE() << "Got" << build_fun.size() << "build method(s) for" << name_str;
    return build_fun;
}

bool TzDeviceBuilder::can_build_device(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id) {
    return !get_build_fun(cmd, dev_id).empty();
}

using Devices = std::vector<std::shared_ptr<TzDevice>>;

template<class T>
std::shared_ptr<T> get_provider(Devices &devices) {
    for (auto &device : devices) {
        auto tdev = std::dynamic_pointer_cast<T>(device);
        if (tdev)
            return tdev;
    }
    return nullptr;
}

bool TzDeviceBuilder::build_devices(std::shared_ptr<BoardCommand> cmd,
                                    Metavision::DeviceBuilder &device_builder, const Metavision::DeviceConfig &config) {
    auto device_count = cmd->get_device_count();
    Devices devices;
    std::shared_ptr<TzDevice> dev;

    for (uint32_t i = 0; i < device_count; i++) {
        std::shared_ptr<TzDevice> next;

        auto build_fun = get_build_fun(cmd, i);
        for (auto build : build_fun) {
            try {
                next = build(cmd, i, dev);
                if (next) {
                    devices.push_back(next);
                    if (dev)
                        dev->set_child(next);
                    dev         = next;
                    auto format = config.get<std::string>("format", "");
                    if ((format != "") && (dev->set_output_format(format).name() != format)) {
                        MV_HAL_LOG_TRACE() << "Unsupported format: " << format << ", using default format for device";
                    }
                    dev->spawn_facilities(device_builder, config);
                    break;
                }
            } catch (const std::system_error &e) { MV_HAL_LOG_TRACE() << "Fail to build device" << i << e.what(); }
        }
    }
    if (devices.empty())
        return false;
    auto hw_identification = device_builder.add_facility(
        std::make_unique<TzHWIdentification>(device_builder.get_plugin_software_info(), cmd, devices));
    auto ctrl = std::make_shared<TzDeviceControl>(devices);
    try {
        size_t raw_size_bytes = 0;
        auto format           = devices[0]->get_output_format();
        auto decoder          = make_decoder(device_builder, format, raw_size_bytes, false);
        device_builder.add_facility(std::make_unique<Metavision::I_EventsStream>(
            cmd->build_data_transfer(raw_size_bytes), hw_identification, decoder, ctrl));
    } catch (std::exception &e) { MV_HAL_LOG_WARNING() << "System can't stream:" << e.what(); }

    std::shared_ptr<TemperatureProvider> temp        = get_provider<TemperatureProvider>(devices);
    std::shared_ptr<IlluminationProvider> illu       = get_provider<IlluminationProvider>(devices);
    std::shared_ptr<PixelDeadTimeProvider> dead_time = get_provider<PixelDeadTimeProvider>(devices);

    if (temp || illu || dead_time)
        device_builder.add_facility(std::make_unique<TzMonitoring>(temp, illu, dead_time));

    auto sync = device_builder.add_facility(std::make_unique<TzCameraSynchronization>(devices, ctrl));
    // Every component should start as standalone, this call goes to the first self-declared main-device,
    // which will in turn set each element of the pipeline in a mode that will make the camera standalone
    sync->set_mode_standalone();

    try {
        device_builder.add_facility(std::make_unique<TzHwRegister>(devices));
    } catch (const std::system_error &e) {
        MV_HAL_LOG_TRACE() << "Did not instantiate a HwRegister facility:" << e.what();
    }

    return true;
}

TzDeviceBuilder::Build_Map &TzDeviceBuilder::generic_map() {
    static Build_Map static_map;
    return static_map;
}

} // namespace Metavision
