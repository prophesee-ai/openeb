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

#include <functional>

#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_control_frame.h"
#include "boards/treuzell/treuzell_command_definition.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

TzDevice::TzDevice(std::shared_ptr<BoardCommand> cmd, uint32_t dev_id, std::shared_ptr<TzDevice> parent) :
    cmd(cmd), tzID(dev_id), parent(parent) {
    try {
        name = get_name();
        MV_HAL_LOG_TRACE() << "Dev" << tzID << "name:" << name;
    } catch (const std::system_error &e) {
        MV_HAL_LOG_TRACE() << "Dev" << tzID << "got no name string:" << e.what();
        name = "Dev" + std::to_string(tzID);
    }
}

TzDevice::~TzDevice() {}

void TzDevice::initialize() {
    TzGenericCtrlFrame enable(TZ_PROP_DEVICE_ENABLE | TZ_WRITE_FLAG);
    enable.push_back32(tzID);
    enable.push_back32(1);
    cmd->transfer_tz_frame(enable);
}

void TzDevice::destroy() {
    TzGenericCtrlFrame disable(TZ_PROP_DEVICE_ENABLE | TZ_WRITE_FLAG);
    disable.push_back32(tzID);
    disable.push_back32(0);
    cmd->transfer_tz_frame(disable);
}

void TzDevice::start() {
    TzGenericCtrlFrame start(TZ_PROP_DEVICE_STREAM | TZ_WRITE_FLAG);
    start.push_back32(tzID);
    start.push_back32(1);
    cmd->transfer_tz_frame(start);
}

void TzDevice::stop() {
    TzGenericCtrlFrame stop(TZ_PROP_DEVICE_STREAM | TZ_WRITE_FLAG);
    stop.push_back32(tzID);
    stop.push_back32(0);
    cmd->transfer_tz_frame(stop);
}

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

DeviceConfigOptionMap TzDevice::get_device_config_options() const {
    const auto formats = get_supported_formats();
    if (formats.size() > 1) {
        std::vector<std::string> values;
        for (auto &fmt : formats) {
            values.push_back(fmt.name());
        }
        return {{"format", DeviceConfigOption(values, values[0])}};
    }
    return {};
}

std::list<StreamFormat> TzDevice::get_supported_formats() const {
    std::list<StreamFormat> formats;
    TzDeviceStringsCtrlFrame format(TZ_PROP_DEVICE_OUTPUT_FORMAT, tzID);
    try {
        cmd->transfer_tz_frame(format);
        formats.push_back(StreamFormat(format.get_strings()[0]));
    } catch (const std::system_error &e) {
        MV_HAL_LOG_TRACE() << name << "did not advertise output format:" << e.what();
    }
    return formats;
}

StreamFormat TzDevice::get_output_format() const {
    TzDeviceStringsCtrlFrame format(TZ_PROP_DEVICE_OUTPUT_FORMAT, tzID);
    try {
        cmd->transfer_tz_frame(format);
        return StreamFormat(format.get_strings()[0]);
    } catch (const std::system_error &e) {
        MV_HAL_LOG_TRACE() << name << "did not advertise output format:" << e.what();
    }
    return StreamFormat("None");
}

StreamFormat TzDevice::set_output_format(const std::string &format_name) {
    TzDeviceStringsCtrlFrame format(TZ_PROP_DEVICE_OUTPUT_FORMAT | TZ_WRITE_FLAG, tzID);
    format.push_back(format_name);
    try {
        cmd->transfer_tz_frame(format);
        return StreamFormat(format.get_strings()[0]);
    } catch (const std::system_error &e) { MV_HAL_LOG_TRACE() << name << "did not set output format:" << e.what(); }
    /* spare the implementation of set_output_format when supporting only one format */
    return get_output_format();
}

void TzDevice::get_device_info(Metavision::I_HW_Identification::SystemInfo &infos, std::string prefix) {
    try {
        infos.insert({prefix + std::to_string(tzID) + " name", get_name()});
    } catch (const std::system_error &e) { MV_HAL_LOG_TRACE() << name << "got no name string:" << e.what(); }

    try {
        for (auto str : get_compatible())
            infos.insert({prefix + std::to_string(tzID) + " compatible", str});
    } catch (const std::system_error &e) { MV_HAL_LOG_TRACE() << name << "got no compat string:" << e.what(); }
}

void TzDevice::set_child(std::shared_ptr<TzDevice> dev) {
    child = dev;
}

} // namespace Metavision
