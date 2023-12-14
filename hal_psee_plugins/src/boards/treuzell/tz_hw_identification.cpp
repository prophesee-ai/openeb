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

#include <ctime>
#include <sstream>

#include "metavision/psee_hw_layer/facilities/psee_device_control.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_hw_identification.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_device.h"
#include "metavision/psee_hw_layer/devices/treuzell/tz_main_device.h"
#include "metavision/psee_hw_layer/boards/rawfile/psee_raw_file_header.h"
#include "devices/utils/device_system_id.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_control_frame.h"
#include "boards/treuzell/treuzell_command_definition.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

TzHWIdentification::TzHWIdentification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info,
                                       const std::shared_ptr<BoardCommand> &cmd,
                                       std::vector<std::shared_ptr<TzDevice>> &devices) :
    I_HW_Identification(plugin_sw_info), icmd_(cmd), sensor_info_({0, 0}), devices_(devices) {
    if (!icmd_) {
        throw(HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Board command is null."));
    }
}

std::string TzHWIdentification::get_serial() const {
    return icmd_->get_serial();
}

long TzHWIdentification::get_system_id() const {
    for (auto dev : devices_) {
        if (auto main_dev = dynamic_cast<TzMainDevice *>(dev.get()))
            return main_dev->get_system_id();
    }
    return 0;
}

I_HW_Identification::SensorInfo TzHWIdentification::get_sensor_info() const {
    for (auto dev : devices_) {
        if (auto main_dev = dynamic_cast<TzMainDevice *>(dev.get()))
            return main_dev->get_sensor_info();
    }
    return sensor_info_;
}

std::vector<std::string> TzHWIdentification::get_available_data_encoding_formats() const {
    std::vector<std::string> available_formats;

    if (!devices_.empty()) {
        for (auto &f : devices_[0]->get_supported_formats()) {
            available_formats.push_back(f.name());
        }
    }
    return available_formats;
}

std::string TzHWIdentification::get_current_data_encoding_format() const {
    return devices_[0]->get_output_format().name();
}

std::string TzHWIdentification::get_integrator() const {
    return icmd_->get_manufacturer();
}

I_HW_Identification::SystemInfo TzHWIdentification::get_system_info() const {
    auto infos = I_HW_Identification::get_system_info();

    long board_version  = icmd_->get_version();
    std::string version = std::to_string((board_version >> 16) & 0xFF) + "." +
                          std::to_string((board_version >> 8) & 0xFF) + "." +
                          std::to_string((board_version >> 0) & 0xFF);
    infos.insert({icmd_->get_name() + " Release Version", version});

    time_t build_date                    = icmd_->get_build_date();
    const char *board_build_date_charptr = asctime(localtime(&build_date));
    std::string board_build_date_str     = board_build_date_charptr == 0 ? "NA" : std::string(board_build_date_charptr);
    board_build_date_str.pop_back();
    infos.insert({icmd_->get_name() + " Build Date", board_build_date_str});

    infos.insert({icmd_->get_name() + " Speed", std::to_string(icmd_->get_board_speed())});

    for (auto dev : devices_)
        dev->get_device_info(infos, "device");

    return infos;
}

std::string TzHWIdentification::get_connection_type() const {
    return "USB";
}

RawFileHeader TzHWIdentification::get_header_impl() const {
    auto format = devices_[0]->get_output_format();
    PseeRawFileHeader header(*this, format);
    return header;
}

DeviceConfigOptionMap TzHWIdentification::get_device_config_options_impl() const {
    DeviceConfigOptionMap res;

    for (const auto &p : devices_[0]->get_device_config_options()) {
        res[p.first] = p.second;
    }

    return res;
}

} // namespace Metavision
