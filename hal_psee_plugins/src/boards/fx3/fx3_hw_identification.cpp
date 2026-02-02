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

#include <iomanip>
#include <sstream>

#include "metavision/psee_hw_layer/boards/rawfile/psee_raw_file_header.h"
#include "metavision/psee_hw_layer/boards/fx3/fx3_libusb_board_command.h"
#include "boards/fx3/fx3_hw_identification.h"
#include "metavision/psee_hw_layer/facilities/psee_device_control.h"
#include "devices/utils/device_system_id.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"
#include "geometries/hd_geometry.h"
#include "geometries/vga_geometry.h"
#include "metavision/hal/utils/hal_log.h"

namespace Metavision {

Fx3HWIdentification::Fx3HWIdentification(const std::shared_ptr<Metavision::I_PluginSoftwareInfo> &plugin_sw_info,
                                         const std::shared_ptr<Fx3LibUSBBoardCommand> &board_cmd,
                                         const std::shared_ptr<PseeDeviceControl> &device_ctrl,
                                         const std::string &integrator) :
    Metavision::I_HW_Identification(plugin_sw_info),
    icmd_(board_cmd),
    dev_ctrl_(device_ctrl),
    integrator_(integrator),
    sensor_info_("Gen0.0") {
    if (!icmd_) {
        throw(Metavision::HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Board command is null."));
    }
    sensor_info_ = get_sensor_info();
}

std::string Fx3HWIdentification::get_serial() const {
    return icmd_->get_serial();
}

std::string Fx3HWIdentification::get_integrator() const {
    return integrator_;
}
std::string Fx3HWIdentification::get_connection_type() const {
    return "USB";
}

std::vector<std::string> Fx3HWIdentification::get_available_data_encoding_formats() const {
    auto sensor_info = get_sensor_info();
    std::vector<std::string> available_formats;
    available_formats.push_back("EVT2");
    if (sensor_info.major_version_ == 4) {
        available_formats.push_back("EVT3");
    }
    return available_formats;
}

std::string Fx3HWIdentification::get_current_data_encoding_format() const {
    return dev_ctrl_->get_evt_format().name();
}

Metavision::I_HW_Identification::SensorInfo Fx3HWIdentification::get_sensor_info() const {
    if (sensor_info_.major_version_ != 0) {
        return sensor_info_;
    }
    Metavision::I_HW_Identification::SensorInfo sensor_info;
    long system_id = icmd_->get_system_id();
    systemid2version(system_id, sensor_info.major_version_, sensor_info.minor_version_);
    return sensor_info;
}

Metavision::I_HW_Identification::SystemInfo Fx3HWIdentification::get_system_info() const {
    auto infos = Metavision::I_HW_Identification::get_system_info();
    infos.insert({"FX3 ID", std::to_string(icmd_->get_board_id())});
    long fx3_version    = icmd_->get_board_release_version();
    std::string version = std::to_string((fx3_version >> 16) & 0xFF) + "." + std::to_string((fx3_version >> 8) & 0xFF) +
                          "." + std::to_string((fx3_version >> 0) & 0xFF);
    infos.insert({"FX3 Release Version", version});
    time_t fx3_build_date              = (time_t)icmd_->get_board_build_date();
    const char *fx3_build_date_charptr = asctime(localtime(&fx3_build_date));
    std::string fx3_build_date_str     = fx3_build_date_charptr == 0 ? "NA" : std::string(fx3_build_date_charptr);
    fx3_build_date_str.pop_back();
    infos.insert({"FX3 Build Date", fx3_build_date_str});
    std::stringstream stream;
    stream << "0x" << std::hex << icmd_->get_board_version_control_id();
    std::string result(stream.str());
    infos.insert({"FX3 Version Control ID", result});
    infos.insert({"FX3 Speed", std::to_string(icmd_->get_board_speed())});
    time_t system_build_date              = (time_t)icmd_->get_system_build_date();
    const char *system_build_date_charptr = asctime(localtime(&system_build_date));
    std::string system_build_date_str = system_build_date_charptr == 0 ? "NA" : std::string(system_build_date_charptr);
    system_build_date_str.pop_back();
    infos.insert({"System Build Date", system_build_date_str});
    std::stringstream system_stream;
    system_stream << "0x" << std::hex << icmd_->get_system_version_control_id();
    std::string system_result(system_stream.str());
    infos.insert({"System Version Control ID", system_result});
    return infos;
}

Metavision::RawFileHeader Fx3HWIdentification::get_header_impl() const {
    const StreamFormat &format = dev_ctrl_->get_evt_format();
    PseeRawFileHeader header(*this, format);
    return header;
}

DeviceConfigOptionMap Fx3HWIdentification::get_device_config_options_impl() const {
    return {};
}

} // namespace Metavision
