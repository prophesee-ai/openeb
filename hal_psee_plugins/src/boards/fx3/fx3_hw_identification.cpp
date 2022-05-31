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

#include "boards/rawfile/psee_raw_file_header.h"
#include "facilities/psee_device_control.h"
#include "boards/utils/psee_libusb_board_command.h"
#include "boards/fx3/fx3_hw_identification.h"
#include "devices/utils/device_system_id.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"
#include "geometries/hd_geometry.h"
#include "geometries/vga_geometry.h"

namespace Metavision {

Fx3HWIdentification::Fx3HWIdentification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info,
                                         const std::shared_ptr<PseeLibUSBBoardCommand> &board_cmd, bool is_EVT3,
                                         long subsystem_ID, const std::string &integrator) :
    I_HW_Identification(plugin_sw_info),
    icmd_(board_cmd),
    sensor_info_({0, 0}),
    is_evt3_(is_EVT3),
    subsystem_ID_(subsystem_ID),
    integrator_(integrator) {
    if (!icmd_) {
        throw(HalException(PseeHalPluginErrorCode::BoardCommandNotFound, "Board command is null."));
    }
    sensor_info_ = get_sensor_info();
}

std::string Fx3HWIdentification::get_serial() const {
    return icmd_->get_serial();
}
long Fx3HWIdentification::get_system_id() const {
    return icmd_->get_system_id();
}

std::string Fx3HWIdentification::get_integrator() const {
    return integrator_;
}
std::string Fx3HWIdentification::get_connection_type() const {
    return "USB";
}

std::vector<std::string> Fx3HWIdentification::get_available_raw_format() const {
    auto sensor_info = get_sensor_info();
    std::vector<std::string> available_formats;
    available_formats.push_back("EVT2");
    if (sensor_info.major_version_ == 4) {
        available_formats.push_back("EVT3");
    }
    return available_formats;
}

I_HW_Identification::SensorInfo Fx3HWIdentification::get_sensor_info() const {
    if (sensor_info_.major_version_ != 0) {
        return sensor_info_;
    }
    I_HW_Identification::SensorInfo sensor_info;
    long system_id = icmd_->get_system_id();
    systemid2version(system_id, sensor_info.major_version_, sensor_info.minor_version_);
    return sensor_info;
}

I_HW_Identification::SystemInfo Fx3HWIdentification::get_system_info() const {
    auto infos = I_HW_Identification::get_system_info();
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

long Fx3HWIdentification::get_system_version() const {
    return icmd_->get_system_version();
}

RawFileHeader Fx3HWIdentification::get_header_impl() const {
    // By chance, we won't build Evk1 with non-HD Gen4.x sensors
    auto is_HD           = (get_sensor_info().major_version_ == 4);
    auto hd              = HDGeometry();
    auto vga             = VGAGeometry();
    I_Geometry &geometry = is_HD ? static_cast<I_Geometry &>(hd) : static_cast<I_Geometry &>(vga);
    PseeRawFileHeader header(*this, geometry);
    header.set_sub_system_id(subsystem_ID_);
    header.set_format(is_evt3_ ? "EVT3" : "EVT2");
    return header;
}

} // namespace Metavision
