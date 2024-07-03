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

#include "metavision/psee_hw_layer/devices/treuzell/tz_psee_fpga_device.h"
#include "metavision/psee_hw_layer/boards/treuzell/tz_libusb_board_command.h"
#include <thread>
#include <sstream>
#include "metavision/hal/utils/hal_log.h"

#define STEREO_SYSTEM_CONFIG_ID_ADDR 0x00000800
#define STEREO_SYSTEM_CONFIG_VERSION_ADDR 0x00000804
#define STEREO_SYSTEM_CONFIG_BUILD_DATE_ADDR 0x00000808
#define STEREO_SYSTEM_CONFIG_VERSION_CONTROL_ID_ADDR 0x0000080c

namespace Metavision {

TzPseeFpgaDevice::TzPseeFpgaDevice() {}

void TzPseeFpgaDevice::get_device_info(I_HW_Identification::SystemInfo &infos, std::string prefix) {
    TzDevice::get_device_info(infos, prefix);

    infos.insert({prefix + std::to_string(tzID) + " system ID", std::to_string(get_system_id())});

    long system_version = get_system_version();
    std::string version = std::to_string((system_version >> 16) & 0xFF) + "." +
                          std::to_string((system_version >> 8) & 0xFF) + "." +
                          std::to_string((system_version >> 0) & 0xFF);
    infos.insert({prefix + std::to_string(tzID) + " version", version});

    time_t system_build_date              = (time_t)get_system_build_date();
    const char *system_build_date_charptr = asctime(localtime(&system_build_date));
    std::string system_build_date_str = system_build_date_charptr == 0 ? "NA" : std::string(system_build_date_charptr);
    system_build_date_str.pop_back();
    infos.insert({prefix + std::to_string(tzID) + " build date", system_build_date_str});

    std::stringstream system_stream;
    system_stream << "0x" << std::hex << get_system_version_control_id();
    std::string system_result(system_stream.str());
    infos.insert({prefix + std::to_string(tzID) + " VCS commit", system_result});
}

uint32_t TzPseeFpgaDevice::get_system_id() const {
    try {
        return cmd->read_device_register(tzID, STEREO_SYSTEM_CONFIG_ID_ADDR)[0];
    } catch (const std::system_error &e) {
        MV_HAL_LOG_WARNING() << "Could not fetch" << name() << "system_id" << e.what();
        return 0;
    }
}

uint32_t TzPseeFpgaDevice::get_system_version() const {
    try {
        return cmd->read_device_register(tzID, STEREO_SYSTEM_CONFIG_VERSION_ADDR)[0];
    } catch (const std::system_error &e) {
        MV_HAL_LOG_WARNING() << "Could not fetch" << name() << "system_version" << e.what();
        return 0;
    }
}

uint32_t TzPseeFpgaDevice::get_system_build_date() const {
    try {
        return cmd->read_device_register(tzID, STEREO_SYSTEM_CONFIG_BUILD_DATE_ADDR)[0];
    } catch (const std::system_error &e) {
        MV_HAL_LOG_WARNING() << "Could not fetch" << name() << "system_build_date" << e.what();
        return 0;
    }
}

uint32_t TzPseeFpgaDevice::get_system_version_control_id() const {
    try {
        return cmd->read_device_register(tzID, STEREO_SYSTEM_CONFIG_VERSION_CONTROL_ID_ADDR)[0];
    } catch (const std::system_error &e) {
        MV_HAL_LOG_WARNING() << "Could not fetch" << name() << "system_version_control_id" << e.what();
        return 0;
    }
}

} // namespace Metavision
