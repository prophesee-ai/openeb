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
#include <iostream>
#include <chrono>
#include <thread>

#include "boards/utils/config_registers_map.h"
#include "devices/gen31/ccam_gen31_helpers.h"
#include "devices/gen31/gen31_device_control.h"
#include "devices/gen31/gen31_evk1_fpga.h"
#include "devices/gen31/gen31_sensor.h"
#include "devices/utils/device_system_id.h"
#include "boards/utils/psee_libusb_board_command.h"
#include "facilities/psee_trigger_in.h"
#include "facilities/psee_trigger_out.h"
#include "geometries/hvga_geometry.h"
#include "geometries/vga_geometry.h"
#include "utils/register_map.h"

namespace Metavision {
using vfield = std::map<std::string, uint32_t>;

Gen31DeviceControl::Gen31DeviceControl(const std::shared_ptr<RegisterMap> &register_map,
                                       const std::shared_ptr<Gen31Fpga> &fpga,
                                       const std::shared_ptr<Gen31Sensor> &sensor) :
    PseeDeviceControl(EvtFormat::EVT2_0), register_map_(register_map), fpga_(fpga), sensor_(sensor) {}

void Gen31DeviceControl::terminate_camera() {
    sensor_->destroy();
    fpga_->destroy();
}

void Gen31DeviceControl::fpga_init() {
    fpga_->init();
}

void Gen31DeviceControl::start_impl() {
    fpga_->start();
    sensor_->start();
}

void Gen31DeviceControl::destroy_camera() {
    sensor_->destroy();
    fpga_->destroy();
}

void Gen31DeviceControl::sensor_init() {
    sensor_->init();
}

void Gen31DeviceControl::start_camera_common(bool is_gen31EM, bool allow_dual_readout) {
    fpga_->start();
    sensor_->start();
}

void Gen31DeviceControl::stop_camera_common() {
    sensor_->stop();
    fpga_->stop();
}

void Gen31DeviceControl::reset() {}

void Gen31DeviceControl::reset_ts_internal() {
    /* fpga_->soft_reset(); */
}

long long Gen31DeviceControl::get_sensor_id() {
    return get_sensor_id((*register_map_));
}

long long Gen31DeviceControl::get_sensor_id(RegisterMap &register_map) {
    return register_map["SENSOR_IF/GEN31/chip_id"].read_value();
}

bool Gen31DeviceControl::is_gen31EM() {
    long sensor_id = get_sensor_id();
    return is_gen31EM(sensor_id);
}

bool Gen31DeviceControl::is_gen31EM(long sensor_id) {
    if (getenv("FORCE_GEN31EM")) {
        return true;
    }
    for (const auto &id : EM_SUBSYSTEM_IDS.at(SystemId::SYSTEM_CCAM3_GEN31)) {
        if ((sensor_id & 0xFFFF) == id) {
            return true;
        }
    }
    return false;
}

bool Gen31DeviceControl::is_gen31EM(RegisterMap &register_map) {
    auto sensor_id = get_sensor_id(register_map);
    return is_gen31EM(sensor_id);
}

bool Gen31DeviceControl::set_evt_format_impl(EvtFormat fmt) {
    return false;
}

bool Gen31DeviceControl::set_mode_standalone_impl() {
    (*register_map_)["SYSTEM_CONTROL/ATIS_CONTROL"]["MASTER_MODE"]   = 0x1;
    (*register_map_)["SYSTEM_CONTROL/ATIS_CONTROL"]["USE_EXT_START"] = 0x0;
    return true;
}

bool Gen31DeviceControl::set_mode_master_impl() {
    if (get_trigger_out()->is_enabled()) {
        return false;
    }
    (*register_map_)["SYSTEM_CONTROL/ATIS_CONTROL"]["MASTER_MODE"]   = 0x1;
    (*register_map_)["SYSTEM_CONTROL/ATIS_CONTROL"]["USE_EXT_START"] = 0x1;
    return true;
}

bool Gen31DeviceControl::set_mode_slave_impl() {
    if (get_trigger_in()->is_enabled(7)) {
        return false;
    }
    (*register_map_)["SYSTEM_CONTROL/ATIS_CONTROL"]["MASTER_MODE"]   = 0x0;
    (*register_map_)["SYSTEM_CONTROL/ATIS_CONTROL"]["USE_EXT_START"] = 0x1;
    return true;
}

} // namespace Metavision
