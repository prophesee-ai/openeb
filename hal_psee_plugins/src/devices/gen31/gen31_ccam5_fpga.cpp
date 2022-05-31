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

#include "devices/gen31/gen31_ccam5_fpga.h"
#include "geometries/vga_geometry.h"
#include "metavision/hal/utils/hal_log.h"
#include "utils/register_map.h"

namespace Metavision {
namespace {
std::string SYSTEM_CONFIG_PREFIX  = "SYSTEM_CONFIG/";
std::string SYSTEM_CONTROL_PREFIX = "SYSTEM_CONTROL/";
std::string SYSTEM_MONITOR_PREFIX = "SYSTEM_MONITOR/";
} // namespace

using vfield = std::map<std::string, uint32_t>;

Gen31CCam5Fpga::Gen31CCam5Fpga(const std::shared_ptr<RegisterMap> &regmap, const std::string &root_prefix,
                               const std::string &sensor_if_prefix) :
    root_prefix_(root_prefix),
    sensor_if_prefix_(sensor_if_prefix),
    register_map_(regmap),
    sys_ctrl_(register_map_, get_system_control_prefix()),
    sensor_if_(register_map_, get_sensor_if_prefix()) {}

void Gen31CCam5Fpga::set_timebase_master(bool enable) {
    timebase_master_ = enable;
}

void Gen31CCam5Fpga::set_timebase_ext_sync(bool enable) {
    timebase_ext_sync_ = enable;
}

bool Gen31CCam5Fpga::get_timebase_master() const {
    return timebase_master_;
}

bool Gen31CCam5Fpga::get_timebase_ext_sync() const {
    return timebase_ext_sync_;
}

std::string Gen31CCam5Fpga::get_root_prefix() const {
    return root_prefix_;
}

std::string Gen31CCam5Fpga::get_sensor_if_prefix() const {
    return sensor_if_prefix_;
}

std::string Gen31CCam5Fpga::get_system_config_prefix() const {
    return root_prefix_ + SYSTEM_CONFIG_PREFIX;
}

std::string Gen31CCam5Fpga::get_system_control_prefix() const {
    return root_prefix_ + SYSTEM_CONTROL_PREFIX;
}

std::string Gen31CCam5Fpga::get_system_monitor_prefix() const {
    return root_prefix_ + SYSTEM_MONITOR_PREFIX;
}

void Gen31CCam5Fpga::init() {
    MV_HAL_LOG_TRACE() << "CCam5 Gen31 Init";
    sys_ctrl_.sensor_atis_control_clear();
    sys_ctrl_.soft_reset("CCAM2_TRIGGER");
    sys_ctrl_.sensor_prepowerup();
    sys_ctrl_.timebase_config(timebase_ext_sync_, timebase_master_);
    // TODO: configure IODELAY
    // self.sensor_if__ioctrl = SensorIfIoctrl(sensor_if__ioctrl_rmap)
    sys_ctrl_.host_if_control(true);
    (*register_map_)[get_root_prefix() + "MIPI_TX/FRAME_PERIOD"]["VALUE_US"]  = 0x3e8;
    (*register_map_)[get_root_prefix() + "MIPI_TX/INTER_FRAME_TIME"]["VALUE"] = 0x640;
    (*register_map_)[get_root_prefix() + "MIPI_TX/CONTROL"]["ENABLE"]         = true;

    (*register_map_)[get_system_control_prefix() + "OUT_OF_FOV_FILTER_SIZE"]["WIDTH"]    = VGAGeometry::width_;
    (*register_map_)[get_system_control_prefix() + "OUT_OF_FOV_FILTER_ORIGIN"]["X"]      = 0;
    (*register_map_)[get_system_control_prefix() + "OUT_OF_FOV_FILTER_SIZE"]["VALUE"]    = VGAGeometry::height_;
    (*register_map_)[get_system_control_prefix() + "OUT_OF_FOV_FILTER_ORIGIN"]["Y"]      = 0;
    (*register_map_)[get_system_control_prefix() + "CCAM2_CONTROL"]["ENABLE_OUT_OF_FOV"] = true;

    sensor_if_.sensor_turn_on_clock();
}

void Gen31CCam5Fpga::start() {
    MV_HAL_LOG_TRACE() << "CCam5 Gen31 Start";
    sys_ctrl_.sensor_powerup();
    sys_ctrl_.timebase_control(true);
}
void Gen31CCam5Fpga::stop() {
    MV_HAL_LOG_TRACE() << "CCam5 Gen31 Stop";
    sys_ctrl_.timebase_control(false);
    sys_ctrl_.sensor_prepowerdown();
}
void Gen31CCam5Fpga::destroy() {
    MV_HAL_LOG_TRACE() << "CCam5 Gen31 Destroy";
    sensor_if_.sensor_turn_off_clock();
    sys_ctrl_.sensor_powerdown();
    sys_ctrl_.soft_reset("CCAM2_TRIGGER");
}
} // namespace Metavision
