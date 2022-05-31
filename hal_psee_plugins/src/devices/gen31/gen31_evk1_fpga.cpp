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

#include "devices/gen31/gen31_evk1_fpga.h"
#include "geometries/vga_geometry.h"
#include "geometries/hvga_geometry.h"
#include "utils/register_map.h"

namespace Metavision {

Gen31Evk1Fpga::Gen31Evk1Fpga(const std::shared_ptr<RegisterMap> &regmap, bool is_em) :
    register_map_(regmap),
    sys_ctrl_(register_map_, "SYSTEM_CONTROL/"),
    sensor_if_(register_map_, "SENSOR_IF/GEN31_IF/"),
    is_em_(is_em) {}

void Gen31Evk1Fpga::init() {
    sys_ctrl_.sensor_atis_control_clear();
    sys_ctrl_.soft_reset("TRIGGERS");
    sys_ctrl_.sensor_prepowerup();
    sys_ctrl_.timebase_config(false, true);
    // TODO: configure IODELAY
    // self.sensor_if__ioctrl = SensorIfIoctrl(sensor_if__ioctrl_rmap)
    sys_ctrl_.host_if_control(true);
    sys_ctrl_.no_blocking_control(true);
    if (is_em_) {
        sys_ctrl_.hvga_remap_control(true);
        (*register_map_)["SYSTEM_CONTROL/OUT_OF_FOV_FILTER_X"]["WIDTH"]       = HVGAGeometry::width_;
        (*register_map_)["SYSTEM_CONTROL/OUT_OF_FOV_FILTER_X"]["ORIGIN"]      = 0;
        (*register_map_)["SYSTEM_CONTROL/OUT_OF_FOV_FILTER_Y"]["HEIGHT"]      = HVGAGeometry::height_;
        (*register_map_)["SYSTEM_CONTROL/OUT_OF_FOV_FILTER_X"]["ORIGIN"]      = 0;
        (*register_map_)["SYSTEM_CONTROL/CCAM2_CONTROL"]["ENABLE_OUT_OF_FOV"] = true;
    } else {
        sys_ctrl_.hvga_remap_control(false);
        (*register_map_)["SYSTEM_CONTROL/OUT_OF_FOV_FILTER_X"]["WIDTH"]       = VGAGeometry::width_;
        (*register_map_)["SYSTEM_CONTROL/OUT_OF_FOV_FILTER_X"]["ORIGIN"]      = 0;
        (*register_map_)["SYSTEM_CONTROL/OUT_OF_FOV_FILTER_Y"]["HEIGHT"]      = VGAGeometry::height_;
        (*register_map_)["SYSTEM_CONTROL/OUT_OF_FOV_FILTER_X"]["ORIGIN"]      = 0;
        (*register_map_)["SYSTEM_CONTROL/CCAM2_CONTROL"]["ENABLE_OUT_OF_FOV"] = true;
    }

    sensor_if_.sensor_turn_on_clock();
    sys_ctrl_.sensor_powerup();
    (*register_map_)["SYSTEM_MONITOR/TEMP_VCC_MONITOR/EXT_TEMP_CONTROL"]["EXT_TEMP_MONITOR_SPI_EN"] = true;
}

void Gen31Evk1Fpga::start() {
    sys_ctrl_.timebase_control(true);
}

void Gen31Evk1Fpga::stop() {
    sys_ctrl_.timebase_control(false);
}

void Gen31Evk1Fpga::destroy() {
    sys_ctrl_.sensor_prepowerdown();
    sensor_if_.sensor_turn_off_clock();
    sys_ctrl_.sensor_powerdown();
    sys_ctrl_.soft_reset("TRIGGERS");
}
} // namespace Metavision
