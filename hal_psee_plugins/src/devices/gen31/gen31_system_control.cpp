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

#include "devices/gen31/gen31_system_control.h"
#include "utils/register_map.h"

namespace Metavision {
using vfield = std::map<std::string, uint32_t>;

Gen31SystemControl::Gen31SystemControl(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix) :
    prefix_(prefix), register_map_(register_map) {}

void Gen31SystemControl::sensor_atis_control_clear(void) {
    (*register_map_)[prefix_ + "ATIS_CONTROL"].write_value(0);
}

void Gen31SystemControl::sensor_prepowerup(void) {
    sensor_roi_td_rstn(false);
    sensor_em_rstn(false);
    sensor_soft_reset(true);
    sensor_enable_vddc(true);
    sensor_enable_vddd(true);
    sensor_soft_reset(false);
}
void Gen31SystemControl::sensor_prepowerdown(void) {
    sensor_roi_td_rstn(false);
    sensor_enable_vdda(false);
}

void Gen31SystemControl::sensor_powerdown(void) {
    sensor_soft_reset(false);
    sensor_enable_vddc(false);
    sensor_enable_vddd(false);
}
void Gen31SystemControl::sensor_enable_vdda(bool enable) {
    (*register_map_)[prefix_ + "ATIS_CONTROL"]["EN_VDDA"].write_value(enable);
}

void Gen31SystemControl::sensor_enable_vddc(bool enable) {
    (*register_map_)[prefix_ + "ATIS_CONTROL"]["EN_VDDC"].write_value(enable);
}

void Gen31SystemControl::sensor_enable_vddd(bool enable) {
    (*register_map_)[prefix_ + "ATIS_CONTROL"]["EN_VDDD"].write_value(enable);
}

void Gen31SystemControl::sensor_soft_reset(bool reset) {
    (*register_map_)[prefix_ + "ATIS_CONTROL"]["SENSOR_SOFT_RESET"].write_value(reset);
}

void Gen31SystemControl::sensor_roi_td_rstn(bool rstn) {
    (*register_map_)[prefix_ + "ATIS_CONTROL"]["TD_RSTN"].write_value(rstn);
}

void Gen31SystemControl::sensor_em_rstn(bool rstn) {
    (*register_map_)[prefix_ + "ATIS_CONTROL"]["EM_RSTN"].write_value(rstn);
}

void Gen31SystemControl::sensor_powerup(void) {
    sensor_enable_vdda(true);
    sensor_roi_td_rstn(true);
    sensor_em_rstn(true);
}

void Gen31SystemControl::soft_reset(std::string reg_obj) {
    // can we please have consistent register names ? thanks !
    (*register_map_)[prefix_ + reg_obj]["SOFT_RESET"].write_value(false);
    (*register_map_)[prefix_ + reg_obj]["SOFT_RESET"].write_value(true);
}

void Gen31SystemControl::hvga_remap_control(bool enable) {
    (*register_map_)[prefix_ + "ATIS_CONTROL"]["SISLEY_HVGA_REMAP_BYPASS"].write_value(!enable);
}

void Gen31SystemControl::no_blocking_control(bool enable) {
    (*register_map_)[prefix_ + "ATIS_CONTROL"]["IN_EVT_NO_BLOCKING_MODE"].write_value(enable);
}

void Gen31SystemControl::host_if_control(bool enable) {
    (*register_map_)[prefix_ + "CCAM2_CONTROL"]["HOST_IF_ENABLE"].write_value(enable);
}

void Gen31SystemControl::timebase_control(bool enable) {
    (*register_map_)[prefix_ + "CCAM2_CONTROL"]["STEREO_MERGE_ENABLE"].write_value(enable);
}

void Gen31SystemControl::timebase_config(bool ext_sync, bool master) {
    (*register_map_)[prefix_ + "ATIS_CONTROL"].write_value(vfield{
        {"MASTER_MODE", master},
        {"USE_EXT_START", ext_sync},
    });
}
} // namespace Metavision
