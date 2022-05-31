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

#include <map>

#include "devices/common/system_control.h"
#include "utils/register_map.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

SystemControl::SystemControl(const std::shared_ptr<RegisterMap> &regmap, const std::string &prefix) :
    prefix_(prefix), register_map_(regmap) {}

bool SystemControl::apply_resets() {
    (*register_map_)[prefix_ + "CLK_CONTROL"].write_value(vfield{{"CORE_SOFT_RST", 1},
                                                                 {"CORE_REG_BANK_RST", 1},
                                                                 {"SENSOR_IF_SOFT_RST", 1},
                                                                 {"SENSOR_IF_REG_BANK_RST", 1},
                                                                 {"HOST_IF_SOFT_RST", 1},
                                                                 {"HOST_IF_REG_BANK_RST", 1}});

    (*register_map_)[prefix_ + "CLK_CONTROL"]["GLOBAL_RST"].write_value(1);

    return (*register_map_)[prefix_ + "CLK_CONTROL"].read_value() == 0;
}

SystemControl::~SystemControl() {}

void SystemControl::set_evt_format(uint32_t fmt) {
    (*register_map_)[prefix_ + "GLOBAL_CONTROL"]["FORMAT"].write_value(fmt);
}

void SystemControl::clk_control(bool enable) {
    (*register_map_)[prefix_ + "CLK_CONTROL"].write_value(
        vfield{{"CORE_EN", enable}, {"SENSOR_IF_EN", enable}, {"HOST_IF_EN", enable}});
}

void SystemControl::time_base_config(bool ext_sync, bool master, bool master_sel, bool fwd_up, bool fwd_down) {
    /*Control the time base and synchronization settings.

    ----------------
    Time-base config
    ----------------
    Internal mode: ext_sync_mode_i = 0
      - Enable with only enable_i

    External mode: ext_sync_mode_i = 1
      - Enable with enable_i and ext_sync_enable_i
      - Sync source is sync_in when ext_sync_master_i = 0
      - Sync source is internal when ext_sync_master_i = 1

    Output internal sync signal on sync_out when ext_sync_mode_i = '1' and ext_sync_master_i = '1'

    ---------------------
    Sync in/out selection
    ---------------------
    When enable_cam_sync = 0:
      - ext0_sync_o = 0
    When enable_cam_sync = 1:
      - ext0_sync_o = ext1_sync_i when ext_sync_master = 0
      - ext0_sync_o = int_sync_i  when ext_sync_master = 1

    When enable_ext_sync = 0:
      - ext1_sync_o = 0
    When enable_ext_sync = 1:
      - ext1_sync_o = ext0_sync_i when ext_sync_master = 0
      - ext1_sync_o = int_sync_i  when ext_sync_master = 1

    int_sync_o = ext0_sync_i when ext_sync_master_sel = 0
    int_sync_o = ext1_sync_i when ext_sync_master_sel = 1

    -----------------------------------
    Sync in/out mapping for CCam3/CCam5
    -----------------------------------
    ext0 = moorea
    ext1 = FPGA sync_in/out header
    int  = Internal timebase

    ext0_sync_i = moorea_sync_out
    ext0_sync_o = moorea_sync_in
    ext1_sync_i = FPGA sync_in
    ext1_sync_o = FPGA sync_out
    int_sync_i = Internal sync_out
    int_sync_o = Internal sync_in

    If enable_cam_sync = 1
        moorea_sync_in = FPGA sync_in (ext_sync_master = 0)
        moorea_sync_in = Internal sync_out (ext_sync_master = 1)

    If enable_ext_sync = 1
        FPGA sync_out = moorea_sync_out (ext_sync_master = 0)
        FPGA sync_out = Internal sync_out (ext_sync_master = 1)

    Internal sync_in = moorea_sync_out (ext_sync_master_sel = 0)
    Internal sync_in = FPGA sync_in (ext_sync_master_sel = 1)
    */

    (*register_map_)[prefix_ + "TIME_BASE_CONTROL"].write_value(vfield{{"ENABLE", 0},
                                                                       {"EXT_SYNC_MODE", ext_sync},
                                                                       {"EXT_SYNC_ENABLE", ext_sync},
                                                                       {"EXT_SYNC_MASTER", master},
                                                                       {"EXT_SYNC_MASTER_SEL", master_sel},
                                                                       {"ENABLE_EXT_SYNC", fwd_up},
                                                                       {"ENABLE_CAM_SYNC", fwd_down}});
}

void SystemControl::time_base_control(bool enable) {
    /*Enable the time base.

    Args:
        enable (bool): Time base state
    */
    (*register_map_)[prefix_ + "TIME_BASE_CONTROL"]["ENABLE"].write_value(enable);
}

void SystemControl::merge_config(bool bypass, int source) {
    /*Merge Config

    Args:
        bypass (bool): Bypasses the event merge
        source (int) : '0' pattern/sensor event stream,
                       '1' monitoring event stream
    */

    (*register_map_)[prefix_ + "EVT_MERGE_CONTROL"].write_value(vfield{{"BYPASS", bypass}, {"SOURCE", source}});
}

void SystemControl::merge_control(bool enable) {
    /*Merge Control

    Arg:
        enable (bool): Enable merge block to process events
    */
    (*register_map_)[prefix_ + "EVT_MERGE_CONTROL"]["ENABLE"].write_value(enable);
}

void SystemControl::th_recovery_config(bool bypass) {
    /*TH Recovery Config.

    Args:
        bypass (bool): Bypasses TH Recovery logic
    */
    (*register_map_)[prefix_ + "TH_RECOVERY_CONTROL"]["BYPASS"].write_value(bypass);
}

void SystemControl::th_recovery_control(bool enable) {
    /*TH Recovery Control.

    Args:
        enable (bool): Enable TH Recovery logic
    */
    (*register_map_)[prefix_ + "TH_RECOVERY_CONTROL"]["ENABLE"].write_value(enable);
}

void SystemControl::data_formatter_config(bool bypass) {
    /*Event Data Formatter Config.

    Args:
        bypass (int, optional): Bypasses the event data formatter
    */
    (*register_map_)[prefix_ + "EVT_DATA_FORMATTER_CONTROL"]["BYPASS"].write_value(bypass);
}

void SystemControl::data_formatter_control(bool enable) {
    /*Event Data Formatter Control.

    Args:
        enable (bool) : Enable Data Formatter logic
    */
    (*register_map_)[prefix_ + "EVT_DATA_FORMATTER_CONTROL"]["ENABLE"].write_value(enable);
}

void SystemControl::set_mode(int mode) {
    /*Set the system mode: Init, Master or Slave.

    Args:
        mode (int, optional): System mode
    */
    (*register_map_)[prefix_ + "GLOBAL_CONTROL"]["MODE"].write_value(mode);
}

void SystemControl::sync_out_pin_control(bool trig_out_override) {
    // Override sync out pin to output trigger out pulse instead
    (*register_map_)[prefix_ + "TIME_BASE_CONTROL"]["EXT_SYNC_OUT_TRIGGER_MODE"].write_value(trig_out_override);
}

void SystemControl::oob_filter_control(bool enable) {
    (*register_map_)[prefix_ + "OOB_FILTER_CONTROL"]["ENABLE"].write_value(enable);
}

void SystemControl::oob_filter_origin(int x, int y) {
    (*register_map_)[prefix_ + "OOB_FILTER_ORIGIN"]["Y"].write_value(y);
    (*register_map_)[prefix_ + "OOB_FILTER_ORIGIN"]["X"].write_value(x);
}

void SystemControl::oob_filter_size(int width, int height) {
    (*register_map_)[prefix_ + "OOB_FILTER_SIZE"]["WIDTH"].write_value(width);
    (*register_map_)[prefix_ + "OOB_FILTER_SIZE"]["HEIGHT"].write_value(height);
}

} // namespace Metavision
