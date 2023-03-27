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

#include "devices/common/evk2_system_control.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

Evk2SystemControl::Evk2SystemControl(const std::shared_ptr<RegisterMap> &regmap) :
    register_map_(regmap),
    sys_ctrl_regbank_("SYSTEM_CONTROL/"),
    sys_mon_regbank_("SYSTEM_MONITOR/"),
    ps_host_if_regbank_("PS_HOST_IF/") {}

bool Evk2SystemControl::apply_resets() {
    (*register_map_)[sys_ctrl_regbank_ + "CLK_CONTROL"].write_value(vfield{{"CORE_SOFT_RST", 1},
                                                                           {"CORE_REG_BANK_RST", 1},
                                                                           {"SENSOR_IF_SOFT_RST", 1},
                                                                           {"SENSOR_IF_REG_BANK_RST", 1},
                                                                           {"HOST_IF_SOFT_RST", 1},
                                                                           {"HOST_IF_REG_BANK_RST", 1}});

    (*register_map_)[sys_ctrl_regbank_ + "CLK_CONTROL"]["GLOBAL_RST"].write_value(1);

    return (*register_map_)[sys_ctrl_regbank_ + "CLK_CONTROL"].read_value() == 0;
}

void Evk2SystemControl::set_evt_format(uint32_t fmt) {
    uint32_t len = 0;

    switch (fmt) {
    case 2:
        len = 4096;
        break;
    case 3:
        len = 8192;
        break;
    default:
        std::cerr << "Unknown event format\n";
        return;
    }

    (*register_map_)[sys_ctrl_regbank_ + "GLOBAL_CONTROL"]["FORMAT"]                    = fmt;
    (*register_map_)[sys_ctrl_regbank_ + "GLOBAL_CONTROL"]["OUTPUT_FORMAT"]             = fmt;
    (*register_map_)[ps_host_if_regbank_ + "AXI_DMA_PACKETIZER/PACKET_LENGTH"]["VALUE"] = len;
}

void Evk2SystemControl::clk_control(bool enable) {
    (*register_map_)[sys_ctrl_regbank_ + "CLK_CONTROL"].write_value(
        vfield{{"CORE_EN", 1}, {"SENSOR_IF_EN", 1}, {"HOST_IF_EN", 1}});
}

void Evk2SystemControl::time_base_config(bool ext_sync, bool master, bool master_sel, bool fwd_up, bool fwd_down) {
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

    ---------------------------------
    Sync in/out mapping for Evk2 Zynq
    ---------------------------------
    ext0 = CCam5
    ext1 = MoBo sync_in/out IOs
    int  = Internal timebase

    ext0_sync_i = CCam5 sync_out
    ext0_sync_o = CCam5 sync_in
    ext1_sync_i = MoBo sync_in
    ext1_sync_o = MoBo sync_out
    int_sync_i = Internal sync_out
    int_sync_o = Internal sync_in

    If enable_cam_sync = 1
        CCam5 sync_in = MoBo sync_in (ext_sync_master = 0)
        CCam5 sync_in = Internal sync_out (ext_sync_master = 1)

    If enable_ext_sync = 1
        MoBo sync_out = CCam5 sync_out (ext_sync_master = 0)
        MoBo sync_out = Internal sync_out (ext_sync_master = 1)

    Internal sync_in = CCam5 sync_out (ext_sync_master_sel = 0)
    Internal sync_in = MoBo sync_in (ext_sync_master_sel = 1)
    */

    (*register_map_)[sys_ctrl_regbank_ + "TIME_BASE_CONTROL"].write_value(vfield{{"ENABLE", 0},
                                                                                 {"EXT_SYNC_MODE", ext_sync},
                                                                                 {"EXT_SYNC_ENABLE", ext_sync},
                                                                                 {"EXT_SYNC_MASTER", master},
                                                                                 {"EXT_SYNC_MASTER_SEL", master_sel},
                                                                                 {"ENABLE_EXT_SYNC", fwd_up},
                                                                                 {"ENABLE_CAM_SYNC", fwd_down}});
}

void Evk2SystemControl::time_base_control(bool enable) {
    /*Enable the time base.

    Args:
        enable (bool): Time base state
    */
    (*register_map_)[sys_ctrl_regbank_ + "TIME_BASE_CONTROL"]["ENABLE"].write_value(enable);
}

void Evk2SystemControl::merge_config(bool bypass, int source) {
    /*Merge Config

    Args:
        bypass (bool): Bypasses the event merge
        source (int) : '0' pattern/sensor event stream,
                       '1' monitoring event stream
    */

    (*register_map_)[sys_ctrl_regbank_ + "EVT_MERGE_CONTROL"].write_value(
        vfield{{"BYPASS", bypass}, {"SOURCE", source}});
}

void Evk2SystemControl::merge_control(bool enable) {
    /*Merge Control

    Arg:
        enable (bool): Enable merge block to process events
    */
    (*register_map_)[sys_ctrl_regbank_ + "EVT_MERGE_CONTROL"]["ENABLE"].write_value(enable);
}

void Evk2SystemControl::th_recovery_config(bool bypass) {
    /*TH Recovery Config.

    Args:
        bypass (bool): Bypasses TH Recovery logic
    */
    (*register_map_)[sys_ctrl_regbank_ + "TH_RECOVERY_CONTROL"]["BYPASS"].write_value(bypass);
}

void Evk2SystemControl::th_recovery_control(bool enable) {
    /*TH Recovery Control.

    Args:
        enable (bool): Enable TH Recovery logic
    */
    (*register_map_)[sys_ctrl_regbank_ + "TH_RECOVERY_CONTROL"]["ENABLE"].write_value(enable);
}

void Evk2SystemControl::out_th_recovery_config(bool bypass) {
    /*Output TH Recovery Config.

    Args:
        bypass (bool): Bypasses Output TH Recovery logic
    */
    (*register_map_)[sys_ctrl_regbank_ + "OUT_TH_RECOVERY_CONTROL"]["BYPASS"].write_value(bypass);
}

void Evk2SystemControl::out_th_recovery_control(bool enable) {
    /*Output TH Recovery Control.

    Args:
        enable (bool): Enable Output TH Recovery logic
    */
    (*register_map_)[sys_ctrl_regbank_ + "OUT_TH_RECOVERY_CONTROL"]["ENABLE"].write_value(enable);
}

void Evk2SystemControl::data_formatter_config(bool bypass) {
    /*Event Data Formatter Config.

    Args:
        bypass (int, optional): Bypasses the event data formatter
    */
    (*register_map_)[sys_ctrl_regbank_ + "EVT_DATA_FORMATTER_CONTROL"]["BYPASS"].write_value(bypass);
}

void Evk2SystemControl::data_formatter_control(bool enable) {
    /*Event Data Formatter Control.

    Args:
        enable (bool) : Enable Data Formatter logic
    */
    (*register_map_)[sys_ctrl_regbank_ + "EVT_DATA_FORMATTER_CONTROL"]["ENABLE"].write_value(enable);
}

void Evk2SystemControl::set_mode(int mode) {
    /*Set the system mode: Init, Master or Slave.

    Args:
        mode (int, optional): System mode
    */
    (*register_map_)[sys_ctrl_regbank_ + "GLOBAL_CONTROL"]["MODE"].write_value(mode);
}

void Evk2SystemControl::monitoring_merge_config(bool bypass, int source) {
    /*Monitoring Merge Config

    Args:
        bypass (bool): Bypasses the event merge
        source (int) : '0' pattern/sensor event stream,
                       '1' monitoring event stream
    */

    (*register_map_)[sys_mon_regbank_ + "CONTROL/EVT_MERGE_CONTROL"].write_value(
        vfield{{"BYPASS", bypass}, {"SOURCE", source}});
}

void Evk2SystemControl::monitoring_merge_control(bool enable) {
    /*Monitoring Merge Control

    Arg:
        enable (bool): Enable merge block to process events
    */
    (*register_map_)[sys_mon_regbank_ + "CONTROL/EVT_MERGE_CONTROL"]["ENABLE"].write_value(enable);
}

void Evk2SystemControl::ts_checker_config(bool bypass) {
    /*TS Checker Config

    Arg:
        bypass (bool): Bypasses the ts checker block
    */
    (*register_map_)[sys_ctrl_regbank_ + "TS_CHECKER_CONTROL"]["BYPASS"].write_value(bypass);
}

void Evk2SystemControl::sync_out_pin_config(bool trig_out_override) {
    /* Sync out IO pin source configuration.
    Override pin to output trigger out pulse instead of sync out signal.

    Arg:
        trig_out_override (bool): select signal to forward to FPGA IO (false -> sync_out, true -> trigget_out)
    */

    // Select mode to forward
    (*register_map_)[sys_ctrl_regbank_ + "IO_CONTROL"]["SYNC_OUT_MODE"].write_value(trig_out_override);

    // Enable multi drivers detector
    // (*register_map_)[sys_ctrl_regbank_ + "IO_CONTROL"]["SYNC_OUT_EN_FLT_CHK"].write_value(1);
}

bool Evk2SystemControl::sync_out_pin_control(bool enable) {
    /*Sync out IO pin external gate control.

    Arg:
        enable (bool) : Enable/disable external line's photo MOS to connect FPGA IO to external connector
    */

    // Enable/disable external MOS gate between FPGA IO and Evk2 connector
    (*register_map_)[sys_ctrl_regbank_ + "IO_CONTROL"]["SYNC_OUT_EN_HSIDE"].write_value(enable);
    return true;

    /* Code removed as some Evk2 MoBo come with fault alert chip not mounted
    // Check if sync out IO pin is not driven
    if (get_sync_out_pin_fault_alert()) {
        MV_HAL_LOG_ERROR() << "SYNC OUT pin is driven by an external signal.";
        return false;
    } else {
        // Enable line's photo MOS to connect FPGA IO to external connector
        (*register_map_)[sys_ctrl_regbank_ + "IO_CONTROL"]["SYNC_OUT_EN_HSIDE"].write_value(1);
        return true;
    }*/
}

bool Evk2SystemControl::get_sync_out_pin_fault_alert() {
    // External sync out connector fault detector (multi drivers detector)
    bool isDriven = false;

    for (int i = 0; i < 20; ++i) {
        isDriven = (*register_map_)[sys_ctrl_regbank_ + "IO_CONTROL"]["SYNC_OUT_FAULT_ALERT"].read_value() == 1;
        MV_HAL_LOG_DEBUG() << "Fault =" << isDriven;
        if (isDriven) {
            return true;
        }
    }
    return false;
}

bool Evk2SystemControl::is_trigger_out_enabled() {
    bool trig_out_en = (*register_map_)[sys_mon_regbank_ + "EXT_TRIGGERS/OUT_ENABLE"].read_value();
    bool io_pin_en   = (*register_map_)[sys_ctrl_regbank_ + "IO_CONTROL"]["SYNC_OUT_MODE"].read_value();
    bool hside_en    = (*register_map_)[sys_ctrl_regbank_ + "IO_CONTROL"]["SYNC_OUT_EN_HSIDE"].read_value();
    return trig_out_en && io_pin_en && hside_en;
}

} // namespace Metavision
