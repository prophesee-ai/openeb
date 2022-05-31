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

#include <thread>
#include <chrono>
#include "devices/gen31/gen31_analog.h"
#include "utils/register_map.h"

namespace Metavision {

Gen31Analog::Gen31Analog(const std::shared_ptr<RegisterMap> &register_map, const std::string &prefix, bool is_em) :
    prefix_(prefix), register_map_(register_map), is_em_(is_em) {}

void Gen31Analog::init() {
    (*register_map_)[prefix_ + "spare_ctrl"]["em_shutter"]    = true;
    (*register_map_)[prefix_ + "global_ctrl"]["bgen_en"]      = true;
    (*register_map_)[prefix_ + "global_ctrl"]["bgen_rstn"]    = true;
    (*register_map_)[prefix_ + "global_ctrl"]["sw_global_en"] = true;

    if (is_em_) {
        (*register_map_)[prefix_ + "acr_ctrl"]["acr_en"]                = false;
        (*register_map_)[prefix_ + "spare_ctrl"]["em_shutter"]          = false;
        (*register_map_)[prefix_ + "global_ctrl"]["td_couple_ctrl"]     = true;
        (*register_map_)[prefix_ + "roi_ctrl"]["roi_em_shadow_trigger"] = true;
    } else {
        (*register_map_)[prefix_ + "rob_ctrl"]["ro_dual_ctrl"] = true;
    }

    (*register_map_)[prefix_ + "bgen_00"]["bias_vdac_val"] = 176;
    (*register_map_)[prefix_ + "bgen_01"]["bias_vdac_val"] = 176;

    // Enable state machine reqY active pull down of TD readout
    (*register_map_)[prefix_ + "readout_ctrl"]["ro_act_pdy_ctrl"] = 1;
    // Disable state machine reqX active pull down of TD readout
    (*register_map_)[prefix_ + "readout_ctrl"]["ro_act_pdx_ctrl"] = 0;
    // Disable pixel reqX active pull up of TD readout
    (*register_map_)[prefix_ + "readout_ctrl"]["ro_act_pux_ctrl"] = 0;
    // Enable state machine reqY active pull down of EM readout
    (*register_map_)[prefix_ + "roe_ctrl"]["ro_em_act_pdy_ctrl"] = 1;
    // Disable state machine reqX active pull down of EM readout
    (*register_map_)[prefix_ + "roe_ctrl"]["ro_em_act_pdx_ctrl"] = 0;
    // Disable pixel reqX active pull up of EM readout
    (*register_map_)[prefix_ + "roe_ctrl"]["ro_em_act_pux_ctrl"] = 0;
    // Enable pixel reqX active pull up from top row for both TD and EM readout
    (*register_map_)[prefix_ + "spare_ctrl"]["ro_top_act_pu_en"] = 1;
    // Enable column reqX active pull down for both TD and EM readout
    (*register_map_)[prefix_ + "spare_ctrl"]["spare0"] = 1;
    // Tune bias del ack array to avoid out of range addresses
    (*register_map_)[prefix_ + "bgen_07"]["bias_vdac_val"] = 0xC0;
    // Set bias_diff_off to VDAC mode to avoid overheating issue
    (*register_map_)[prefix_ + "bgen_19"]["bias_type"]     = 1;
    (*register_map_)[prefix_ + "bgen_19"]["bias_vdac_val"] = 0x1E;
    // Set bias_diff_on to VDAC mode to avoid overheating issue
    // Tune bias_diff_on voltage after overheating issue and validation campaign
    (*register_map_)[prefix_ + "bgen_20"]["bias_type"]     = 1;
    (*register_map_)[prefix_ + "bgen_20"]["bias_vdac_val"] = 0x35;
    // Set bias_diff to VDAC mode to avoid overheating issue
    // Tune bias_diff voltage    after overheating issue and validation campaign
    (*register_map_)[prefix_ + "bgen_21"]["bias_type"]     = 1;
    (*register_map_)[prefix_ + "bgen_21"]["bias_vdac_val"] = 0x29;
    // Tune bias_fo value to improve latency.
    (*register_map_)[prefix_ + "bgen_22"]["bias_idac_val"] = 0x3D;

    // Enable output clock gating to save power
    (*register_map_)[prefix_ + "clk_out_ctrl"]["clk_gate_bypass"] = 0;
    // Enable out of range checks
    (*register_map_)[prefix_ + "oor_ctrl"]["oor_en"] = 1;

    analog_td_rstn(false);

    (*register_map_)[prefix_ + "roi_ctrl"]["roi_td_en"]             = true;
    (*register_map_)[prefix_ + "td_roi_x20"]["val"]                 = false;
    (*register_map_)[prefix_ + "td_roi_y15"]["val"]                 = false;
    (*register_map_)[prefix_ + "roi_ctrl"]["roi_td_shadow_trigger"] = true;
}

void Gen31Analog::start() {
    // Save user configuration given to the facility
    uint32_t evt_thresh = (*register_map_)[prefix_ + "nfl_thresh"]["evt_thresh"].read_value();
    uint32_t period_cnt = (*register_map_)[prefix_ + "nfl_thresh"]["period_cnt_thresh"].read_value();
    bool enable         = (*register_map_)[prefix_ + "nfl_ctrl"]["nfl_en"].read_value();

    // Enable the noise filter and set to max (dropping all events).
    // This will prevent the initial event burst when starting.
    (*register_map_)[prefix_ + "nfl_thresh"]["evt_thresh"]        = 0x3FFF;
    (*register_map_)[prefix_ + "nfl_thresh"]["period_cnt_thresh"] = 0xA;
    (*register_map_)[prefix_ + "nfl_ctrl"]["nfl_en"]              = true;

    analog_td_rstn(true);

    // Wait for event burst to be filtered out
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Disable the noise filter (allowing events to pass again) and restore user settings.
    (*register_map_)[prefix_ + "nfl_ctrl"]["nfl_en"]              = false;
    (*register_map_)[prefix_ + "nfl_thresh"]["evt_thresh"]        = evt_thresh;
    (*register_map_)[prefix_ + "nfl_thresh"]["period_cnt_thresh"] = period_cnt;
    (*register_map_)[prefix_ + "nfl_ctrl"]["nfl_en"]              = enable;
}

void Gen31Analog::stop() {
    analog_td_rstn(false);
}

void Gen31Analog::destroy() {}

void Gen31Analog::analog_td_rstn(bool rstn) {
    (*register_map_)[prefix_ + "roi_ctrl"]["roi_td_rstn"] = rstn;
}

void Gen31Analog::analog_em_rstn(bool rstn) {
    (*register_map_)[prefix_ + "roi_ctrl"]["em_shutter"] = rstn;
}

void Gen31Analog::enable_lifo_measurement() {
    (*register_map_)[prefix_ + "lifo_ctrl"]["lifo_en"] = true;
}

uint32_t Gen31Analog::lifo_counter() {
    (*register_map_)[prefix_ + "lifo_ctrl"]["lifo_cnt_en"] = true;
    bool valid                                             = false;
    uint16_t retries                                       = 0;
    uint32_t counter                                       = 0;
    while (valid == false && retries < 10) {
        auto reg_val = (*register_map_)[prefix_ + "lifo_ctrl"].read_value();
        valid        = reg_val & 1 << 29;
        counter      = reg_val & ((1 << 27) - 1);
        retries += 1;
    }
    if (valid)
        return -1;
    return counter;
}

} // namespace Metavision
