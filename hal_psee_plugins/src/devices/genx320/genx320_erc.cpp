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

#include <fstream>
#include <sstream>
#include <iomanip>

#include "metavision/psee_hw_layer/devices/genx320/genx320_erc.h"
#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

GenX320Erc::GenX320Erc(const std::shared_ptr<RegisterMap> &register_map) :
    register_map_(register_map), cd_event_count_shadow_(CD_EVENT_COUNT_DEFAULT) {
    (*register_map_)["erc/td_target_event_count"]["val"].write_value(cd_event_count_shadow_);
    (*register_map_)["erc/ref_period_flavor"]["reference_period"].write_value(REF_PERIOD);
}

std::map<std::string, uint32_t> GenX320Erc::is_powered_up_dyn() {
    uint32_t erc_dl_pd    = (*register_map_)["sram_pd1"]["erc_dl_pd"].read_value();
    uint32_t erc_ilg_pd   = (*register_map_)["sram_pd1"]["erc_ilg_pd"].read_value();
    uint32_t erc_tdrop_pd = (*register_map_)["sram_pd1"]["erc_tdrop_pd"].read_value();

    uint32_t erc_dl_initn    = (*register_map_)["sram_initn"]["erc_dl_initn"].read_value();
    uint32_t erc_ilg_initn   = (*register_map_)["sram_initn"]["erc_ilg_initn"].read_value();
    uint32_t erc_tdrop_initn = (*register_map_)["sram_initn"]["erc_tdrop_initn"].read_value();

    std::map<std::string, uint32_t> status = {{"erc_dfifo", ~(erc_dl_pd)&erc_dl_initn},
                                              {"erc_ilg", ~(erc_ilg_pd)&erc_ilg_initn},
                                              {"erc_tdrop", ~(erc_tdrop_pd)&erc_tdrop_initn}};

    return status;
}

bool GenX320Erc::dfifo_disable_bypass_dyn() {
    // Enable path through the delay fifo for events going through ERC

    uint32_t value = is_powered_up_dyn()["erc_dfifo"];

    if (value) {
        (*register_map_)["erc/delay_fifo_flush_and_bypass"]["en"].write_value(0);
        return true;
    } else {
        MV_HAL_LOG_ERROR() << "ERC pipe try to use delay fifo whereas dfifo sram is not powered up";
        return false;
    }
}

bool GenX320Erc::set_evt_rate_dyn(uint32_t ref_period, uint32_t td_target_vx_cnt, uint32_t adr_delayed,
                                  uint32_t dfifo_non_td_area) {
    /*
    ERC set target event rate
     - Warning: delay fifo sram is not powered down since it might be used by the event period(s) already stored in the
    delay fifo
    */
    bool ret_code = false;
    if (adr_delayed != 0) {
        ret_code = dfifo_disable_bypass_dyn();
    }

    if (ret_code) {
        (*register_map_)["erc/ref_period_flavor"].write_value(
            vfield{{"avg_drop_rate_delayed", adr_delayed}, {"reference_period", ref_period}});
        (*register_map_)["erc/td_target_event_count"]["val"].write_value(td_target_vx_cnt);

        /* Delay fifo is used. Set reserved area size for non-TD events, and allow auto increment
           of the area size if a non-TD event has to be dropped since fifo is full.
           By default, preserve 5128 - 50 * (1 + 100 + 1) entries for non-TD events
        */
        (*register_map_)["erc/delay_fifo_non_td_rsvd_area"].write_value(
            vfield{{"val", dfifo_non_td_area}, {"auto_raise", 1}});
    }

    return true;
}

bool GenX320Erc::wait_status() {
    return (*register_map_)["erc/ahvt_dropping_control"].read_value();
}

bool GenX320Erc::activate_dyn(const uint32_t &td_target_vx_cnt) {
    // First force hold mode to flush ERC in case it is processing an event period
    (*register_map_)["erc/pipeline_control"].write_value(
        vfield{{"enable", 0}, {"drop_nbackpressure", 0}, {"bypass", 0}});

    // Go in bypass mode while setting configuration
    (*register_map_)["erc/pipeline_control"].write_value(
        vfield{{"enable", 1}, {"drop_nbackpressure", 0}, {"bypass", 1}});

    // SRAM dfifo powerup
    (*register_map_)["sram_initn"]["erc_dl_initn"].write_value(1);
    (*register_map_)["sram_pd1"]["erc_dl_pd"].write_value(0);

    bool ret_code = set_evt_rate_dyn(REF_PERIOD, td_target_vx_cnt, 1,
                                     28); // adr_delayed = 1, dfifo_non_td_area = 28

    if (!ret_code) {
        return false;
    }

    // minimum monitoring events (AVG_DROP_RATE is the "start" flag of the an event period while TD_EVENT_COUNT is the
    // last tag)
    (*register_map_)["erc/monitoring_event_control"].write_value(
        vfield{{"avg_drop_rate_en", 1}, {"in_td_cnt_en", 1}, {"erc_td_evt_cnt_en", 1}});

    // To set future drop configuration, need to have ahvt_dropping_control.status equal to 1
    ret_code = wait_status();

    if (!ret_code) {
        return false;
    }

    (*register_map_)["erc/ahvt_dropping_control"].write_value(vfield{{"h_dropping_en", 0},
                                                                     {"v_dropping_en", 0},
                                                                     {"t_dropping_en", 1},
                                                                     {"t_dropping_lut_en", 0},
                                                                     {"drop_all_td_when_drop_geq", 512}});

    // Do not reset tdrop counter between event periods, since it mostly preserves events at the beginming of the lines.
    (*register_map_)["erc/reset_tdrop_counter_on_mtag_first"]["en"].write_value(0);

    // Enable ERC
    (*register_map_)["erc/pipeline_control"].write_value(
        vfield{{"enable", 1}, {"drop_nbackpressure", 0}, {"bypass", 0}});

    return true;
}

bool GenX320Erc::enable(bool en) {
    (*register_map_)["erc/ahvt_dropping_control"].write_value({"t_dropping_en", en});

    if (en) {
        set_cd_event_count(cd_event_count_shadow_);
        activate_dyn(cd_event_count_shadow_);
    }

    return true;
}

bool GenX320Erc::is_enabled() const {
    bool t_dropping_en = (*register_map_)["erc/ahvt_dropping_control"]["t_dropping_en"].read_value();
    return t_dropping_en;
}

void GenX320Erc::erc_from_file(const std::string &file_path) {
    MV_HAL_LOG_ERROR() << "ERC configuration from file not implemented";
}

uint32_t GenX320Erc::get_count_period() const {
    return (*register_map_)["erc/ref_period_flavor"]["reference_period"].read_value();
}

bool GenX320Erc::set_cd_event_count(uint32_t count) {
    if (count > CD_EVENT_COUNT_MAX) {
        std::stringstream ss;
        ss << "Cannot set CD event count to :" << count << ". Value should be in the range [0, " << CD_EVENT_COUNT_MAX
           << "]";
        throw HalException(HalErrorCode::ValueOutOfRange, ss.str());
    }
    (*register_map_)["erc/td_target_event_count"]["val"].write_value(count);
    cd_event_count_shadow_ = count;

    return true;
}

uint32_t GenX320Erc::get_min_supported_cd_event_count() const {
    return 0;
}

uint32_t GenX320Erc::get_max_supported_cd_event_count() const {
    return CD_EVENT_COUNT_MAX;
}

uint32_t GenX320Erc::get_cd_event_count() const {
    return (*register_map_)["erc/td_target_event_count"]["val"].read_value();
}

} // namespace Metavision
