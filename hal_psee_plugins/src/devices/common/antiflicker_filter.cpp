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

#include <cmath>

#include "metavision/hal/utils/hal_exception.h"

#include "metavision/psee_hw_layer/devices/common/antiflicker_filter.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

AntiFlickerFilter::AntiFlickerFilter(const std::shared_ptr<RegisterMap> &register_map,
                                     const I_HW_Identification::SensorInfo &sensor_info,
                                     const std::string &sensor_prefix) :
    register_map_(register_map), sensor_prefix_(sensor_prefix) {
    if (sensor_info.name_ == "GenX320") {
        is_sensor_saphir = true;
        flag_done_       = "flag_init_done";
        afk_param_       = "afk/afk_param";
    } else {
        is_sensor_saphir = false;
        flag_done_       = "afk_flag_init_done";
        afk_param_       = "afk/param";
    }
}

uint32_t AntiFlickerFilter::freq_to_period(const uint32_t &freq) {
    int period = 1e6 / freq;
    return period >> 7;
}

std::pair<uint32_t, uint32_t> AntiFlickerFilter::compute_invalidation(const uint32_t &max_cutoff_period,
                                                                      const uint32_t &clk_freq) {
    /* computation of invalidation speed parameter based on processing frequency and max_cut_freq
           max_cutoff_period : in unit of 128us
           evt_clk_freq; processing frequency, in unit of MHz */

    // Proccessing clock period in nanoseconds
    uint32_t clk_period_ns = 1000 / clk_freq;
    uint32_t in_parallel   = 5;

    float invalidation_period =
        (((2 << 15) - (128 * (3 + max_cutoff_period))) * in_parallel * 1000) / (160 * 5 * clk_period_ns);

    uint32_t df_wait_time = 4;

    uint32_t df_timeout = std::floor(invalidation_period) - df_wait_time;

    if (df_timeout > 4095) {
        df_timeout = 4095;
    }

    return std::make_pair(df_wait_time, df_timeout);
}

bool AntiFlickerFilter::enable(bool b) {
    // Always disable in order to configure AFK parameters
    (*register_map_)[sensor_prefix_ + "afk/pipeline_control"].write_value(0b101);
    if (!b) {
        return true;
    }

    if (is_sensor_saphir) {
        // SRAM powerup
        (*register_map_)[sensor_prefix_ + "sram_initn"]["afk_initn"].write_value(1);
        (*register_map_)[sensor_prefix_ + "sram_pd0"].write_value(
            vfield{{"afk_alr_pd", 0}, {"afk_str0_pd", 0}, {"afk_str1_pd", 0}});
    }

    bool init_done = 0;
    for (int i = 0; i < 3; i++) {
        init_done = (*register_map_)[sensor_prefix_ + "afk/initialization"][flag_done_].read_value();
        if (init_done == 1) {
            break;
        }
    }
    if (init_done == 0) {
        throw HalException(HalErrorCode::InternalInitializationError, "Bad AFK initialization");
    }

    // Set frequency bandwidth
    uint32_t min_cutoff_period = freq_to_period(high_freq_);
    uint32_t max_cutoff_period = freq_to_period(low_freq_);

    std::pair<uint32_t, uint32_t> invalidation_cfg;
    if (is_sensor_saphir) {
        invalidation_cfg = compute_invalidation(max_cutoff_period, 25);
        (*register_map_)[sensor_prefix_ + "afk/invalidation"].write_value(
            vfield{{"dt_fifo_wait_time", std::get<0>(invalidation_cfg)},
                   {"dt_fifo_timeout", std::get<1>(invalidation_cfg)},
                   {"in_parallel", 5}});
    } else {
        (*register_map_)[sensor_prefix_ + "afk/invalidation"]["dt_fifo_wait_time"].write_value(1630);
    }

    // Set duty cycle
    (*register_map_)[sensor_prefix_ + "afk/filter_period"].write_value(vfield{{"min_cutoff_period", min_cutoff_period},
                                                                      {"max_cutoff_period", max_cutoff_period},
                                                                      {"inverted_duty_cycle", inverted_duty_cycle_}});

    // Set filtering mode
    (*register_map_)[sensor_prefix_ + afk_param_]["invert"].write_value(mode_ == BAND_STOP ? 0 : 1);

    // Set hysteresis counters
    (*register_map_)[sensor_prefix_ + afk_param_]["counter_high"].write_value(start_threshold_);
    (*register_map_)[sensor_prefix_ + afk_param_]["counter_low"].write_value(stop_threshold_);

    (*register_map_)[sensor_prefix_ + "afk/pipeline_control"].write_value(0b001);

    return true;
}

bool AntiFlickerFilter::is_enabled() {
    return (*register_map_)[sensor_prefix_ + "afk/pipeline_control"].read_value() == 0b001;
}

bool AntiFlickerFilter::reset() {
    if (is_enabled()) {
        if (!enable(false)) {
            return false;
        }
        if (!enable(true)) {
            return false;
        }
    }
    return true;
}

bool AntiFlickerFilter::set_frequency_band(uint32_t low_freq, uint32_t high_freq) {
    // Check frequency values
    if (low_freq < get_min_supported_frequency() || high_freq > get_max_supported_frequency() || low_freq > high_freq) {
        std::stringstream ss;
        ss << "Invalid input frequencies. Expected: " << get_min_supported_frequency() << " <= low_freq (= " << low_freq
           << ") < high_freq (= " << high_freq << ") <= " << get_max_supported_frequency();
        throw HalException(HalErrorCode::ValueOutOfRange, ss.str());
    }

    low_freq_  = low_freq;
    high_freq_ = high_freq;

    // Reset if needed
    if (!reset()) {
        return false;
    }

    return true;
}

uint32_t AntiFlickerFilter::get_min_supported_frequency() const {
    return 50;
}

uint32_t AntiFlickerFilter::get_max_supported_frequency() const {
    return 520;
}

uint32_t AntiFlickerFilter::get_band_low_frequency() const {
    return low_freq_;
}

uint32_t AntiFlickerFilter::get_band_high_frequency() const {
    return high_freq_;
}

bool AntiFlickerFilter::set_filtering_mode(I_AntiFlickerModule::AntiFlickerMode mode) {
    mode_ = mode;

    // Reset if needed
    if (!reset()) {
        return false;
    }

    return true;
}

I_AntiFlickerModule::AntiFlickerMode AntiFlickerFilter::get_filtering_mode() const {
    return mode_;
}

bool AntiFlickerFilter::set_duty_cycle(float duty_cycle) {
    if (duty_cycle <= 0.0 || duty_cycle > get_max_supported_duty_cycle()) {
        std::stringstream ss;
        ss << "Invalid input duty cycle. Expected: " << 0.0 << " < duty_cycle (= " << duty_cycle << ") <= " << 100.0;
        throw HalException(HalErrorCode::ValueOutOfRange, ss.str());
    }

    inverted_duty_cycle_ = std::roundf((16 * (100.0 - duty_cycle)) / 100.0);
    // Lowest duty cycle is 1/16th
    if (inverted_duty_cycle_ >= 16) {
        inverted_duty_cycle_ = 15;
    }

    // Reset if needed
    if (!reset()) {
        return false;
    }

    return true;
}

float AntiFlickerFilter::get_duty_cycle() const {
    return 100.0 - (inverted_duty_cycle_ * 100.0) / 16.0;
}

float AntiFlickerFilter::get_min_supported_duty_cycle() const {
    return 1.0 / 16.0;
}

float AntiFlickerFilter::get_max_supported_duty_cycle() const {
    return 100.0;
}

bool AntiFlickerFilter::set_start_threshold(uint32_t threshold) {
    if (threshold < get_min_supported_start_threshold() || threshold > get_max_supported_start_threshold()) {
        std::stringstream ss;
        ss << "Invalid start threshold. Expected: " << get_min_supported_start_threshold()
           << " <= threshold (= " << threshold << ") <= " << get_max_supported_start_threshold();

        throw HalException(HalErrorCode::ValueOutOfRange, ss.str());
    }

    start_threshold_ = threshold;

    // Reset if needed
    if (!reset()) {
        return false;
    }

    return true;
}

bool AntiFlickerFilter::set_stop_threshold(uint32_t threshold) {
    if (threshold < get_min_supported_stop_threshold() || threshold > get_max_supported_stop_threshold()) {
        std::stringstream ss;
        ss << "Invalid stop threshold. Expected: " << get_min_supported_stop_threshold()
           << " <= threshold (= " << threshold << ") <= " << get_max_supported_stop_threshold();

        throw HalException(HalErrorCode::ValueOutOfRange, ss.str());
    }

    stop_threshold_ = threshold;

    // Reset if needed
    if (!reset()) {
        return false;
    }

    return true;
}

uint32_t AntiFlickerFilter::get_start_threshold() const {
    return start_threshold_;
}

uint32_t AntiFlickerFilter::get_stop_threshold() const {
    return stop_threshold_;
}

uint32_t AntiFlickerFilter::get_min_supported_start_threshold() const {
    return 0;
}

uint32_t AntiFlickerFilter::get_max_supported_start_threshold() const {
    return (1 << 3) - 1;
}

uint32_t AntiFlickerFilter::get_min_supported_stop_threshold() const {
    return 0;
}

uint32_t AntiFlickerFilter::get_max_supported_stop_threshold() const {
    return (1 << 3) - 1;
}

} // namespace Metavision
