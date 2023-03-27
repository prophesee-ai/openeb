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

#include "metavision/psee_hw_layer/devices/gen41/gen41_antiflicker_module.h"
#include "metavision/psee_hw_layer/utils/register_map.h"

namespace Metavision {

namespace {
// Invalidation Speed
int invalidation_dt_fifo_wait_time = 1630;
} // namespace

Gen41AntiFlickerModule::Gen41AntiFlickerModule(const std::shared_ptr<RegisterMap> &register_map,
                                               const std::string &sensor_prefix) :
    register_map_(register_map), sensor_prefix_(sensor_prefix) {}

bool Gen41AntiFlickerModule::enable(bool b) {
    // Always disable in order to configure AFK parameters
    (*register_map_)[sensor_prefix_ + "afk/pipeline_control"].write_value(0b101);
    if (!b) {
        return true;
    }

    bool init_done = 0;
    for (int i = 0; i < 3; i++) {
        init_done = (*register_map_)[sensor_prefix_ + "afk/initialization"]["afk_flag_init_done"].read_value();
        if (init_done == 1) {
            break;
        }
    }
    if (init_done == 0) {
        throw HalException(HalErrorCode::InternalInitializationError, "Bad AFK initialization");
    }

    // Set frequency bandwidth
    int filter_period_min_cutoff_period;
    filter_period_min_cutoff_period = 1e6 / high_freq_;
    filter_period_min_cutoff_period = filter_period_min_cutoff_period >> 7;

    int filter_period_max_cutoff_period;
    filter_period_max_cutoff_period = 1e6 / low_freq_;
    filter_period_max_cutoff_period = filter_period_max_cutoff_period >> 7;

    (*register_map_)[sensor_prefix_ + "afk/invalidation"].write_value(
        {"dt_fifo_wait_time", invalidation_dt_fifo_wait_time});
    (*register_map_)[sensor_prefix_ + "afk/filter_period"].write_value(
        {{"min_cutoff_period", filter_period_min_cutoff_period},
         {"max_cutoff_period", filter_period_max_cutoff_period},
         {"inverted_duty_cycle", inverted_duty_cycle_}});

    // Set filtering mode
    (*register_map_)[sensor_prefix_ + "afk/param"]["invert"].write_value(mode_ == BAND_STOP ? 0 : 1);

    // Set duty cycle
    (*register_map_)[sensor_prefix_ + "afk/filter_period"]["inverted_duty_cycle"].write_value(inverted_duty_cycle_);

    // Set hysteresis counters
    (*register_map_)[sensor_prefix_ + "afk/param"]["counter_high"].write_value(start_threshold_);
    (*register_map_)[sensor_prefix_ + "afk/param"]["counter_low"].write_value(stop_threshold_);

    (*register_map_)[sensor_prefix_ + "afk/pipeline_control"].write_value(0b001);

    return true;
}

bool Gen41AntiFlickerModule::is_enabled() {
    return (*register_map_)[sensor_prefix_ + "afk/pipeline_control"].read_value() == 0b001;
}

bool Gen41AntiFlickerModule::reset() {
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

bool Gen41AntiFlickerModule::set_frequency_band(uint32_t low_freq, uint32_t high_freq) {
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

uint32_t Gen41AntiFlickerModule::get_min_supported_frequency() const {
    return 50;
}

uint32_t Gen41AntiFlickerModule::get_max_supported_frequency() const {
    return 520;
}

uint32_t Gen41AntiFlickerModule::get_band_low_frequency() const {
    return low_freq_;
}

uint32_t Gen41AntiFlickerModule::get_band_high_frequency() const {
    return high_freq_;
}

bool Gen41AntiFlickerModule::set_filtering_mode(I_AntiFlickerModule::AntiFlickerMode mode) {
    mode_ = mode;

    // Reset if needed
    if (!reset()) {
        return false;
    }

    return true;
}

I_AntiFlickerModule::AntiFlickerMode Gen41AntiFlickerModule::get_filtering_mode() const {
    return mode_;
}

bool Gen41AntiFlickerModule::set_duty_cycle(float duty_cycle) {
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

float Gen41AntiFlickerModule::get_duty_cycle() const {
    return 100.0 - (inverted_duty_cycle_ * 100.0) / 16.0;
}

float Gen41AntiFlickerModule::get_min_supported_duty_cycle() const {
    return 1.0 / 16.0;
}

float Gen41AntiFlickerModule::get_max_supported_duty_cycle() const {
    return 100.0;
}

bool Gen41AntiFlickerModule::set_start_threshold(uint32_t threshold) {
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

bool Gen41AntiFlickerModule::set_stop_threshold(uint32_t threshold) {
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

uint32_t Gen41AntiFlickerModule::get_start_threshold() const {
    return start_threshold_;
}

uint32_t Gen41AntiFlickerModule::get_stop_threshold() const {
    return stop_threshold_;
}

uint32_t Gen41AntiFlickerModule::get_min_supported_start_threshold() const {
    return 0;
}

uint32_t Gen41AntiFlickerModule::get_max_supported_start_threshold() const {
    return (1 << 3) - 1;
}

uint32_t Gen41AntiFlickerModule::get_min_supported_stop_threshold() const {
    return 0;
}

uint32_t Gen41AntiFlickerModule::get_max_supported_stop_threshold() const {
    return (1 << 3) - 1;
}

} // namespace Metavision
