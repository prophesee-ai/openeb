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

#include "devices/gen41/gen41_antiflicker_module.h"
#include "utils/register_map.h"

namespace Metavision {

namespace {
// Invalidation Speed
int invalidation_dt_fifo_wait_time = 1630;
} // namespace

Gen41AntiFlickerModule::Gen41AntiFlickerModule(const std::shared_ptr<RegisterMap> &register_map,
                                               const std::string &sensor_prefix) :
    register_map_(register_map), sensor_prefix_(sensor_prefix) {}

void Gen41AntiFlickerModule::enable() {
    (*register_map_)[sensor_prefix_ + "afk/pipeline_control"]["afk_en"].write_value(0b001);
}

void Gen41AntiFlickerModule::disable() {
    (*register_map_)[sensor_prefix_ + "afk/pipeline_control"]["afk_en"].write_value(0b101);
}

void Gen41AntiFlickerModule::set_frequency(uint32_t frequency_center, uint32_t bandwidth, bool stop) {
    uint32_t freq_min;
    uint32_t freq_max;
    freq_min = frequency_center - bandwidth / 2;
    freq_max = frequency_center + bandwidth / 2;
    set_frequency_band(freq_min, freq_max);
}

void Gen41AntiFlickerModule::set_frequency_band(uint32_t min_freq, uint32_t max_freq, bool stop) {
    // Check frequency values
    if (min_freq < 50 || max_freq > 500 || min_freq > max_freq) {
        MV_HAL_LOG_ERROR() << "Bad AFK frequencies";
        return;
    }

    // Store AFK status
    auto afk_en_prev = (*register_map_)[sensor_prefix_ + "afk/pipeline_control"]["afk_en"].read_value();

    // Disable AFK before changing frequency
    disable();

    // Check sram init done
    bool init_done = 0;
    for (int i = 0; i < 3; i++) {
        init_done = (*register_map_)[sensor_prefix_ + "afk/initialization"]["afk_flag_init_done"].read_value();
        if (init_done == 1) {
            break;
        }
    }
    if (init_done == 0) {
        MV_HAL_LOG_ERROR() << "Bad AFK initialization";
        return;
    }

    // Set new algorithm configuration
    int filter_period_min_cutoff_period;
    filter_period_min_cutoff_period = 1e6 / max_freq;
    filter_period_min_cutoff_period = filter_period_min_cutoff_period >> 7;

    int filter_period_max_cutoff_period;
    filter_period_max_cutoff_period = 1e6 / min_freq;
    filter_period_max_cutoff_period = filter_period_max_cutoff_period >> 7;

    (*register_map_)[sensor_prefix_ + "afk/invalidation"].write_value(
        {"dt_fifo_wait_time", invalidation_dt_fifo_wait_time});
    (*register_map_)[sensor_prefix_ + "afk/filter_period"].write_value(
        {{"min_cutoff_period", filter_period_min_cutoff_period},
         {"max_cutoff_period", filter_period_max_cutoff_period},
         {"Reserved_19_16", 8}});

    (*register_map_)[sensor_prefix_ + "afk/Reserved_C004"]["Reserved_6"].write_value(int(!stop));

    // Restore AFK status
    (*register_map_)[sensor_prefix_ + "afk/pipeline_control"]["afk_en"].write_value(afk_en_prev);
}

} // namespace Metavision
