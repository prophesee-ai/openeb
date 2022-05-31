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

#include <memory>

#include "facilities/experimental/psee_afk_registers_api_v0.h"
#include "metavision/hal/utils/hal_error_code.h"
#include "metavision/hal/utils/hal_exception.h"

namespace Metavision {

void PseeAFKRegistersAPIv0::enable() {
    if (!initialized_) {
        initialize();
        initialized_ = true;
    }

    set_mode(AFKModes::ON);
    start();
}

void PseeAFKRegistersAPIv0::disable() {
    set_mode(AFKModes::BYPASS);
    stop();
    wait_for_init();
}

void PseeAFKRegistersAPIv0::set_frequency(uint32_t frequency_center, uint32_t bandwidth, bool stop) {
    if (frequency_center < 50 || frequency_center > 500) {
        throw HalException(HalErrorCode::InvalidAFKValue,
                           "Antiflicker frequency center: " + std::to_string(frequency_center) +
                               " is not in the range [50, 500]");
    }

    // Divide the bandwith by 2 to ease the computation
    bandwidth = bandwidth / 2;

    if (frequency_center - bandwidth < 50) {
        throw HalException(HalErrorCode::InvalidAFKValue,
                           "Antiflicker lower frequency: " + std::to_string(frequency_center - bandwidth) +
                               " is lower than 50." +
                               " Either increase the frequency_center to: " + std::to_string(bandwidth + 50) +
                               " or decrease the bandwidth to: " + std::to_string((frequency_center - 50) * 2));
    }
    if (frequency_center + bandwidth > 500) {
        throw HalException(HalErrorCode::InvalidAFKValue,
                           "Antiflicker lower frequency: " + std::to_string(frequency_center + bandwidth) +
                               " is greater than 500." +
                               " Either decrease the frequency_center to: " + std::to_string(500 - bandwidth) +
                               " or decrease the bandwidth to: " + std::to_string((500 - frequency_center) * 2));
    }

    set_frequency_band(frequency_center - bandwidth, frequency_center + bandwidth, stop);
}

void PseeAFKRegistersAPIv0::set_frequency_band(uint32_t min_freq, uint32_t max_freq, bool stop) {
    if (min_freq >= max_freq) {
        throw HalException(HalErrorCode::InvalidAFKValue,
                           "Antiflicker min frequency: " + std::to_string(min_freq) +
                               " should be less than max frequency: " + std::to_string(max_freq));
    }

    if (min_freq < 50 || max_freq > 500) {
        throw HalException(HalErrorCode::InvalidAFKValue, "Antiflicker min frequency: " + std::to_string(min_freq) +
                                                              " max frequency: " + std::to_string(max_freq) +
                                                              " is not in the range [50, 500]");
    }

    if (!initialized_) {
        initialize();
        initialized_ = true;
    }

    // Keep mode to restart after the configuration in the same mode
    auto mode = get_mode();

    // Disable the afk while reconfiguring it
    disable();

    uint32_t min_period = to_period(max_freq);
    uint32_t max_period = to_period(min_freq);

    set_filter_period(min_period, max_period, to_inverted_duty_cycle(0.5));
    uint32_t counter_low  = 0;
    uint32_t counter_high = 0;
    bool invert           = 0;
    bool drop_disable     = 0;
    get_afk_param(counter_low, counter_high, invert, drop_disable);
    set_afk_param(counter_low, counter_high, !stop, false);
    uint32_t in_parallel     = 10; // As recommended before characterization
    uint32_t dt_fifo_timeout = 90; // As recommended

    uint32_t dt_fifo_wait_time = ((253 - max_period) * 2 ^ 7) / ((720 * 10) / (0.01 * in_parallel)) - dt_fifo_timeout;
    if (dt_fifo_wait_time < 4) {
        dt_fifo_timeout -= 4 - dt_fifo_wait_time;
        dt_fifo_wait_time = 4;
    }
    set_invalidation(dt_fifo_wait_time >> 7, dt_fifo_timeout, in_parallel, true);

    // Restart the AFK in the same mode as before
    set_mode(mode);
    start();
}

} // namespace Metavision
