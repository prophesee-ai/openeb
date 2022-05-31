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
#include <limits>

#include "facilities/experimental/psee_stc_registers_api_v0.h"

namespace {

void noise_filter_parameters(uint32_t threshold, uint32_t &best_prescaler, uint32_t &best_multiplier,
                             uint32_t &max_invalidation_period, uint32_t nbits_prescaler = 5,
                             uint32_t nbits_multiplier = 4, uint32_t bitdepth = 4, uint32_t refresh_speed = 2592) {
    double best_precision   = std::numeric_limits<double>::max();
    best_prescaler          = std::numeric_limits<uint32_t>::max();
    best_multiplier         = std::numeric_limits<uint32_t>::max();
    max_invalidation_period = std::numeric_limits<uint32_t>::max();
    for (uint32_t prescaler = 1; prescaler < (1UL << nbits_prescaler); prescaler++) {
        uint32_t min_multiplier = ((1UL << (prescaler - 1)) / threshold);
        min_multiplier          = min_multiplier <= 1 ? 1 : min_multiplier - 1;
        uint64_t max_multiplier =
            (((uint64_t)((1UL << bitdepth) - 2) << prescaler) - ((1UL << (prescaler - 1)))) / threshold;

        max_multiplier = max_multiplier >= (1UL << nbits_multiplier) ? (1UL << nbits_multiplier) : max_multiplier;
        for (uint32_t multiplier = min_multiplier; multiplier < (1UL << nbits_multiplier); multiplier++) {
            uint32_t low_res_threshold = ((multiplier * threshold) + ((1UL << (prescaler - 1)))) >> prescaler;
            if (low_res_threshold >= (uint32_t)((1UL << bitdepth) - 2)) {
                // increasing the multiplier will increase low_res_threshold no need
                // to go futher
                continue;
            }
            if (low_res_threshold == 0) {
                // should not happen a lot since the min_multiplier is computed
                continue;
            }
            double precision = (1L << (prescaler)) / multiplier - 1;

            if (precision >= best_precision) {
                continue;
            }

            uint32_t invalidation_period = ((1UL << bitdepth) - 2 - low_res_threshold) * precision;
            if (invalidation_period < refresh_speed) {
                continue;
            }
            if (invalidation_period == 0) {
                continue;
            }

            max_invalidation_period = invalidation_period;
            best_prescaler          = prescaler;
            best_multiplier         = multiplier;
            best_precision          = precision;
        }
    }
}

} // anonymous namespace

namespace Metavision {

void PseeSTCRegistersAPIv0::enable(Type type, uint32_t threshold) {
    disable();
    uint32_t prescaler           = 0;
    uint32_t multiplier          = 0;
    uint32_t invalidation_period = 0;

    noise_filter_parameters(threshold, prescaler, multiplier, invalidation_period);
    int32_t dt_fifo_timeout = (double)(100 / (4 * 720)) * 0.9 * invalidation_period - 4;
    if (dt_fifo_timeout < 0) {
        dt_fifo_timeout = 0;
    }

    if (type == I_NoiseFilterModule::Type::STC) {
        set_trail_param(false, threshold);
        set_stc_param(true, threshold);
        set_timestamping(prescaler, multiplier, true);
    }
    if (type == I_NoiseFilterModule::Type::TRAIL) {
        set_stc_param(false, threshold);
        set_trail_param(true, threshold);
        set_timestamping(prescaler, multiplier, true);
    }
    set_invalidation(4, dt_fifo_timeout, 10, false);
    set_mode(STCModes::ON);
    start();
}

void PseeSTCRegistersAPIv0::disable() {
    stop();
}

} // namespace Metavision
