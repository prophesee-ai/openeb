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

#ifndef METAVISION_HAL_GENX320_NOISE_FILTER_H
#define METAVISION_HAL_GENX320_NOISE_FILTER_H

#include <string>

#include "metavision/hal/facilities/i_event_rate_noise_filter_module.h"

namespace Metavision {

class RegisterMap;

class GenX320NoiseFilter : public I_EventRateNoiseFilterModule {
public:
    GenX320NoiseFilter(const std::shared_ptr<RegisterMap> &register_map);

    virtual bool enable(bool enable_filter) override;
    virtual bool set_event_rate_threshold(uint32_t threshold_Kev_s) override;
    virtual uint32_t get_event_rate_threshold() override;

    static constexpr uint32_t min_time_window_us_            = 1;
    static constexpr uint32_t max_time_window_us_            = 1023;
    static constexpr uint32_t min_event_rate_threshold_kev_s = 10;
    static constexpr uint32_t max_event_rate_threshold_kev_s = 10000;

private:
    bool set_time_window(uint32_t window_length_us);
    uint32_t get_time_window();
    uint32_t current_threshold_kev_s_{0};

    static constexpr uint32_t max_pixel_event_threshold_on  = 0xFFFFFFFF;
    static constexpr uint32_t max_pixel_event_threshold_off = 0xFFFFFFFF;

    std::shared_ptr<RegisterMap> register_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_NOISE_FILTER_H
