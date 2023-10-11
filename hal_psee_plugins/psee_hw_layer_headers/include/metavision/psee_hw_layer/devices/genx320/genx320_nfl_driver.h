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

#ifndef METAVISION_HAL_GENX320_NFL_DRIVER_H
#define METAVISION_HAL_GENX320_NFL_DRIVER_H

#include <string>

#include "metavision/hal/facilities/i_registrable_facility.h"
#include "metavision/hal/facilities/i_event_rate_activity_filter_module.h"

namespace Metavision {

class RegisterMap;

class GenX320NflDriver : public I_RegistrableFacility<GenX320NflDriver> {
public:
    using NflThresholds = I_EventRateActivityFilterModule::thresholds;

    GenX320NflDriver(const std::shared_ptr<RegisterMap> &register_map);

    bool enable(bool enable_filter);
    bool is_enabled() const;

    NflThresholds is_thresholds_supported() const;
    bool set_thresholds(const NflThresholds &thresholds_ev_s);
    NflThresholds get_thresholds() const;

    NflThresholds get_min_supported_thresholds() const;
    NflThresholds get_max_supported_thresholds() const;

private:
    uint32_t compute_cd_threshold(uint32_t event_rate_ev_s) const;
    uint32_t compute_event_rate(uint32_t threshold) const;

    bool set_time_window(uint32_t window_length_us);
    uint32_t get_time_window() const;

    static constexpr uint32_t min_time_window_us            = 1;
    static constexpr uint32_t max_time_window_us            = 1024;
    static constexpr uint32_t min_pixel_event_threshold_on  = 0;
    static constexpr uint32_t min_pixel_event_threshold_off = 0;
    static constexpr uint32_t max_pixel_event_threshold_on  = 0x190000;
    static constexpr uint32_t max_pixel_event_threshold_off = 0x190000;
    static constexpr uint32_t max_register_value            = 0x1FFFFF;

    std::shared_ptr<RegisterMap> register_map_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GENX320_NFL_DRIVER_H
