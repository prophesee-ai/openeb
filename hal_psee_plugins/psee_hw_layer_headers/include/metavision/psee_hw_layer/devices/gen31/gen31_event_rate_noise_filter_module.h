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

#ifndef METAVISION_HAL_GEN31_EVENT_RATE_NOISE_FILTER_MODULE_H
#define METAVISION_HAL_GEN31_EVENT_RATE_NOISE_FILTER_MODULE_H

#include <string>

#include "metavision/hal/facilities/i_event_rate_activity_filter_module.h"

namespace Metavision {

class I_HW_Register;

class Gen31_EventRateNoiseFilterModule : public I_EventRateActivityFilterModule {
public:
    using NflThresholds = I_EventRateActivityFilterModule::thresholds;

    Gen31_EventRateNoiseFilterModule(const std::shared_ptr<I_HW_Register> &i_hw_register, const std::string &prefix);

    bool enable(bool enable_filter) override;
    bool is_enabled() const override;

    NflThresholds is_thresholds_supported() const override;
    bool set_thresholds(const NflThresholds &thresholds_ev_s) override;
    NflThresholds get_thresholds() const override;

    NflThresholds get_min_supported_thresholds() const override;
    NflThresholds get_max_supported_thresholds() const override;

    static constexpr uint32_t min_time_window_us_            = 1;
    static constexpr uint32_t max_time_window_us_            = 1023;
    static constexpr uint32_t min_event_rate_threshold_kev_s = 10;
    static constexpr uint32_t max_event_rate_threshold_kev_s = 10000;

protected:
    const std::shared_ptr<I_HW_Register> &get_hw_register() const;

private:
    bool set_time_window(uint32_t window_length_us);
    uint32_t get_time_window() const;

    std::shared_ptr<I_HW_Register> i_hw_register_;
    const std::string base_name_;
};

} // namespace Metavision

#endif // METAVISION_HAL_GEN31_EVENT_RATE_NOISE_FILTER_MODULE_H
