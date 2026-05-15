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

#include <iostream>
#include <cmath>

#include "metavision/psee_hw_layer/devices/gen31/gen31_event_rate_noise_filter_module.h"
#include "metavision/hal/facilities/i_hw_register.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

namespace Metavision {

constexpr uint32_t Gen31_EventRateNoiseFilterModule::min_time_window_us_;
constexpr uint32_t Gen31_EventRateNoiseFilterModule::max_time_window_us_;
constexpr uint32_t Gen31_EventRateNoiseFilterModule::min_event_rate_threshold_kev_s;
constexpr uint32_t Gen31_EventRateNoiseFilterModule::max_event_rate_threshold_kev_s;

// Gen3.1 event rate noise filter class
//
// The number of events per event period is programmable via the register evt_thresh.
// Period length in ‘us’ is programmable in the period_cnt_thresh register.
// Valid values range from 1 to 1023.
// The microsecond tick can be adjusted using the “pre_cnt_thresh” (i.e. in case the sensor runs at frequency lower
// than 100MHz). Longer is the period, smoother is the event rate calculation, but higher is the latency. Using small
// “period_cnt_thresh”, on the contrary, will result in shorter latency but also shorter integration causing less
// precise event rate estimation.
// User has to program the event threshold (measured in number of event per period) below which events are not
// transmitted. The event rate supported ranges from 25kevt/s to 10Mevt/s. By default the block is configured to output
// events above 500kevt/s (50 events every 100us). The user is responsible for programming in a consistent way the
// above mentioned parameters in order to reach the best tradeoff between latency and precision.

Gen31_EventRateNoiseFilterModule::Gen31_EventRateNoiseFilterModule(const std::shared_ptr<I_HW_Register> &i_hw_register,
                                                                   const std::string &prefix) :
    i_hw_register_(i_hw_register), base_name_(prefix) {
    if (!i_hw_register_) {
        throw(HalException(PseeHalPluginErrorCode::HWRegisterNotFound, "HW Register facility is null."));
    }
}

bool Gen31_EventRateNoiseFilterModule::enable(bool enable_filter) {
    get_hw_register()->write_register(base_name_ + "nfl_ctrl", "nfl_en", enable_filter);
    get_thresholds();

    return true;
}

bool Gen31_EventRateNoiseFilterModule::is_enabled() const {
    return get_hw_register()->read_register(base_name_ + "nfl_ctrl", "nfl_en");
}

bool Gen31_EventRateNoiseFilterModule::set_time_window(uint32_t window_length_us) {
    if (window_length_us < min_time_window_us_ || window_length_us > max_time_window_us_) {
        return false;
    }

    get_hw_register()->write_register(base_name_ + "nfl_thresh", "period_cnt_thresh", window_length_us);
    return true;
}

uint32_t Gen31_EventRateNoiseFilterModule::get_time_window() const {
    return get_hw_register()->read_register(base_name_ + "nfl_thresh", "period_cnt_thresh");
}

bool Gen31_EventRateNoiseFilterModule::set_thresholds(
    const Gen31_EventRateNoiseFilterModule::NflThresholds &thresholds_ev_s) {
    uint32_t threshold_kev_s = std::round(thresholds_ev_s.lower_bound_start / 1000.0);
    if (threshold_kev_s < min_event_rate_threshold_kev_s || threshold_kev_s > max_event_rate_threshold_kev_s) {
        return false;
    }

    set_time_window(max_time_window_us_); // Maximum time resolution to ensures we are closest to the user input
                                          // threshold (1023 us latency) though it is invisible in term of user
                                          // experience.

    auto min_event_count_in_time_shifting_window =
        std::round((threshold_kev_s / 1000.) * get_time_window()); // ev per microseconds
    get_hw_register()->write_register(base_name_ + "nfl_thresh", "evt_thresh", min_event_count_in_time_shifting_window);
    get_thresholds();
    return true;
}

Gen31_EventRateNoiseFilterModule::NflThresholds Gen31_EventRateNoiseFilterModule::get_thresholds() const {
    const uint32_t current_threshold_kev_s = std::round(
         (get_hw_register()->read_register(base_name_ + "nfl_thresh", "evt_thresh") * 1000.) / get_time_window());

    return {(current_threshold_kev_s * 1000), 0, 0, 0};
}

Gen31_EventRateNoiseFilterModule::NflThresholds Gen31_EventRateNoiseFilterModule::is_thresholds_supported() const {
    return {1, 0, 0, 0};
}

Gen31_EventRateNoiseFilterModule::NflThresholds Gen31_EventRateNoiseFilterModule::get_min_supported_thresholds() const {
    return {(min_event_rate_threshold_kev_s * 1000), 0, 0, 0};
}

Gen31_EventRateNoiseFilterModule::NflThresholds Gen31_EventRateNoiseFilterModule::get_max_supported_thresholds() const {
    return {(max_event_rate_threshold_kev_s * 1000), 0, 0, 0};
}

const std::shared_ptr<I_HW_Register> &Gen31_EventRateNoiseFilterModule::get_hw_register() const {
    return i_hw_register_;
}

} // namespace Metavision
