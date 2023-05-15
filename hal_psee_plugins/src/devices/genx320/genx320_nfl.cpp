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

#include "metavision/psee_hw_layer/devices/genx320/genx320_nfl.h"
#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

constexpr uint32_t GenX320NoiseFilter::min_time_window_us_;
constexpr uint32_t GenX320NoiseFilter::max_time_window_us_;
constexpr uint32_t GenX320NoiseFilter::min_event_rate_threshold_kev_s;
constexpr uint32_t GenX320NoiseFilter::max_event_rate_threshold_kev_s;

GenX320NoiseFilter::GenX320NoiseFilter(const std::shared_ptr<RegisterMap> &register_map) :
    register_map_(register_map) {}

bool GenX320NoiseFilter::enable(bool enable_filter) {
    (*register_map_)["nfl/pipeline_control"].write_value(vfield{{"enable", 1}, {"bypass", 1}});

    (*register_map_)["nfl/insert_drop_monitoring"]["en"].write_value(0);
    (*register_map_)["nfl/max_voxel_threshold_on"]["val"].write_value(max_pixel_event_threshold_on);
    (*register_map_)["nfl/max_voxel_threshold_off"]["val"].write_value(max_pixel_event_threshold_off);
    get_event_rate_threshold();

    (*register_map_)["nfl/pipeline_control"]["bypass"].write_value(!enable_filter);

    return true;
}

bool GenX320NoiseFilter::set_time_window(uint32_t window_length_us) {
    if (window_length_us < min_time_window_us_ || window_length_us > max_time_window_us_) {
        return false;
    }

    (*register_map_)["nfl/reference_period"]["val"].write_value(window_length_us);
    return true;
}

uint32_t GenX320NoiseFilter::get_time_window() {
    return (*register_map_)["nfl/reference_period"]["val"].read_value();
}

bool GenX320NoiseFilter::set_event_rate_threshold(uint32_t threshold_kev_s) {
    if (threshold_kev_s < min_event_rate_threshold_kev_s || threshold_kev_s > max_event_rate_threshold_kev_s) {
        return false;
    }

    set_time_window(max_time_window_us_); // Maximum time resolution to ensures we are closest to the user input
                                          // threshold (1023 us latency) though it is invisible in term of user
                                          // experience.

    auto min_event_count_in_time_shifting_window =
        std::round((threshold_kev_s / 1000.) * get_time_window()); // ev per microseconds

    (*register_map_)["nfl/min_voxel_threshold_off"]["val"].write_value(min_event_count_in_time_shifting_window + 100);
    (*register_map_)["nfl/min_voxel_threshold_on"]["val"].write_value(min_event_count_in_time_shifting_window);

    get_event_rate_threshold();
    return true;
}

uint32_t GenX320NoiseFilter::get_event_rate_threshold() {
    current_threshold_kev_s_ =
        std::round(((*register_map_)["nfl/min_voxel_threshold_off"]["val"].read_value() * 1000.) / get_time_window());
    return current_threshold_kev_s_;
}

} // namespace Metavision
