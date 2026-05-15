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

#include "metavision/psee_hw_layer/devices/genx320/genx320_nfl_driver.h"
#include "metavision/psee_hw_layer/utils/register_map.h"
#include "metavision/hal/utils/hal_exception.h"
#include "utils/psee_hal_plugin_error_code.h"

using vfield = std::map<std::string, uint32_t>;

namespace Metavision {

GenX320NflDriver::GenX320NflDriver(const std::shared_ptr<RegisterMap> &register_map) : register_map_(register_map) {
    set_time_window(1000);
    auto min_thres = get_min_supported_thresholds();
    auto max_thres = get_max_supported_thresholds();
    set_thresholds({min_thres.lower_bound_start, min_thres.lower_bound_stop, max_thres.upper_bound_start,
                    max_thres.upper_bound_stop});
}

bool GenX320NflDriver::enable(bool enable_filter) {
    (*register_map_)["nfl/pipeline_control"].write_value(vfield{{"enable", 1}, {"bypass", 1}});

    (*register_map_)["nfl/insert_drop_monitoring"]["en"].write_value(0);
    (*register_map_)["nfl/pipeline_control"]["bypass"].write_value(!enable_filter);

    return true;
}

bool GenX320NflDriver::is_enabled() const {
    return !(*register_map_)["nfl/pipeline_control"]["bypass"].read_value();
}

bool GenX320NflDriver::set_time_window(uint32_t window_length_us) {
    // Using higher time resolution will ensure we are closest to the user input
    // threshold. Higher time window also increase latency. Though it is mostly invisible for the user given the
    // maximum value is 1024 us.

    if (window_length_us < min_time_window_us || window_length_us > max_time_window_us) {
        return false;
    }

    (*register_map_)["nfl/reference_period"]["val"].write_value(window_length_us);
    return true;
}

uint32_t GenX320NflDriver::get_time_window() const {
    return (*register_map_)["nfl/reference_period"]["val"].read_value();
}

uint32_t GenX320NflDriver::compute_cd_threshold(uint32_t event_rate_ev_s) const {
    return std::round((event_rate_ev_s / 1000000.) * get_time_window());
}

uint32_t GenX320NflDriver::compute_event_rate(uint32_t threshold) const {
    return std::round((threshold * 1000000.) / get_time_window());
}

GenX320NflDriver::NflThresholds GenX320NflDriver::is_thresholds_supported() const {
    NflThresholds thresholds_valid;

    thresholds_valid.lower_bound_start = 1;
    thresholds_valid.lower_bound_stop  = 1;
    thresholds_valid.upper_bound_start = 1;
    thresholds_valid.upper_bound_stop  = 1;

    return thresholds_valid;
}

bool GenX320NflDriver::set_thresholds(const GenX320NflDriver::NflThresholds &thresholds_ev_s) {
    // Set lower bound start
    auto thres = compute_cd_threshold(thresholds_ev_s.lower_bound_start);
    (*register_map_)["nfl/min_voxel_threshold_on"]["val"].write_value(thres);

    // Set lower bound stop
    thres = compute_cd_threshold(thresholds_ev_s.lower_bound_stop);
    (*register_map_)["nfl/min_voxel_threshold_off"]["val"].write_value(thres);

    // Set upper bound start
    auto max_evt_rate = get_max_supported_thresholds().upper_bound_start;
    thres             = compute_cd_threshold(thresholds_ev_s.upper_bound_start);

    if (thres > max_register_value) {
        std::ostringstream err_msg;
        err_msg << "NFL upper bound event rate threshold '" << thresholds_ev_s.upper_bound_start
                << " evt/s' exceeds register maximum allowed value.";
        MV_HAL_LOG_ERROR() << err_msg.str();
        return false;
    }

    if (thresholds_ev_s.upper_bound_start > max_evt_rate) {
        std::ostringstream warn_msg;
        warn_msg << "NFL upper bound event rate threshold selected '" << thresholds_ev_s.upper_bound_start
                 << " evt/s' exceeds highest settings.";
        MV_HAL_LOG_WARNING() << warn_msg.str();
        warn_msg.clear();
        warn_msg.str("");
        warn_msg << "NFL upper bound will be capped at '" << max_evt_rate << " evt/s'";
        MV_HAL_LOG_WARNING() << warn_msg.str();
        thres = compute_cd_threshold(max_evt_rate);
        MV_HAL_LOG_INFO() << "Threshold = 0x" << std::hex << thres << std::dec;
    }

    (*register_map_)["nfl/max_voxel_threshold_on"]["val"].write_value(thres);

    // Set upper bound stop
    max_evt_rate = get_max_supported_thresholds().upper_bound_stop;
    thres        = compute_cd_threshold(thresholds_ev_s.upper_bound_stop);

    if (thres > max_register_value) {
        std::ostringstream err_msg;
        err_msg << "NFL upper bound event rate threshold '" << thresholds_ev_s.upper_bound_stop
                << " evt/s' exceeds register maximum allowed value.";
        MV_HAL_LOG_ERROR() << err_msg.str();
        return false;
    }

    if (thresholds_ev_s.upper_bound_stop > max_evt_rate) {
        std::ostringstream warn_msg;
        warn_msg << "NFL upper bound event rate threshold selected '" << thresholds_ev_s.upper_bound_stop
                 << " evt/s' exceeds highest settings.";
        MV_HAL_LOG_WARNING() << warn_msg.str();
        warn_msg.clear();
        warn_msg.str("");
        warn_msg << "NFL upper bound will be capped at '" << max_evt_rate << " evt/s'";
        MV_HAL_LOG_WARNING() << warn_msg.str();
        thres = compute_cd_threshold(max_evt_rate);
        MV_HAL_LOG_INFO() << "Threshold = 0x" << std::hex << thres << std::dec;
    }

    (*register_map_)["nfl/max_voxel_threshold_off"]["val"].write_value(thres);

    return true;
}

GenX320NflDriver::NflThresholds GenX320NflDriver::get_thresholds() const {
    GenX320NflDriver::NflThresholds thres_list;

    thres_list.lower_bound_start =
        compute_event_rate((*register_map_)["nfl/min_voxel_threshold_on"]["val"].read_value());
    thres_list.lower_bound_stop =
        compute_event_rate((*register_map_)["nfl/min_voxel_threshold_off"]["val"].read_value());
    thres_list.upper_bound_start =
        compute_event_rate((*register_map_)["nfl/max_voxel_threshold_on"]["val"].read_value());
    thres_list.upper_bound_stop =
        compute_event_rate((*register_map_)["nfl/max_voxel_threshold_off"]["val"].read_value());

    return thres_list;
}

GenX320NflDriver::NflThresholds GenX320NflDriver::get_min_supported_thresholds() const {
    GenX320NflDriver::NflThresholds thres;

    thres.lower_bound_start = compute_event_rate(min_pixel_event_threshold_on);
    thres.lower_bound_stop  = compute_event_rate(min_pixel_event_threshold_off);
    thres.upper_bound_start = compute_event_rate(min_pixel_event_threshold_on);
    thres.upper_bound_stop  = compute_event_rate(min_pixel_event_threshold_off);

    return thres;
}

GenX320NflDriver::NflThresholds GenX320NflDriver::get_max_supported_thresholds() const {
    GenX320NflDriver::NflThresholds thres;

    thres.lower_bound_start = compute_event_rate(max_pixel_event_threshold_on);
    thres.lower_bound_stop  = compute_event_rate(max_pixel_event_threshold_off);
    thres.upper_bound_start = compute_event_rate(max_pixel_event_threshold_on);
    thres.upper_bound_stop  = compute_event_rate(max_pixel_event_threshold_off);

    return thres;
}

} // namespace Metavision
