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

#include <map>
#include <cmath>

#include "metavision/hal/utils/hal_exception.h"
#include "metavision/psee_hw_layer/devices/common/event_trail_filter.h"

using vfield = std::map<std::string, uint32_t>;
namespace Metavision {

namespace {
std::map<const int, std::map<const std::string, const int>> gen41_stc_threshold_params = {
    {1, {{"presc", 12}, {"mult", 15}, {"dt_fifo_timeout", 90}}},
    {2, {{"presc", 10}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {3, {{"presc", 11}, {"mult", 5}, {"dt_fifo_timeout", 95}}},
    {4, {{"presc", 9}, {"mult", 1}, {"dt_fifo_timeout", 102}}},
    {5, {{"presc", 13}, {"mult", 15}, {"dt_fifo_timeout", 90}}},
    {6, {{"presc", 11}, {"mult", 3}, {"dt_fifo_timeout", 114}}},
    {7, {{"presc", 11}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {8, {{"presc", 12}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {9, {{"presc", 13}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {10, {{"presc", 13}, {"mult", 9}, {"dt_fifo_timeout", 90}}},
    {11, {{"presc", 10}, {"mult", 1}, {"dt_fifo_timeout", 102}}},
    {12, {{"presc", 14}, {"mult", 15}, {"dt_fifo_timeout", 109}}},
    {13, {{"presc", 13}, {"mult", 7}, {"dt_fifo_timeout", 117}}},
    {14, {{"presc", 14}, {"mult", 13}, {"dt_fifo_timeout", 127}}},
    {15, {{"presc", 12}, {"mult", 3}, {"dt_fifo_timeout", 138}}},
    {16, {{"presc", 12}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {17, {{"presc", 12}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {18, {{"presc", 14}, {"mult", 11}, {"dt_fifo_timeout", 99}}},
    {19, {{"presc", 13}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {20, {{"presc", 13}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {21, {{"presc", 14}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {22, {{"presc", 14}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {23, {{"presc", 11}, {"mult", 1}, {"dt_fifo_timeout", 209}}},
    {24, {{"presc", 11}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {25, {{"presc", 11}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {26, {{"presc", 15}, {"mult", 15}, {"dt_fifo_timeout", 147}}},
    {27, {{"presc", 15}, {"mult", 15}, {"dt_fifo_timeout", 147}}},
    {28, {{"presc", 14}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {29, {{"presc", 14}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {30, {{"presc", 15}, {"mult", 13}, {"dt_fifo_timeout", 170}}},
    {31, {{"presc", 15}, {"mult", 13}, {"dt_fifo_timeout", 170}}},
    {32, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {33, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {34, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {35, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {36, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 90}}},
    {37, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {38, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 99}}},
    {39, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 99}}},
    {40, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 99}}},
    {41, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {42, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {43, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {44, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 109}}},
    {45, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {46, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {47, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {48, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {49, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 122}}},
    {50, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 280}}},
    {51, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 280}}},
    {52, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {53, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {54, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {55, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 138}}},
    {56, {{"presc", 16}, {"mult", 15}, {"dt_fifo_timeout", 147}}},
    {57, {{"presc", 16}, {"mult", 15}, {"dt_fifo_timeout", 147}}},
    {58, {{"presc", 16}, {"mult", 15}, {"dt_fifo_timeout", 147}}},
    {59, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {60, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {61, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {62, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {63, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 158}}},
    {64, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 171}}},
    {65, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 171}}},
    {66, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 171}}},
    {67, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 171}}},
    {68, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 171}}},
    {69, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {70, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {71, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {72, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {73, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 185}}},
    {74, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 409}}},
    {75, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {76, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {77, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {78, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {79, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {80, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 202}}},
    {81, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 451}}},
    {82, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {83, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {84, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {85, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {86, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {87, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {88, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 223}}},
    {89, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 501}}},
    {90, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 501}}},
    {91, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 501}}},
    {92, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {93, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {94, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {95, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {96, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {97, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {98, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 248}}},
    {99, {{"presc", 13}, {"mult", 1}, {"dt_fifo_timeout", 564}}},
    {100, {{"presc", 13}, {"mult", 1}, {"dt_fifo_timeout", 564}}}};

std::map<const int, std::map<const std::string, const int>> saphir_25_stc_threshold_params = {
    {1, {{"presc", 11}, {"mult", 15}, {"dt_fifo_timeout", 32}}},
    {2, {{"presc", 10}, {"mult", 5}, {"dt_fifo_timeout", 27}}},
    {3, {{"presc", 12}, {"mult", 15}, {"dt_fifo_timeout", 27}}},
    {4, {{"presc", 12}, {"mult", 11}, {"dt_fifo_timeout", 39}}},
    {5, {{"presc", 11}, {"mult", 5}, {"dt_fifo_timeout", 27}}},
    {6, {{"presc", 9}, {"mult", 1}, {"dt_fifo_timeout", 35}}},
    {7, {{"presc", 12}, {"mult", 7}, {"dt_fifo_timeout", 41}}},
    {8, {{"presc", 11}, {"mult", 3}, {"dt_fifo_timeout", 49}}},
    {9, {{"presc", 13}, {"mult", 11}, {"dt_fifo_timeout", 54}}},
    {10, {{"presc", 12}, {"mult", 5}, {"dt_fifo_timeout", 59}}},
    {11, {{"presc", 12}, {"mult", 5}, {"dt_fifo_timeout", 27}}},
    {12, {{"presc", 13}, {"mult", 9}, {"dt_fifo_timeout", 31}}},
    {13, {{"presc", 10}, {"mult", 1}, {"dt_fifo_timeout", 35}}},
    {14, {{"presc", 14}, {"mult", 15}, {"dt_fifo_timeout", 38}}},
    {15, {{"presc", 13}, {"mult", 7}, {"dt_fifo_timeout", 41}}},
    {16, {{"presc", 14}, {"mult", 13}, {"dt_fifo_timeout", 45}}},
    {17, {{"presc", 14}, {"mult", 13}, {"dt_fifo_timeout", 45}}},
    {18, {{"presc", 12}, {"mult", 3}, {"dt_fifo_timeout", 49}}},
    {19, {{"presc", 14}, {"mult", 11}, {"dt_fifo_timeout", 54}}},
    {20, {{"presc", 14}, {"mult", 11}, {"dt_fifo_timeout", 54}}},
    {21, {{"presc", 13}, {"mult", 5}, {"dt_fifo_timeout", 59}}},
    {22, {{"presc", 13}, {"mult", 5}, {"dt_fifo_timeout", 59}}},
    {23, {{"presc", 14}, {"mult", 9}, {"dt_fifo_timeout", 67}}},
    {24, {{"presc", 14}, {"mult", 9}, {"dt_fifo_timeout", 67}}},
    {25, {{"presc", 11}, {"mult", 1}, {"dt_fifo_timeout", 155}}},
    {26, {{"presc", 11}, {"mult", 1}, {"dt_fifo_timeout", 75}}},
    {27, {{"presc", 11}, {"mult", 1}, {"dt_fifo_timeout", 75}}},
    {28, {{"presc", 15}, {"mult", 15}, {"dt_fifo_timeout", 81}}},
    {29, {{"presc", 15}, {"mult", 15}, {"dt_fifo_timeout", 81}}},
    {30, {{"presc", 14}, {"mult", 7}, {"dt_fifo_timeout", 87}}},
    {31, {{"presc", 14}, {"mult", 7}, {"dt_fifo_timeout", 87}}},
    {32, {{"presc", 15}, {"mult", 13}, {"dt_fifo_timeout", 94}}},
    {33, {{"presc", 15}, {"mult", 13}, {"dt_fifo_timeout", 94}}},
    {34, {{"presc", 15}, {"mult", 13}, {"dt_fifo_timeout", 94}}},
    {35, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 102}}},
    {36, {{"presc", 13}, {"mult", 3}, {"dt_fifo_timeout", 102}}},
    {37, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 228}}},
    {38, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 112}}},
    {39, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 112}}},
    {40, {{"presc", 15}, {"mult", 11}, {"dt_fifo_timeout", 112}}},
    {41, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 123}}},
    {42, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 123}}},
    {43, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 123}}},
    {44, {{"presc", 14}, {"mult", 5}, {"dt_fifo_timeout", 123}}},
    {45, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 280}}},
    {46, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 138}}},
    {47, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 138}}},
    {48, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 138}}},
    {49, {{"presc", 15}, {"mult", 9}, {"dt_fifo_timeout", 138}}},
    {50, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 315}}},
    {51, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 315}}},
    {52, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 155}}},
    {53, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 155}}},
    {54, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 155}}},
    {55, {{"presc", 12}, {"mult", 1}, {"dt_fifo_timeout", 155}}},
    {56, {{"presc", 16}, {"mult", 15}, {"dt_fifo_timeout", 166}}},
    {57, {{"presc", 16}, {"mult", 15}, {"dt_fifo_timeout", 166}}},
    {58, {{"presc", 16}, {"mult", 15}, {"dt_fifo_timeout", 166}}},
    {59, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 178}}},
    {60, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 178}}},
    {61, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 178}}},
    {62, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 178}}},
    {63, {{"presc", 15}, {"mult", 7}, {"dt_fifo_timeout", 178}}},
    {64, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 192}}},
    {65, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 192}}},
    {66, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 192}}},
    {67, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 192}}},
    {68, {{"presc", 16}, {"mult", 13}, {"dt_fifo_timeout", 192}}},
    {69, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 209}}},
    {70, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 209}}},
    {71, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 209}}},
    {72, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 209}}},
    {73, {{"presc", 14}, {"mult", 3}, {"dt_fifo_timeout", 209}}},
    {74, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 461}}},
    {75, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 228}}},
    {76, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 228}}},
    {77, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 228}}},
    {78, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 228}}},
    {79, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 228}}},
    {80, {{"presc", 16}, {"mult", 11}, {"dt_fifo_timeout", 228}}},
    {81, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 507}}},
    {82, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 251}}},
    {83, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 251}}},
    {84, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 251}}},
    {85, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 251}}},
    {86, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 251}}},
    {87, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 251}}},
    {88, {{"presc", 15}, {"mult", 5}, {"dt_fifo_timeout", 251}}},
    {89, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 564}}},
    {90, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 564}}},
    {91, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 564}}},
    {92, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 280}}},
    {93, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 280}}},
    {94, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 280}}},
    {95, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 280}}},
    {96, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 280}}},
    {97, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 280}}},
    {98, {{"presc", 16}, {"mult", 9}, {"dt_fifo_timeout", 280}}},
    {99, {{"presc", 13}, {"mult", 1}, {"dt_fifo_timeout", 635}}},
    {100, {{"presc", 13}, {"mult", 1}, {"dt_fifo_timeout", 635}}}};
} // namespace

EventTrailFilter::EventTrailFilter(const std::shared_ptr<RegisterMap> &register_map,
                                   const I_HW_Identification::SensorInfo &sensor_info,
                                   const std::string &sensor_prefix) :
    register_map_(register_map), sensor_prefix_(sensor_prefix) {
    if (sensor_info.name_ == "GenX320") {
        stc_prefix_   = "";
        trail_prefix_ = "";
        threshold_params_.insert(saphir_25_stc_threshold_params.begin(), saphir_25_stc_threshold_params.end());
        is_sensor_saphir = true;
    } else {
        stc_prefix_   = "stc_";
        trail_prefix_ = "trail_";
        threshold_params_.insert(gen41_stc_threshold_params.begin(), gen41_stc_threshold_params.end());
        is_sensor_saphir = false;
    }

    // Select available modes based on sensor type
    if (sensor_info.name_ == "Gen4.1") {
        // STC v1.1
        is_stc_improved = false;
        modes_          = {I_EventTrailFilterModule::Type::STC_CUT_TRAIL, I_EventTrailFilterModule::Type::TRAIL};
    } else {
        // STC v1.2
        is_stc_improved = true;
        modes_ = {I_EventTrailFilterModule::Type::STC_CUT_TRAIL, I_EventTrailFilterModule::Type::STC_KEEP_TRAIL,
                  I_EventTrailFilterModule::Type::TRAIL};
    }
}

std::set<I_EventTrailFilterModule::Type> EventTrailFilter::get_available_types() const {
    return modes_;
}

bool EventTrailFilter::set_threshold(uint32_t threshold) {
    if (threshold < get_min_supported_threshold() || threshold > get_max_supported_threshold()) {
        std::stringstream ss;
        ss << "Bad STC threshold value: " << threshold << ". Value should be in range ["
           << std::to_string(get_min_supported_threshold()) << ", " << std::to_string(get_max_supported_threshold())
           << "].";
        throw HalException(HalErrorCode::ValueOutOfRange, ss.str());
    }

    threshold_ms_ = std::roundf(threshold / 1000.0);

    // Reset if needed
    if (is_enabled()) {
        enable(false);
        enable(true);
    }

    return true;
}

bool EventTrailFilter::set_type(I_EventTrailFilterModule::Type type) {
    if (get_available_types().count(type) <= 0) {
        throw HalException(HalErrorCode::UnsupportedValue);
    }

    filtering_type_ = type;

    // Reset if needed
    if (is_enabled()) {
        enable(false);
        enable(true);
    }

    return true;
}

bool EventTrailFilter::enable(bool state) {
    // Bypass filter
    (*register_map_)[sensor_prefix_ + "stc/pipeline_control"].write_value(0b101);
    enabled_ = false;

    if (!state)
        return true; // To deactivate STC filter, we set it in bypass mode

    // Start sram init
    // Clear init flag
    (*register_map_)[sensor_prefix_ + "stc/initialization"][stc_prefix_ + "flag_init_done"].write_value(1);

    if (is_sensor_saphir) {
        // SRAM power up
        (*register_map_)[sensor_prefix_ + "sram_initn"]["ehc_stc_initn"].write_value(1);
        (*register_map_)[sensor_prefix_ + "sram_pd0"]["stc0_pd"].write_value(0);
    }

    (*register_map_)[sensor_prefix_ + "stc/initialization"][stc_prefix_ + "req_init"].write_value(1);

    // Setup new configuration
    if (filtering_type_ == I_EventTrailFilterModule::Type::STC_CUT_TRAIL ||
        filtering_type_ == I_EventTrailFilterModule::Type::STC_KEEP_TRAIL) {
        vfield fields = {{stc_prefix_ + "enable", 1}, {stc_prefix_ + "threshold", threshold_ms_ * 1000}};

        if (is_stc_improved) {
            fields.insert({"disable_stc_cut_trail", filtering_type_ == I_EventTrailFilterModule::Type::STC_KEEP_TRAIL});
        }
        (*register_map_)[sensor_prefix_ + "stc/stc_param"].write_value(fields);
        (*register_map_)[sensor_prefix_ + "stc/trail_param"][trail_prefix_ + "enable"].write_value(0);

    } else if (filtering_type_ == I_EventTrailFilterModule::Type::TRAIL) {
        (*register_map_)[sensor_prefix_ + "stc/stc_param"][stc_prefix_ + "enable"].write_value(0);
        (*register_map_)[sensor_prefix_ + "stc/trail_param"].write_value(
            vfield{{trail_prefix_ + "enable", 1}, {trail_prefix_ + "threshold", threshold_ms_ * 1000}});
    }

    vfield fields = {{"prescaler", threshold_params_[threshold_ms_]["presc"]},
                     {"multiplier", threshold_params_[threshold_ms_]["mult"]}};

    if (is_stc_improved) {
        fields.insert({"enable_last_ts_update_at_every_event", 1});
    }

    (*register_map_)[sensor_prefix_ + "stc/timestamping"].write_value(fields);
    (*register_map_)[sensor_prefix_ + "stc/invalidation"]["dt_fifo_timeout"].write_value(
        threshold_params_[threshold_ms_]["dt_fifo_timeout"]);

    // Check sram init done
    bool init_done = 0;
    for (int i = 0; i < 3; i++) {
        init_done = (*register_map_)[sensor_prefix_ + "stc/initialization"][stc_prefix_ + "flag_init_done"].read_value();
        if (init_done == 1) {
            break;
        }
    }
    if (init_done == 0) {
        throw HalException(HalErrorCode::InternalInitializationError, "Bad STC initialization");
    }

    // Enable filter
    (*register_map_)[sensor_prefix_ + "stc/pipeline_control"].write_value(0b001);
    enabled_ = true;

    return true;
}

bool EventTrailFilter::is_enabled() const {
    return enabled_;
}

I_EventTrailFilterModule::Type EventTrailFilter::get_type() const {
    return filtering_type_;
}

uint32_t EventTrailFilter::get_threshold() const {
    return threshold_ms_ * 1000;
}

uint32_t EventTrailFilter::get_max_supported_threshold() const {
    return 100000;
}

uint32_t EventTrailFilter::get_min_supported_threshold() const {
    return 1000;
}


} // namespace Metavision
